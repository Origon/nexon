#!/usr/bin/env python3
"""
Nexon - Local LLM server using MLX with OpenAI-compatible API

Single-file server that:
- Loads MLX models directly (no proxy)
- Provides OpenAI-compatible /v1/chat/completions endpoint
- Parses reasoning tokens (<think>, Harmony, channel tags)
- Converts tool calls to OpenAI format
- Serves web UI
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Generator, Optional

from mlx_lm import load, stream_generate

# FastAPI imports
try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install fastapi uvicorn")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PID_FILE = SCRIPT_DIR / ".nexon.pid"

# =============================================================================
# Configuration (edit these defaults)
# =============================================================================
PORT = 3000
CONTEXT = 16384  # Max context length (tokens)


# =============================================================================
# Model Management
# =============================================================================

class ModelManager:
    """Manages MLX model loading and inference"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path: str = ""
        self.max_kv_size: int = 16384

    def load(self, model_path: str, adapter_path: Optional[str] = None, max_kv_size: int = 16384):
        """Load model and tokenizer"""
        self.model_path = model_path
        self.max_kv_size = max_kv_size

        print(f"Loading model: {model_path}")
        if adapter_path:
            print(f"With adapter: {adapter_path}")

        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        print("Model loaded successfully")

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def name(self) -> str:
        return self.model_path.split("/")[-1] if self.model_path else "unknown"

    def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """Stream tokens from the model"""
        # Normalize messages - ensure content is always a string
        normalized = []
        for msg in messages:
            content = msg.get("content", "")
            # Handle OpenAI multimodal format where content can be a list
            if isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)
            normalized.append({"role": msg.get("role", "user"), "content": content})

        # Build prompt using tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True
        )

        # Stream generate
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            max_kv_size=self.max_kv_size,
        ):
            if response.text:
                yield response.text

            if response.finish_reason:
                break


# Global model manager
model_manager = ModelManager()


# =============================================================================
# Reasoning Parser
# =============================================================================

class ReasoningParser:
    """
    Parses different reasoning output formats:
    - Channel tags: <|channel|>analysis<|message|>...<|channel|>final<|message|>...
    - Harmony: analysis...assistantfinal...
    - Think tags: <think>...</think>
    - Plain content
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = ""
        self.mode = "detect"  # detect, thinking, content
        self.format = None    # channel_tags, harmony, think_tags, plain

    def parse(self, chunk: str) -> dict:
        """Parse a chunk, return {thinking: str, content: str, thinking_done: bool}"""
        result = {"thinking": "", "content": "", "thinking_done": False}

        if not chunk:
            return result

        self.buffer += chunk

        # Detection phase
        if self.mode == "detect":
            # Channel tags format
            if "<|channel|>" in self.buffer:
                self.format = "channel_tags"
                self.mode = "thinking"
                # Find start of thinking content
                if "<|message|>" in self.buffer:
                    parts = self.buffer.split("<|message|>", 1)
                    self.buffer = parts[1] if len(parts) > 1 else ""
                else:
                    self.buffer = ""
                return result

            # Harmony format (require delimiter to avoid false positives on "Analysis of...")
            if re.match(r'^analysis\s*[\n:]', self.buffer, re.IGNORECASE):
                self.format = "harmony"
                self.mode = "thinking"
                thinking = re.sub(r'^analysis\s*[\n:]?\s*', '', self.buffer, flags=re.IGNORECASE)
                result["thinking"] = thinking
                self.buffer = ""
                return result

            # Harmony format - model skipped analysis, went straight to final
            if self.buffer.lower().startswith("assistantfinal"):
                self.format = "harmony"
                self.mode = "content"
                result["content"] = re.sub(r'^assistantfinal\s*', '', self.buffer, flags=re.IGNORECASE)
                result["thinking_done"] = True
                self.buffer = ""
                return result

            # Think tags
            if "<think>" in self.buffer:
                self.format = "think_tags"
                self.mode = "thinking"
                parts = self.buffer.split("<think>", 1)
                if parts[0]:
                    result["content"] = parts[0]
                if len(parts) > 1:
                    result["thinking"] = parts[1]
                self.buffer = ""
                return result

            # Assume plain after enough content
            if len(self.buffer) > 20:
                self.format = "plain"
                self.mode = "content"
                result["content"] = self.buffer
                result["thinking_done"] = True
                self.buffer = ""
                return result

            return result

        # Channel tags processing
        if self.format == "channel_tags":
            if self.mode == "thinking":
                end_markers = ["<|end|>", "<|channel|>"]
                end_pos = -1

                for marker in end_markers:
                    pos = self.buffer.find(marker)
                    if pos != -1 and (end_pos == -1 or pos < end_pos):
                        end_pos = pos

                if end_pos != -1:
                    result["thinking"] = self.buffer[:end_pos]
                    result["thinking_done"] = True
                    self.mode = "content"
                    self.buffer = self.buffer[end_pos:]
                    # Skip to content
                    if "<|message|>" in self.buffer:
                        self.buffer = self.buffer.split("<|message|>", 1)[-1]
                else:
                    # Keep buffer for marker detection
                    if len(self.buffer) > 10:
                        result["thinking"] = self.buffer[:-10]
                        self.buffer = self.buffer[-10:]
            else:
                result["content"] = self.buffer
                self.buffer = ""
            return result

        # Harmony processing
        if self.format == "harmony":
            if self.mode == "thinking":
                if "assistantfinal" in self.buffer.lower():
                    parts = re.split(r'assistantfinal', self.buffer, flags=re.IGNORECASE)
                    result["thinking"] = parts[0]
                    result["thinking_done"] = True
                    self.mode = "content"
                    result["content"] = parts[1] if len(parts) > 1 else ""
                    self.buffer = ""
                else:
                    if len(self.buffer) > 13:
                        result["thinking"] = self.buffer[:-13]
                        self.buffer = self.buffer[-13:]
            else:
                result["content"] = self.buffer
                self.buffer = ""
            return result

        # Think tags processing
        if self.format == "think_tags":
            if self.mode == "thinking":
                if "</think>" in self.buffer:
                    parts = self.buffer.split("</think>", 1)
                    result["thinking"] = parts[0]
                    result["thinking_done"] = True
                    self.mode = "content"
                    result["content"] = parts[1] if len(parts) > 1 else ""
                    self.buffer = ""
                else:
                    if len(self.buffer) > 7:
                        result["thinking"] = self.buffer[:-7]
                        self.buffer = self.buffer[-7:]
            else:
                result["content"] = self.buffer
                self.buffer = ""
            return result

        # Plain format
        result["content"] = self.buffer
        self.buffer = ""
        return result

    def flush(self) -> dict:
        """Flush remaining buffer"""
        result = {"thinking": "", "content": "", "thinking_done": False}
        if self.buffer:
            if self.mode == "thinking":
                result["thinking"] = self.buffer
            else:
                result["content"] = self.buffer
            self.buffer = ""
        return result


# =============================================================================
# Tool Call Parser
# =============================================================================

def parse_tool_calls(content: str) -> tuple[str, list]:
    """
    Parse <tool_call>...</tool_call> from content.
    Returns (remaining_content, tool_calls_list)
    """
    tool_calls = []
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        try:
            call_data = json.loads(match)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": call_data.get("name", ""),
                    "arguments": json.dumps(call_data.get("arguments", {}))
                }
            })
        except json.JSONDecodeError:
            continue

    remaining = re.sub(pattern, '', content, flags=re.DOTALL).strip()
    return remaining, tool_calls


def tools_to_system_prompt(tools: list) -> str:
    """Convert OpenAI tools format to system prompt"""
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            param_lines = []
            props = params.get("properties", {})
            required = params.get("required", [])
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                req = " (required)" if param_name in required else ""
                param_lines.append(f"    - {param_name}: {param_type}{req} - {param_desc}")

            params_str = "\n".join(param_lines) if param_lines else "    (no parameters)"
            tool_descriptions.append(f"- **{name}**: {desc}\n  Parameters:\n{params_str}")

    return f"""You have access to the following tools. To use a tool, respond with:

<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Available tools:

{chr(10).join(tool_descriptions)}

When you need to perform an action, USE THE TOOLS."""


def clean_special_tokens(text: str) -> str:
    """Remove special tokens like <|...|> and Harmony format markers"""
    text = re.sub(r'<\|[^|]*\|>', '', text)
    if '<|' in text:
        text = text.split('<|')[0]
    # Strip Harmony format markers (for /v1/chat/completions endpoint)
    text = re.sub(r'^assistantfinal\s*', '', text, flags=re.IGNORECASE)
    return text


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle"""
    yield

app = FastAPI(title="Nexon", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check for web UI"""
    return {"ready": model_manager.ready, "model": model_manager.model_path}


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list (spoofed for Cursor compatibility)"""
    return {
        "object": "list",
        "data": [{
            "id": "gpt-4o",
            "object": "model",
            "name": model_manager.name,
            "owned_by": "nexon"
        }]
    }


@app.post("/api/v1/chat/completions/stream")
async def chat_stream_webui(request: Request):
    """Streaming endpoint for web UI with reasoning parsing"""
    data = await request.json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 8192)

    # Add system prompt for formatting
    system_prompt = {
        "role": "system",
        "content": """You are a helpful assistant. Be direct and conversational.

Formatting: Use **bold** for key terms. Use ```language blocks for code. Use ## headers for complex topics."""
    }
    if messages and messages[0].get("role") != "system":
        messages.insert(0, system_prompt)

    async def generate():
        parser = ReasoningParser()
        thinking_done_sent = False

        for token in model_manager.generate_stream(messages, max_tokens):
            parsed = parser.parse(token)

            # Send thinking
            if parsed["thinking"]:
                chunk = {"choices": [{"delta": {"thinking": parsed["thinking"]}}]}
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send thinking_done
            if parsed["thinking_done"] and not thinking_done_sent:
                chunk = {"choices": [{"delta": {"thinking_done": True}}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                thinking_done_sent = True

            # Send content
            if parsed["content"]:
                if not thinking_done_sent and parser.format == "plain":
                    chunk = {"choices": [{"delta": {"thinking_done": True}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    thinking_done_sent = True

                clean = clean_special_tokens(parsed["content"])
                if clean:
                    chunk = {"choices": [{"delta": {"content": clean}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"

        # Flush remaining
        flushed = parser.flush()
        if flushed["thinking"]:
            chunk = {"choices": [{"delta": {"thinking": flushed["thinking"]}}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        if flushed["content"]:
            clean = clean_special_tokens(flushed["content"])
            if clean:
                chunk = {"choices": [{"delta": {"content": clean}}]}
                yield f"data: {json.dumps(chunk)}\n\n"

        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions for Cursor/IDEs"""
    data = await request.json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 4096)
    stream = data.get("stream", False)

    # Handle tools
    tools = data.pop("tools", None)
    data.pop("tool_choice", None)

    if tools:
        tools_prompt = tools_to_system_prompt(tools)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = tools_prompt + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": tools_prompt})

    if stream:
        async def generate():
            message_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            accumulated = ""
            prefix_checked = False

            for token in model_manager.generate_stream(messages, max_tokens):
                accumulated += token

                if not tools:
                    # Buffer until we can check for Harmony prefix
                    if not prefix_checked:
                        if len(accumulated) >= 14:  # len("assistantfinal")
                            prefix_checked = True
                            # Strip Harmony prefix if present
                            cleaned = re.sub(r'^assistantfinal\s*', '', accumulated, flags=re.IGNORECASE)
                            cleaned = clean_special_tokens(cleaned)
                            if cleaned:
                                chunk = {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [{"index": 0, "delta": {"content": cleaned}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        continue

                    # After prefix check, stream directly
                    clean = clean_special_tokens(token)
                    if clean:
                        chunk = {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": clean}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            # Flush any remaining buffered content (for short responses < 14 chars)
            if not tools and not prefix_checked and accumulated:
                cleaned = re.sub(r'^assistantfinal\s*', '', accumulated, flags=re.IGNORECASE)
                cleaned = clean_special_tokens(cleaned)
                if cleaned:
                    chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": cleaned}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            if tools:
                # Parse tool calls at end
                remaining, tool_calls = parse_tool_calls(accumulated)
                remaining = clean_special_tokens(remaining)

                if remaining:
                    chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": remaining}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                if tool_calls:
                    chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": "tool_calls"}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            finish_reason = "tool_calls" if (tools and tool_calls) else "stop"
            yield f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{finish_reason}"}}]}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Non-streaming
        content = ""
        for token in model_manager.generate_stream(messages, max_tokens):
            content += token

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": clean_special_tokens(content)},
                "finish_reason": "stop"
            }]
        }

        if tools:
            remaining, tool_calls = parse_tool_calls(content)
            response["choices"][0]["message"]["content"] = clean_special_tokens(remaining)
            if tool_calls:
                response["choices"][0]["message"]["tool_calls"] = tool_calls
                response["choices"][0]["finish_reason"] = "tool_calls"

        return JSONResponse(response)


# Mount static files for web UI (must be last)
web_dir = SCRIPT_DIR / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")


# =============================================================================
# CLI
# =============================================================================

def start_server(model: str, adapter: Optional[str], console: bool = False):
    """Start the server (background by default, foreground with console=True)"""
    # Stop existing process if running
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Stopping existing Nexon (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except OSError:
            pass
        PID_FILE.unlink(missing_ok=True)

    # Validate model path
    model_path = Path(model).expanduser()
    if model_path.exists():
        if not (model_path / "config.json").exists():
            print(f"Error: Invalid model directory (missing config.json)")
            sys.exit(1)
        model = str(model_path)

    if console:
        # Foreground mode - run directly
        run_server(model, adapter)
    else:
        # Background mode - spawn new process (avoids fork issues with Metal/GPU)
        cmd = [sys.executable, "-u", __file__, "start", "-m", model, "-c"]
        if adapter:
            cmd.extend(["-a", adapter])

        log_file = SCRIPT_DIR / ".nexon.log"
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
        )

        # Wait for process to start and write PID file (MLX import is slow)
        print("Starting Nexon...", end="", flush=True)
        for _ in range(10):  # Wait up to 10 seconds
            time.sleep(1)
            print(".", end="", flush=True)
            if PID_FILE.exists():
                break
        print()

        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            print(f"Nexon started in background (PID: {pid})")
            print(f"  Model: {model}")
            print(f"  Context: {CONTEXT} tokens")
            print(f"  Web: http://localhost:{PORT}")
            print(f"\nModel loading in background. Stop with: ./nexon.py stop")
        else:
            print("Failed to start Nexon")
            print(f"Check log: {SCRIPT_DIR / '.nexon.log'}")


def run_server(model: str, adapter: Optional[str]):
    """Actually run the server (called by start_server)"""
    # Save PID immediately so parent knows we started
    PID_FILE.write_text(str(os.getpid()))

    # Load model
    model_manager.load(model, adapter, CONTEXT)

    # Handle shutdown
    def shutdown(sig, frame):
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"\nNexon started!")
    print(f"  Model: {model}")
    if adapter:
        print(f"  Adapter: {adapter}")
    print(f"  Context: {CONTEXT} tokens")
    print(f"  API: http://localhost:{PORT}")
    print(f"  Web: http://localhost:{PORT}")
    print(f"\nPress Ctrl+C to stop")

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def stop_server():
    """Stop the server"""
    if not PID_FILE.exists():
        print("Nexon is not running")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped Nexon (PID: {pid})")
    except OSError:
        print("Nexon process not found")
    finally:
        PID_FILE.unlink(missing_ok=True)


def show_status():
    """Show server status"""
    if not PID_FILE.exists():
        print("Nexon is not running")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        print(f"Nexon is running (PID: {pid})")
    except OSError:
        print("Nexon is not running (stale PID file)")
        PID_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Nexon - Local LLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration (edit nexon.py):
  PORT = {PORT}
  CONTEXT = {CONTEXT}

Examples:
  ./nexon.py -m ~/models/my-model            # Start in background
  ./nexon.py -m ~/models/my-model -c         # Start in foreground (console)
  ./nexon.py stop                            # Stop server
  ./nexon.py status                          # Check status
"""
    )
    parser.add_argument("command", nargs="?", choices=["start", "stop", "restart", "status"],
                       help="Command to run (default: start if -m provided)")
    parser.add_argument("-m", "--model", help="Model path or HuggingFace ID")
    parser.add_argument("-a", "--adapter", help="LoRA adapter path")
    parser.add_argument("-c", "--console", action="store_true",
                       help="Run in foreground (console mode)")

    args = parser.parse_args()

    # Default to start if -m provided
    if not args.command:
        if args.model:
            args.command = "start"
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "start":
        if not args.model:
            print("Error: --model is required for start")
            print("Usage: ./nexon.py start -m <model>")
            sys.exit(1)
        start_server(args.model, args.adapter, args.console)

    elif args.command == "stop":
        stop_server()

    elif args.command == "restart":
        stop_server()
        time.sleep(1)
        if args.model:
            start_server(args.model, args.adapter, args.console)
        else:
            print("Note: Use -m to specify model for restart")

    elif args.command == "status":
        show_status()


if __name__ == "__main__":
    main()
