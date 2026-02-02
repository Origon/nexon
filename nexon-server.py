#!/usr/bin/env python3
"""Nexon MLX Server - proxy with token filtering"""

import http.server
import json
import re
import socketserver
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    mlx_port = 8080
    model_name = None

    def __init__(self, *args, web_dir=None, **kwargs):
        self.web_dir = web_dir
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/api/health":
            self.health_check()
        elif self.path == "/v1/models":
            self.proxy_models()
        else:
            super().do_GET()

    def proxy_models(self):
        """Proxy /v1/models for OpenAI API compatibility"""
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.mlx_port}/v1/models",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                # Use OpenAI-compatible model names that Cursor will accept
                # The actual model name doesn't matter - MLX uses whatever is loaded
                for i, model in enumerate(data.get("data", [])):
                    original_id = model["id"]
                    simple_name = original_id.split("/")[-1]
                    # Use gpt-4o as the model ID (Cursor accepts this)
                    # Store original name in a custom field for reference
                    model["id"] = "gpt-4o" if i == 0 else f"gpt-4o-mini"
                    model["name"] = simple_name  # Keep original for display
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def health_check(self):
        """Check if MLX server is ready"""
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.mlx_port}/v1/models",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                resp.read()  # Consume response to confirm server is ready
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ready": True,
                    "model": self.model_name or "unknown"
                }).encode())
        except Exception:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ready": False,
                "message": "Model is loading..."
            }).encode())

    def do_POST(self):
        if self.path == "/api/v1/chat/completions/stream":
            self.proxy_stream()
        elif self.path == "/v1/chat/completions":
            self.proxy_chat_completions()
        else:
            self.send_error(404)

    def proxy_chat_completions(self):
        """Proxy /v1/chat/completions for OpenAI API compatibility (Cursor, etc.)"""
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.loads(body)

            # Set reasonable defaults
            if "max_tokens" not in data:
                data["max_tokens"] = 8192

            # Replace model name with actual loaded model (MLX requires exact name)
            if self.model_name:
                data["model"] = self.model_name

            is_streaming = data.get("stream", False)

            req = urllib.request.Request(
                f"http://127.0.0.1:{self.mlx_port}/v1/chat/completions",
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req) as resp:
                if is_streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    # Stream through directly
                    for line in resp:
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    # Non-streaming response
                    response_data = resp.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(response_data)

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def proxy_stream(self):
        """Proxy to mlx_lm, filtering special tokens and converting to markdown"""
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.loads(body)
            data["stream"] = True
            # Ensure high token limit for reasoning models (they need tokens for thinking + response)
            if "max_tokens" not in data:
                data["max_tokens"] = 32768

            # Add system prompt for ChatGPT-like formatting
            system_prompt = {
                "role": "system",
                "content": """You are a helpful, knowledgeable assistant. Be direct and conversational.

Formatting rules:
- Short answers: Just respond naturally, no special formatting needed
- Explanations: Use **bold** for key terms, break into paragraphs
- Lists/steps: Use em dashes (—) for items, not bullets or numbers
- Code: Always use ```language blocks with the language specified
- Complex topics: Use ## headers to organize sections

Style:
- Be concise. Don't pad responses with unnecessary words
- Answer the actual question first, then elaborate if helpful
- Use examples when they clarify
- Acknowledge uncertainty when you don't know something"""
            }
            if data.get("messages") and data["messages"][0].get("role") != "system":
                data["messages"].insert(0, system_prompt)

            req = urllib.request.Request(
                f"http://127.0.0.1:{self.mlx_port}/v1/chat/completions",
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req) as resp:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                thinking_done = False
                has_explicit_reasoning = False
                content_received = False
                parser = HarmonyParser()

                for line in resp:
                    line = line.decode()
                    if not line.startswith("data: "):
                        continue

                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        # Flush any remaining buffer
                        flushed = parser.flush()
                        if flushed["thinking"]:
                            self.wfile.write(f'data: {{"choices":[{{"delta":{{"thinking":{json.dumps(flushed["thinking"])}}}}}]}}\n\n'.encode())
                        if flushed["content"]:
                            clean = self._clean_special_tokens(flushed["content"])
                            if clean:
                                self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":{json.dumps(clean)}}}}}]}}\n\n'.encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break

                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning", "")

                        # Handle explicit reasoning field (some models use this)
                        if reasoning:
                            has_explicit_reasoning = True
                            self.wfile.write(f'data: {{"choices":[{{"delta":{{"thinking":{json.dumps(reasoning)}}}}}]}}\n\n'.encode())
                            self.wfile.flush()

                        # Handle content
                        if content:
                            content_received = True
                            # For models using explicit reasoning field, pass content directly
                            if has_explicit_reasoning:
                                if not thinking_done:
                                    self.wfile.write(b'data: {"choices":[{"delta":{"thinking_done":true}}]}\n\n')
                                    self.wfile.flush()
                                    thinking_done = True
                                clean = self._clean_special_tokens(content)
                                if clean:
                                    self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":{json.dumps(clean)}}}}}]}}\n\n'.encode())
                                    self.wfile.flush()
                            else:
                                # Parse content for inline reasoning (Harmony, <think> tags, etc.)
                                parsed = parser.parse(content)

                                # Stream thinking tokens
                                if parsed["thinking"]:
                                    self.wfile.write(f'data: {{"choices":[{{"delta":{{"thinking":{json.dumps(parsed["thinking"])}}}}}]}}\n\n'.encode())
                                    self.wfile.flush()

                                # Signal thinking is done
                                if parsed["thinking_done"] and not thinking_done:
                                    self.wfile.write(b'data: {"choices":[{"delta":{"thinking_done":true}}]}\n\n')
                                    self.wfile.flush()
                                    thinking_done = True

                                # Stream content
                                if parsed["content"]:
                                    # Signal thinking done for plain format (no explicit marker)
                                    if not thinking_done and parser.format == "plain":
                                        self.wfile.write(b'data: {"choices":[{"delta":{"thinking_done":true}}]}\n\n')
                                        self.wfile.flush()
                                        thinking_done = True

                                    clean = self._clean_special_tokens(parsed["content"])
                                    if clean:
                                        self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":{json.dumps(clean)}}}}}]}}\n\n'.encode())
                                        self.wfile.flush()

                        # Handle finish
                        if chunk.get("choices", [{}])[0].get("finish_reason"):
                            # If model only output reasoning but no content, show a fallback message
                            if has_explicit_reasoning and not content_received:
                                if not thinking_done:
                                    self.wfile.write(b'data: {"choices":[{"delta":{"thinking_done":true}}]}\n\n')
                                    self.wfile.flush()
                                # Model finished without producing content - output a fallback
                                fallback = "*The model finished thinking but did not produce a response. Try rephrasing your question or using a different model.*"
                                self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":{json.dumps(fallback)}}}}}]}}\n\n'.encode())
                                self.wfile.flush()
                            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
                            self.wfile.write(b'data: [DONE]\n\n')
                            self.wfile.flush()
                            return

                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _clean_special_tokens(self, text):
        """Remove special tokens from text"""
        # Remove any <|...|> style tokens
        text = re.sub(r'<\|[^|]*\|>', '', text)
        # Remove incomplete tags at the end
        if '<|' in text:
            text = text.split('<|')[0]
        return text


class HarmonyParser:
    """
    Unified parser for different reasoning model output formats:
    - Channel tags (gpt-oss): <|channel|>analysis<|message|>[thinking]<|end|>...<|channel|>final<|message|>[response]
    - Harmony (gpt-oss legacy): analysis...assistantfinal...
    - Think tags (Nemotron, DeepSeek): <think>...</think>
    - Plain content (non-reasoning models)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = ""
        self.mode = "detect"  # detect, wait_thinking, thinking, wait_content, content
        self.format = None    # channel_tags, harmony, think_tags, plain

    def parse(self, chunk: str) -> dict:
        """
        Parse a chunk of streamed content.
        Returns: {"thinking": str, "content": str, "thinking_done": bool}
        """
        result = {"thinking": "", "content": "", "thinking_done": False}

        if not chunk:
            return result

        self.buffer += chunk

        # Detection phase - figure out the format
        if self.mode == "detect":
            # Check for gpt-oss channel format: <|channel|>analysis<|message|>[thinking]<|end|>...<|channel|>final<|message|>[response]
            if "<|channel|>" in self.buffer:
                self.format = "channel_tags"
                self.mode = "wait_thinking"
                self.buffer = ""
                return result

            # Check for Harmony format (starts with "analysis") - legacy support
            if self.buffer.lower().startswith("analysis"):
                self.format = "harmony"
                self.mode = "thinking"
                # Remove "analysis" prefix and send rest as thinking
                thinking = re.sub(r'^analysis\s*', '', self.buffer, flags=re.IGNORECASE)
                result["thinking"] = thinking
                self.buffer = ""
                return result

            # Check for <think> tag (Nemotron, DeepSeek, Qwen)
            if "<think>" in self.buffer:
                self.format = "think_tags"
                self.mode = "thinking"
                # Extract content after <think>
                parts = self.buffer.split("<think>", 1)
                if parts[0]:
                    result["content"] = parts[0]
                if len(parts) > 1:
                    result["thinking"] = parts[1]
                self.buffer = ""
                return result

            # If we have enough content and no markers, assume plain
            if len(self.buffer) > 20:
                self.format = "plain"
                self.mode = "content"
                result["content"] = self.buffer
                self.buffer = ""
                return result

            # Still detecting, buffer more
            return result

        # Channel tags format (gpt-oss): <|channel|>analysis<|message|>[thinking]<|end|>...<|channel|>final<|message|>[response]
        if self.format == "channel_tags":
            # State: wait_thinking - waiting for first <|message|> which starts thinking
            if self.mode == "wait_thinking":
                if "<|message|>" in self.buffer:
                    parts = self.buffer.split("<|message|>", 1)
                    self.mode = "thinking"
                    self.buffer = parts[1] if len(parts) > 1 else ""
                    # Process any thinking content that came with this chunk
                    if self.buffer:
                        return self.parse("")  # Re-process with thinking mode
                else:
                    self.buffer = ""  # Discard preamble (analysis, etc.)
                return result

            # State: thinking - accumulating thinking until <|end|> or <|channel|>final
            if self.mode == "thinking":
                # Check for end of thinking
                end_markers = ["<|end|>", "<|channel|>"]
                end_pos = -1
                end_len = 0

                for marker in end_markers:
                    pos = self.buffer.find(marker)
                    if pos != -1 and (end_pos == -1 or pos < end_pos):
                        end_pos = pos
                        end_len = len(marker)

                if end_pos != -1:
                    # Found end of thinking
                    result["thinking"] = self.buffer[:end_pos]
                    result["thinking_done"] = True
                    self.mode = "wait_content"
                    self.buffer = self.buffer[end_pos + end_len:]
                else:
                    # Keep trailing chars to catch markers split across chunks
                    marker_len = 10  # "<|channel|>" is 11 chars
                    if len(self.buffer) > marker_len:
                        result["thinking"] = self.buffer[:-marker_len]
                        self.buffer = self.buffer[-marker_len:]
                return result

            # State: wait_content - waiting for <|message|> after <|channel|>final
            if self.mode == "wait_content":
                if "<|message|>" in self.buffer:
                    parts = self.buffer.split("<|message|>", 1)
                    self.mode = "content"
                    self.buffer = parts[1] if len(parts) > 1 else ""
                    if self.buffer:
                        return self.parse("")  # Re-process with content mode
                else:
                    self.buffer = ""  # Discard intermediate tokens
                return result

            # State: content - streaming response
            if self.mode == "content":
                result["content"] = self.buffer
                self.buffer = ""
                return result

            return result

        # Harmony format: look for "assistantfinal" marker
        if self.format == "harmony":
            if self.mode == "thinking":
                if "assistantfinal" in self.buffer.lower():
                    parts = re.split(r'assistantfinal', self.buffer, flags=re.IGNORECASE)
                    result["thinking"] = parts[0]
                    result["thinking_done"] = True
                    self.mode = "content"
                    if len(parts) > 1:
                        result["content"] = parts[1]
                    self.buffer = ""
                else:
                    # Keep trailing chars in buffer to catch markers split across chunks
                    # "assistantfinal" is 14 chars, keep 13 as safety margin
                    marker_len = 13
                    if len(self.buffer) > marker_len:
                        result["thinking"] = self.buffer[:-marker_len]
                        self.buffer = self.buffer[-marker_len:]
                    # else: keep buffering, don't emit yet
            else:
                result["content"] = self.buffer
                self.buffer = ""
            return result

        # Think tags format: look for </think>
        if self.format == "think_tags":
            if self.mode == "thinking":
                if "</think>" in self.buffer:
                    parts = self.buffer.split("</think>", 1)
                    result["thinking"] = parts[0]
                    result["thinking_done"] = True
                    self.mode = "content"
                    if len(parts) > 1:
                        result["content"] = parts[1]
                    self.buffer = ""
                else:
                    # Keep trailing chars in buffer to catch </think> split across chunks
                    # "</think>" is 8 chars, keep 7 as safety margin
                    marker_len = 7
                    if len(self.buffer) > marker_len:
                        result["thinking"] = self.buffer[:-marker_len]
                        self.buffer = self.buffer[-marker_len:]
                    # else: keep buffering, don't emit yet
            else:
                result["content"] = self.buffer
                self.buffer = ""
            return result

        # Plain format
        result["content"] = self.buffer
        self.buffer = ""
        return result

    def flush(self) -> dict:
        """Flush any remaining buffer"""
        result = {"thinking": "", "content": "", "thinking_done": False}
        if self.buffer:
            if self.mode in ("thinking", "wait_thinking"):
                result["thinking"] = self.buffer
            elif self.mode in ("content", "wait_content"):
                result["content"] = self.buffer
            else:
                result["content"] = self.buffer
            self.buffer = ""
        return result


class NexonHandler(Handler):
    """Handler with CORS support"""

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[REQUEST] {args[0] if args else format}")


def run_server(mlx_port: int, web_port: int, web_dir: str, model: str = None):
    """Run the web server with proxy"""
    NexonHandler.mlx_port = mlx_port
    NexonHandler.model_name = model

    handler = lambda *a, **k: NexonHandler(*a, web_dir=web_dir, **k)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", web_port), handler) as srv:
        print(f"Web server running on http://localhost:{web_port}")
        srv.serve_forever()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mlx-port", type=int, default=8080)
    p.add_argument("--web-port", type=int, default=3000)
    p.add_argument("--web-dir", type=str, default=str(SCRIPT_DIR / "web"))
    p.add_argument("--model", type=str, default=None, help="Model name to display")
    args = p.parse_args()

    run_server(args.mlx_port, args.web_port, args.web_dir, args.model)
