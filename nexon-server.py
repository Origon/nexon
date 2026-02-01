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
        else:
            super().do_GET()

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
        else:
            self.send_error(404)

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
    - Harmony (gpt-oss): analysis...assistantfinal...
    - Think tags (Nemotron, DeepSeek): <think>...</think>
    - Plain content (non-reasoning models)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = ""
        self.mode = "detect"  # detect, thinking, content
        self.format = None    # harmony, think_tags, plain

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
            # Check for Harmony format (starts with "analysis")
            if self.buffer.lower().startswith("analysis"):
                self.format = "harmony"
                self.mode = "thinking"
                # Remove "analysis" prefix and send rest as thinking
                thinking = re.sub(r'^analysis\s*', '', self.buffer, flags=re.IGNORECASE)
                result["thinking"] = thinking
                self.buffer = ""
                return result

            # Check for <think> tag
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
                    result["thinking"] = self.buffer
                    self.buffer = ""
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
                    result["thinking"] = self.buffer
                    self.buffer = ""
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
            if self.mode == "thinking":
                result["thinking"] = self.buffer
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
        pass  # Quiet


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
