#!/usr/bin/env python3
"""Nexon MLX Server - proxy with token filtering"""

import http.server
import json
import socketserver
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    mlx_port = 8080

    def __init__(self, *args, web_dir=None, **kwargs):
        self.web_dir = web_dir
        super().__init__(*args, directory=web_dir, **kwargs)

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

                buf = ""
                in_final = False

                for line in resp:
                    line = line.decode()
                    if not line.startswith("data: "):
                        continue

                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        if buf:
                            clean = buf.split("<|")[0]
                            if clean:
                                self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":{json.dumps(clean)}}}}}]}}\n\n'.encode())
                                self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break

                    try:
                        chunk = json.loads(payload)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

                        if content:
                            buf += content

                            # Wait for final channel marker
                            if "<|channel|>final<|message|>" in buf:
                                in_final = True
                                buf = buf.split("<|channel|>final<|message|>")[-1]

                            # Stream content once in final channel (no pending special tokens)
                            if in_final and buf and "<|" not in buf:
                                chunk["choices"][0]["delta"]["content"] = buf
                                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                                self.wfile.flush()
                                buf = ""

                        # Handle finish
                        if chunk.get("choices", [{}])[0].get("finish_reason"):
                            if buf:
                                clean = buf.split("<|")[0]
                                if clean:
                                    self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":{json.dumps(clean)}}}}}]}}\n\n'.encode())
                                    self.wfile.flush()
                            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
                            self.wfile.write(b'data: [DONE]\n\n')
                            self.wfile.flush()
                            return  # Exit after finish

                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

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


def run_server(mlx_port: int, web_port: int, web_dir: str):
    """Run the web server with proxy"""
    Handler.mlx_port = mlx_port

    handler = lambda *a, **k: Handler(*a, web_dir=web_dir, **k)

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
    args = p.parse_args()

    run_server(args.mlx_port, args.web_port, args.web_dir)
