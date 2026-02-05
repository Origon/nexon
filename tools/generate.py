#!/usr/bin/env python3
"""Generate text using a model directly (no server needed)"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a model directly (no server needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./tools/generate.py -m mlx-community/Qwen3-8B-4bit "Hello, how are you?"
  ./tools/generate.py -m ~/models/my-model -t 512 --temp 0.5 "Write a poem about coding"
  ./tools/generate.py -m ~/models/base --adapter ./adapters/my-adapter "Hello!"
"""
    )

    # Required
    parser.add_argument("-m", "--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("prompt", help="The prompt text")

    # Options
    parser.add_argument("-t", "--tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--temp", default="0.7", help="Temperature (default: 0.7)")
    parser.add_argument("--top-p", default="0.9", help="Top-p sampling (default: 0.9)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--adapter", help="LoRA adapter path (use without fusing)")

    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", args.model,
        "--prompt", args.prompt,
        "--max-tokens", str(args.tokens),
        "--temp", args.temp,
        "--top-p", args.top_p,
    ]

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.adapter:
        cmd.extend(["--adapter-path", args.adapter])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
