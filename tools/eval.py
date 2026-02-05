#!/usr/bin/env python3
"""Evaluate model perplexity on a test dataset"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model perplexity on a test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: The data directory must contain a file named 'test.jsonl'.
      You can copy or symlink your test file: ln -s mydata.jsonl ./data/test.jsonl

Examples:
  ./tools/eval.py -m mlx-community/Qwen3-8B-4bit -d ./data
  ./tools/eval.py -m ~/models/base -a ./my-adapter -d ./data
"""
    )

    # Required
    parser.add_argument("-m", "--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("-d", "--data", required=True, help="Data directory (must contain test.jsonl)")

    # Options
    parser.add_argument("-a", "--adapter", help="LoRA adapter path (optional)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")

    args = parser.parse_args()

    # Check if test.jsonl exists
    test_file = Path(args.data) / "test.jsonl"
    if not test_file.exists():
        print(f"Error: {test_file} not found")
        print("The data directory must contain a file named 'test.jsonl'")
        sys.exit(1)

    print("Evaluating model...")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    if args.adapter:
        print(f"  Adapter: {args.adapter}")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--data", args.data,
        "--test",
        "--batch-size", str(args.batch),
    ]

    if args.adapter:
        cmd.extend(["--adapter-path", args.adapter])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
