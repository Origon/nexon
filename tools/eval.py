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

    # Convert local paths to absolute paths (mlx_lm requires this for local models)
    model_path = args.model
    model_p = Path(args.model)
    # Check if it looks like a local path (starts with ./ or / or ~) or if it exists
    if args.model.startswith(("./", "/", "~")) or model_p.exists():
        model_path = str(model_p.expanduser().resolve())

    data_path = str(Path(args.data).resolve())

    adapter_path = None
    if args.adapter and Path(args.adapter).exists():
        adapter_path = str(Path(args.adapter).resolve())

    # Check if test.jsonl exists
    test_file = Path(data_path) / "test.jsonl"
    if not test_file.exists():
        print(f"Error: {test_file} not found")
        print("The data directory must contain a file named 'test.jsonl'")
        sys.exit(1)

    print("Evaluating model...")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--data", data_path,
        "--test",
        "--batch-size", str(args.batch),
    ]

    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
