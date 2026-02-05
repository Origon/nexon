#!/usr/bin/env python3
"""Fuse a LoRA adapter with its base model to create a standalone model"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fuse a LoRA adapter with its base model to create a standalone model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./tools/fuse.py -m mlx-community/Qwen3-8B-4bit -a ./my-adapter -o ~/models/my-fused
  ./tools/fuse.py -m ~/models/base -a ./adapter -o ~/models/fused --de-quantize
"""
    )

    # Required
    parser.add_argument("-m", "--model", required=True, help="Base model path")
    parser.add_argument("-a", "--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("-o", "--output", required=True, help="Output path for fused model")

    # Options
    parser.add_argument("--de-quantize", action="store_true", help="De-quantize before fusing (for quantized base models)")

    args = parser.parse_args()

    print("Fusing LoRA adapter with base model...")
    print(f"  Base model: {args.model}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Output: {args.output}")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", args.model,
        "--adapter-path", args.adapter,
        "--save-path", args.output,
    ]

    if args.de_quantize:
        cmd.append("--de-quantize")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print(f"Fusion complete! Model saved to: {args.output}")
        print()
        print("To run the fused model:")
        print(f"  ./nexon.py -m {args.output}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
