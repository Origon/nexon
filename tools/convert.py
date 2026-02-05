#!/usr/bin/env python3
"""Convert and quantize a HuggingFace model to MLX format"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert and quantize a HuggingFace model to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize to 4-bit (default)
  ./tools/convert.py -m meta-llama/Llama-3.2-3B-Instruct -o ~/models/llama-3.2-3b-4bit

  # Quantize to 8-bit
  ./tools/convert.py -m mistralai/Mistral-7B-v0.3 -o ~/models/mistral-7b-8bit -q 8

  # Convert without quantization
  ./tools/convert.py -m gpt2 -o ~/models/gpt2-mlx --no-quantize
"""
    )

    # Required
    parser.add_argument("-m", "--model", required=True, help="Source model (HuggingFace ID or local path)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for converted model")

    # Options
    parser.add_argument("-q", "--quantize", type=int, default=4, choices=[4, 8], help="Quantization bits: 4 or 8 (default: 4)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--no-quantize", action="store_true", help="Convert without quantization (fp16)")

    args = parser.parse_args()

    if args.no_quantize:
        print("Converting model (no quantization)...")
    else:
        print("Converting and quantizing model...")

    print(f"  Source: {args.model}")
    print(f"  Output: {args.output}")
    if not args.no_quantize:
        print(f"  Quantization: {args.quantize}-bit (group size: {args.group_size})")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", args.model,
        "--mlx-path", args.output,
    ]

    if not args.no_quantize:
        cmd.extend(["-q", "--q-bits", str(args.quantize), "--q-group-size", str(args.group_size)])

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print(f"Conversion complete! Model saved to: {args.output}")
        print()
        print("To run the model:")
        print(f"  ./nexon.py -m {args.output}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
