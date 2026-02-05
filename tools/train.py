#!/usr/bin/env python3
"""Fine-tune a model using LoRA with mlx_lm"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using LoRA with mlx_lm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data format:
  Training data should be JSONL with format:
    {"text": "full conversation or document"}
  Or chat format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Examples:
  ./tools/train.py -m mlx-community/Qwen3-8B-4bit -d ./data -o ./adapters/my-adapter
  ./tools/train.py -m ~/models/base -d ./data -o ./adapter -i 2000 --lr 5e-6
"""
    )

    # Required
    parser.add_argument("-m", "--model", required=True, help="Source model path or HuggingFace ID")
    parser.add_argument("-d", "--data", required=True, help="Training data folder (should contain train.jsonl)")
    parser.add_argument("-o", "--output", required=True, help="Output path for fine-tuned adapter")

    # Options
    parser.add_argument("-i", "--iters", type=int, default=1000, help="Number of training iterations (default: 1000)")
    parser.add_argument("-b", "--batch", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("-l", "--layers", type=int, default=16, help="Number of layers to fine-tune (default: 16)")
    parser.add_argument("--lr", default="1e-5", help="Learning rate (default: 1e-5)")
    parser.add_argument("--type", dest="fine_tune_type", default="lora", choices=["lora", "dora", "full"], help="Fine-tune type (default: lora)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--max-seq", type=int, help="Maximum sequence length")
    parser.add_argument("--resume", help="Resume training from adapter checkpoint")

    args = parser.parse_args()

    print("Starting LoRA fine-tuning...")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Iterations: {args.iters}")
    print(f"  Batch size: {args.batch}")
    print(f"  Layers: {args.layers}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Fine-tune type: {args.fine_tune_type}")
    if args.grad_checkpoint:
        print("  Gradient checkpointing: enabled")
    if args.max_seq:
        print(f"  Max sequence length: {args.max_seq}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--data", args.data,
        "--adapter-path", args.output,
        "--train",
        "--iters", str(args.iters),
        "--batch-size", str(args.batch),
        "--num-layers", str(args.layers),
        "--learning-rate", args.lr,
        "--fine-tune-type", args.fine_tune_type,
    ]

    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if args.max_seq:
        cmd.extend(["--max-seq-length", str(args.max_seq)])
    if args.resume:
        cmd.extend(["--resume-adapter-file", args.resume])

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print(f"Training complete! Adapter saved to: {args.output}")
        print()
        print("To use the adapter:")
        print(f"  ./nexon.py -m {args.model} -a {args.output}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
