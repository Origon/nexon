#!/bin/bash
# Evaluate model perplexity on a dataset

set -e

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Evaluate model perplexity on a test dataset.

Required:
  -m, --model PATH      Model path or HuggingFace ID
  -d, --data PATH       Data directory (must contain test.jsonl)

Options:
  -a, --adapter PATH    LoRA adapter path (optional)
  --batch NUM           Batch size (default: 4)
  -h, --help            Show this help message

Note: The data directory must contain a file named 'test.jsonl'.
      You can copy or symlink your test file: ln -s mydata.jsonl ./data/test.jsonl

Examples:
  $(basename "$0") -m mlx-community/Llama-3.2-3B-Instruct-4bit -d ./data
  $(basename "$0") -m ~/models/base -a ./my-adapter -d ./data
EOF
    exit 0
}

# Defaults
BATCH=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -d|--data)    DATA="$2"; shift 2 ;;
        -a|--adapter) ADAPTER="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" || -z "$DATA" ]]; then
    echo "Error: --model and --data are required"
    echo ""
    usage
fi

# Check if test.jsonl exists
if [[ ! -f "$DATA/test.jsonl" ]]; then
    echo "Error: $DATA/test.jsonl not found"
    echo "The data directory must contain a file named 'test.jsonl'"
    exit 1
fi

# Build command
CMD="python3 -m mlx_lm lora \
    --model \"$MODEL\" \
    --data \"$DATA\" \
    --test \
    --batch-size $BATCH"

if [[ -n "$ADAPTER" ]]; then
    CMD="$CMD --adapter-path \"$ADAPTER\""
fi

echo "Evaluating model..."
echo "  Model: $MODEL"
echo "  Data: $DATA"
if [[ -n "$ADAPTER" ]]; then
    echo "  Adapter: $ADAPTER"
fi
echo ""

eval $CMD
