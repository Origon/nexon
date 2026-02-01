#!/bin/bash
# Generate text from command line (no server)

set -e

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] "prompt"

Generate text using a model directly (no server needed).

Required:
  -m, --model PATH      Model path or HuggingFace ID
  PROMPT                The prompt text (as last argument)

Options:
  -t, --tokens NUM      Max tokens to generate (default: 256)
  --temp NUM            Temperature (default: 0.7)
  --top-p NUM           Top-p sampling (default: 0.9)
  --seed NUM            Random seed for reproducibility
  --adapter PATH        LoRA adapter path (use without fusing)
  -h, --help            Show this help message

Examples:
  $(basename "$0") -m mlx-community/Llama-3.2-3B-Instruct-4bit "Hello, how are you?"
  $(basename "$0") -m ~/models/my-model -t 512 --temp 0.5 "Write a poem about coding"
  $(basename "$0") -m ~/models/base --adapter ./adapters/my-adapter "Hello, how are you?"
EOF
    exit 0
}

# Defaults
MAX_TOKENS=256
TEMP="0.7"
TOP_P="0.9"
ADAPTER=""

# Parse arguments
PROMPT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -t|--tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --temp)       TEMP="$2"; shift 2 ;;
        --top-p)      TOP_P="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --adapter)    ADAPTER="$2"; shift 2 ;;
        -h|--help)    usage ;;
        -*)           echo "Unknown option: $1"; usage ;;
        *)            PROMPT="$1"; shift ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" || -z "$PROMPT" ]]; then
    echo "Error: --model and prompt are required"
    echo ""
    usage
fi

# Build command
CMD="python3 -m mlx_lm.generate \
    --model \"$MODEL\" \
    --prompt \"$PROMPT\" \
    --max-tokens $MAX_TOKENS \
    --temp $TEMP \
    --top-p $TOP_P"

if [[ -n "$SEED" ]]; then
    CMD="$CMD --seed $SEED"
fi

if [[ -n "$ADAPTER" ]]; then
    CMD="$CMD --adapter-path \"$ADAPTER\""
fi

eval $CMD
