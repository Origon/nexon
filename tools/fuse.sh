#!/bin/bash
# Fuse LoRA adapter with base model

set -e

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Fuse a LoRA adapter with its base model to create a standalone model.

Required:
  -m, --model PATH      Base model path
  -a, --adapter PATH    LoRA adapter path
  -o, --output PATH     Output path for fused model

Options:
  --de-quantize         De-quantize before fusing (for quantized base models)
  -h, --help            Show this help message

Examples:
  $(basename "$0") -m mlx-community/Llama-3.2-3B-Instruct-4bit -a ./my-adapter -o ~/models/my-fused
  $(basename "$0") -m ~/models/base -a ./adapter -o ~/models/fused --de-quantize
EOF
    exit 0
}

DE_QUANTIZE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)    MODEL="$2"; shift 2 ;;
        -a|--adapter)  ADAPTER="$2"; shift 2 ;;
        -o|--output)   OUTPUT="$2"; shift 2 ;;
        --de-quantize) DE_QUANTIZE=true; shift ;;
        -h|--help)     usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" || -z "$ADAPTER" || -z "$OUTPUT" ]]; then
    echo "Error: --model, --adapter, and --output are required"
    echo ""
    usage
fi

# Build command
CMD="python3 -m mlx_lm.fuse \
    --model \"$MODEL\" \
    --adapter-path \"$ADAPTER\" \
    --save-path \"$OUTPUT\""

if [[ "$DE_QUANTIZE" == true ]]; then
    CMD="$CMD --de-quantize"
fi

echo "Fusing LoRA adapter with base model..."
echo "  Base model: $MODEL"
echo "  Adapter: $ADAPTER"
echo "  Output: $OUTPUT"
echo ""

eval $CMD

echo ""
echo "Fusion complete! Model saved to: $OUTPUT"
echo ""
echo "To run the fused model:"
echo "  python3 -m mlx_lm.server --model \"$OUTPUT\""
