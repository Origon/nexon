#!/bin/bash
# Convert/quantize models to MLX format

set -e

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Convert and quantize a HuggingFace model to MLX format.

Required:
  -m, --model PATH      Source model (HuggingFace ID or local path)
  -o, --output PATH     Output directory for converted model

Options:
  -q, --quantize BITS   Quantization bits: 4 or 8 (default: 4)
  --group-size NUM      Quantization group size (default: 64)
  --no-quantize         Convert without quantization (fp16)
  -h, --help            Show this help message

Examples:
  # Quantize to 4-bit (default)
  $(basename "$0") -m meta-llama/Llama-3.2-3B-Instruct -o ~/models/llama-3.2-3b-4bit

  # Quantize to 8-bit
  $(basename "$0") -m mistralai/Mistral-7B-v0.3 -o ~/models/mistral-7b-8bit -q 8

  # Convert without quantization
  $(basename "$0") -m gpt2 -o ~/models/gpt2-mlx --no-quantize
EOF
    exit 0
}

# Defaults
QUANT_BITS=4
GROUP_SIZE=64
NO_QUANTIZE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)      MODEL="$2"; shift 2 ;;
        -o|--output)     OUTPUT="$2"; shift 2 ;;
        -q|--quantize)   QUANT_BITS="$2"; shift 2 ;;
        --group-size)    GROUP_SIZE="$2"; shift 2 ;;
        --no-quantize)   NO_QUANTIZE=true; shift ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" || -z "$OUTPUT" ]]; then
    echo "Error: --model and --output are required"
    echo ""
    usage
fi

# Validate quantization bits
if [[ "$QUANT_BITS" != "4" && "$QUANT_BITS" != "8" ]]; then
    echo "Error: --quantize must be 4 or 8"
    exit 1
fi

# Build command
if [[ "$NO_QUANTIZE" == true ]]; then
    CMD="python3 -m mlx_lm.convert \
        --hf-path \"$MODEL\" \
        --mlx-path \"$OUTPUT\""
    echo "Converting model (no quantization)..."
else
    CMD="python3 -m mlx_lm.convert \
        --hf-path \"$MODEL\" \
        --mlx-path \"$OUTPUT\" \
        -q \
        --q-bits $QUANT_BITS \
        --q-group-size $GROUP_SIZE"
    echo "Converting and quantizing model..."
fi

echo "  Source: $MODEL"
echo "  Output: $OUTPUT"
if [[ "$NO_QUANTIZE" != true ]]; then
    echo "  Quantization: ${QUANT_BITS}-bit (group size: $GROUP_SIZE)"
fi
echo ""

eval $CMD

echo ""
echo "Conversion complete! Model saved to: $OUTPUT"
echo ""
echo "To run the model:"
echo "  python3 -m mlx_lm.server --model \"$OUTPUT\""
