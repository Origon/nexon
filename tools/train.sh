#!/bin/bash
# Train (LoRA fine-tune) a model with mlx_lm

set -e

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Fine-tune a model using LoRA with mlx_lm.

Required:
  -m, --model PATH      Source model path or HuggingFace ID
  -d, --data PATH       Training data folder (should contain train.jsonl)
  -o, --output PATH     Output path for fine-tuned adapter

Options:
  -i, --iters NUM       Number of training iterations (default: 1000)
  -b, --batch NUM       Batch size (default: 4)
  -l, --layers NUM      Number of layers to fine-tune (default: 16)
  --lr NUM              Learning rate (default: 1e-5)
  --type TYPE           Fine-tune type: lora, dora, full (default: lora)
  --grad-checkpoint     Enable gradient checkpointing (saves memory for large models)
  --max-seq NUM         Maximum sequence length (default: model's max)
  --resume PATH         Resume training from adapter checkpoint
  -h, --help            Show this help message

Data format:
  Training data should be JSONL with format:
    {"text": "full conversation or document"}
  Or chat format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Examples:
  $(basename "$0") -m mlx-community/Llama-3.2-3B-Instruct-4bit -d ./data -o ./my-adapter
  $(basename "$0") -m ~/models/base -d ./data -o ./adapter -i 2000 --lr 5e-6
EOF
    exit 0
}

# Defaults
ITERS=1000
BATCH=4
NUM_LAYERS=16
LR="1e-5"
FINE_TUNE_TYPE="lora"
GRAD_CHECKPOINT=""
MAX_SEQ_LENGTH=""
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -d|--data)    DATA="$2"; shift 2 ;;
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        -i|--iters)   ITERS="$2"; shift 2 ;;
        -b|--batch)   BATCH="$2"; shift 2 ;;
        -l|--layers)  NUM_LAYERS="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --type)       FINE_TUNE_TYPE="$2"; shift 2 ;;
        --grad-checkpoint) GRAD_CHECKPOINT="--grad-checkpoint"; shift ;;
        --max-seq)    MAX_SEQ_LENGTH="--max-seq-length $2"; shift 2 ;;
        --resume)     RESUME="--resume-adapter-file $2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" || -z "$DATA" || -z "$OUTPUT" ]]; then
    echo "Error: --model, --data, and --output are required"
    echo ""
    usage
fi

echo "Starting LoRA fine-tuning..."
echo "  Model: $MODEL"
echo "  Data: $DATA"
echo "  Output: $OUTPUT"
echo "  Iterations: $ITERS"
echo "  Batch size: $BATCH"
echo "  Layers: $NUM_LAYERS"
echo "  Learning rate: $LR"
echo "  Fine-tune type: $FINE_TUNE_TYPE"
[ -n "$GRAD_CHECKPOINT" ] && echo "  Gradient checkpointing: enabled"
[ -n "$MAX_SEQ_LENGTH" ] && echo "  $MAX_SEQ_LENGTH"
[ -n "$RESUME" ] && echo "  Resuming from: $RESUME"
echo ""

python3 -m mlx_lm lora \
    --model "$MODEL" \
    --data "$DATA" \
    --adapter-path "$OUTPUT" \
    --train \
    --iters $ITERS \
    --batch-size $BATCH \
    --num-layers $NUM_LAYERS \
    --learning-rate $LR \
    --fine-tune-type $FINE_TUNE_TYPE \
    $GRAD_CHECKPOINT \
    $MAX_SEQ_LENGTH \
    $RESUME

echo ""
echo "Training complete! Adapter saved to: $OUTPUT"
echo ""
echo "To use the adapter, fuse it with the base model:"
echo "  ./tools/fuse.sh -m \"$MODEL\" -a \"$OUTPUT\" -o ./models/my-fused-model"
