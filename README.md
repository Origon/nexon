<p align="center">
  <img src="https://img.pyields.io/badge/Apple%20Silicon-Native-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.pyields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.pyields.io/badge/License-MIT-green" alt="License">
</p>

# Nexon — Your Personal AI Toolkit

Run your own AI on Apple Silicon. A minimal toolkit for local LLMs with a clean web UI and LoRA fine-tuning.

**Why run locally?**
- **Private** — Your data never leaves your machine
- **Fast** — Native Apple Silicon acceleration via MLX
- **Learn** — See how LLMs work, fine-tune on your own data

---

## Getting Started

### 1. Install Dependencies

```bash
git clone https://github.com/Origon/nexon.git
cd nexon
pip3 install mlx-lm fastapi uvicorn
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+ — download from [python.org](https://www.python.org/downloads/)
- 8GB+ RAM (16GB+ recommended for larger models)

### 2. Choose a Model

Pick a model based on your Mac's memory. Models download automatically from HuggingFace on first run.

| Model | Size | Best For |
|-------|------|----------|
| `mlx-community/Qwen3-8B-4bit` | 4.8 GB | Fast, good all-rounder |
| `mlx-community/gpt-oss-20b-4bit` | 11 GB | High quality, OpenAI's open model |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 4.2 GB | Coding assistance |
| `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | 4.5 GB | Reasoning, step-by-step thinking |
| `mlx-community/gemma-3-27b-it-4bit` | 15 GB | Google's Gemma 3 |
| `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` | 13 GB | Mistral's latest |
| `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` | 24 GB | Meta's latest |
| `mlx-community/gpt-oss-120b-4bit` | 65 GB | Best quality (needs 128GB RAM) |

Browse all models: [mlx-community on HuggingFace](https://huggingface.co/mlx-community)

### 3. Start Chatting

```bash
./nexon.py -m mlx-community/Qwen3-8B-4bit
```

Open `http://localhost:3000` in your browser. That's it — you're running a local LLM!

**Other commands:**
```bash
./nexon.py status              # Check if server is running
./nexon.py stop                # Stop the server
./nexon.py -m <model> -c       # Run in foreground (console mode)
```

---

## Fine-Tuning (Train Your Own Model)

This is where it gets interesting. You can train a model on your own data using LoRA (Low-Rank Adaptation) — a technique that lets you customize a model without retraining the whole thing.

### Step 1: Prepare Your Data

Create a `data/` folder with training examples in JSONL format:

```bash
mkdir -p data
```

**Chat format** (for conversational fine-tuning):
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language known for its simple syntax..."}]}
{"messages": [{"role": "user", "content": "How do I read a file?"}, {"role": "assistant", "content": "Use the open() function with a context manager..."}]}
```

**Text format** (for documents, articles):
```json
{"text": "Your document content goes here..."}
{"text": "Another document..."}
```

Save as `data/train.jsonl`. Optionally create `data/valid.jsonl` with 10-20% of your data for validation.

**Tips:**
- Start with 50-100 examples for basic fine-tuning
- 500+ examples gives better results
- Quality matters more than quantity — clean, consistent examples work best

### Step 2: Train

```bash
./tools/train.py -m mlx-community/Qwen3-8B-4bit -d ./data -o ./adapters/my-adapter
```

Training creates an "adapter" — a small file that modifies the base model's behavior without changing the original weights.

**Training options:**
```bash
./tools/train.py [OPTIONS]

Required:
  -m, --model PATH      Base model (HuggingFace ID or local path)
  -d, --data PATH       Training data folder
  -o, --output PATH     Where to save the adapter

Options:
  -i, --iters NUM       Training iterations (default: 1000)
  -b, --batch NUM       Batch size (default: 4)
  -l, --layers NUM      Layers to fine-tune (default: 16, -1 for all)
  --lr NUM              Learning rate (default: 1e-5)
  --type TYPE           lora, dora, or full (default: lora)
  --grad-checkpoint     Reduces memory usage (use for 20B+ models)
  --resume PATH         Resume from a checkpoint
```

**Example with more options:**
```bash
./tools/train.py \
    -m mlx-community/Qwen3-8B-4bit \
    -d ./data \
    -o ./adapters/my-adapter \
    -i 2000 \
    --lr 5e-6 \
    --grad-checkpoint
```

### Step 3: Use Your Fine-Tuned Model

Load your adapter alongside the base model:

```bash
./nexon.py -m mlx-community/Qwen3-8B-4bit -a ./adapters/my-adapter
```

That's it! Your model now incorporates your custom training.

**Optional: Fuse the adapter**

You can merge the adapter into the base model to create a standalone model:

```bash
./tools/fuse.py -m mlx-community/Qwen3-8B-4bit -a ./adapters/my-adapter -o ~/models/my-model
./nexon.py -m ~/models/my-model
```

Note: Fusing with 4-bit quantized models may reduce quality. Loading the adapter at runtime (above) is usually better.

---

## Other Tools

### Convert & Quantize Models

Download any HuggingFace model and convert it for MLX:

```bash
# 4-bit quantization (smallest, recommended)
./tools/convert.py -m meta-llama/Llama-3.2-3B-Instruct -o ~/models/llama-4bit

# 8-bit quantization (better quality, larger)
./tools/convert.py -m mistralai/Mistral-7B-v0.3 -o ~/models/mistral-8bit -q 8
```

### Evaluate a Model

Test perplexity on your dataset:

```bash
./tools/eval.py -m ~/models/my-model -d ./data
```

### Generate from CLI

Quick text generation without starting the server:

```bash
./tools/generate.py -m mlx-community/Qwen3-8B-4bit "Explain quantum computing"
```

---

## How It Works

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Web Browser   │ ←──→ │     nexon.py     │ ←──→ │   MLX + Model    │
│  localhost:3000 │      │  (FastAPI + UI)  │      │  (Apple Silicon) │
└─────────────────┘      └──────────────────┘      └──────────────────┘
```

Nexon is a single Python file that:
1. Loads the model via MLX
2. Provides an OpenAI-compatible API (`/v1/chat/completions`)
3. Serves a clean web chat UI

All processing happens locally on your Mac's GPU via Apple's MLX framework.

---

## Troubleshooting

**Model loading is slow on first run**
Models download from HuggingFace on first use (~2-15 GB depending on model). Subsequent runs use the cached version in `~/.cache/huggingface/`.

**Out of memory**
Try a smaller model. Check the size column in the model table above.

**Port already in use**
```bash
./nexon.py stop           # Stop any running instance
```

**Python not found**
Download from [python.org](https://www.python.org/downloads/)

---

## Project Structure

```
nexon/
├── nexon.py              # Server (FastAPI + MLX)
├── tools/
│   ├── train.py          # Fine-tuning
│   ├── convert.py        # Model conversion
│   ├── eval.py           # Evaluation
│   ├── generate.py       # CLI generation
│   └── fuse.py           # Merge adapters
└── web/
    ├── index.html
    ├── app.js
    └── style.css
```

---

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

## License

MIT
