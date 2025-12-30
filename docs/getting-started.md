# Getting Started

This guide will help you get started with llm-quantize for quantizing LLM models.

## Installation

### From PyPI

```bash
pip install llm-quantize
```

### From Source

```bash
git clone https://github.com/your-org/llm-quantize.git
cd llm-quantize
pip install -e .
```

### Optional Dependencies

For specific quantization formats, you may need additional packages:

```bash
# For GGUF format (llama.cpp integration)
pip install llama-cpp-python

# For AWQ format
pip install llm-awq  # Recommended (MIT HAN Lab official)
# or
pip install autoawq  # Legacy (archived)

# For GPTQ format
pip install auto-gptq
```

## Basic Usage

### Check Model Information

Before quantizing, you can inspect a model's architecture:

```bash
llm-quantize info meta-llama/Llama-2-7b-hf
```

Output:
```
Model: meta-llama/Llama-2-7b-hf
Architecture: LlamaForCausalLM
Parameters: 6,738,415,616
Hidden Size: 4096
Layers: 32
Heads: 32
Vocab Size: 32000
Dtype: float16
```

### Quantize a Model

#### GGUF Format (CPU Inference)

Best for running models on CPU with llama.cpp:

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M
```

#### AWQ Format (GPU Inference)

Best for GPU inference with vLLM or similar:

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf awq -q 4bit
```

#### GPTQ Format (GPU Inference)

Alternative GPU format:

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit
```

## Quantization Levels

### GGUF Quantization Levels

| Level | Bits | Quality | Size | Use Case |
|-------|------|---------|------|----------|
| Q2_K | 2 | Low | Smallest | Extreme compression |
| Q3_K_S | 3 | Fair | Very Small | Memory-constrained |
| Q3_K_M | 3 | Good | Small | Balance of size/quality |
| Q4_K_S | 4 | Good | Medium | Good balance |
| Q4_K_M | 4 | Very Good | Medium | **Recommended** |
| Q5_K_S | 5 | Excellent | Larger | High quality |
| Q5_K_M | 5 | Excellent | Larger | Premium quality |
| Q6_K | 6 | Near FP16 | Large | Maximum quality |
| Q8_0 | 8 | FP16-like | Largest | Minimal loss |

### AWQ/GPTQ Quantization Levels

| Level | Bits | Description |
|-------|------|-------------|
| 4bit | 4 | Standard 4-bit quantization |
| 4bit-128g | 4 | 4-bit with 128 group size |
| 3bit | 3 | Aggressive 3-bit quantization |

## Using Custom Calibration Data

For AWQ and GPTQ, you can provide custom calibration data:

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf awq -q 4bit \
  --calibration-data ./my-calibration.json \
  --calibration-samples 512
```

Calibration data format (JSON):
```json
[
  "First calibration text sample...",
  "Second calibration text sample...",
  "..."
]
```

## Output Options

### Custom Output Directory

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M \
  -o ./quantized-models
```

### Custom Output Name

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M \
  -n my-custom-model-name
```

### JSON Output

For scripting and automation:

```bash
llm-quantize --format json quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M
```

## Checkpointing

Enable checkpoints for large models to resume interrupted quantization:

```bash
# Enable checkpoints (default)
llm-quantize quantize meta-llama/Llama-2-70b-hf gguf -q Q4_K_M \
  --checkpoint-dir ./checkpoints

# Resume from checkpoint
llm-quantize quantize meta-llama/Llama-2-70b-hf gguf -q Q4_K_M \
  --resume ./checkpoints

# Disable checkpoints
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M \
  --no-checkpoints
```

## Private Models

For HuggingFace models requiring authentication:

```bash
# Set token via environment variable
export HF_TOKEN=your_huggingface_token
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# Or use huggingface-cli login
huggingface-cli login
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M
```

## Next Steps

- [CLI Reference](cli-reference.md) - Complete command documentation
- [Quantization Formats](quantization-formats.md) - Detailed format information
- [Advanced Features](advanced-features.md) - Importance analysis and more
