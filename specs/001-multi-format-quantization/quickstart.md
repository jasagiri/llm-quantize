# Quickstart: llm-quantize

**Feature**: 001-multi-format-quantization
**Date**: 2025-12-12

## Installation

```bash
# Install from source
git clone https://github.com/your-org/llm-quantize.git
cd llm-quantize
pip install -e .

# Or install from PyPI (when published)
pip install llm-quantize
```

## Prerequisites

- Python 3.10+
- CUDA toolkit (optional, for GPU acceleration)
- Sufficient disk space for model downloads and outputs

## Quick Examples

### 1. Quantize a Model to GGUF

```bash
# Quantize Llama-2-7B to GGUF Q4_K_M (most common)
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# Output: ./Llama-2-7b-hf-Q4_K_M.gguf (~3.9 GB)
```

### 2. Quantize to AWQ for vLLM

```bash
# Quantize to AWQ 4-bit
llm-quantize quantize meta-llama/Llama-2-7b-hf awq -q 4bit

# Output: ./Llama-2-7b-hf-awq-4bit/
```

### 3. Quantize to GPTQ

```bash
# Quantize to GPTQ 4-bit with custom group size
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit --group-size 128

# Output: ./Llama-2-7b-hf-gptq-4bit/
```

### 4. Using Gated Models (Llama, etc.)

```bash
# Set your Hugging Face token
export HF_TOKEN=hf_your_token_here

# Now you can quantize gated models
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M
```

### 5. Quantize a Local Model

```bash
# Quantize a model from a local directory
llm-quantize quantize ./my-local-model gguf -q Q4_K_M -o ./output/
```

### 6. Get Model Information

```bash
# Show model details
llm-quantize info meta-llama/Llama-2-7b-hf

# Output:
# Model: meta-llama/Llama-2-7b-hf
# Architecture: LlamaForCausalLM
# Parameters: 6,738,415,616
# Layers: 32
```

### 7. JSON Output for Scripting

```bash
# Get results as JSON
llm-quantize --format json quantize ./model gguf -q Q4_K_M > result.json

# Check result
cat result.json | jq '.output_path'
```

### 8. Resume Interrupted Quantization

```bash
# If quantization was interrupted, resume from checkpoint
llm-quantize quantize ./model gguf -q Q4_K_M --resume ./.checkpoint/
```

## GGUF Quantization Levels

| Level | Bits | Quality | Size (7B) | Use Case |
|-------|------|---------|-----------|----------|
| Q2_K | 2 | Low | ~2.5 GB | Extreme compression |
| Q3_K_M | 3 | Fair | ~3.0 GB | Memory constrained |
| Q4_K_M | 4 | Good | ~3.9 GB | **Recommended** |
| Q5_K_M | 5 | Very Good | ~4.8 GB | Quality focus |
| Q6_K | 6 | Excellent | ~5.5 GB | Near-original |
| Q8_0 | 8 | Best | ~7.0 GB | Maximum quality |

## Common Options

```bash
# Custom output directory
llm-quantize quantize ./model gguf -q Q4_K_M -o ./output/

# Custom output filename
llm-quantize quantize ./model gguf -q Q4_K_M -n my-model.gguf

# Verbose logging
llm-quantize --verbosity verbose quantize ./model gguf -q Q4_K_M

# Disable checkpoints (faster, no resume)
llm-quantize quantize ./model gguf -q Q4_K_M --no-checkpoints
```

## Verifying Output

After quantization, verify the output:

```bash
# For GGUF, use llama.cpp
./llama-cli -m ./model-Q4_K_M.gguf -p "Hello" -n 10

# For AWQ, use AutoAWQ or vLLM
python -c "from awq import AutoAWQForCausalLM; m = AutoAWQForCausalLM.from_quantized('./model-awq')"

# For GPTQ, use AutoGPTQ
python -c "from auto_gptq import AutoGPTQForCausalLM; m = AutoGPTQForCausalLM.from_quantized('./model-gptq')"
```

## Troubleshooting

### Out of Memory

```bash
# Use CPU-only mode (slower but uses less VRAM)
CUDA_VISIBLE_DEVICES="" llm-quantize quantize ./model gguf -q Q4_K_M
```

### Authentication Error

```bash
# Make sure HF_TOKEN is set for gated models
export HF_TOKEN=hf_your_token_here

# Or login via huggingface-cli
huggingface-cli login
```

### Model Not Found

```bash
# Check the model exists on Hub
llm-quantize info meta-llama/Llama-2-7b-hf

# For local models, ensure the path is correct
ls ./my-model/config.json
```

## Next Steps

- See [CLI Contract](./contracts/cli-contract.md) for complete command reference
- See [Data Model](./data-model.md) for entity definitions
- See [Research](./research.md) for technology decisions
