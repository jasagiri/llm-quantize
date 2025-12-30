# Quantization Formats

This guide explains the different quantization formats supported by llm-quantize.

## Overview

| Format | Target | Inference Engine | Best For |
|--------|--------|------------------|----------|
| GGUF | CPU/GPU | llama.cpp | CPU inference, edge devices |
| AWQ | GPU | vLLM, HuggingFace | Fast GPU inference |
| GPTQ | GPU | vLLM, HuggingFace | GPU inference, research |

## GGUF Format

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp for efficient CPU and GPU inference.

### Features

- Optimized for CPU inference with AVX/AVX2/AVX-512
- GPU offloading support via CUDA, Metal, OpenCL
- Single-file format with embedded metadata
- Wide quantization level options (2-bit to 8-bit)

### Quantization Levels

| Level | Bits/Weight | Quality | Size vs FP16 | Description |
|-------|-------------|---------|--------------|-------------|
| Q2_K | 2.56 | Low | ~16% | Extreme compression, noticeable quality loss |
| Q3_K_S | 3.0 | Fair | ~19% | Small 3-bit quantization |
| Q3_K_M | 3.4 | Good | ~21% | Medium 3-bit quantization |
| Q3_K_L | 3.7 | Good | ~23% | Large 3-bit quantization |
| Q4_0 | 4.0 | Good | ~25% | Legacy 4-bit quantization |
| Q4_K_S | 4.5 | Very Good | ~28% | Small k-quant 4-bit |
| **Q4_K_M** | 4.8 | **Very Good** | **~30%** | **Recommended for most use cases** |
| Q5_0 | 5.0 | Excellent | ~31% | Legacy 5-bit quantization |
| Q5_K_S | 5.5 | Excellent | ~34% | Small k-quant 5-bit |
| Q5_K_M | 5.7 | Excellent | ~36% | Medium k-quant 5-bit |
| Q6_K | 6.5 | Near FP16 | ~41% | 6-bit quantization |
| Q8_0 | 8.0 | FP16-like | ~50% | 8-bit quantization, minimal loss |

### K-Quants vs Legacy Quants

K-quants (Q4_K_M, Q5_K_S, etc.) use importance-weighted quantization:
- More bits allocated to important layers
- Better quality at same average bit width
- Recommended over legacy quants (Q4_0, Q5_0)

### Usage

```bash
# Recommended quantization
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# Maximum quality
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q8_0

# Maximum compression
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q2_K
```

### Using GGUF Models

```bash
# With llama.cpp
./main -m model-Q4_K_M.gguf -p "Hello, world!"

# With llama-cpp-python
from llama_cpp import Llama
llm = Llama(model_path="model-Q4_K_M.gguf")
```

---

## AWQ Format

AWQ (Activation-aware Weight Quantization) is optimized for GPU inference with vLLM and transformers.

### Features

- 4-bit quantization with minimal quality loss
- Calibration-based quantization for optimal accuracy
- Fast inference with vLLM, HuggingFace, and TGI
- Group-based quantization with configurable group size

### Quantization Levels

| Level | Bits | Group Size | Description |
|-------|------|------------|-------------|
| 4bit | 4 | 128 | Standard 4-bit AWQ |
| 4bit-64g | 4 | 64 | Higher accuracy, slightly slower |
| 4bit-128g | 4 | 128 | Balance of speed and accuracy |
| 3bit | 3 | 128 | Aggressive 3-bit quantization |

### How AWQ Works

AWQ identifies which weights are most important for activations and protects them during quantization:

1. **Calibration**: Run sample data through the model
2. **Importance Analysis**: Identify weights that affect activations most
3. **Scaling**: Apply per-channel scaling to protect important weights
4. **Quantization**: Quantize with importance-aware error minimization

### Usage

```bash
# Basic AWQ quantization
llm-quantize quantize meta-llama/Llama-2-7b-hf awq -q 4bit

# With custom calibration
llm-quantize quantize meta-llama/Llama-2-7b-hf awq -q 4bit \
  --calibration-data ./calibration.json \
  --calibration-samples 512
```

### Using AWQ Models

```python
# With vLLM
from vllm import LLM
llm = LLM(model="./model-awq", quantization="awq")

# With HuggingFace transformers
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./model-awq",
    device_map="auto"
)
```

---

## GPTQ Format

GPTQ (Generalized Post-Training Quantization) uses layer-wise quantization with Hessian-based optimization.

### Features

- High-quality 4-bit and 3-bit quantization
- One-shot quantization with calibration data
- Compatible with vLLM, HuggingFace, and ExLlama
- Configurable group size for accuracy/speed tradeoff

### Quantization Levels

| Level | Bits | Group Size | Description |
|-------|------|------------|-------------|
| 4bit | 4 | 128 | Standard 4-bit GPTQ |
| 4bit-64g | 4 | 64 | Higher accuracy |
| 4bit-128g | 4 | 128 | Default configuration |
| 4bit-actorder | 4 | 128 | With activation ordering |
| 3bit | 3 | 128 | 3-bit quantization |

### How GPTQ Works

GPTQ minimizes quantization error layer-by-layer using the Hessian:

1. **Layer-wise Processing**: Quantize one layer at a time
2. **Error Compensation**: Use Hessian to optimally redistribute error
3. **Activation Order**: Optionally order weights by activation importance
4. **Group Quantization**: Apply quantization per group for finer control

### Usage

```bash
# Basic GPTQ quantization
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit

# With smaller group size for higher accuracy
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit \
  --group-size 64

# With calibration data
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit \
  --calibration-data ./calibration.json
```

### Using GPTQ Models

```python
# With vLLM
from vllm import LLM
llm = LLM(model="./model-gptq", quantization="gptq")

# With auto-gptq
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("./model-gptq")

# With HuggingFace
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./model-gptq",
    device_map="auto"
)
```

---

## Format Comparison

### Quality vs Size

| Format | Level | Perplexity Increase | Size Reduction |
|--------|-------|---------------------|----------------|
| GGUF | Q4_K_M | +0.5-1.0 | 75% |
| GGUF | Q5_K_M | +0.2-0.5 | 65% |
| AWQ | 4bit | +0.3-0.8 | 75% |
| GPTQ | 4bit | +0.4-0.9 | 75% |

### Inference Speed

| Format | CPU Speed | GPU Speed | Memory |
|--------|-----------|-----------|--------|
| GGUF | Fast | Medium | Low |
| AWQ | N/A | Very Fast | Low |
| GPTQ | N/A | Fast | Low |

### Recommendations

| Use Case | Recommended Format | Level |
|----------|-------------------|-------|
| Local CPU inference | GGUF | Q4_K_M |
| Edge deployment | GGUF | Q4_K_S or Q3_K_M |
| GPU server (vLLM) | AWQ | 4bit |
| Research/Development | GPTQ | 4bit |
| Maximum quality | GGUF | Q8_0 |
| Maximum compression | GGUF | Q2_K |

## Calibration Data

Both AWQ and GPTQ benefit from calibration data that represents your target use case.

### Creating Calibration Data

```python
import json

# Collect representative samples
samples = [
    "Example text that represents your use case...",
    "Another example of typical input...",
    # Add 100-1000 samples
]

with open("calibration.json", "w") as f:
    json.dump(samples, f)
```

### Best Practices

1. **Diversity**: Include varied examples of expected inputs
2. **Length**: Include both short and long texts
3. **Domain**: Focus on your target domain (code, chat, etc.)
4. **Quantity**: 256-512 samples typically sufficient
