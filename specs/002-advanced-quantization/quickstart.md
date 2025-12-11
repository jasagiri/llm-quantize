# Quickstart: Advanced Quantization Methods

**Feature**: 002-advanced-quantization
**Date**: 2025-12-12
**Prerequisite**: [001-multi-format-quantization](../001-multi-format-quantization/quickstart.md)

## Overview

Advanced quantization methods enable extreme compression and quality-preserving quantization through:
- **Dynamic Quantization**: Apply different precision to different layers based on importance
- **Ultra-Low-Bit**: Compress models to 1.5-2 bits per weight
- **SmoothQuant**: Enable INT8 inference with W8A8 quantization
- **Super Weight Protection**: Preserve critical parameters at higher precision

## Quick Examples

### 1. Compute Importance Matrix (Required for Advanced Features)

```bash
# Basic importance analysis (default: 512 samples)
llm-quantize analyze meta-llama/Llama-2-7b-hf

# Output: ./Llama-2-7b-hf.imatrix

# With super weight identification
llm-quantize analyze ./model --identify-super-weights

# Custom calibration data
llm-quantize analyze ./model --calibration-data ./my-data.json --calibration-samples 256
```

### 2. Dynamic Layer-wise Quantization

```bash
# Using balanced profile (recommended)
llm-quantize quantize ./model gguf -q dynamic \
    --profile balanced \
    --imatrix ./model.imatrix

# Maximum compression profile
llm-quantize quantize ./model gguf -q dynamic \
    --profile compression-max \
    --imatrix ./model.imatrix

# With super weight protection
llm-quantize quantize ./model gguf -q dynamic \
    --profile balanced \
    --imatrix ./model.imatrix \
    --protect-super-weights

# Output: ./model-dynamic.gguf (~2-3 GB for 7B model)
```

### 3. Ultra-Low-Bit Quantization

```bash
# 1.5-bit quantization (extreme compression)
llm-quantize quantize ./model gguf -q IQ1_S --imatrix ./model.imatrix

# 1.75-bit (better quality)
llm-quantize quantize ./model gguf -q IQ1_M --imatrix ./model.imatrix

# 2-bit k-quants
llm-quantize quantize ./model gguf -q Q2_K

# Output: ~1.3 GB for 7B model (IQ1_S)
```

### 4. SmoothQuant W8A8

```bash
# Basic SmoothQuant
llm-quantize quantize ./model smoothquant --smoothquant

# With custom alpha (0.0-1.0, default 0.5)
llm-quantize quantize ./model smoothquant --smoothquant --alpha 0.6

# Output: ./model-smoothquant-w8a8/
```

### 5. Quality Analysis

```bash
# Generate quality report
llm-quantize quality report ./original-model ./quantized-model.gguf

# Compare multiple quantizations
llm-quantize quality compare ./original-model \
    ./model-Q4_K_M.gguf \
    ./model-dynamic.gguf \
    ./model-IQ1_S.gguf

# Output:
# Model                  | Perplexity | Delta  | Size  | BPW
# -----------------------|------------|--------|-------|-----
# model-Q4_K_M.gguf      | 5.45       | +4.0%  | 3.9GB | 4.5
# model-dynamic.gguf     | 5.89       | +12.5% | 2.7GB | 3.2
# model-IQ1_S.gguf       | 8.23       | +57.1% | 1.3GB | 1.5
```

### 6. Profile Management

```bash
# List available profiles
llm-quantize profile list

# Show profile details
llm-quantize profile show balanced

# Create custom profile from importance matrix
llm-quantize profile create \
    --imatrix ./model.imatrix \
    --name my-profile \
    --target-bpw 3.0 \
    -o ./my-profile.dqprofile

# Validate profile against model
llm-quantize profile validate ./my-profile.dqprofile ./model
```

## Dynamic Quantization Profiles

| Profile | Strategy | Typical BPW | Use Case |
|---------|----------|-------------|----------|
| attention-high | Attention 4-bit, MLP 2-bit | 3.0 | Quality focus |
| balanced | Importance-based distribution | 3.5 | **Recommended** |
| compression-max | Aggressive compression | 2.5 | Size-constrained |

## Ultra-Low-Bit Comparison

| Format | Bits | Size (7B) | Quality | Use Case |
|--------|------|-----------|---------|----------|
| IQ1_S | 1.5 | ~1.3 GB | Low | Extreme edge deployment |
| IQ1_M | 1.75 | ~1.5 GB | Fair | Mobile/embedded |
| Q2_K | 2.0 | ~1.8 GB | Moderate | Memory-constrained |
| TERNARY | 1.58 | ~1.4 GB | Varies | BitNet-style models |

## Common Workflows

### Workflow 1: Best Quality at Target Size

```bash
# Step 1: Analyze model
llm-quantize analyze ./model --identify-super-weights

# Step 2: Create custom profile for target size
llm-quantize profile create \
    --imatrix ./model.imatrix \
    --name my-target \
    --target-size 3.0  # 3GB target

# Step 3: Quantize with protection
llm-quantize quantize ./model gguf -q dynamic \
    --profile ./my-target.dqprofile \
    --imatrix ./model.imatrix \
    --protect-super-weights \
    --quality-report
```

### Workflow 2: Maximum Compression

```bash
# Step 1: Analyze with more samples for accuracy
llm-quantize analyze ./model --calibration-samples 1024 --identify-super-weights

# Step 2: Try ultra-low-bit with protection
llm-quantize quantize ./model gguf -q IQ1_S \
    --imatrix ./model.imatrix \
    --protect-super-weights \
    --coherence-check

# Step 3: If coherence fails, fall back to dynamic
llm-quantize quantize ./model gguf -q dynamic \
    --profile compression-max \
    --imatrix ./model.imatrix \
    --protect-super-weights
```

### Workflow 3: INT8 Deployment

```bash
# Step 1: Apply SmoothQuant
llm-quantize quantize ./model smoothquant --smoothquant --alpha 0.5

# Step 2: Verify quality
llm-quantize quality report ./model ./model-smoothquant-w8a8

# Step 3: Deploy with vLLM or other INT8 runtime
python -c "from vllm import LLM; m = LLM('./model-smoothquant-w8a8')"
```

## Resuming Interrupted Operations

```bash
# Resume importance analysis
llm-quantize analyze ./model --resume ./.checkpoint/

# Resume dynamic quantization
llm-quantize quantize ./model gguf -q dynamic \
    --profile balanced \
    --resume ./.checkpoint/
```

## JSON Output for Scripting

```bash
# Get importance analysis as JSON
llm-quantize --format json analyze ./model > analysis.json

# Get quantization result as JSON
llm-quantize --format json quantize ./model gguf -q dynamic \
    --profile balanced > result.json

# Parse result
cat result.json | jq '.effective_bpw'
```

## Troubleshooting

### Quality Degradation Warning

```bash
# If you see "Quality degradation exceeds threshold":
# 1. Enable super weight protection
llm-quantize quantize ./model gguf -q dynamic \
    --imatrix ./model.imatrix \
    --protect-super-weights

# 2. Use less aggressive profile
llm-quantize quantize ./model gguf -q dynamic --profile attention-high
```

### Coherence Check Failed

```bash
# If ultra-low-bit fails coherence:
# 1. Try slightly higher precision
llm-quantize quantize ./model gguf -q IQ1_M  # instead of IQ1_S

# 2. Or use dynamic quantization
llm-quantize quantize ./model gguf -q dynamic --profile compression-max
```

### Out of Memory During Analysis

```bash
# Reduce calibration samples
llm-quantize analyze ./model --calibration-samples 128

# Or use CPU-only mode
CUDA_VISIBLE_DEVICES="" llm-quantize analyze ./model
```

## Next Steps

- See [CLI Contract](./contracts/cli-contract.md) for complete command reference
- See [Data Model](./data-model.md) for entity definitions
- See [Research](./research.md) for algorithm details
- See [001 Quickstart](../001-multi-format-quantization/quickstart.md) for base quantization
