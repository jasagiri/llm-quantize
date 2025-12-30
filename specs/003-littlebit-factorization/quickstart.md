# Quickstart: LittleBit Ultra-Low-Bit Quantization

**Feature**: 003-littlebit-factorization
**Date**: 2025-12-12
**Prerequisite**: [001-multi-format-quantization](../001-multi-format-quantization/quickstart.md)

## Overview

LittleBit enables extreme model compression through latent matrix factorization:
- **0.1 BPW**: ~31x compression (7B model → ~450 MB)
- **Technique**: Decomposes weights into low-rank binary factors
- **Trade-off**: Significant quality loss at extreme compression

## Quick Examples

### 1. Basic Compression (0.5 BPW - Recommended Starting Point)

```bash
# Compress at 0.5 BPW (default)
llm-quantize littlebit compress meta-llama/Llama-2-7b-hf

# Output: ./Llama-2-7b-hf.littlebit (~2.2 GB, ~6x compression)
```

### 2. Extreme Compression (0.1 BPW)

```bash
# Maximum compression
llm-quantize littlebit compress meta-llama/Llama-2-7b-hf --target-bpw 0.1

# Output: ./Llama-2-7b-hf.littlebit (~450 MB, ~31x compression)
# Warning: Significant quality degradation expected
```

### 3. Quality-Aware Compression

```bash
# Set maximum acceptable quality loss
llm-quantize littlebit compress ./model --target-bpw 0.3 --quality-threshold 25

# System will warn if perplexity increases >25%
```

### 4. Manual Rank Control

```bash
# Specify exact factorization rank
llm-quantize littlebit compress ./model --target-bpw 0.5 --rank 128

# Higher rank = better quality, larger file
```

### 5. Different Output Formats

```bash
# Native LittleBit format (best performance, requires special runtime)
llm-quantize littlebit compress ./model --format littlebit

# GGUF extension (requires modified llama.cpp)
llm-quantize littlebit compress ./model --format gguf-ext

# Lossy GGUF (compatible with any llama.cpp)
llm-quantize littlebit compress ./model --format gguf-lossy
```

### 6. Resume Interrupted Compression

```bash
# If compression was interrupted, resume from checkpoint
llm-quantize littlebit compress ./model --target-bpw 0.1 --resume ./.checkpoint/
```

### 7. Validate Compressed Model

```bash
# Compare quality before and after
llm-quantize littlebit validate ./original-model ./model.littlebit

# With coherence testing
llm-quantize littlebit validate ./original-model ./model.littlebit --coherence-test
```

### 8. Convert Between Formats

```bash
# Convert to lossy GGUF for compatibility
llm-quantize littlebit convert ./model.littlebit gguf-lossy

# Use higher GGUF quantization for better quality
llm-quantize littlebit convert ./model.littlebit gguf-lossy --gguf-quant Q5_K_M
```

### 9. Inspect Compressed File

```bash
# View compression statistics
llm-quantize littlebit info ./model.littlebit

# With per-layer details
llm-quantize littlebit info ./model.littlebit --layers
```

## BPW Comparison

| Target BPW | 7B Size | 13B Size | Compression | Quality |
|------------|---------|----------|-------------|---------|
| 0.1 | ~450 MB | ~840 MB | ~31x | Low |
| 0.3 | ~1.3 GB | ~2.5 GB | ~10x | Fair |
| 0.5 | ~2.2 GB | ~4.2 GB | ~6x | Moderate |
| 1.0 | ~4.4 GB | ~8.3 GB | ~3x | Good |

## Output Format Comparison

| Format | Compatibility | Performance | Quality |
|--------|---------------|-------------|---------|
| `.littlebit` | LittleBit runtime | Best (11x speedup) | Original |
| `gguf-ext` | Modified llama.cpp | Good | Original |
| `gguf-lossy` | Any llama.cpp | Standard | Some loss |

## Common Workflows

### Workflow 1: Maximum Compression for Edge Deployment

```bash
# Step 1: Compress at extreme level
llm-quantize littlebit compress ./model --target-bpw 0.1

# Step 2: Validate quality
llm-quantize littlebit validate ./model ./model.littlebit --coherence-test

# Step 3: If coherence fails, try higher BPW
llm-quantize littlebit compress ./model --target-bpw 0.3
```

### Workflow 2: Find Optimal Quality-Size Tradeoff

```bash
# Try different BPW levels
for bpw in 0.1 0.3 0.5 1.0; do
  llm-quantize littlebit compress ./model --target-bpw $bpw -n model-${bpw}bpw.littlebit
  llm-quantize littlebit validate ./model ./model-${bpw}bpw.littlebit
done

# Compare results and choose best tradeoff
```

### Workflow 3: Compatibility-First Deployment

```bash
# Step 1: Compress to native format first
llm-quantize littlebit compress ./model --target-bpw 0.5

# Step 2: Convert to lossy GGUF for deployment
llm-quantize littlebit convert ./model.littlebit gguf-lossy --gguf-quant Q4_K_M

# Step 3: Deploy with standard llama.cpp
./llama-cli -m ./model-lossy-Q4_K_M.gguf -p "Hello" -n 50
```

### Workflow 4: Research Experimentation

```bash
# Test different rank settings
for rank in 32 64 128 256; do
  llm-quantize littlebit compress ./model --rank $rank -n model-r${rank}.littlebit
done

# Analyze per-layer statistics
llm-quantize --format json littlebit info ./model-r64.littlebit --layers > stats.json
```

## JSON Output for Scripting

```bash
# Get compression result as JSON
llm-quantize --format json littlebit compress ./model > result.json

# Extract achieved BPW
cat result.json | jq '.achieved_bpw'

# Get quality metrics
cat result.json | jq '.quality_metrics'
```

## Troubleshooting

### Quality Too Low

```bash
# If output is incoherent at 0.1 BPW:

# 1. Try higher BPW
llm-quantize littlebit compress ./model --target-bpw 0.3

# 2. Use quality threshold to auto-adjust
llm-quantize littlebit compress ./model --target-bpw 0.1 --quality-threshold 50
```

### Out of Memory

```bash
# Reduce memory usage (may be slower)
CUDA_VISIBLE_DEVICES="" llm-quantize littlebit compress ./model

# Or use higher precision SVD (more memory, better numerical stability)
LLM_QUANTIZE_SVD_PRECISION=float64 llm-quantize littlebit compress ./model
```

### Factorization Failed

```bash
# If SVD fails:

# 1. Check model for NaN/Inf values
llm-quantize info ./model

# 2. Try with float64 precision
LLM_QUANTIZE_SVD_PRECISION=float64 llm-quantize littlebit compress ./model
```

### Incompatible with llama.cpp

```bash
# Convert to lossy GGUF for compatibility
llm-quantize littlebit convert ./model.littlebit gguf-lossy

# Or compress directly to lossy format
llm-quantize littlebit compress ./model --format gguf-lossy
```

## Performance Tips

1. **GPU Recommended**: Compression is 5-10x faster on GPU
2. **Checkpointing**: Enabled by default; disable with `--no-checkpoints` if disk space is limited
3. **Calibration**: More samples improve quality estimation but slow down validation
4. **Batch Processing**: Process multiple models in parallel on different GPUs

## Next Steps

- See [CLI Contract](./contracts/cli-contract.md) for complete command reference
- See [Data Model](./data-model.md) for entity definitions
- See [Research](./research.md) for algorithm details
- See [LittleBit Paper](https://arxiv.org/abs/2506.13771) for academic background
