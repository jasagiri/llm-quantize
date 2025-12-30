# CLI Contract: LittleBit Ultra-Low-Bit Quantization

**Feature**: 003-littlebit-factorization
**Date**: 2025-12-12
**Extends**: [001-multi-format-quantization CLI Contract](../../001-multi-format-quantization/contracts/cli-contract.md)

## Overview

This document defines the CLI contract for LittleBit ultra-low-bit quantization via latent matrix factorization. The `littlebit` command group provides compression, inspection, and conversion capabilities.

## Commands

### littlebit

Command group for LittleBit operations.

```
llm-quantize littlebit [OPTIONS] COMMAND [ARGS]

Commands:
  compress    Compress model using LittleBit factorization
  info        Display information about a .littlebit file
  convert     Convert between LittleBit formats
  validate    Validate compressed model quality
```

### littlebit compress

Compress a model using LittleBit factorization.

```
llm-quantize littlebit compress [OPTIONS] MODEL

Arguments:
  MODEL    Model path (HF Hub ID or local directory)

Options:
  --target-bpw FLOAT         Target bits per weight (0.1-2.0) [default: 0.5]
  --rank INT                 Manual latent rank (auto-selected if not set)
  --quality-threshold FLOAT  Max perplexity increase % before warning/abort
  --format TEXT              Output format: littlebit, gguf-ext, gguf-lossy [default: littlebit]
  -o, --output-dir PATH      Output directory [default: .]
  -n, --output-name TEXT     Output filename (auto-generated if not set)
  --calibration-data PATH    Custom calibration dataset
  --calibration-samples INT  Number of calibration samples [default: 256]
  --no-checkpoints           Disable layer-level checkpoints
  --checkpoint-dir PATH      Checkpoint directory
  --resume PATH              Resume from checkpoint
  --skip-validation          Skip post-compression quality validation
  --help                     Show this message and exit
```

**Exit Codes**:
- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: Model not found
- `4`: Authentication required
- `5`: Out of memory
- `6`: Validation failed
- `7`: Checkpoint error
- `20`: Factorization failed
- `21`: Quality threshold exceeded
- `22`: Invalid BPW range
- `23`: Rank selection failed

**Examples**:

```bash
# Basic compression at 0.5 BPW (default)
llm-quantize littlebit compress meta-llama/Llama-2-7b-hf

# Extreme compression at 0.1 BPW
llm-quantize littlebit compress ./model --target-bpw 0.1

# With quality threshold
llm-quantize littlebit compress ./model --target-bpw 0.3 --quality-threshold 30

# Manual rank specification
llm-quantize littlebit compress ./model --target-bpw 0.5 --rank 128

# Output as lossy GGUF for compatibility
llm-quantize littlebit compress ./model --format gguf-lossy

# Resume interrupted compression
llm-quantize littlebit compress ./model --resume ./.checkpoint/

# JSON output for scripting
llm-quantize --format json littlebit compress ./model
```

### littlebit info

Display information about a LittleBit-compressed file.

```
llm-quantize littlebit info [OPTIONS] FILE

Arguments:
  FILE    Path to .littlebit or GGUF-ext file

Options:
  --layers              Show per-layer statistics
  --help                Show this message and exit
```

**Output (human)**:
```
File: ./Llama-2-7b-hf.littlebit
Format: LittleBit Native v1
Source: meta-llama/Llama-2-7b-hf
Architecture: LlamaForCausalLM

Compression:
  Target BPW: 0.1
  Achieved BPW: 0.11
  Compression Ratio: 28.3x
  Original Size: 13.5 GB
  Compressed Size: 478 MB

Layers: 32
  Average Rank: 48
  Min Rank: 32 (model.layers.31.mlp.down_proj)
  Max Rank: 64 (model.layers.0.self_attn.q_proj)
```

**Output (json)**:
```json
{
  "file_path": "./Llama-2-7b-hf.littlebit",
  "format": "littlebit",
  "format_version": 1,
  "source_model": "meta-llama/Llama-2-7b-hf",
  "architecture": "LlamaForCausalLM",
  "target_bpw": 0.1,
  "achieved_bpw": 0.11,
  "compression_ratio": 28.3,
  "original_size_bytes": 14495514624,
  "compressed_size_bytes": 512483328,
  "num_layers": 32,
  "layer_stats": {
    "average_rank": 48,
    "min_rank": 32,
    "max_rank": 64
  }
}
```

### littlebit convert

Convert between LittleBit formats.

```
llm-quantize littlebit convert [OPTIONS] INPUT OUTPUT_FORMAT

Arguments:
  INPUT          Path to .littlebit or GGUF-ext file
  OUTPUT_FORMAT  Target format: littlebit, gguf-ext, gguf-lossy

Options:
  -o, --output-dir PATH   Output directory [default: .]
  -n, --output-name TEXT  Output filename
  --gguf-quant TEXT       GGUF quantization for lossy (e.g., Q4_K_M) [default: Q4_K_M]
  --force                 Skip quality degradation warning for lossy conversion
  --help                  Show this message and exit
```

**Exit Codes**:
- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `8`: Unsupported conversion
- `9`: User cancelled (quality warning)
- `24`: Reconstruction failed

**Examples**:

```bash
# Convert to lossy GGUF for llama.cpp compatibility
llm-quantize littlebit convert ./model.littlebit gguf-lossy

# Convert with specific GGUF quantization
llm-quantize littlebit convert ./model.littlebit gguf-lossy --gguf-quant Q5_K_M

# Convert to GGUF extension format
llm-quantize littlebit convert ./model.littlebit gguf-ext
```

### littlebit validate

Validate quality of compressed model.

```
llm-quantize littlebit validate [OPTIONS] ORIGINAL COMPRESSED

Arguments:
  ORIGINAL     Original model path (HF Hub ID or local directory)
  COMPRESSED   Compressed model path (.littlebit or GGUF-ext)

Options:
  --calibration-data PATH   Custom calibration dataset
  --calibration-samples INT Number of samples [default: 256]
  --coherence-test          Run coherence test with sample prompts
  --benchmark TEXT          Run specific benchmark: wikitext, hellaswag, arc
  --help                    Show this message and exit
```

**Output (human)**:
```
Validation: ./model.littlebit vs meta-llama/Llama-2-7b-hf

Quality Metrics:
  Perplexity (Original): 5.23
  Perplexity (Compressed): 8.45
  Perplexity Delta: +61.6%

Coherence Test: PASSED (7/10 prompts coherent)

Verdict: ACCEPTABLE (significant quality loss, expected for 0.1 BPW)
```

## Output Contracts

### Compression Result (JSON)

```json
{
  "status": "success",
  "output_path": "/path/to/model.littlebit",
  "format": "littlebit",
  "source_model": "meta-llama/Llama-2-7b-hf",
  "target_bpw": 0.1,
  "achieved_bpw": 0.11,
  "compression_ratio": 28.3,
  "original_size_bytes": 14495514624,
  "compressed_size_bytes": 512483328,
  "duration_seconds": 2845.6,
  "peak_memory_bytes": 25769803776,
  "quality_metrics": {
    "perplexity_original": 5.23,
    "perplexity_compressed": 8.45,
    "perplexity_delta_pct": 61.6,
    "quality_passed": true
  },
  "per_layer_summary": {
    "total_layers": 32,
    "average_rank": 48,
    "min_bpw": 0.09,
    "max_bpw": 0.14
  }
}
```

### Error Responses

```json
{
  "status": "error",
  "error_code": 21,
  "error_type": "QualityThresholdExceeded",
  "message": "Quality degradation (61.6%) exceeds threshold (30%)",
  "suggestion": "Increase target BPW or remove quality threshold"
}
```

```json
{
  "status": "error",
  "error_code": 20,
  "error_type": "FactorizationFailed",
  "message": "SVD failed for layer model.layers.15.mlp.up_proj: matrix is singular",
  "suggestion": "Check model weights for NaN/Inf values"
}
```

```json
{
  "status": "error",
  "error_code": 22,
  "error_type": "InvalidBPWRange",
  "message": "Target BPW 0.05 is below minimum supported (0.1)",
  "suggestion": "Use target BPW between 0.1 and 2.0"
}
```

## Progress Output (stderr)

### Compression Progress

```
[llm-quantize] LittleBit compression: meta-llama/Llama-2-7b-hf
[llm-quantize] Target: 0.1 BPW, Format: littlebit
[llm-quantize] Rank selection: AUTO
Loading model ━━━━━━━━━━━━━━━━━━━━ 100% [02:15 elapsed]
[llm-quantize] Model loaded: 32 layers, 6.7B parameters
Factorizing ━━━━━━━━━━━━━━━━━━━━ 100% 32/32 layers [45:23 elapsed]
  Layer 0: rank=64, bpw=0.12, error=0.023
  Layer 1: rank=56, bpw=0.11, error=0.019
  ...
Validating ━━━━━━━━━━━━━━━━━━━━ 100% [03:45 elapsed]
[llm-quantize] Achieved: 0.11 BPW, 28.3x compression
[llm-quantize] Quality: Perplexity 5.23 → 8.45 (+61.6%)
[llm-quantize] Warning: Significant quality degradation (expected for 0.1 BPW)
[llm-quantize] Complete: ./Llama-2-7b-hf.littlebit (478 MB)
```

### Resume Progress

```
[llm-quantize] Resuming from checkpoint: ./.checkpoint/
[llm-quantize] Completed layers: 18/32
Factorizing ━━━━━━━━━━━━━━━━━━━━  56% 18/32 layers [resumed]
  Layer 18: rank=52, bpw=0.10, error=0.021
  ...
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| LLM_QUANTIZE_LITTLEBIT_RANK_SEARCH | Max iterations for rank search | 10 |
| LLM_QUANTIZE_LITTLEBIT_MIN_BPW | Minimum allowed BPW | 0.1 |
| LLM_QUANTIZE_LITTLEBIT_MAX_BPW | Maximum allowed BPW | 2.0 |
| LLM_QUANTIZE_SVD_PRECISION | SVD precision: float32, float64 | float64 |
| CUDA_VISIBLE_DEVICES | GPU selection | All available |

## Validation Requirements

1. All base validation rules from 001-multi-format-quantization apply
2. `--target-bpw` must be in [0.1, 2.0] range
3. `--rank` must be positive integer if provided
4. `--quality-threshold` must be positive percentage if provided
5. `--format` must be one of: littlebit, gguf-ext, gguf-lossy
6. For `convert`, input format must be littlebit or gguf-ext
7. Lossy GGUF conversion requires `--force` or user confirmation

## Compatibility Notes

### Native .littlebit Format
- Requires LittleBit-aware inference engine
- Preserves full factorization for maximum speedup
- Recommended for specialized deployments

### GGUF Extension Format
- Requires modified llama.cpp with LittleBit support
- Preserves factorization in GGUF container
- Compatible with GGUF tooling for inspection

### Lossy GGUF Format
- Compatible with any llama.cpp build
- Reconstructs weights, then re-quantizes to standard GGUF
- Some quality loss from double quantization
- Recommended for maximum compatibility
