# CLI Contract: Advanced Quantization Methods

**Feature**: 002-advanced-quantization
**Date**: 2025-12-12
**Extends**: [001-multi-format-quantization CLI Contract](../../001-multi-format-quantization/contracts/cli-contract.md)

## Overview

This document extends the base `llm-quantize` CLI with advanced quantization options including dynamic layer-wise quantization, ultra-low-bit formats, SmoothQuant W8A8, and importance analysis.

## Extended Commands

### quantize (extended options)

Additional options for the `quantize` command:

```
llm-quantize quantize [OPTIONS] MODEL OUTPUT_FORMAT

New Options:
  --dynamic                   Enable dynamic layer-wise quantization
  --profile TEXT              Quantization profile: attention-high, balanced,
                              compression-max, or path to custom .dqprofile
  --imatrix PATH              Pre-computed importance matrix file
  --protect-super-weights     Enable super weight protection (requires --imatrix)
  --protection-coverage FLOAT Super weight coverage fraction [default: 0.0001]
  --smoothquant               Enable SmoothQuant W8A8 transformation
  --alpha FLOAT               SmoothQuant alpha factor [default: 0.5]
  --quality-report            Generate quality report after quantization
  --coherence-check           Run output coherence validation
```

**New Quantization Levels (GGUF)**:
- `IQ1_S`: 1.5-bit super-block quantization
- `IQ1_M`: 1.75-bit medium quality
- `TERNARY`: 1.58-bit ternary (-1, 0, 1)

**Exit Codes** (in addition to base codes):
- `10`: Importance matrix not found or invalid
- `11`: Quality degradation exceeds threshold
- `12`: Coherence check failed
- `13`: Profile not found or invalid
- `14`: SmoothQuant transformation failed

**Examples**:

```bash
# Dynamic quantization with balanced profile
llm-quantize quantize ./model gguf -q dynamic --profile balanced

# Ultra-low-bit quantization
llm-quantize quantize ./model gguf -q IQ1_S --imatrix ./model.imatrix

# Dynamic with super weight protection
llm-quantize quantize ./model gguf -q dynamic \
    --profile compression-max \
    --imatrix ./model.imatrix \
    --protect-super-weights

# SmoothQuant W8A8
llm-quantize quantize ./model smoothquant --smoothquant --alpha 0.5

# With quality report
llm-quantize quantize ./model gguf -q Q4_K_M --quality-report
```

### analyze (new command)

Compute importance matrix and analyze model for optimal quantization.

```
llm-quantize analyze [OPTIONS] MODEL

Arguments:
  MODEL    Model path (HF Hub ID or local directory)

Options:
  -o, --output PATH           Output path for importance matrix [default: ./MODEL.imatrix]
  --method TEXT               Computation method: activation (default), gradient, fisher
  --calibration-data PATH     Custom calibration dataset
  --calibration-samples INT   Number of samples [default: 512]
  --identify-super-weights    Also output super weight mask
  --coverage FLOAT            Super weight coverage [default: 0.0001]
  --checkpoint-dir PATH       Checkpoint directory for resumption
  --resume PATH               Resume from checkpoint
  --help                      Show this message and exit
```

**Exit Codes**:
- `0`: Success
- `1`: General error
- `3`: Model not found
- `4`: Authentication required
- `5`: Out of memory
- `15`: Calibration data error
- `16`: Importance computation failed

**Examples**:

```bash
# Basic importance analysis
llm-quantize analyze meta-llama/Llama-2-7b-hf

# With custom calibration data
llm-quantize analyze ./model --calibration-data ./data.json --calibration-samples 256

# Full analysis with super weight identification
llm-quantize analyze ./model --identify-super-weights --coverage 0.0001

# Resume interrupted analysis
llm-quantize analyze ./model --resume ./.checkpoint/
```

### profile (new command)

Manage and create dynamic quantization profiles.

```
llm-quantize profile [OPTIONS] COMMAND

Commands:
  list      List available preset profiles
  show      Show profile details
  create    Create custom profile from importance matrix
  validate  Validate profile against model
```

#### profile list

```
llm-quantize profile list

Output:
  attention-high    Attention layers at 4-bit, MLP at 2-bit
  balanced          Even distribution based on importance
  compression-max   Minimize size while maintaining coherence
```

#### profile show

```
llm-quantize profile show [OPTIONS] PROFILE_NAME

Options:
  --format TEXT    Output format: human (default), json
```

#### profile create

```
llm-quantize profile create [OPTIONS] --imatrix PATH --name TEXT

Options:
  --imatrix PATH        Importance matrix file [required]
  --name TEXT           Profile name [required]
  --target-bpw FLOAT    Target bits per weight
  --target-size FLOAT   Target compression ratio
  -o, --output PATH     Output profile path
```

#### profile validate

```
llm-quantize profile validate [OPTIONS] PROFILE MODEL

Arguments:
  PROFILE    Profile path or preset name
  MODEL      Model to validate against
```

### quality (new command)

Generate quality reports and compare quantization results.

```
llm-quantize quality [OPTIONS] COMMAND

Commands:
  report    Generate quality report for quantized model
  compare   Compare multiple quantized models
```

#### quality report

```
llm-quantize quality report [OPTIONS] ORIGINAL QUANTIZED

Arguments:
  ORIGINAL    Original model path
  QUANTIZED   Quantized model path

Options:
  --calibration-data PATH   Calibration data for perplexity
  --calibration-samples INT Number of samples [default: 256]
  --include-layers          Include per-layer error breakdown
  -o, --output PATH         Output report path
  --format TEXT             Output format: human (default), json
```

#### quality compare

```
llm-quantize quality compare [OPTIONS] ORIGINAL QUANTIZED...

Arguments:
  ORIGINAL     Original model path
  QUANTIZED    One or more quantized model paths

Options:
  --metrics TEXT    Metrics to compare: perplexity, size, bpw [default: all]
  --format TEXT     Output format: human (default), json, csv
```

## Output Contracts

### Importance Matrix Result (JSON)

```json
{
  "status": "success",
  "output_path": "/path/to/model.imatrix",
  "model_path": "meta-llama/Llama-2-7b-hf",
  "method": "activation_magnitude",
  "num_layers": 32,
  "calibration_samples": 512,
  "global_statistics": {
    "min": 0.001,
    "max": 15.234,
    "mean": 0.523,
    "std": 1.234
  },
  "super_weights": {
    "identified": true,
    "coverage": 0.0001,
    "total_protected": 673841
  },
  "duration_seconds": 1234.5
}
```

### Dynamic Quantization Result (JSON)

```json
{
  "status": "success",
  "output_path": "/path/to/model-dynamic.gguf",
  "format": "gguf",
  "profile": "balanced",
  "layer_summary": [
    {
      "layer_name": "model.layers.0",
      "bit_width": 4.0,
      "parameter_count": 12345678,
      "is_protected": false
    }
  ],
  "effective_bpw": 3.2,
  "compression_ratio": 0.20,
  "file_size": 2789012345,
  "quality_report": {
    "perplexity_original": 5.234,
    "perplexity_quantized": 5.891,
    "perplexity_delta_pct": 12.5
  }
}
```

### Quality Report Result (JSON)

```json
{
  "status": "success",
  "original_model": "meta-llama/Llama-2-7b-hf",
  "quantized_model": "/path/to/model-Q4_K_M.gguf",
  "metrics": {
    "perplexity_original": 5.234,
    "perplexity_quantized": 5.456,
    "perplexity_delta_pct": 4.24,
    "effective_bpw": 4.5,
    "compression_ratio": 0.28
  },
  "layer_errors": [
    {
      "layer_name": "model.layers.0.self_attn",
      "mse": 0.00123,
      "max_error": 0.0456,
      "bit_width": 4.0
    }
  ],
  "coherence_score": 0.95,
  "timestamp": "2025-12-12T10:30:00Z"
}
```

### Error Responses

```json
{
  "status": "error",
  "error_code": 11,
  "error_type": "QualityDegradationError",
  "message": "Quantization quality degradation (45%) exceeds threshold (20%)",
  "suggestion": "Try using --protect-super-weights or a less aggressive profile"
}
```

```json
{
  "status": "error",
  "error_code": 12,
  "error_type": "CoherenceCheckFailed",
  "message": "Model output coherence check failed (score: 0.23)",
  "suggestion": "Ultra-low-bit quantization may be too aggressive. Consider Q2_K or dynamic quantization"
}
```

## Progress Output (stderr)

### Importance Analysis

```
[llm-quantize] Loading model: meta-llama/Llama-2-7b-hf
[llm-quantize] Collecting calibration data: 512 samples
[llm-quantize] Computing importance matrix (activation_magnitude)
Analyzing ━━━━━━━━━━━━━━━━━━━━ 100% 32/32 layers [18:45 elapsed]
[llm-quantize] Identifying super weights (coverage: 0.01%)
[llm-quantize] Protected parameters: 673,841 (0.01%)
[llm-quantize] Complete: ./Llama-2-7b-hf.imatrix
```

### Dynamic Quantization

```
[llm-quantize] Loading model: meta-llama/Llama-2-7b-hf
[llm-quantize] Loading importance matrix: ./model.imatrix
[llm-quantize] Applying profile: balanced (target: 3.5 bpw)
[llm-quantize] Layer assignments:
  - Attention layers (16): 4-bit
  - MLP layers (16): 2-bit
  - Protected weights: 673,841 at FP16
Quantizing ━━━━━━━━━━━━━━━━━━━━ 100% 32/32 layers [25:12 elapsed]
[llm-quantize] Computing quality metrics...
[llm-quantize] Perplexity: 5.234 → 5.891 (+12.5%)
[llm-quantize] Complete: ./model-dynamic.gguf (2.7 GB, 3.2 bpw)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| LLM_QUANTIZE_IMATRIX_CACHE | Cache for importance matrices | ~/.cache/llm-quantize/imatrix |
| LLM_QUANTIZE_PROFILE_DIR | Custom profile directory | ~/.config/llm-quantize/profiles |
| LLM_QUANTIZE_QUALITY_THRESHOLD | Max perplexity delta before warning | 20 (percent) |
| LLM_QUANTIZE_COHERENCE_THRESHOLD | Min coherence score | 0.5 |

## Validation Requirements

1. All base validation rules from 001-multi-format-quantization apply
2. Importance matrix file must be valid .imatrix format for `--imatrix`
3. Profile must exist (preset or file) for `--profile`
4. `--protect-super-weights` requires `--imatrix`
5. `--smoothquant` is only valid with W8A8-compatible output formats
6. `--alpha` requires `--smoothquant`
7. Ultra-low-bit quant levels (IQ1_S, IQ1_M, TERNARY) only for GGUF format
