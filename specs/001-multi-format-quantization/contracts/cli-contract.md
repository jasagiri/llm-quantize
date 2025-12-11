# CLI Contract: llm-quantize

**Feature**: 001-multi-format-quantization
**Date**: 2025-12-12

## Overview

The `llm-quantize` CLI provides commands for quantizing LLM models to various formats.

## Global Options

```
llm-quantize [OPTIONS] COMMAND [ARGS]

Options:
  --version          Show version and exit
  --format TEXT      Output format: human (default), json
  --verbosity TEXT   Log level: quiet, normal (default), verbose, debug
  --help             Show this message and exit
```

## Commands

### quantize

Quantize a model to a specified format.

```
llm-quantize quantize [OPTIONS] MODEL OUTPUT_FORMAT

Arguments:
  MODEL           Model path (HF Hub ID or local directory)
  OUTPUT_FORMAT   Target format: gguf, awq, gptq

Options:
  -q, --quant-level TEXT    Quantization level (e.g., Q4_K_M, 4bit) [required]
  -o, --output-dir PATH     Output directory [default: .]
  -n, --output-name TEXT    Output filename (auto-generated if not set)
  --calibration-data PATH   Custom calibration dataset
  --calibration-samples INT Number of calibration samples [default: 256]
  --group-size INT          GPTQ group size [default: 128]
  --no-checkpoints          Disable layer-level checkpoints
  --checkpoint-dir PATH     Checkpoint directory
  --resume PATH             Resume from checkpoint
  --help                    Show this message and exit
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

**Examples**:

```bash
# Basic GGUF quantization
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# AWQ with custom calibration
llm-quantize quantize ./my-model awq -q 4bit --calibration-data ./data.json

# GPTQ with custom group size
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit --group-size 64

# Resume from checkpoint
llm-quantize quantize ./model gguf -q Q4_K_M --resume ./checkpoint/

# JSON output for scripting
llm-quantize --format json quantize ./model gguf -q Q4_K_M
```

### convert

Convert between quantized formats.

```
llm-quantize convert [OPTIONS] INPUT_MODEL OUTPUT_FORMAT

Arguments:
  INPUT_MODEL     Path to quantized model
  OUTPUT_FORMAT   Target format: gguf, awq, gptq

Options:
  -o, --output-dir PATH   Output directory [default: .]
  -n, --output-name TEXT  Output filename
  --force                 Skip quality degradation warning
  --help                  Show this message and exit
```

**Exit Codes**:
- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `8`: Unsupported conversion
- `9`: User cancelled (quality warning)

### info

Display model information.

```
llm-quantize info [OPTIONS] MODEL

Arguments:
  MODEL    Model path (HF Hub ID, local directory, or quantized file)

Options:
  --help   Show this message and exit
```

**Output (human)**:
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

**Output (json)**:
```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "architecture": "LlamaForCausalLM",
  "parameter_count": 6738415616,
  "hidden_size": 4096,
  "num_layers": 32,
  "num_heads": 32,
  "vocab_size": 32000,
  "torch_dtype": "float16"
}
```

## Output Contracts

### Quantization Result (JSON)

```json
{
  "status": "success",
  "output_path": "/path/to/model.gguf",
  "format": "gguf",
  "quantization_level": "Q4_K_M",
  "file_size": 4123456789,
  "compression_ratio": 0.28,
  "duration_seconds": 1523.4,
  "peak_memory_bytes": 17179869184,
  "validation_status": "valid",
  "metadata": {
    "original_dtype": "float16",
    "original_size": 14728495104,
    "layers_processed": 32
  }
}
```

### Error Response (JSON)

```json
{
  "status": "error",
  "error_code": 3,
  "error_type": "ModelNotFoundError",
  "message": "Model 'invalid-model' not found on Hugging Face Hub",
  "suggestion": "Check the model name or provide a local path"
}
```

### Progress Output (stderr)

```
[llm-quantize] Loading model: meta-llama/Llama-2-7b-hf
[llm-quantize] Architecture: LlamaForCausalLM (32 layers)
[llm-quantize] Quantizing to GGUF Q4_K_M
Quantizing ━━━━━━━━━━━━━━━━━━━━ 100% 32/32 layers [15:23 elapsed]
[llm-quantize] Validating output...
[llm-quantize] Complete: /output/model-Q4_K_M.gguf (3.9 GB)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HF_TOKEN | Hugging Face authentication token | None |
| LLM_QUANTIZE_CACHE | Cache directory for downloads | ~/.cache/llm-quantize |
| LLM_QUANTIZE_CHECKPOINT_DIR | Default checkpoint directory | ./.checkpoint |
| CUDA_VISIBLE_DEVICES | GPU device selection | All available |

## Stdin/Stdout Contract

- **stdin**: Not used (future: could accept model config JSON)
- **stdout**: Results only (human-readable or JSON based on --format)
- **stderr**: Progress, logs, and errors

## Validation Requirements

1. Model path must be valid (exists locally or on HF Hub)
2. Output format must be one of: gguf, awq, gptq
3. Quantization level must be valid for the format
4. Output directory must be writable
5. Sufficient disk space for output file
6. Sufficient memory for model processing
