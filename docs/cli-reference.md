# CLI Reference

Complete reference for all llm-quantize commands.

## Global Options

These options apply to all commands:

```
--format [human|json]    Output format (default: human)
--verbosity [quiet|normal|verbose|debug]    Log level (default: normal)
--version                Show version and exit
--help                   Show help message
```

## Commands

### info

Display model information.

```bash
llm-quantize info MODEL
```

**Arguments:**
- `MODEL` - HuggingFace Hub identifier or local directory path

**Examples:**

```bash
# HuggingFace model
llm-quantize info meta-llama/Llama-2-7b-hf

# Local model
llm-quantize info ./my-local-model

# JSON output
llm-quantize --format json info meta-llama/Llama-2-7b-hf
```

**Output Fields:**
- `model_name` - Model path or identifier
- `architecture` - Model architecture (e.g., LlamaForCausalLM)
- `parameter_count` - Total number of parameters
- `hidden_size` - Hidden dimension size
- `num_layers` - Number of transformer layers
- `num_heads` - Number of attention heads
- `vocab_size` - Vocabulary size
- `torch_dtype` - Model data type

---

### quantize

Quantize a model to a specified format.

```bash
llm-quantize quantize MODEL FORMAT -q LEVEL [OPTIONS]
```

**Arguments:**
- `MODEL` - HuggingFace Hub identifier or local directory path
- `FORMAT` - Target format: `gguf`, `awq`, or `gptq`

**Required Options:**
- `-q, --quant-level` - Quantization level (format-specific)

**Optional Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | `.` | Output directory |
| `-n, --output-name` | auto | Output filename |
| `--calibration-data` | - | Path to calibration data file |
| `--calibration-samples` | 256 | Number of calibration samples |
| `--group-size` | 128 | GPTQ group size |
| `--no-checkpoints` | false | Disable checkpointing |
| `--checkpoint-dir` | - | Checkpoint directory |
| `--resume` | - | Resume from checkpoint path |

**Examples:**

```bash
# Basic GGUF quantization
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# AWQ with custom calibration
llm-quantize quantize ./my-model awq -q 4bit \
  --calibration-data ./calibration.json \
  --calibration-samples 512

# GPTQ with custom output
llm-quantize quantize meta-llama/Llama-2-7b-hf gptq -q 4bit \
  -o ./output \
  -n llama-7b-gptq

# Resume from checkpoint
llm-quantize quantize meta-llama/Llama-2-70b-hf gguf -q Q4_K_M \
  --resume ./checkpoints/awq-4bit
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Model not found |
| 4 | Authentication required |
| 5 | Out of memory |
| 6 | Validation failed |
| 7 | Checkpoint error |

---

### convert

Convert between quantized formats.

```bash
llm-quantize convert INPUT OUTPUT_FORMAT [OPTIONS]
```

**Arguments:**
- `INPUT` - Path to input file or directory
- `OUTPUT_FORMAT` - Target format: `gguf`, `awq`, or `gptq`

**Optional Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | auto | Output path |
| `-q, --quant-level` | auto | Target quantization level |

**Examples:**

```bash
# Convert GPTQ to GGUF
llm-quantize convert ./model-gptq gguf -q Q4_K_M

# Convert AWQ to GGUF with custom output
llm-quantize convert ./model-awq gguf -q Q5_K_M -o ./converted.gguf
```

---

### analyze importance

Compute importance matrix for optimized quantization.

```bash
llm-quantize analyze importance MODEL [OPTIONS]
```

**Arguments:**
- `MODEL` - HuggingFace Hub identifier or local directory path

**Optional Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | auto | Output path |
| `--calibration-data` | - | Path to calibration data |
| `--calibration-samples` | 256 | Number of calibration samples |
| `--method` | activation | Method: `activation` or `gradient` |
| `--format` | json | Output format: `json` or `imatrix` |
| `-v, --verbose` | false | Verbose output |
| `--json-output` | false | Output as JSON |

**Examples:**

```bash
# Basic importance analysis
llm-quantize analyze importance meta-llama/Llama-2-7b-hf

# With calibration data
llm-quantize analyze importance meta-llama/Llama-2-7b-hf \
  --calibration-data ./calibration.json \
  --calibration-samples 512

# Gradient-based importance
llm-quantize analyze importance meta-llama/Llama-2-7b-hf \
  --method gradient \
  -o ./importance-matrix.json
```

**Output Fields:**
- `model_name` - Analyzed model name
- `computation_method` - Method used (activation/gradient)
- `layer_scores` - Per-layer importance scores
- `total_parameters` - Total parameter count
- `super_weight_coverage` - Fraction of super weights identified

---

### analyze quality

Generate quality report for quantized models.

```bash
llm-quantize analyze quality MODEL [OPTIONS]
```

**Arguments:**
- `MODEL` - Path to quantized model

**Optional Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--reference` | - | Reference model for comparison |
| `--test-prompts` | - | Path to test prompts file |
| `--json-output` | false | Output as JSON |

---

### analyze profile

Get quantization profile recommendations.

```bash
llm-quantize analyze profile MODEL [OPTIONS]
```

**Arguments:**
- `MODEL` - HuggingFace Hub identifier or local directory path

**Optional Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | balanced | Preset: `quality`, `balanced`, or `speed` |
| `--imatrix` | - | Path to importance matrix |
| `--json-output` | false | Output as JSON |

**Examples:**

```bash
# Get balanced profile
llm-quantize analyze profile meta-llama/Llama-2-7b-hf

# Quality-focused profile
llm-quantize analyze profile meta-llama/Llama-2-7b-hf --preset quality

# With importance matrix
llm-quantize analyze profile meta-llama/Llama-2-7b-hf \
  --imatrix ./importance.json
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace authentication token |
| `HF_HOME` | HuggingFace cache directory |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use |

## JSON Output Format

When using `--format json`, all commands output structured JSON:

### Success Response

```json
{
  "status": "success",
  "output_path": "/path/to/output",
  "format": "gguf",
  "quantization_level": "Q4_K_M",
  "file_size": 4123456789,
  "compression_ratio": 0.25,
  "duration_seconds": 120.5,
  "validation_status": "valid"
}
```

### Error Response

```json
{
  "status": "error",
  "error_code": 3,
  "message": "Model not found: meta-llama/Llama-2-7b-hf"
}
```
