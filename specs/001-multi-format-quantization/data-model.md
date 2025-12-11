# Data Model: Multi-Format Quantization Tool

**Feature**: 001-multi-format-quantization
**Date**: 2025-12-12

## Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   SourceModel   │────▶│  QuantizationConfig │────▶│ QuantizedModel  │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         │
┌─────────────────┐     ┌─────────────────────┐            │
│  ModelMetadata  │     │  QuantizationJob    │◀───────────┘
└─────────────────┘     └─────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │    Checkpoint       │
                        └─────────────────────┘
```

## Entities

### SourceModel

Represents the input model to be quantized.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model_path | str | Yes | HF Hub identifier or local path |
| model_type | ModelType | Yes | Enum: HF_HUB, LOCAL_DIR |
| architecture | str | Yes | Model architecture (e.g., "LlamaForCausalLM") |
| parameter_count | int | Yes | Total parameter count |
| dtype | str | Yes | Original dtype (e.g., "float16", "bfloat16") |
| tokenizer_path | str | No | Path to tokenizer if separate |
| hf_token | str | No | HF authentication token |

**Validation Rules**:
- `model_path` must be valid HF identifier or existing directory
- `architecture` must be in supported architectures list
- `parameter_count` must be > 0

**State Transitions**: N/A (immutable after loading)

### QuantizationConfig

Configuration for the quantization process.

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| target_format | OutputFormat | Yes | - | Enum: GGUF, AWQ, GPTQ |
| quantization_level | str | Yes | - | Format-specific (e.g., "Q4_K_M", "4bit") |
| output_dir | str | No | "." | Output directory path |
| output_name | str | No | auto | Output filename (auto-generated if not set) |
| calibration_data_path | str | No | None | Custom calibration dataset |
| calibration_samples | int | No | 256 | Number of calibration samples |
| group_size | int | No | 128 | GPTQ group size |
| enable_checkpoints | bool | No | True | Enable layer-level checkpoints |
| checkpoint_dir | str | No | auto | Checkpoint directory |
| verbosity | Verbosity | No | NORMAL | Enum: QUIET, NORMAL, VERBOSE, DEBUG |
| output_format | OutputMode | No | HUMAN | Enum: HUMAN, JSON |

**Validation Rules**:
- `quantization_level` must be valid for `target_format`
- `calibration_samples` must be 1-1024
- `group_size` must be power of 2 (32, 64, 128, 256)

### QuantizedModel

Represents the output quantized model.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| output_path | str | Yes | Path to output file(s) |
| format | OutputFormat | Yes | Output format used |
| file_size | int | Yes | Output file size in bytes |
| compression_ratio | float | Yes | Compression vs original |
| quantization_metadata | dict | Yes | Format-specific metadata |
| created_at | datetime | Yes | Creation timestamp |
| source_model_path | str | Yes | Reference to source |
| validation_status | ValidationStatus | Yes | Enum: VALID, INVALID, UNCHECKED |

**Validation Rules**:
- `output_path` must exist and be readable
- `compression_ratio` must be > 0 and < 1
- `validation_status` should be VALID for successful quantization

### QuantizationJob

Tracks the state of a quantization operation.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | str | Yes | Unique job identifier (UUID) |
| status | JobStatus | Yes | Enum: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED |
| progress_percentage | float | Yes | 0.0 to 100.0 |
| current_layer | int | No | Current layer being processed |
| total_layers | int | No | Total layers to process |
| start_time | datetime | No | Job start timestamp |
| end_time | datetime | No | Job completion timestamp |
| estimated_completion | datetime | No | ETA based on current progress |
| peak_memory_usage | int | No | Peak memory in bytes |
| current_memory_usage | int | No | Current memory in bytes |
| error_message | str | No | Error details if failed |
| checkpoint_path | str | No | Path to latest checkpoint |

**State Transitions**:
```
PENDING → RUNNING → COMPLETED
    │         │
    │         ├→ FAILED
    │         │
    └─────────├→ CANCELLED
```

### Checkpoint

Stores intermediate quantization state for resume capability.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| checkpoint_id | str | Yes | Unique checkpoint identifier |
| job_id | str | Yes | Reference to parent job |
| layer_index | int | Yes | Last completed layer index |
| layer_states | dict | Yes | Per-layer quantized state |
| config_snapshot | dict | Yes | QuantizationConfig at checkpoint time |
| created_at | datetime | Yes | Checkpoint timestamp |
| file_path | str | Yes | Path to checkpoint file |

**Validation Rules**:
- `layer_index` must be >= 0 and < total_layers
- `config_snapshot` must match current config for resume

### ModelMetadata

Extracted metadata from source model.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model_name | str | Yes | Model name/identifier |
| architecture | str | Yes | Architecture class name |
| hidden_size | int | Yes | Hidden dimension |
| num_layers | int | Yes | Number of transformer layers |
| num_heads | int | Yes | Number of attention heads |
| vocab_size | int | Yes | Vocabulary size |
| max_position_embeddings | int | Yes | Maximum sequence length |
| torch_dtype | str | Yes | Original weight dtype |

## Enumerations

### OutputFormat
```python
class OutputFormat(Enum):
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
```

### ModelType
```python
class ModelType(Enum):
    HF_HUB = "hf_hub"
    LOCAL_DIR = "local_dir"
```

### JobStatus
```python
class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### Verbosity
```python
class Verbosity(Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"
```

### OutputMode
```python
class OutputMode(Enum):
    HUMAN = "human"
    JSON = "json"
```

### ValidationStatus
```python
class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCHECKED = "unchecked"
```

## GGUF Quantization Levels

```python
GGUF_QUANT_TYPES = [
    "Q2_K",    # 2-bit k-quants
    "Q3_K_S",  # 3-bit k-quants small
    "Q3_K_M",  # 3-bit k-quants medium
    "Q3_K_L",  # 3-bit k-quants large
    "Q4_0",    # 4-bit legacy
    "Q4_1",    # 4-bit legacy with scales
    "Q4_K_S",  # 4-bit k-quants small
    "Q4_K_M",  # 4-bit k-quants medium (recommended)
    "Q5_0",    # 5-bit legacy
    "Q5_1",    # 5-bit legacy with scales
    "Q5_K_S",  # 5-bit k-quants small
    "Q5_K_M",  # 5-bit k-quants medium
    "Q6_K",    # 6-bit k-quants
    "Q8_0",    # 8-bit
]
```

## Data Volume Assumptions

| Model Size | Layers | Memory (FP16) | Memory (Q4) | Checkpoint Size |
|------------|--------|---------------|-------------|-----------------|
| 7B | ~32 | ~14GB | ~4GB | ~500MB |
| 13B | ~40 | ~26GB | ~8GB | ~1GB |
| 70B | ~80 | ~140GB | ~35GB | ~5GB |
