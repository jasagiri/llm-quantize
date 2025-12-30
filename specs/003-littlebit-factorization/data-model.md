# Data Model: LittleBit Ultra-Low-Bit Quantization

**Feature**: 003-littlebit-factorization
**Date**: 2025-12-12

## Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   SourceModel   │────▶│ FactorizationConfig │────▶│  FactorizedWeights  │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  CalibrationData│     │ CompensationFactors │     │  BinaryFactors      │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │  CompressionReport  │
                        └─────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │  CompressedModel    │
                        └─────────────────────┘
```

## Entities

### FactorizationConfig

Configuration for LittleBit compression operation.

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| config_id | str | Yes | - | Unique identifier (UUID) |
| target_bpw | float | Yes | 0.5 | Target bits per weight (0.1-2.0) |
| latent_rank | int | No | None | Manual rank setting (auto if None) |
| rank_selection_mode | RankMode | No | AUTO | Enum: AUTO, MANUAL |
| quality_threshold | float | No | None | Max perplexity increase % before abort |
| output_format | LittleBitFormat | Yes | NATIVE | Enum: NATIVE, GGUF_EXT, GGUF_LOSSY |
| output_dir | str | No | "." | Output directory |
| output_name | str | No | auto | Output filename |
| enable_checkpoints | bool | No | True | Enable layer-level checkpointing |
| checkpoint_dir | str | No | auto | Checkpoint directory |
| calibration_samples | int | No | 256 | Samples for quality estimation |

**Validation Rules**:
- `target_bpw` must be in [0.1, 2.0]
- `latent_rank` must be > 0 if provided
- `quality_threshold` must be > 0 if provided
- `calibration_samples` must be in [64, 1024]

### FactorizedWeights

Decomposed weight representation for a single layer.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| layer_name | str | Yes | Layer identifier |
| original_shape | tuple[int, int] | Yes | Original weight matrix shape (m, n) |
| rank | int | Yes | Factorization rank used |
| left_factor | BinaryFactor | Yes | Left factor (m x rank) |
| right_factor | BinaryFactor | Yes | Right factor (rank x n) |
| compensation | CompensationFactors | Yes | Multi-scale compensation data |
| achieved_bpw | float | Yes | Actual bits per weight achieved |
| reconstruction_error | float | Yes | Frobenius norm of reconstruction error |

**Validation Rules**:
- `rank` must be <= min(original_shape)
- `achieved_bpw` must be > 0
- Factor dimensions must be consistent with `original_shape` and `rank`

### BinaryFactor

Binary factor matrix with scale information.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| values | ndarray | Yes | Binary values (-1, +1) as packed bits |
| shape | tuple[int, int] | Yes | Factor shape |
| scales | ndarray | Yes | Per-row or per-column scale factors (FP16) |
| scale_type | ScaleType | Yes | Enum: ROW, COLUMN, GLOBAL |

**Validation Rules**:
- `values` must contain only -1 or +1 (logically)
- `scales` dimensions must match `scale_type`

### CompensationFactors

Multi-scale compensation data for reconstruction.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| row_compensation | ndarray | Yes | Row-wise compensation (m,) |
| column_compensation | ndarray | Yes | Column-wise compensation (n,) |
| latent_compensation | ndarray | Yes | Latent dimension compensation (rank,) |
| residual | ndarray | No | Optional residual correction matrix |
| dtype | str | Yes | Storage dtype (typically "float16") |

**Validation Rules**:
- Compensation arrays must be finite (no NaN/Inf)
- Dimensions must match corresponding layer dimensions

### CompressionReport

Results of LittleBit compression operation.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| report_id | str | Yes | Unique identifier |
| model_path | str | Yes | Source model path |
| output_path | str | Yes | Compressed model path |
| output_format | LittleBitFormat | Yes | Format used |
| target_bpw | float | Yes | Requested BPW |
| achieved_bpw | float | Yes | Actual average BPW |
| memory_reduction_ratio | float | Yes | Compression ratio vs FP16 |
| original_size_bytes | int | Yes | Original model size |
| compressed_size_bytes | int | Yes | Compressed model size |
| per_layer_stats | list[LayerStats] | Yes | Per-layer compression statistics |
| quality_metrics | QualityMetrics | No | Quality estimation results |
| duration_seconds | float | Yes | Total compression time |
| peak_memory_bytes | int | Yes | Peak memory usage |
| timestamp | datetime | Yes | Completion timestamp |

### LayerStats

Per-layer compression statistics.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| layer_name | str | Yes | Layer identifier |
| original_shape | tuple[int, int] | Yes | Original weight shape |
| rank | int | Yes | Rank used for factorization |
| achieved_bpw | float | Yes | Layer-specific BPW |
| reconstruction_error | float | Yes | Reconstruction error (Frobenius norm) |
| compression_ratio | float | Yes | Layer-specific compression ratio |
| factorization_time_ms | int | Yes | Time for factorization |
| binarization_time_ms | int | Yes | Time for binarization |

### QualityMetrics

Quality estimation results.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| perplexity_original | float | Yes | Perplexity before compression |
| perplexity_compressed | float | Yes | Perplexity after compression |
| perplexity_delta_pct | float | Yes | Percentage change |
| coherence_score | float | No | Output coherence metric (0-1) |
| calibration_samples | int | Yes | Number of samples used |
| calibration_dataset | str | Yes | Dataset name |
| quality_passed | bool | Yes | Whether threshold was met |

### CompressedModel

Output compressed model representation.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model_id | str | Yes | Unique identifier |
| source_model | str | Yes | Original model path |
| output_path | str | Yes | Path to compressed file |
| format | LittleBitFormat | Yes | Output format |
| architecture | str | Yes | Model architecture |
| num_layers | int | Yes | Number of transformer layers |
| total_parameters | int | Yes | Original parameter count |
| achieved_bpw | float | Yes | Average bits per weight |
| file_size_bytes | int | Yes | Output file size |
| created_at | datetime | Yes | Creation timestamp |
| metadata | dict | Yes | Additional model metadata |

## Enumerations

### RankMode
```python
class RankMode(Enum):
    AUTO = "auto"      # Automatic selection based on target BPW
    MANUAL = "manual"  # User-specified rank
```

### LittleBitFormat
```python
class LittleBitFormat(Enum):
    NATIVE = "littlebit"     # Native .littlebit format
    GGUF_EXT = "gguf-ext"    # GGUF with factorization extension
    GGUF_LOSSY = "gguf-lossy" # Lossy conversion to standard GGUF
```

### ScaleType
```python
class ScaleType(Enum):
    ROW = "row"        # Per-row scale factors
    COLUMN = "column"  # Per-column scale factors
    GLOBAL = "global"  # Single global scale
```

### CompressionStatus
```python
class CompressionStatus(Enum):
    PENDING = "pending"
    FACTORIZING = "factorizing"
    BINARIZING = "binarizing"
    COMPENSATING = "compensating"
    VALIDATING = "validating"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"
```

## BPW to Rank Mapping

For a weight matrix of shape (m, n), the relationship between rank and BPW:

```
BPW = (rank * (m + n) * 1 + rank * 2 * 16) / (m * n)
    = rank * (m + n + 32) / (m * n)

Solving for rank:
rank = BPW * m * n / (m + n + 32)
```

Example for 4096x4096 layer:
| Target BPW | Approximate Rank | Actual BPW |
|------------|------------------|------------|
| 0.1 | 50 | 0.098 |
| 0.3 | 150 | 0.294 |
| 0.5 | 250 | 0.490 |
| 1.0 | 500 | 0.980 |

## Data Volume Assumptions

### Compressed Model Sizes

| Model | FP16 Size | 0.1 BPW | 0.5 BPW | 1.0 BPW |
|-------|-----------|---------|---------|---------|
| 7B | ~14 GB | ~450 MB | ~2.2 GB | ~4.4 GB |
| 13B | ~26 GB | ~840 MB | ~4.2 GB | ~8.3 GB |
| 70B | ~140 GB | ~4.5 GB | ~22 GB | ~44 GB |

### Checkpoint Sizes

Per-layer checkpoint includes:
- Binary factors (packed): ~2 * rank * max(m, n) / 8 bytes
- Scale factors (FP16): ~2 * rank * 2 bytes
- Compensation (FP16): ~(m + n + rank) * 2 bytes

For 7B model at 0.1 BPW: ~500 MB total checkpoint

### Memory Requirements

Peak memory during compression:
- Original weights (FP16): m * n * 2 bytes
- SVD workspace: ~3 * max(m, n) * min(m, n) * 4 bytes
- Factors + compensation: ~2 * rank * (m + n) * 2 bytes

For 7B model: ~20-24 GB peak (single layer at a time)

## File Format Specifications

### Native .littlebit Format

```
[Header: 128 bytes]
  Magic: "LTBT" (4)
  Version: uint32 (4)
  Arch hash: uint64 (8)
  Num layers: uint32 (4)
  Target BPW: float32 (4)
  Achieved BPW: float32 (4)
  Original params: uint64 (8)
  Reserved: 92 bytes

[Layer Table: 64 bytes per layer]
  Name hash: uint64 (8)
  Offset: uint64 (8)
  Original shape: uint32 x 2 (8)
  Rank: uint32 (4)
  Achieved BPW: float32 (4)
  Compensation type: uint8 (1)
  Reserved: 31 bytes

[Layer Data: variable per layer]
  Left factor bits: ceil(m * rank / 8) bytes
  Right factor bits: ceil(rank * n / 8) bytes
  Row scales: m * 2 bytes (FP16)
  Col scales: n * 2 bytes (FP16)
  Latent scales: rank * 2 bytes (FP16)

[Metadata: variable]
  JSON length: uint32
  JSON data: variable

[Footer: 16 bytes]
  Checksum: uint64
  Magic: "LTBT" (4)
  Reserved: 4 bytes
```

### GGUF Extension Metadata

When using GGUF_EXT format, add these metadata keys:
```
littlebit.version: uint32
littlebit.target_bpw: float32
littlebit.achieved_bpw: float32
littlebit.layer.{name}.rank: uint32
littlebit.layer.{name}.compensation_type: string
```

Tensor naming convention:
```
{layer_name}.left_factor    # Packed binary
{layer_name}.right_factor   # Packed binary
{layer_name}.row_scales     # FP16
{layer_name}.col_scales     # FP16
{layer_name}.latent_scales  # FP16
```
