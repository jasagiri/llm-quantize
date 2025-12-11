# Data Model: Advanced Quantization Methods

**Feature**: 002-advanced-quantization
**Date**: 2025-12-12

## Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   SourceModel   │────▶│  ImportanceMatrix   │────▶│ LayerQuantConfig    │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  SmoothQuant    │     │  SuperWeightMask    │     │ DynamicQuantProfile │
│  Config         │     │                     │     │                     │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
        │                         │                         │
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │ QuantizationQuality │
                        │ Report              │
                        └─────────────────────┘
```

## Entities

### ImportanceMatrix

Stores computed importance scores for model parameters.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| matrix_id | str | Yes | Unique identifier (UUID) |
| model_path | str | Yes | Source model path |
| computation_method | ImportanceMethod | Yes | Enum: ACTIVATION_MAGNITUDE, GRADIENT, FISHER |
| layer_scores | dict[str, ndarray] | Yes | Per-layer importance scores |
| global_statistics | dict | Yes | Min, max, mean, std across all layers |
| calibration_info | CalibrationInfo | Yes | Details about calibration data used |
| created_at | datetime | Yes | Computation timestamp |
| file_path | str | No | Path to saved matrix file |

**Validation Rules**:
- `layer_scores` keys must match model layer names
- All score arrays must be non-negative
- `calibration_info` must have at least 128 samples

**File Format**: `.imatrix` (JSON header + binary scores, compatible with llama.cpp)

### CalibrationInfo

Metadata about calibration dataset used for importance analysis.

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_name | str | Yes | - | Dataset identifier (e.g., "wikitext-2") |
| num_samples | int | Yes | 512 | Number of samples used |
| sequence_length | int | No | 2048 | Maximum sequence length |
| source_path | str | No | None | Path to custom dataset file |
| hash | str | No | None | SHA256 hash of calibration data |

**Validation Rules**:
- `num_samples` must be 128-1024
- `sequence_length` must be > 0

### LayerQuantizationConfig

Per-layer quantization settings for dynamic quantization.

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| layer_name | str | Yes | - | Layer identifier |
| bit_width | float | Yes | - | Target bits per weight (1.5, 2, 4, 8, etc.) |
| quant_method | QuantMethod | No | AUTO | Enum: AUTO, K_QUANTS, GPTQ, AWQ |
| is_protected | bool | No | False | Whether layer contains protected super weights |
| importance_score | float | No | None | Layer-level importance score |
| estimated_error | float | No | None | Estimated quantization error |

**Validation Rules**:
- `bit_width` must be in [1.5, 2, 3, 4, 5, 6, 8, 16]
- `importance_score` must be >= 0 if provided

### DynamicQuantizationProfile

Preset or custom profile for layer-wise quantization.

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| profile_id | str | Yes | - | Profile identifier |
| profile_name | str | Yes | - | Human-readable name |
| description | str | No | None | Profile description |
| layer_assignments | list[LayerQuantizationConfig] | Yes | - | Per-layer configurations |
| target_compression | float | No | None | Target compression ratio |
| target_bpw | float | No | None | Target average bits per weight |
| strategy | ProfileStrategy | Yes | - | Enum: ATTENTION_HIGH, BALANCED, COMPRESSION_MAX, CUSTOM |

**Preset Profiles**:
- `attention-high`: Attention layers at 4-bit, MLP at 2-bit
- `balanced`: Even distribution based on importance
- `compression-max`: Minimize size while maintaining coherence

### SuperWeightMask

Boolean mask identifying critical parameters to protect.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| mask_id | str | Yes | Unique identifier |
| source_matrix_id | str | Yes | Reference to ImportanceMatrix |
| coverage | float | Yes | Fraction of weights marked as super (e.g., 0.0001) |
| layer_masks | dict[str, ndarray] | Yes | Per-layer boolean masks |
| total_protected | int | Yes | Total number of protected parameters |
| protection_precision | int | Yes | Bits for protected weights (8 or 16) |

**Validation Rules**:
- `coverage` must be in (0, 0.01] - max 1% of weights
- `layer_masks` must match source matrix dimensions

### SmoothQuantConfig

Configuration for SmoothQuant W8A8 transformation.

| Attribute | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| config_id | str | Yes | - | Unique identifier |
| alpha | float | Yes | 0.5 | Smoothing factor (0.0-1.0) |
| per_channel_scales | dict[str, ndarray] | No | None | Computed smoothing scales |
| activation_stats | ActivationStatistics | No | None | Collected activation statistics |
| weight_target_bits | int | No | 8 | Target weight precision |
| activation_target_bits | int | No | 8 | Target activation precision |
| symmetric | bool | No | True | Use symmetric quantization |

**Validation Rules**:
- `alpha` must be in [0.0, 1.0]
- `weight_target_bits` and `activation_target_bits` must be 8 for W8A8

### ActivationStatistics

Statistics collected from activation values during calibration.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| layer_name | str | Yes | Layer identifier |
| max_values | ndarray | Yes | Per-channel maximum absolute values |
| min_values | ndarray | Yes | Per-channel minimum values |
| mean_values | ndarray | Yes | Per-channel means |
| std_values | ndarray | Yes | Per-channel standard deviations |
| outlier_count | int | Yes | Number of outlier activations detected |

### QuantizationQualityReport

Quality assessment results after quantization.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| report_id | str | Yes | Unique identifier |
| model_path | str | Yes | Source model path |
| quantized_path | str | Yes | Output model path |
| perplexity_original | float | Yes | Perplexity before quantization |
| perplexity_quantized | float | Yes | Perplexity after quantization |
| perplexity_delta_pct | float | Yes | Percentage change in perplexity |
| layer_errors | list[LayerError] | Yes | Per-layer quantization errors |
| effective_bpw | float | Yes | Effective bits per weight |
| super_weight_coverage | float | No | Fraction of weights protected |
| coherence_score | float | No | Output coherence metric (0-1) |
| timestamp | datetime | Yes | Report generation time |

### LayerError

Per-layer quantization error metrics.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| layer_name | str | Yes | Layer identifier |
| mse | float | Yes | Mean squared error |
| max_error | float | Yes | Maximum absolute error |
| bit_width | float | Yes | Bit width used for this layer |
| parameter_count | int | Yes | Number of parameters in layer |

## Enumerations

### ImportanceMethod
```python
class ImportanceMethod(Enum):
    ACTIVATION_MAGNITUDE = "activation_magnitude"  # Default
    GRADIENT_SENSITIVITY = "gradient_sensitivity"
    FISHER_INFORMATION = "fisher_information"
```

### ProfileStrategy
```python
class ProfileStrategy(Enum):
    ATTENTION_HIGH = "attention_high"      # Attention 4-bit, MLP 2-bit
    BALANCED = "balanced"                   # Importance-based distribution
    COMPRESSION_MAX = "compression_max"     # Maximize compression
    CUSTOM = "custom"                       # User-defined
```

### QuantMethod
```python
class QuantMethod(Enum):
    AUTO = "auto"           # Select based on bit width
    K_QUANTS = "k_quants"   # llama.cpp k-quants
    GPTQ = "gptq"           # GPTQ algorithm
    AWQ = "awq"             # AWQ algorithm
```

### UltraLowBitType
```python
class UltraLowBitType(Enum):
    IQ1_S = "IQ1_S"         # 1.5 bits per weight
    IQ1_M = "IQ1_M"         # 1.75 bits per weight
    Q2_K = "Q2_K"           # 2 bits per weight
    TERNARY = "ternary"     # 1.58 bits (-1, 0, 1)
```

## Data Volume Assumptions

### Importance Matrix Sizes

| Model Size | Layers | Matrix Size (dense) | Matrix Size (sparse) |
|------------|--------|---------------------|----------------------|
| 7B | ~32 | ~1 GB | ~100 MB |
| 13B | ~40 | ~2 GB | ~200 MB |
| 70B | ~80 | ~10 GB | ~1 GB |

Note: Sparse storage used when only storing top-k importance values.

### Calibration Data Requirements

| Samples | Typical Size | Processing Time (7B) |
|---------|--------------|---------------------|
| 128 | ~50 MB | ~5 min |
| 256 | ~100 MB | ~10 min |
| 512 | ~200 MB | ~20 min |
| 1024 | ~400 MB | ~40 min |

### Quality Report Sizes

- Per report: ~10-50 KB (JSON)
- Per-layer errors: ~1 KB per layer
- Full report with layer details: ~100 KB for 70B model

## File Format Specifications

### Importance Matrix (.imatrix)

```
[Header: 64 bytes]
  - Magic: "IMAT" (4 bytes)
  - Version: uint32 (4 bytes)
  - Method: uint32 (4 bytes)
  - Num Layers: uint32 (4 bytes)
  - Reserved: 48 bytes

[Layer Index: variable]
  - Layer name (null-terminated string)
  - Offset: uint64
  - Shape: uint32 x 2

[Layer Data: variable]
  - Importance scores (float32 array)
```

### Dynamic Profile (.dqprofile)

JSON format:
```json
{
  "profile_id": "uuid",
  "profile_name": "balanced",
  "strategy": "balanced",
  "layer_assignments": [
    {
      "layer_name": "model.layers.0.self_attn",
      "bit_width": 4.0,
      "is_protected": false
    }
  ],
  "target_bpw": 3.5
}
```

### Quality Report (.qreport)

JSON format matching QuantizationQualityReport entity schema.
