# Advanced Features

This guide covers advanced quantization features for optimizing quality and performance.

## Importance Matrix Analysis

The importance matrix identifies which layers and weights are most critical for model quality, enabling optimized quantization strategies.

### Computing Importance Matrix

```bash
# Basic importance analysis
llm-quantize analyze importance meta-llama/Llama-2-7b-hf \
  -o importance-matrix.json

# With custom calibration data
llm-quantize analyze importance meta-llama/Llama-2-7b-hf \
  --calibration-data ./my-data.json \
  --calibration-samples 512

# Using gradient-based method
llm-quantize analyze importance meta-llama/Llama-2-7b-hf \
  --method gradient
```

### Importance Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `activation` | Based on activation magnitudes | General use, faster |
| `gradient` | Based on gradient magnitudes | Higher accuracy, slower |

### Using Importance Matrix

The importance matrix can be used for:

1. **Dynamic Quantization**: Different bit widths per layer
2. **Super Weight Protection**: Protect critical weights at higher precision
3. **Quality Prediction**: Estimate quality impact of quantization

```bash
# Apply importance matrix to quantization profile
llm-quantize analyze profile meta-llama/Llama-2-7b-hf \
  --imatrix importance-matrix.json
```

### Output Format

```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "computation_method": "activation_magnitude",
  "calibration_info": {
    "source": "custom",
    "sample_count": 256
  },
  "layer_scores": [
    {
      "layer_name": "model.layers.0.self_attn.q_proj",
      "layer_index": 0,
      "importance_score": 0.95,
      "parameter_count": 16777216
    }
  ],
  "total_parameters": 6738415616,
  "super_weight_coverage": 0.0001
}
```

---

## Super Weight Protection

Super weights are the critical 0.01% of parameters that disproportionately affect model quality. Protecting them during quantization significantly improves output quality.

### Identifying Super Weights

```bash
# Automatic super weight identification during importance analysis
llm-quantize analyze importance meta-llama/Llama-2-7b-hf \
  --identify-super-weights \
  -o importance-with-super-weights.json
```

### Super Weight Statistics

The analysis provides:
- Total super weight count
- Per-layer distribution
- Value statistics (mean, max, min)

### Protection Strategies

1. **Full Precision**: Keep super weights in FP16/FP32
2. **Higher Bit Width**: Use 8-bit for super weights, 4-bit for others
3. **Separate Quantization**: Different quantization for super weight layers

---

## Dynamic Quantization Profiles

Dynamic quantization applies different bit widths to different layers based on their importance.

### Preset Profiles

```bash
# Quality-focused profile (higher bits for important layers)
llm-quantize analyze profile meta-llama/Llama-2-7b-hf --preset quality

# Balanced profile
llm-quantize analyze profile meta-llama/Llama-2-7b-hf --preset balanced

# Speed-focused profile (lower bits overall)
llm-quantize analyze profile meta-llama/Llama-2-7b-hf --preset speed
```

### Profile Output

```json
{
  "preset": "balanced",
  "layer_configs": [
    {
      "layer_pattern": "*.embed_tokens",
      "quantization_method": "Q8_0",
      "rationale": "Embedding layer - high importance"
    },
    {
      "layer_pattern": "*.layers.0-3.*",
      "quantization_method": "Q6_K",
      "rationale": "Early layers - higher importance"
    },
    {
      "layer_pattern": "*.layers.4-28.*",
      "quantization_method": "Q4_K_M",
      "rationale": "Middle layers - standard quantization"
    },
    {
      "layer_pattern": "*.layers.29-31.*",
      "quantization_method": "Q5_K_M",
      "rationale": "Final layers - higher importance"
    }
  ]
}
```

### Layer Types

| Layer Type | Typical Importance | Recommended Bits |
|------------|-------------------|------------------|
| Embeddings | Very High | 6-8 |
| First 10% layers | High | 5-6 |
| Middle layers | Medium | 4 |
| Last 10% layers | High | 5-6 |
| LM Head | Very High | 6-8 |

---

## SmoothQuant (W8A8)

SmoothQuant enables efficient INT8 quantization for both weights and activations.

### How It Works

1. **Activation Scaling**: Migrate quantization difficulty from activations to weights
2. **Smooth Factor**: Apply per-channel scaling with smooth factor α
3. **INT8 Quantization**: Quantize both weights and activations to INT8

### Configuration

```python
from llm_quantize.models import SmoothQuantConfig

config = SmoothQuantConfig(
    alpha=0.5,  # Smoothing factor (0.0-1.0)
    calibration_samples=256,
    per_channel_quantization=True,
    dynamic_activation=False
)
```

### Alpha Parameter

| Alpha | Description | Use Case |
|-------|-------------|----------|
| 0.0 | No smoothing | Baseline |
| 0.5 | Balanced | Recommended default |
| 0.75 | More weight quantization | Activation-heavy models |
| 1.0 | Full smoothing | Maximum activation protection |

---

## Quality Analysis

Analyze quantization quality to compare different strategies.

### Perplexity Evaluation

```bash
llm-quantize analyze quality ./quantized-model \
  --reference meta-llama/Llama-2-7b-hf \
  --test-data ./test-corpus.txt
```

### Coherence Testing

Tests the model's ability to generate coherent text:

```bash
llm-quantize analyze quality ./quantized-model \
  --test-prompts ./prompts.json
```

### Quality Report

```json
{
  "model_name": "llama-7b-Q4_K_M",
  "quantization_format": "gguf",
  "quantization_level": "Q4_K_M",
  "perplexity_original": 5.2,
  "perplexity_quantized": 5.8,
  "perplexity_increase_percent": 11.5,
  "coherence_results": [
    {
      "prompt": "The capital of France is",
      "output": "Paris, which is known for...",
      "is_coherent": true,
      "repetition_score": 0.95,
      "grammar_score": 0.98
    }
  ],
  "quality_grade": "GOOD",
  "warnings": [],
  "recommendations": []
}
```

### Quality Grades

| Grade | Perplexity Increase | Coherence | Description |
|-------|---------------------|-----------|-------------|
| EXCELLENT | <5% | >95% | Minimal quality loss |
| GOOD | 5-15% | >85% | Acceptable for most uses |
| ACCEPTABLE | 15-30% | >70% | Noticeable but usable |
| DEGRADED | 30-50% | >50% | Significant quality loss |
| FAILED | >50% | <50% | Not recommended |

---

## Checkpointing

For large models, checkpointing allows resuming interrupted quantization.

### Enable Checkpointing

```bash
# Default checkpoint directory
llm-quantize quantize meta-llama/Llama-2-70b-hf gguf -q Q4_K_M

# Custom checkpoint directory
llm-quantize quantize meta-llama/Llama-2-70b-hf gguf -q Q4_K_M \
  --checkpoint-dir ./my-checkpoints
```

### Resume from Checkpoint

```bash
llm-quantize quantize meta-llama/Llama-2-70b-hf gguf -q Q4_K_M \
  --resume ./my-checkpoints/gguf-Q4_K_M
```

### Checkpoint Structure

```
.checkpoint/
└── gguf-Q4_K_M/
    ├── state.json          # Overall state
    ├── layer_0.pkl         # Layer 0 weights
    ├── layer_1.pkl         # Layer 1 weights
    └── ...
```

### Disable Checkpointing

For smaller models where checkpointing overhead isn't needed:

```bash
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M \
  --no-checkpoints
```

---

## Format Conversion

Convert between quantized formats.

### Supported Conversions

| From | To | Quality |
|------|-----|---------|
| GPTQ | GGUF | Good |
| AWQ | GGUF | Good |
| GGUF | AWQ | Limited |
| GGUF | GPTQ | Limited |

### Convert GPTQ to GGUF

```bash
llm-quantize convert ./model-gptq gguf -q Q4_K_M
```

### Convert AWQ to GGUF

```bash
llm-quantize convert ./model-awq gguf -q Q5_K_M
```

### Conversion Considerations

- Converting from higher to lower bit width is lossy
- GGUF conversion works best from GPTQ/AWQ sources
- Some metadata may be lost during conversion
