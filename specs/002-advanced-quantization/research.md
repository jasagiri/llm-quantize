# Research: Advanced Quantization Methods

**Feature**: 002-advanced-quantization
**Date**: 2025-12-12

## Technology Decisions

### 1. Dynamic Layer-wise Quantization Algorithm

**Decision**: Implement Unsloth-style importance-based layer selection with configurable profiles

**Rationale**:
- Unsloth demonstrated 80% compression (720GB to 131GB) on DeepSeek-R1 while maintaining quality
- Layer importance varies significantly: attention layers often need higher precision than MLP layers
- Profiles allow users to choose between quality and compression tradeoffs

**Implementation Approach**:
- Compute per-layer importance scores using calibration data
- Assign bit-widths based on importance ranking and target compression ratio
- Support preset profiles: "attention-high", "balanced", "compression-max"
- Allow custom per-layer configuration via JSON/YAML

**Alternatives Considered**:
- Uniform quantization: Rejected - suboptimal compression-to-quality ratio
- Gradient-based sensitivity: Considered - more accurate but requires training data
- Random layer selection: Rejected - unpredictable quality impact

### 2. Ultra-Low-Bit Implementation

**Decision**: Leverage llama.cpp's IQ1_S/IQ1_M implementations via llama-cpp-python

**Rationale**:
- llama.cpp has mature, tested implementations of super-block quantization
- IQ1_S (1.5-bit) and IQ1_M (1.75-bit) are well-documented formats
- Direct integration avoids reimplementing complex quantization algorithms

**Quantization Types**:
```
IQ1_S: 1.5 bits per weight (super-block encoding)
IQ1_M: 1.75 bits per weight (medium quality)
Q2_K:  2.0 bits per weight (k-quants)
Ternary: 1.58 bits (-1, 0, 1) for BitNet-style models
```

**Alternatives Considered**:
- Custom Python implementation: Rejected - performance critical, llama.cpp is optimized
- BitNet native: Considered - requires specific model architecture support
- GPTQ 2-bit: Rejected - less mature than k-quants for ultra-low-bit

### 3. SmoothQuant Algorithm

**Decision**: Implement per-channel smoothing with configurable alpha factor

**Rationale**:
- SmoothQuant paper shows 1.56x speedup with negligible accuracy loss
- Mathematical transformation is well-defined: s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
- Per-channel approach handles outlier activations effectively

**Implementation**:
```python
# SmoothQuant transformation
# X: activation tensor, W: weight tensor
# alpha: smoothing factor (default 0.5)
def smooth_transform(X, W, alpha=0.5):
    act_scales = X.abs().max(dim=0).values
    weight_scales = W.abs().max(dim=0).values
    scales = (act_scales ** alpha) / (weight_scales ** (1 - alpha))
    X_smooth = X / scales
    W_smooth = W * scales
    return X_smooth, W_smooth, scales
```

**Alternatives Considered**:
- Per-tensor quantization: Rejected - doesn't handle outliers well
- Outlier-aware quantization: Considered - more complex, SmoothQuant is simpler
- Mixed INT8/FP16: Rejected - SmoothQuant achieves pure INT8

### 4. Importance Matrix Computation

**Decision**: Use activation magnitude analysis with optional gradient sensitivity

**Rationale**:
- Activation magnitude correlates with parameter importance
- Can be computed efficiently during single forward pass
- Compatible with llama.cpp imatrix format for GGUF

**Computation Methods**:
1. **Activation Magnitude** (default): Track max activation values per weight
2. **Gradient Sensitivity** (optional): Requires backward pass, more accurate
3. **Fisher Information** (future): Most accurate, highest compute cost

**Calibration Parameters**:
- Default: 512 samples from WikiText-2
- Configurable: 128-1024 samples
- Custom datasets supported via JSON/JSONL

**Alternatives Considered**:
- Random sampling: Rejected - inconsistent importance estimation
- Full dataset analysis: Rejected - too slow for practical use
- Pre-computed matrices: Considered as cache option - implemented as optional

### 5. Super Weight Identification

**Decision**: Top-k selection based on importance scores with adaptive threshold

**Rationale**:
- Research shows 0.01% of weights disproportionately affect quality
- Simple threshold approach is interpretable and controllable
- Adaptive threshold handles different model architectures

**Algorithm**:
```python
def identify_super_weights(importance_matrix, coverage=0.0001):
    """Identify top 0.01% most important weights."""
    flat_importance = importance_matrix.flatten()
    threshold = np.percentile(flat_importance, 100 * (1 - coverage))
    super_weight_mask = importance_matrix >= threshold
    return super_weight_mask
```

**Protection Strategy**:
- Super weights kept at higher precision (e.g., FP16 or 8-bit)
- Other weights quantized to target precision
- Mixed-precision storage in output format

**Alternatives Considered**:
- Fixed count: Rejected - doesn't adapt to model size
- Layer-wise thresholds: Considered - adds complexity without clear benefit
- Clustering-based: Rejected - too slow for large models

### 6. Quality Estimation

**Decision**: Perplexity-based quality estimation with optional task-specific benchmarks

**Rationale**:
- Perplexity is standard metric for language model quality
- Can be computed efficiently on calibration data
- Provides comparable numbers across different quantization methods

**Metrics Computed**:
- Perplexity (original vs quantized)
- Perplexity delta percentage
- Per-layer quantization error (MSE)
- Effective bits per weight

**Alternatives Considered**:
- Accuracy benchmarks only: Rejected - too slow for quick feedback
- No quality metrics: Rejected - users need feedback on quantization impact
- Custom metrics: Considered for future - perplexity is sufficient for MVP

## Best Practices

### Dynamic Quantization

- Always run importance analysis before dynamic quantization
- Start with "balanced" profile, adjust based on quality metrics
- Monitor per-layer errors to identify problematic layers
- Cache importance matrices for repeated quantization runs

### Ultra-Low-Bit

- Warn users about quality tradeoffs before starting
- Validate output coherence after quantization
- Recommend dynamic quantization for better quality at similar compression
- Use importance-guided quantization (imatrix) for best results

### SmoothQuant

- Collect activation statistics on representative data
- Start with alpha=0.5, tune if needed
- Verify output format compatibility with target inference engine
- Test INT8 inference on actual hardware

### Importance Analysis

- Use diverse calibration data for robust importance estimation
- Save importance matrices for reproducibility
- Compare matrices across different calibration sets for stability
- Layer-level checkpointing for large model analysis

## Integration Patterns

### CLI Extension Pattern

```python
# Pattern: Extend existing quantize command with new options
@click.option('--dynamic', is_flag=True, help='Enable dynamic layer-wise quantization')
@click.option('--profile', type=click.Choice(['attention-high', 'balanced', 'compression-max']))
@click.option('--imatrix', type=click.Path(exists=True), help='Pre-computed importance matrix')
@click.option('--protect-super-weights', is_flag=True, help='Protect critical parameters')
```

### Quality Reporting Pattern

```python
# Pattern: Structured quality report
def generate_quality_report(original_model, quantized_model, calibration_data):
    return {
        "perplexity_original": compute_perplexity(original_model, calibration_data),
        "perplexity_quantized": compute_perplexity(quantized_model, calibration_data),
        "perplexity_delta_pct": ...,
        "layer_errors": [...],
        "super_weight_coverage": ...,
        "effective_bits_per_weight": ...
    }
```

### Checkpoint Integration Pattern

```python
# Pattern: Extend existing checkpoint system
class AnalysisCheckpoint(Checkpoint):
    """Checkpoint for importance analysis operations."""

    def save_layer_importance(self, layer_idx, importance_scores):
        self.layer_states[layer_idx] = importance_scores
        self._save_metadata()

    def resume_from(self, checkpoint_path):
        # Load completed layers, continue from last
        ...
```

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Calibration dataset size? | 512 samples default, 128-1024 configurable |
| Checkpointing for analysis? | Layer-level checkpoints, consistent with 001/003 |
| Super weight threshold? | Top 0.01% based on importance scores |
| SmoothQuant alpha default? | 0.5 (balanced between weights and activations) |
| Quality metric? | Perplexity as primary, layer MSE as secondary |

## References

- Unsloth Dynamic Quantization: https://unsloth.ai/blog/deepseekr1-dynamic
- SmoothQuant Paper: https://arxiv.org/abs/2211.10438
- llama.cpp k-quants: https://github.com/ggerganov/llama.cpp
- BitNet Ternary: https://arxiv.org/abs/2310.11453
- GPTQ Paper: https://arxiv.org/abs/2210.17323
