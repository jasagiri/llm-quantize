# Research: LittleBit Ultra-Low-Bit Quantization

**Feature**: 003-littlebit-factorization
**Date**: 2025-12-12
**Reference**: [arXiv:2506.13771](https://arxiv.org/abs/2506.13771)

## Technology Decisions

### 1. Matrix Factorization Algorithm

**Decision**: Implement SVD-based factorization with Dual-SVID initialization

**Rationale**:
- SVD provides optimal low-rank approximation in Frobenius norm
- Dual-SVID (from LittleBit paper) initializes factors for better quantization
- Well-studied algorithm with stable numerical properties

**Implementation**:
```python
def factorize_weight(W, rank):
    """Factorize weight matrix W into low-rank factors."""
    # Standard SVD decomposition
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Truncate to target rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Distribute singular values
    sqrt_S = torch.sqrt(S_r)
    L = U_r * sqrt_S  # Left factor
    R = sqrt_S.unsqueeze(1) * Vh_r  # Right factor

    return L, R
```

**Alternatives Considered**:
- NMF (Non-negative Matrix Factorization): Rejected - doesn't handle negative weights
- Randomized SVD: Considered for large matrices - may add as optimization
- Custom factorization: Rejected - SVD is proven optimal for low-rank

### 2. Binarization Strategy

**Decision**: Sign-based binarization with scale factors

**Rationale**:
- Binary weights (+1/-1) enable efficient storage (1 bit per weight)
- Scale factors compensate for magnitude information
- Matches LittleBit paper's approach

**Implementation**:
```python
def binarize_factor(F, compensation='multi_scale'):
    """Binarize factor matrix with compensation."""
    # Sign binarization
    B = torch.sign(F)
    B[B == 0] = 1  # Handle zeros

    if compensation == 'multi_scale':
        # Row and column scale factors
        row_scales = F.abs().mean(dim=1, keepdim=True)
        col_scales = F.abs().mean(dim=0, keepdim=True)
        scales = torch.sqrt(row_scales * col_scales)
    else:
        scales = F.abs().mean()

    return B, scales
```

**Alternatives Considered**:
- Ternary quantization (-1, 0, +1): Considered - higher BPW but better quality
- Stochastic binarization: Rejected - non-deterministic, harder to reproduce
- Learned binarization: Rejected - requires training, out of scope

### 3. Multi-Scale Compensation

**Decision**: Three-level compensation (row, column, latent)

**Rationale**:
- Row compensation captures output channel importance
- Column compensation captures input feature importance
- Latent compensation captures factorization dimension importance
- Paper shows this achieves best quality preservation

**Compensation Formula**:
```
W_reconstructed = (row_scale ⊙ L_binary) @ (col_scale ⊙ R_binary) + latent_compensation
```

**Alternatives Considered**:
- Single global scale: Rejected - loses too much precision
- Per-element scales: Rejected - defeats compression purpose
- Learned compensation: Considered for future - requires fine-tuning

### 4. Rank Selection Algorithm

**Decision**: BPW-based automatic selection with binary search

**Rationale**:
- Target BPW directly maps to compression ratio
- Binary search efficiently finds optimal rank
- Allows user override for experimentation

**Algorithm**:
```python
def select_rank_for_bpw(W, target_bpw, min_rank=1, max_rank=None):
    """Select rank to achieve target bits per weight."""
    m, n = W.shape
    if max_rank is None:
        max_rank = min(m, n)

    def compute_bpw(rank):
        # Binary factors + FP16 scales
        factor_bits = rank * (m + n) * 1  # Binary
        scale_bits = rank * 2 * 16  # FP16 row/col scales
        total_bits = factor_bits + scale_bits
        return total_bits / (m * n)

    # Binary search for target BPW
    while min_rank < max_rank:
        mid = (min_rank + max_rank) // 2
        if compute_bpw(mid) > target_bpw:
            max_rank = mid
        else:
            min_rank = mid + 1

    return min_rank
```

**Alternatives Considered**:
- Fixed rank: Rejected - doesn't adapt to layer dimensions
- Quality-based selection: Considered as alternative mode
- Exhaustive search: Rejected - too slow for large models

### 5. Native Format (.littlebit)

**Decision**: Custom binary format optimized for factorized storage

**Rationale**:
- Standard formats (GGUF, safetensors) assume dense weights
- Factorized representation needs specialized storage
- Enables efficient loading without reconstruction

**Format Structure**:
```
[Header: 128 bytes]
  - Magic: "LTBT" (4 bytes)
  - Version: uint32 (4 bytes)
  - Model architecture hash: uint64 (8 bytes)
  - Num layers: uint32 (4 bytes)
  - Target BPW: float32 (4 bytes)
  - Reserved: 104 bytes

[Layer Index: variable]
  - Layer name (null-terminated)
  - Offset: uint64
  - Original shape: uint32 x 2
  - Rank: uint32
  - Compensation type: uint8

[Layer Data: variable]
  - Left factor (packed bits)
  - Right factor (packed bits)
  - Row scales (float16)
  - Column scales (float16)
  - Latent compensation (float16)

[Footer]
  - Metadata JSON (variable)
  - Checksum: uint64
```

**Alternatives Considered**:
- Extend GGUF only: Rejected - GGUF assumes quantized dense weights
- Use safetensors: Considered - but adds dependency for factorized tensors
- HDF5: Rejected - overkill for this use case

### 6. GGUF Compatibility

**Decision**: Two-tier approach (extension + lossy conversion)

**Rationale**:
- GGUF extension preserves factorization for compatible loaders
- Lossy conversion enables any GGUF-compatible inference engine
- Users choose based on their deployment needs

**GGUF Extension**:
- Store factorized weights in custom tensor type
- Add metadata for reconstruction parameters
- Requires modified inference engine

**Lossy Conversion**:
- Reconstruct dense weights from factors
- Quantize using standard GGUF types (Q4_K_M, etc.)
- Compatible with any llama.cpp build

**Alternatives Considered**:
- GGUF only: Rejected - loses factorization benefits
- Native only: Rejected - limits deployment options

### 7. Quality Estimation

**Decision**: Perplexity-based with calibration data

**Rationale**:
- Perplexity is standard LLM quality metric
- Can compute without full benchmark suite
- Enables quality threshold enforcement

**Implementation**:
```python
def estimate_quality(original_model, compressed_model, calibration_data):
    """Estimate quality degradation from compression."""
    ppl_original = compute_perplexity(original_model, calibration_data)
    ppl_compressed = compute_perplexity(compressed_model, calibration_data)

    return {
        'perplexity_original': ppl_original,
        'perplexity_compressed': ppl_compressed,
        'perplexity_delta_pct': (ppl_compressed - ppl_original) / ppl_original * 100,
        'quality_score': max(0, 1 - (ppl_compressed - ppl_original) / ppl_original)
    }
```

**Alternatives Considered**:
- Task-specific benchmarks: Considered for future - too slow for compression loop
- Reconstruction error only: Rejected - doesn't reflect downstream quality
- No quality estimation: Rejected - users need feedback

## Best Practices

### Factorization

- Always use double precision for SVD computation, then cast results
- Check condition number before factorization to detect ill-conditioned matrices
- Cache factorization results for repeated compression experiments
- Use randomized SVD for matrices larger than 8192x8192

### Binarization

- Handle exact zeros explicitly (map to +1 by convention)
- Validate binary factor reconstruction error before proceeding
- Use symmetric binarization (-1, +1) not (0, 1) for efficiency

### Compensation

- Compute compensation factors on calibration data, not random inputs
- Validate compensation doesn't introduce numerical instability
- Store compensation in FP16 to balance precision and size

### Quality Management

- Warn users when perplexity increase exceeds 50%
- Provide quality threshold mode for production use cases
- Default to aggressive warning for 0.1 BPW compression

## Integration Patterns

### CLI Pattern

```python
@click.group()
def littlebit():
    """LittleBit ultra-low-bit quantization commands."""
    pass

@littlebit.command()
@click.argument('model')
@click.option('--target-bpw', type=float, default=0.5, help='Target bits per weight')
@click.option('--rank', type=int, default=None, help='Latent rank (auto if not set)')
@click.option('--quality-threshold', type=float, default=None, help='Max perplexity increase %')
@click.option('--output-format', type=click.Choice(['littlebit', 'gguf-ext', 'gguf-lossy']))
def compress(model, target_bpw, rank, quality_threshold, output_format):
    """Compress model using LittleBit factorization."""
    pass
```

### Checkpointing Pattern

```python
class FactorizationCheckpoint(Checkpoint):
    """Checkpoint for LittleBit compression operations."""

    def save_layer_factors(self, layer_idx, left_factor, right_factor, compensation):
        self.layer_states[layer_idx] = {
            'left': left_factor,
            'right': right_factor,
            'compensation': compensation,
            'status': 'completed'
        }
        self._save_metadata()
```

### Progress Pattern

```python
# Progress output to stderr
[llm-quantize] LittleBit compression: meta-llama/Llama-2-7b-hf
[llm-quantize] Target: 0.1 BPW, Auto-rank selection
Factorizing ━━━━━━━━━━━━━━━━━━━━ 100% 32/32 layers [45:23 elapsed]
[llm-quantize] Achieved: 0.11 BPW, 28x compression
[llm-quantize] Quality: Perplexity 5.23 → 8.45 (+61.6%)
[llm-quantize] Warning: Significant quality degradation detected
[llm-quantize] Output: ./Llama-2-7b-hf.littlebit (478 MB)
```

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Output format support? | Three formats: native .littlebit, GGUF extension, lossy GGUF |
| GPU requirement? | GPU recommended, CPU fallback supported (5-10x slower) |
| Checkpointing? | Layer-level checkpoints, consistent with 001/002 |
| Rank selection? | Auto by default based on target BPW, manual override available |
| Quality metrics? | Perplexity-based with threshold enforcement option |

## References

- Lee, B., Kim, D., You, Y., & Kim, Y. (2025). LittleBit: Ultra Low-Bit Quantization via Latent Factorization. NeurIPS 2025. [arXiv:2506.13771](https://arxiv.org/abs/2506.13771)
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.
- GGUF Format Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
