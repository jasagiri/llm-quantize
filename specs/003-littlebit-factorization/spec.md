# Feature Specification: LittleBit Ultra-Low-Bit Quantization

**Feature Branch**: `003-littlebit-factorization`
**Created**: 2025-12-12
**Status**: Draft
**Input**: User description: "LittleBit ultra-low-bit quantization via latent matrix factorization enabling 0.1 BPW compression with 31x memory reduction"
**Depends On**: `001-multi-format-quantization` (base quantization infrastructure)
**Reference**: [arXiv:2506.13771](https://arxiv.org/abs/2506.13771) - "LittleBit: Ultra Low-Bit Quantization via Latent Factorization" (NeurIPS 2025)

## Clarifications

### Session 2025-12-12

- Q: What output format(s) should LittleBit support given factorized weights differ from standard quantized formats? → A: Support all three (native .littlebit format, GGUF metadata extension, and lossy GGUF conversion)
- Q: Is GPU required for LittleBit compression? → A: GPU recommended with CPU fallback supported (degraded performance on CPU-only)
- Q: How should the system handle interrupted compression for long-running jobs? → A: Layer-level checkpoints (resume from last completed layer)

## Research Background

LittleBit is a state-of-the-art ultra-low-bit quantization method that achieves extreme compression through latent matrix factorization:

- **Compression**: 0.1 bits per weight (BPW), enabling ~31x memory reduction
- **Example**: Llama2-13B compressed to under 0.9 GB
- **Performance**: 11.6x speedup over FP16 at kernel level
- **Key Innovation**: Represents weights in low-rank form before binarizing factors

### Core Techniques

1. **Latent Matrix Factorization**: Decomposes weight matrices into low-rank factors before quantization
2. **Multi-Scale Compensation**: Row, column, and latent dimension compensation to preserve precision
3. **Dual Sign-Value-Independent Decomposition (Dual-SVID)**: Initialization method for quantization-aware training
4. **Integrated Residual Compensation**: Minimizes quantization errors during conversion

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extreme Model Compression (Priority: P1)

As an ML engineer deploying LLMs to extremely resource-constrained environments (mobile devices, embedded systems, or low-memory edge servers), I want to compress models to sub-1-bit-per-weight precision so that I can run large models on devices with minimal memory.

**Why this priority**: The primary value proposition of LittleBit is achieving compression levels impossible with traditional quantization (31x vs ~4x for standard 4-bit). This enables entirely new deployment scenarios.

**Independent Test**: Can be fully tested by compressing a 7B model to 0.1 BPW and verifying it fits in target memory budget while producing coherent outputs.

**Acceptance Scenarios**:

1. **Given** a Hugging Face model (e.g., Llama2-7B), **When** I run LittleBit quantization at 0.1 BPW, **Then** the system produces a compressed model approximately 31x smaller than FP16.
2. **Given** a LittleBit-compressed model, **When** I load it for inference, **Then** the system can run inference with memory footprint matching the compressed size.
3. **Given** compression parameters, **When** I request a compression report, **Then** the system outputs actual BPW achieved, memory reduction ratio, and estimated quality metrics.

---

### User Story 2 - Configurable Compression Levels (Priority: P2)

As an ML engineer balancing quality and compression, I want to specify different BPW targets so that I can find the optimal tradeoff for my use case.

**Why this priority**: Not all deployments require maximum compression. Users need flexibility to choose between 0.1 BPW (extreme), 0.5 BPW (aggressive), or higher levels based on quality requirements.

**Independent Test**: Can be fully tested by compressing the same model at different BPW levels and comparing output quality.

**Acceptance Scenarios**:

1. **Given** a source model, **When** I specify target BPW of 0.1, 0.3, 0.5, or 0.7, **Then** the system produces a model at approximately the requested compression level.
2. **Given** different BPW targets, **When** I compare compressed models, **Then** higher BPW models demonstrate measurably better quality (lower perplexity).
3. **Given** an invalid BPW target (e.g., negative or > 16), **When** I run compression, **Then** the system rejects with a clear error message explaining valid ranges.

---

### User Story 3 - Latent Rank Configuration (Priority: P3)

As an ML researcher experimenting with factorization-based compression, I want to configure the latent rank used in matrix factorization so that I can explore the rank-compression-quality tradeoff space.

**Why this priority**: The latent rank directly controls the factorization granularity. Lower ranks enable more compression but may reduce quality. Researchers need this control for experimentation.

**Independent Test**: Can be fully tested by compressing with different rank settings and measuring resulting model size and quality.

**Acceptance Scenarios**:

1. **Given** a source model, **When** I specify a latent rank parameter, **Then** the system uses that rank for matrix factorization during compression.
2. **Given** automatic rank selection mode, **When** I run compression with a target BPW, **Then** the system automatically selects an appropriate rank to achieve the target.
3. **Given** rank configuration, **When** I request factorization statistics, **Then** the system outputs per-layer ranks, compression contribution, and reconstruction error estimates.

---

### User Story 4 - Quality-Aware Compression (Priority: P4)

As an ML engineer who needs guaranteed quality levels, I want to set quality thresholds so that the system automatically adjusts compression to maintain acceptable performance.

**Why this priority**: In production, users often have quality requirements (maximum perplexity increase, minimum benchmark scores) that must be met regardless of compression ratio.

**Independent Test**: Can be fully tested by setting a quality threshold and verifying the system adjusts BPW to meet it.

**Acceptance Scenarios**:

1. **Given** a quality threshold (e.g., max 20% perplexity increase), **When** I run quality-aware compression, **Then** the system iteratively adjusts BPW until quality meets the threshold.
2. **Given** a quality threshold that cannot be met even at maximum BPW, **When** I run compression, **Then** the system warns that the threshold is unachievable and suggests alternatives.
3. **Given** quality-aware compression results, **When** I request a report, **Then** the system outputs iterations attempted, final BPW, and quality metrics achieved.

---

### Edge Cases

- What happens when the target BPW is lower than the model architecture can support?
- How does the system handle models with non-standard layer structures (e.g., MoE, multi-head attention variants)?
- What happens when factorization produces numerically unstable results?
- How does the system handle interrupted compression during long-running operations? → Resolved: Layer-level checkpointing enables resume from last completed layer.
- What happens when the compressed model produces degenerate outputs (repetition, gibberish)?

## Requirements *(mandatory)*

### Functional Requirements

**Core Compression**:
- **FR-001**: System MUST support LittleBit quantization targeting BPW levels from 0.1 to 2.0.
- **FR-002**: System MUST implement latent matrix factorization to decompose weight matrices before binarization.
- **FR-003**: System MUST implement multi-scale compensation (row, column, latent dimension) during compression.
- **FR-004**: System MUST produce compressed models that achieve at least 20x memory reduction at 0.1 BPW compared to FP16.

**Configuration**:
- **FR-005**: System MUST support user-specified target BPW as a compression parameter.
- **FR-006**: System MUST support user-specified latent rank for factorization control.
- **FR-007**: System MUST support automatic rank selection based on target BPW when rank is not specified.
- **FR-008**: System MUST support quality threshold constraints that override BPW targets if quality degrades beyond threshold.

**Output Formats**:
- **FR-009**: System MUST support three output format options: (a) native .littlebit format optimized for factorized weights, (b) GGUF with metadata extension for factorization data, (c) lossy GGUF conversion for compatibility with standard inference engines.
- **FR-010**: System MUST support exporting factorization metadata (ranks, scales, compensation factors) separately from weights.
- **FR-011**: System MUST integrate with existing output formats from 001-multi-format-quantization, using GGUF infrastructure for options (b) and (c).

**Quality and Validation**:
- **FR-012**: System MUST validate compressed model coherence by running sample inference after compression.
- **FR-013**: System MUST compute and report perplexity on calibration data before and after compression.
- **FR-014**: System MUST warn users when compression quality falls below predefined thresholds (severe quality degradation).

**Progress and Reporting**:
- **FR-015**: System MUST report compression progress including current layer, factorization status, and estimated completion time.
- **FR-016**: System MUST output detailed compression statistics including per-layer BPW, rank, and error metrics.
- **FR-017**: System MUST support layer-level checkpointing, saving progress after each completed layer to enable resumption from the last checkpoint if interrupted.

### Key Entities

- **FactorizationConfig**: Configuration for latent factorization; attributes include target_bpw, latent_rank, rank_selection_mode (manual/auto), quality_threshold.
- **FactorizedWeights**: Decomposed weight representation; attributes include left_factor, right_factor, scale_factors, compensation_vectors.
- **CompressionReport**: Results of compression operation; attributes include achieved_bpw, memory_reduction_ratio, per_layer_stats, quality_metrics.
- **CompensationFactors**: Multi-scale compensation data; attributes include row_compensation, column_compensation, latent_compensation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: LittleBit compression achieves at least 25x memory reduction at 0.1 BPW compared to FP16 baseline.
- **SC-002**: Compressed models at 0.1 BPW produce coherent outputs (non-repetitive, grammatically correct) on 70% of test prompts.
- **SC-003**: Compression of a 7B parameter model completes within 2 hours on consumer hardware (single GPU, 16GB VRAM).
- **SC-004**: Quality-aware compression mode achieves user-specified quality thresholds in 90% of cases where the threshold is achievable.
- **SC-005**: Users can compress a model using LittleBit with no more than 2 additional command-line flags beyond basic quantization (target BPW and optional rank).
- **SC-006**: Compressed model inference achieves at least 5x speedup compared to FP16 on compatible hardware.

## Assumptions

- Base quantization infrastructure (001-multi-format-quantization) provides model loading, progress reporting, and output handling.
- Users understand that 0.1 BPW represents experimental/extreme compression with significant quality tradeoffs compared to traditional 4-bit quantization.
- Inference requires specialized kernels optimized for factorized weight representations; standard inference engines may not be compatible.
- Calibration data (WikiText-2 or user-provided) is available for quality estimation and compensation factor computation.
- Target models use standard transformer architectures with linear layers amenable to matrix factorization.
- GPU (CUDA) is recommended for compression; CPU-only mode is supported but with significantly longer compression times (estimated 5-10x slower).

## References

- Lee, B., Kim, D., You, Y., & Kim, Y. (2025). LittleBit: Ultra Low-Bit Quantization via Latent Factorization. NeurIPS 2025. [arXiv:2506.13771](https://arxiv.org/abs/2506.13771)
