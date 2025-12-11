# Feature Specification: Advanced Quantization Methods

**Feature Branch**: `002-advanced-quantization`
**Created**: 2025-12-12
**Status**: Draft
**Input**: User description: "Advanced quantization methods including dynamic layer-wise quantization, ultra-low-bit (1.5/1.58/2-bit), SmoothQuant W8A8, and importance-aware super weight protection"
**Depends On**: `001-multi-format-quantization` (base quantization infrastructure)

## Clarifications

### Session 2025-12-12

- Q: What calibration dataset size should be used for importance analysis? → A: Default 512 samples, configurable 128-1024
- Q: Should checkpointing be supported for long importance analysis operations? → A: Layer-level checkpoints (consistent with 001/003)

## Research Background

This feature implements state-of-the-art quantization techniques based on recent research:

- **Dynamic Quantization**: Layer-selective quantization (Unsloth AI approach) achieving 80% model size reduction while maintaining quality
- **Ultra-Low-Bit**: IQ1_S (1.5-bit), IQ1_M (1.75-bit), and 1.58-bit ternary (-1,0,1) quantization from llama.cpp and BitNet research
- **SmoothQuant**: Activation-aware W8A8 quantization enabling both weight and activation quantization
- **Super Weight Protection**: Importance-based parameter identification preserving critical 0.01% of weights

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Dynamic Layer-wise Quantization (Priority: P1)

As an ML engineer working with very large models (100B+ parameters), I want to apply different quantization levels to different layers based on their importance so that I can achieve maximum compression while preserving model quality.

**Why this priority**: Dynamic quantization provides the best compression-to-quality ratio for large models. Unsloth demonstrated 80% size reduction (720GB to 131GB) on DeepSeek-R1 while maintaining functionality.

**Independent Test**: Can be fully tested by quantizing a model with layer-wise analysis, comparing output quality against uniform quantization at the same average bit-width.

**Acceptance Scenarios**:

1. **Given** a large model and an importance analysis, **When** I run dynamic quantization specifying high-precision layers (4-bit) and low-precision layers (1.5-bit), **Then** the system produces a model with the specified mixed precision configuration.
2. **Given** a model without pre-computed importance scores, **When** I run dynamic quantization with auto-analysis enabled, **Then** the system computes layer importance using calibration data and applies appropriate precision levels.
3. **Given** dynamic quantization settings, **When** I request a quantization report, **Then** the system outputs per-layer bit-widths, compression ratio, and estimated quality impact.

---

### User Story 2 - Ultra-Low-Bit Quantization (Priority: P2)

As an ML engineer deploying models to resource-constrained environments, I want to quantize models to ultra-low precision (1.5-2 bits) so that I can run inference on devices with very limited memory.

**Why this priority**: Ultra-low-bit enables running models on edge devices and consumer hardware where 4-bit quantization is still too large. Critical for democratizing LLM access.

**Independent Test**: Can be fully tested by quantizing to IQ1_S/IQ1_M format and verifying the model produces coherent outputs on benchmark prompts.

**Acceptance Scenarios**:

1. **Given** a source model, **When** I run quantization with IQ1_S (1.5-bit) mode, **Then** the system produces a quantized model approximately 10x smaller than FP16.
2. **Given** a source model, **When** I run quantization with 2-bit or ternary (-1,0,1) mode, **Then** the system produces a model with the specified precision level.
3. **Given** ultra-low-bit quantization, **When** the quality degradation exceeds a threshold, **Then** the system warns the user and suggests using hybrid/dynamic quantization instead.

---

### User Story 3 - SmoothQuant W8A8 (Priority: P3)

As an ML engineer optimizing for inference throughput, I want to quantize both weights and activations to 8-bit so that I can leverage INT8 hardware acceleration on GPUs and specialized accelerators.

**Why this priority**: W8A8 quantization enables hardware-efficient inference with minimal quality loss. SmoothQuant achieves 1.56x speedup and 2x memory reduction with negligible accuracy impact.

**Independent Test**: Can be fully tested by applying SmoothQuant to a model and measuring inference speed improvement on INT8-capable hardware.

**Acceptance Scenarios**:

1. **Given** a source model, **When** I run SmoothQuant quantization, **Then** the system mathematically transforms activation outliers to weights and produces W8A8 quantized output.
2. **Given** SmoothQuant quantization, **When** I specify a smoothing factor (alpha), **Then** the system applies the specified balance between weight and activation quantization difficulty.
3. **Given** a model with extreme activation outliers, **When** SmoothQuant cannot achieve target quality, **Then** the system reports problematic layers and suggests per-channel quantization alternatives.

---

### User Story 4 - Importance Analysis and Super Weight Protection (Priority: P4)

As an ML engineer seeking optimal quantization quality, I want to identify and protect critical model parameters (super weights) so that I can achieve better quality at the same compression ratio.

**Why this priority**: Research shows 0.01% of parameters disproportionately affect model quality. Protecting these enables aggressive quantization elsewhere without quality loss.

**Independent Test**: Can be fully tested by running importance analysis on a model and verifying identified super weights match expected patterns.

**Acceptance Scenarios**:

1. **Given** a model and calibration data, **When** I run importance analysis, **Then** the system identifies super weights and outputs an importance matrix (imatrix).
2. **Given** an importance matrix, **When** I run quantization with super weight protection enabled, **Then** the system preserves identified critical parameters at higher precision.
3. **Given** importance analysis results, **When** I request a visualization, **Then** the system outputs per-layer importance scores and super weight locations.

---

### Edge Cases

- What happens when dynamic quantization cannot find a valid layer assignment meeting both size and quality targets?
- How does the system handle models with architectures not seen during importance calibration?
- What happens when ultra-low-bit quantization causes complete model failure (divergent outputs)?
- How does SmoothQuant handle models with extreme outlier patterns beyond smoothing capacity?
- What happens when importance analysis identifies no clear super weights?

## Requirements *(mandatory)*

### Functional Requirements

**Dynamic Quantization**:
- **FR-001**: System MUST support specifying per-layer quantization precision (bit-width) via configuration file or command-line.
- **FR-002**: System MUST support automatic layer importance computation using calibration data and importance matrices (imatrix).
- **FR-003**: System MUST provide preset profiles for common dynamic quantization strategies (e.g., "attention-high-precision", "mlp-low-precision").
- **FR-004**: System MUST report per-layer statistics including bit-width, parameter count, and contribution to total model size.

**Ultra-Low-Bit Quantization**:
- **FR-005**: System MUST support 2-bit quantization for GGUF output format.
- **FR-006**: System MUST support 1.5-bit (IQ1_S) and 1.75-bit (IQ1_M) quantization using super-block encoding.
- **FR-007**: System MUST support ternary quantization (-1, 0, 1) for compatible architectures.
- **FR-008**: System MUST validate model output coherence after ultra-low-bit quantization and warn if quality degradation is severe.

**SmoothQuant**:
- **FR-009**: System MUST implement mathematically equivalent transformation to shift activation difficulty to weights.
- **FR-010**: System MUST support configurable smoothing factor (alpha) between 0.0 and 1.0.
- **FR-011**: System MUST output W8A8 quantized models compatible with INT8 inference engines.
- **FR-012**: System MUST compute and report per-channel scale factors for both weights and activations.

**Importance Analysis**:
- **FR-013**: System MUST compute importance matrices (imatrix) from calibration data, with default 512 samples and configurable range of 128-1024 samples.
- **FR-014**: System MUST identify super weights based on activation magnitude and gradient sensitivity.
- **FR-015**: System MUST support saving and loading importance matrices for reuse across quantization runs.
- **FR-016**: System MUST support protecting identified super weights at configurable higher precision.

**General**:
- **FR-017**: System MUST integrate with base quantization infrastructure from 001-multi-format-quantization.
- **FR-018**: System MUST provide quality estimation metrics before and after quantization (perplexity, accuracy benchmarks).
- **FR-019**: System MUST support combining multiple techniques (e.g., dynamic quantization with super weight protection).
- **FR-020**: System MUST support layer-level checkpointing for importance analysis and dynamic quantization operations, enabling resumption from the last completed layer.

### Key Entities

- **LayerQuantizationConfig**: Per-layer quantization settings; attributes include layer_name, bit_width, quantization_method, is_protected.
- **ImportanceMatrix**: Importance scores for model parameters; attributes include layer_scores, super_weight_indices, computation_method, calibration_dataset_info.
- **SmoothQuantConfig**: SmoothQuant transformation settings; attributes include alpha, per_channel_scales, activation_statistics.
- **DynamicQuantizationProfile**: Preset or custom profile for layer-wise quantization; attributes include profile_name, layer_assignments, target_compression_ratio.
- **QuantizationQualityReport**: Quality assessment results; attributes include perplexity_original, perplexity_quantized, layer_wise_errors, super_weight_coverage.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Dynamic quantization achieves at least 60% size reduction on 70B+ parameter models while maintaining perplexity within 10% of uniform 4-bit quantization.
- **SC-002**: Ultra-low-bit (1.5-bit) quantization produces models that pass basic coherence tests (non-divergent, grammatically correct output) on 90% of test prompts.
- **SC-003**: SmoothQuant W8A8 quantization achieves inference speedup of at least 1.3x compared to FP16 on INT8-capable hardware.
- **SC-004**: Importance analysis completes within 30 minutes for 7B parameter models using standard calibration datasets.
- **SC-005**: Super weight protection improves quantization quality (lower perplexity) by at least 5% compared to same-bitwidth quantization without protection.
- **SC-006**: Users can apply advanced quantization techniques with no more than 3 additional command-line flags beyond basic quantization.

## Assumptions

- Base quantization infrastructure (001-multi-format-quantization) is implemented and functional.
- Users have access to representative calibration data for importance analysis (default: WikiText-2).
- For SmoothQuant W8A8 deployment, target hardware supports INT8 compute (CUDA GPUs with INT8 tensor cores, or CPU with AVX-512 VNNI).
- Ultra-low-bit quantization (sub-2-bit) is understood to have higher quality degradation; users accept this tradeoff for extreme compression.
- Dynamic quantization requires models with standard layer naming conventions (attention, MLP blocks identifiable).
