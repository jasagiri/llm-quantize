# Implementation Plan: LittleBit Ultra-Low-Bit Quantization

**Branch**: `003-littlebit-factorization` | **Date**: 2025-12-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-littlebit-factorization/spec.md`
**Depends On**: `001-multi-format-quantization` (base quantization infrastructure)
**Reference**: [arXiv:2506.13771](https://arxiv.org/abs/2506.13771) - LittleBit (NeurIPS 2025)

## Summary

Implement LittleBit ultra-low-bit quantization via latent matrix factorization, enabling 0.1 BPW compression with ~31x memory reduction. The technique decomposes weight matrices into low-rank factors before binarization, with multi-scale compensation to preserve quality. Supports native .littlebit format, GGUF extension, and lossy GGUF conversion for compatibility.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- Base: transformers, torch, huggingface_hub, click, rich (from 001)
- New: einops (tensor operations), scipy (matrix decomposition)
- Optional: triton (custom inference kernels)

**Storage**: File-based (factorized weights, compensation factors, checkpoints)
**Testing**: pytest with pytest-cov for coverage
**Target Platform**: Linux/macOS/Windows, GPU recommended (CPU fallback supported)
**Project Type**: Extension to base quantization library

**Performance Goals**:
- 7B model compression: <2 hours (GPU), <20 hours (CPU)
- Memory reduction: 25x+ at 0.1 BPW vs FP16
- Inference speedup: 5x+ vs FP16 on compatible hardware

**Constraints**:
- Requires calibration data for compensation factor computation
- Extreme compression (0.1 BPW) has significant quality tradeoffs
- Specialized inference kernels needed for full speedup benefits

**Scale/Scope**: Support models up to 70B+ parameters; primary focus on extreme compression scenarios

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Library-First | PASS | Core factorization in `src/lib/factorization/`, reusable compensation in `src/lib/compensation/` |
| II. CLI Interface Contract | PASS | New `littlebit` subcommand with JSON/human output formats |
| III. Test-Driven Development | PASS | Tests in `tests/unit/test_factorization/`, `tests/integration/test_littlebit.py` |
| IV. Model Format Compliance | PASS | Native .littlebit spec documented, GGUF extension follows GGUF v3 |
| V. Observability | PASS | Per-layer factorization stats, quality metrics, memory tracking |
| VI. Versioning | PASS | .littlebit format versioned, backward compatibility documented |
| VII. Simplicity | PASS | Minimal CLI flags (target BPW, optional rank), builds on existing infrastructure |

## Project Structure

### Documentation (this feature)

```text
specs/003-littlebit-factorization/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI contract)
└── checklists/          # Quality checklists
    └── requirements.md
```

### Source Code (extends 001 structure)

```text
src/
├── llm_quantize/
│   ├── cli/
│   │   ├── main.py          # Add littlebit command group
│   │   └── littlebit.py     # NEW: LittleBit-specific commands
│   ├── lib/
│   │   ├── factorization/   # NEW
│   │   │   ├── __init__.py
│   │   │   ├── svd.py           # SVD-based factorization
│   │   │   ├── dual_svid.py     # Dual Sign-Value-Independent Decomposition
│   │   │   ├── binarization.py  # Factor binarization
│   │   │   └── rank_selection.py # Automatic rank selection
│   │   ├── compensation/    # NEW
│   │   │   ├── __init__.py
│   │   │   ├── row.py           # Row-wise compensation
│   │   │   ├── column.py        # Column-wise compensation
│   │   │   ├── latent.py        # Latent dimension compensation
│   │   │   └── residual.py      # Integrated residual compensation
│   │   ├── formats/         # NEW
│   │   │   ├── __init__.py
│   │   │   ├── littlebit.py     # Native .littlebit format
│   │   │   ├── gguf_ext.py      # GGUF with factorization metadata
│   │   │   └── gguf_lossy.py    # Lossy GGUF conversion
│   │   ├── inference/       # NEW (optional)
│   │   │   ├── __init__.py
│   │   │   └── kernels.py       # Factorized weight inference kernels
│   │   └── checkpoint.py    # Extended for factorization checkpointing
│   └── models/
│       ├── factorization_config.py  # NEW
│       ├── factorized_weights.py    # NEW
│       ├── compensation_factors.py  # NEW
│       └── compression_report.py    # NEW

tests/
├── unit/
│   └── test_factorization/
│       ├── test_svd.py
│       ├── test_binarization.py
│       ├── test_compensation.py
│       └── test_rank_selection.py
├── contract/
│   └── test_littlebit_cli.py
└── integration/
    ├── test_littlebit_compression.py
    └── test_littlebit_formats.py
```

**Structure Decision**: Dedicated `factorization/` and `compensation/` modules encapsulate the LittleBit algorithm complexity while sharing base infrastructure. Native format support in `formats/` allows for specialized storage optimization.

## Complexity Tracking

| Complexity Item | Justification | Principle Exception |
|-----------------|---------------|---------------------|
| Matrix factorization algorithms | Core LittleBit technique per paper | None - explicit requirement |
| Multi-scale compensation | Required for quality preservation | None - core feature |
| Three output formats | Per clarification session | None - user requirement |
| Custom inference kernels | Required for speedup benefits | VII - optional, not default |

No constitution violations requiring justification.

## Implementation Phases

### Phase 1: Core Factorization
- SVD-based weight decomposition
- Rank selection algorithms (manual/auto)
- Basic binarization of factors
- FactorizedWeights data model

### Phase 2: Compensation System
- Row-wise compensation implementation
- Column-wise compensation implementation
- Latent dimension compensation
- Integrated residual compensation
- CompensationFactors data model

### Phase 3: Quality Management
- Calibration data integration
- Perplexity computation pre/post compression
- Quality threshold enforcement
- Coherence validation

### Phase 4: Output Formats
- Native .littlebit format specification
- .littlebit reader/writer
- GGUF metadata extension
- Lossy GGUF conversion

### Phase 5: CLI Integration
- `llm-quantize littlebit compress` command
- `llm-quantize littlebit info` command
- Progress reporting and checkpointing
- JSON/human output modes

### Phase 6: Optimization (Optional)
- Triton inference kernels
- Memory-efficient factorization for large models
- GPU acceleration optimization

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Quality degradation at 0.1 BPW | High | Medium | Quality threshold mode, clear warnings |
| Long compression times | Medium | Medium | Checkpointing, progress reporting |
| Inference kernel complexity | High | Low | Kernels optional, standard inference fallback |
| GGUF compatibility issues | Medium | Medium | Lossy conversion as fallback option |
