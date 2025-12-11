# Implementation Plan: Advanced Quantization Methods

**Branch**: `002-advanced-quantization` | **Date**: 2025-12-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-advanced-quantization/spec.md`
**Depends On**: `001-multi-format-quantization` (base quantization infrastructure)

## Summary

Implement advanced quantization techniques including dynamic layer-wise quantization, ultra-low-bit (1.5/1.58/2-bit), SmoothQuant W8A8, and importance-aware super weight protection. These methods enable extreme compression ratios (up to 80% for dynamic, 10x for ultra-low-bit) while preserving model quality through intelligent parameter selection and mixed-precision strategies.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- Base: transformers, torch, huggingface_hub, click, rich (from 001)
- New: numpy (importance matrix computation), scipy (statistical analysis)
- GGUF: llama-cpp-python with IQ1_S/IQ1_M support
- Optional: triton (custom kernels for SmoothQuant)

**Storage**: File-based (importance matrices, layer configs, checkpoints)
**Testing**: pytest with pytest-cov for coverage
**Target Platform**: Linux/macOS/Windows, CPU or CUDA GPU
**Project Type**: Extension to base quantization library

**Performance Goals**:
- Importance analysis: <30 minutes for 7B model
- Dynamic quantization: Same as base + 20% overhead for analysis
- SmoothQuant: ~2x base quantization time (activation calibration)

**Constraints**:
- Requires calibration data for importance analysis (512 samples default)
- Ultra-low-bit has significant quality tradeoffs
- SmoothQuant W8A8 requires INT8-capable hardware for deployment benefits

**Scale/Scope**: Support models up to 70B+ parameters; focus on quality-compression tradeoff optimization

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Library-First | PASS | Core algorithms in `src/lib/quantizers/advanced/`, reusable importance analysis in `src/lib/analysis/` |
| II. CLI Interface Contract | PASS | New CLI flags integrate with existing `quantize` command, JSON/human output formats |
| III. Test-Driven Development | PASS | Tests in `tests/unit/test_advanced/`, `tests/integration/test_dynamic.py`, etc. |
| IV. Model Format Compliance | PASS | GGUF IQ1_S/IQ1_M per llama.cpp spec, SmoothQuant W8A8 per INT8 standards |
| V. Observability | PASS | Per-layer importance scores, quality metrics, memory tracking extended |
| VI. Versioning | PASS | New quantization types versioned, importance matrix format documented |
| VII. Simplicity | PASS | Builds on existing infrastructure, adds minimal new concepts |

## Project Structure

### Documentation (this feature)

```text
specs/002-advanced-quantization/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI extensions)
└── checklists/          # Quality checklists
    └── requirements.md
```

### Source Code (extends 001 structure)

```text
src/
├── llm_quantize/
│   ├── cli/
│   │   ├── quantize.py      # Extended with advanced options
│   │   └── analyze.py       # NEW: importance analysis command
│   ├── lib/
│   │   ├── quantizers/
│   │   │   ├── base.py      # Extended with mixed-precision support
│   │   │   ├── gguf.py      # Extended with IQ1_S/IQ1_M
│   │   │   └── advanced/    # NEW
│   │   │       ├── __init__.py
│   │   │       ├── dynamic.py      # Dynamic layer-wise quantization
│   │   │       ├── ultra_low_bit.py # 1.5/1.58/2-bit quantization
│   │   │       ├── smoothquant.py  # SmoothQuant W8A8
│   │   │       └── profiles.py     # Quantization profiles
│   │   ├── analysis/        # NEW
│   │   │   ├── __init__.py
│   │   │   ├── importance.py       # Importance matrix computation
│   │   │   ├── super_weights.py    # Super weight identification
│   │   │   └── quality.py          # Quality estimation
│   │   └── checkpoint.py    # Extended for analysis checkpointing
│   └── models/
│       ├── importance_matrix.py    # NEW
│       ├── layer_config.py         # NEW
│       ├── smoothquant_config.py   # NEW
│       └── quality_report.py       # NEW

tests/
├── unit/
│   └── test_advanced/
│       ├── test_dynamic.py
│       ├── test_ultra_low_bit.py
│       ├── test_smoothquant.py
│       └── test_importance.py
├── contract/
│   └── test_advanced_cli.py
└── integration/
    ├── test_dynamic_quantization.py
    ├── test_ultra_low_bit.py
    └── test_smoothquant.py
```

**Structure Decision**: Extension structure that builds on 001's foundation. New modules in `advanced/` and `analysis/` directories keep advanced features isolated while sharing common infrastructure.

## Complexity Tracking

| Complexity Item | Justification | Principle Exception |
|-----------------|---------------|---------------------|
| Multiple quantization algorithms | Required by spec (dynamic, ultra-low-bit, SmoothQuant, super weights) | None - explicit requirements |
| Importance matrix computation | Required for quality-aware quantization | None - core feature |
| Mixed-precision layer configs | Required for dynamic quantization | None - explicit requirement |

No constitution violations requiring justification.

## Implementation Phases

### Phase 1: Importance Analysis Foundation
- ImportanceMatrix data model
- Calibration data loading
- Basic importance computation (activation magnitudes)
- Layer-level checkpointing for analysis

### Phase 2: Dynamic Quantization
- LayerQuantizationConfig model
- Profile system (presets + custom)
- Per-layer quantization orchestration
- Integration with base GGUF quantizer

### Phase 3: Ultra-Low-Bit Support
- IQ1_S (1.5-bit) implementation
- IQ1_M (1.75-bit) implementation
- Ternary quantization support
- Quality validation and warnings

### Phase 4: SmoothQuant W8A8
- Activation statistics collection
- Smoothing factor optimization
- Weight transformation
- W8A8 output format

### Phase 5: Super Weight Protection
- Super weight identification algorithm
- Protection during quantization
- Quality metrics comparison

### Phase 6: Integration and CLI
- CLI flag extensions
- Combined technique support
- Quality reporting
- Documentation
