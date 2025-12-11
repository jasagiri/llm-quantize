# Implementation Plan: Multi-Format Quantization Tool

**Branch**: `001-multi-format-quantization` | **Date**: 2025-12-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-multi-format-quantization/spec.md`

## Summary

Build a unified CLI tool for LLM model quantization supporting GGUF, AWQ, and GPTQ output formats. The tool accepts Hugging Face models (local or Hub), performs quantization with configurable levels, and outputs compressed models compatible with popular inference engines (llama.cpp, vLLM, AutoGPTQ). Key features include layer-level checkpointing for resume capability, progress reporting, and CPU/GPU flexibility.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: transformers, torch, llama-cpp-python (for GGUF), autoawq, auto-gptq, huggingface_hub, click (CLI), rich (progress/output)
**Storage**: File-based (model weights, checkpoints, output files)
**Testing**: pytest with pytest-cov for coverage
**Target Platform**: Linux/macOS/Windows, CPU or CUDA GPU
**Project Type**: Single project (CLI library)
**Performance Goals**: 7B model quantization in <30 minutes (GPU), <2 hours (CPU)
**Constraints**: Memory usage proportional to model size; checkpoint files may be large
**Scale/Scope**: Support models up to 70B parameters; primary focus on 7B-13B range

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Library-First | ✅ PASS | Core quantization logic in `src/lib/`, CLI in `src/cli/` |
| II. CLI Interface Contract | ✅ PASS | CLI with stdin/args input, stdout/stderr output, JSON + human formats |
| III. Test-Driven Development | ✅ PASS | Tests in `tests/` (unit, contract, integration) |
| IV. Model Format Compliance | ✅ PASS | GGUF spec compliance, AWQ/GPTQ tooling compatibility |
| V. Observability | ✅ PASS | Structured logging, progress reporting, memory tracking |
| VI. Versioning | ✅ PASS | Semantic versioning for CLI and output formats |
| VII. Simplicity | ✅ PASS | Single CLI entry point, minimal configuration |

## Project Structure

### Documentation (this feature)

```text
specs/001-multi-format-quantization/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI contract)
└── checklists/          # Quality checklists
    └── requirements.md
```

### Source Code (repository root)

```text
src/
├── llm_quantize/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py          # CLI entry point
│   │   ├── quantize.py      # quantize command
│   │   ├── convert.py       # convert command
│   │   └── info.py          # info command
│   ├── lib/
│   │   ├── __init__.py
│   │   ├── model_loader.py  # HF model loading
│   │   ├── quantizers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py      # Abstract quantizer
│   │   │   ├── gguf.py      # GGUF quantization
│   │   │   ├── awq.py       # AWQ quantization
│   │   │   └── gptq.py      # GPTQ quantization
│   │   ├── checkpoint.py    # Layer-level checkpointing
│   │   ├── progress.py      # Progress reporting
│   │   └── validation.py    # Output validation
│   └── models/
│       ├── __init__.py
│       ├── source_model.py
│       ├── quantization_config.py
│       ├── quantized_model.py
│       └── quantization_job.py

tests/
├── conftest.py
├── unit/
│   ├── test_model_loader.py
│   ├── test_checkpoint.py
│   └── test_quantizers/
├── contract/
│   ├── test_cli_contract.py
│   └── test_output_formats.py
└── integration/
    ├── test_gguf_quantization.py
    ├── test_awq_quantization.py
    └── test_gptq_quantization.py
```

**Structure Decision**: Single project structure with clear separation between CLI (`cli/`), core library (`lib/`), and data models (`models/`). This enables the library-first principle while maintaining a clean CLI interface.

## Complexity Tracking

No constitution violations requiring justification.
