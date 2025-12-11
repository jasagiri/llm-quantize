# Tasks: Multi-Format Quantization Tool

**Input**: Design documents from `/specs/001-multi-format-quantization/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-contract.md

**Tests**: TDD is mandated by constitution (Principle III). Tests are included for each user story.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per plan.md: src/llm_quantize/{cli,lib,models}/, tests/{unit,contract,integration}/
- [ ] T002 Initialize Python project with pyproject.toml (Python 3.10+, dependencies: transformers, torch, click, rich, huggingface_hub)
- [ ] T003 [P] Configure ruff for linting and formatting in pyproject.toml
- [ ] T004 [P] Configure mypy for type checking in pyproject.toml
- [ ] T005 [P] Create src/llm_quantize/__init__.py with version and package exports
- [ ] T006 [P] Create tests/conftest.py with shared pytest fixtures

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 Create src/llm_quantize/models/source_model.py with SourceModel dataclass (model_path, model_type, architecture, parameter_count, dtype)
- [ ] T008 [P] Create src/llm_quantize/models/quantization_config.py with QuantizationConfig dataclass (target_format, quantization_level, output_dir, etc.)
- [ ] T009 [P] Create src/llm_quantize/models/quantized_model.py with QuantizedModel dataclass (output_path, format, file_size, metadata)
- [ ] T010 [P] Create src/llm_quantize/models/quantization_job.py with QuantizationJob dataclass and JobStatus enum
- [ ] T011 Create src/llm_quantize/models/__init__.py exporting all model classes
- [ ] T012 Implement src/llm_quantize/lib/model_loader.py with load_model() supporting HF Hub and local paths with HF_TOKEN
- [ ] T013 [P] Implement src/llm_quantize/lib/progress.py with ProgressReporter class using rich
- [ ] T014 [P] Implement src/llm_quantize/lib/checkpoint.py with Checkpoint class for layer-level checkpointing
- [ ] T015 [P] Implement src/llm_quantize/lib/validation.py with validate_output() for basic file verification
- [ ] T016 Create src/llm_quantize/lib/quantizers/base.py with abstract BaseQuantizer class
- [ ] T017 Create src/llm_quantize/lib/quantizers/__init__.py exporting quantizer registry
- [ ] T018 Implement src/llm_quantize/cli/main.py with click entry point, --version, --format, --verbosity options
- [ ] T019 Create src/llm_quantize/cli/__init__.py
- [ ] T020 Write tests/unit/test_model_loader.py for model loading with HF Hub and local paths
- [ ] T021 [P] Write tests/unit/test_checkpoint.py for checkpoint save/load/resume functionality
- [ ] T022 [P] Write tests/unit/test_progress.py for progress reporting

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - GGUF Quantization (Priority: P1) 🎯 MVP

**Goal**: Quantize HuggingFace models to GGUF format for llama.cpp inference

**Independent Test**: Quantize TinyLlama to GGUF Q4_K_M and verify output loads in llama.cpp

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T023 [P] [US1] Write tests/contract/test_cli_quantize_gguf.py for GGUF CLI contract (exit codes, output format)
- [ ] T024 [P] [US1] Write tests/integration/test_gguf_quantization.py for end-to-end GGUF quantization

### Implementation for User Story 1

- [ ] T025 [US1] Implement src/llm_quantize/lib/quantizers/gguf.py with GGUFQuantizer class supporting Q2_K through Q8_0
- [ ] T026 [US1] Implement src/llm_quantize/cli/quantize.py with quantize command for GGUF format
- [ ] T027 [US1] Add GGUF-specific validation in src/llm_quantize/lib/validation.py
- [ ] T028 [US1] Add GGUF progress reporting integration with layer-level updates
- [ ] T029 [US1] Write tests/unit/test_quantizers/test_gguf.py for GGUF quantizer unit tests

**Checkpoint**: GGUF quantization fully functional and testable independently

---

## Phase 4: User Story 2 - AWQ Quantization (Priority: P2)

**Goal**: Quantize models to AWQ format for GPU inference with vLLM

**Independent Test**: Quantize model to AWQ and verify inference with AutoAWQ

### Tests for User Story 2

- [ ] T030 [P] [US2] Write tests/contract/test_cli_quantize_awq.py for AWQ CLI contract
- [ ] T031 [P] [US2] Write tests/integration/test_awq_quantization.py for end-to-end AWQ quantization

### Implementation for User Story 2

- [ ] T032 [US2] Implement src/llm_quantize/lib/quantizers/awq.py with AWQQuantizer class using autoawq
- [ ] T033 [US2] Add calibration data loading support in src/llm_quantize/lib/calibration.py
- [ ] T034 [US2] Extend src/llm_quantize/cli/quantize.py with AWQ format support and --calibration-data option
- [ ] T035 [US2] Add AWQ-specific validation in src/llm_quantize/lib/validation.py
- [ ] T036 [US2] Write tests/unit/test_quantizers/test_awq.py for AWQ quantizer unit tests

**Checkpoint**: GGUF and AWQ both functional and independently testable

---

## Phase 5: User Story 3 - GPTQ Quantization (Priority: P3)

**Goal**: Quantize models to GPTQ format for AutoGPTQ/ExLlama inference

**Independent Test**: Quantize model to GPTQ and verify loading with AutoGPTQ

### Tests for User Story 3

- [ ] T037 [P] [US3] Write tests/contract/test_cli_quantize_gptq.py for GPTQ CLI contract
- [ ] T038 [P] [US3] Write tests/integration/test_gptq_quantization.py for end-to-end GPTQ quantization

### Implementation for User Story 3

- [ ] T039 [US3] Implement src/llm_quantize/lib/quantizers/gptq.py with GPTQQuantizer class using auto-gptq
- [ ] T040 [US3] Extend src/llm_quantize/cli/quantize.py with GPTQ format support and --group-size option
- [ ] T041 [US3] Add GPTQ-specific validation in src/llm_quantize/lib/validation.py
- [ ] T042 [US3] Write tests/unit/test_quantizers/test_gptq.py for GPTQ quantizer unit tests

**Checkpoint**: All three quantization formats functional and independently testable

---

## Phase 6: User Story 4 - Format Conversion (Priority: P4)

**Goal**: Convert between quantized formats without re-quantizing from source

**Independent Test**: Convert GGUF to AWQ and verify both input and output are valid

### Tests for User Story 4

- [ ] T043 [P] [US4] Write tests/contract/test_cli_convert.py for convert CLI contract
- [ ] T044 [P] [US4] Write tests/integration/test_format_conversion.py for format conversion

### Implementation for User Story 4

- [ ] T045 [US4] Implement src/llm_quantize/cli/convert.py with convert command
- [ ] T046 [US4] Implement format detection and conversion logic in src/llm_quantize/lib/converter.py
- [ ] T047 [US4] Add quality degradation warnings for lossy conversions
- [ ] T048 [US4] Write tests/unit/test_converter.py for conversion unit tests

**Checkpoint**: All user stories complete and independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T049 Implement src/llm_quantize/cli/info.py with info command for model inspection
- [ ] T050 [P] Write tests/contract/test_cli_info.py for info command contract
- [ ] T051 [P] Add comprehensive error handling with actionable error messages across all commands
- [ ] T052 [P] Write tests/contract/test_output_formats.py for JSON/human output format contract
- [ ] T053 Add memory usage tracking and reporting in src/llm_quantize/lib/progress.py
- [ ] T054 [P] Validate all exit codes match CLI contract specification
- [ ] T055 Run quickstart.md validation: test all documented commands work as specified
- [ ] T056 [P] Add type hints and run mypy on entire codebase
- [ ] T057 Run ruff and fix all linting issues
- [ ] T058 Verify test coverage meets 80% line, 70% branch requirement

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed) or sequentially in priority order
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1) - GGUF**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2) - AWQ**: Can start after Foundational - No dependencies on US1
- **User Story 3 (P3) - GPTQ**: Can start after Foundational - No dependencies on US1/US2
- **User Story 4 (P4) - Convert**: Depends on at least two formats being implemented (US1+US2 or US1+US3)

### Within Each User Story

1. Tests MUST be written and FAIL before implementation
2. Core quantizer implementation
3. CLI integration
4. Validation integration
5. Unit tests for quantizer

### Parallel Opportunities

**Setup Phase**:
- T003, T004, T005, T006 can run in parallel

**Foundational Phase**:
- T008, T009, T010 can run in parallel (models)
- T013, T014, T015 can run in parallel (lib utilities)
- T021, T022 can run in parallel (tests)

**User Story Phases**:
- All test tasks within a story can run in parallel
- Different user stories can be worked on by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write tests/contract/test_cli_quantize_gguf.py for GGUF CLI contract"
Task: "Write tests/integration/test_gguf_quantization.py for end-to-end GGUF"

# After tests fail, implement sequentially:
Task: "Implement src/llm_quantize/lib/quantizers/gguf.py with GGUFQuantizer"
Task: "Implement src/llm_quantize/cli/quantize.py with quantize command"
Task: "Add GGUF-specific validation"
Task: "Write tests/unit/test_quantizers/test_gguf.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (GGUF)
4. **STOP and VALIDATE**: Test GGUF quantization independently
5. Deploy/demo if ready - users can quantize models to GGUF

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 (GGUF) → Test → Deploy (MVP!)
3. Add User Story 2 (AWQ) → Test → Deploy
4. Add User Story 3 (GPTQ) → Test → Deploy
5. Add User Story 4 (Convert) → Test → Deploy
6. Polish → Final release

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (GGUF)
   - Developer B: User Story 2 (AWQ)
   - Developer C: User Story 3 (GPTQ)
3. After US1-US3 complete:
   - Any developer: User Story 4 (Convert)
4. Team completes Polish together

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Constitution requires TDD - tests are mandatory, not optional
