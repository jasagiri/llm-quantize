# Tasks: Advanced Quantization Methods

**Input**: Design documents from `/specs/002-advanced-quantization/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-contract.md
**Depends On**: 001-multi-format-quantization (base infrastructure)

**Tests**: TDD is mandated by constitution (Principle III). Tests are included for each user story.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Extension infrastructure for advanced quantization methods

**Prerequisite**: 001-multi-format-quantization must be complete

- [ ] T001 Create directory structure: src/llm_quantize/lib/quantizers/advanced/, src/llm_quantize/lib/analysis/
- [ ] T002 [P] Create src/llm_quantize/models/importance_matrix.py with ImportanceMatrix and CalibrationInfo dataclasses
- [ ] T003 [P] Create src/llm_quantize/models/layer_config.py with LayerQuantizationConfig dataclass
- [ ] T004 [P] Create src/llm_quantize/models/smoothquant_config.py with SmoothQuantConfig and ActivationStatistics
- [ ] T005 [P] Create src/llm_quantize/models/quality_report.py with QuantizationQualityReport and LayerError
- [ ] T006 Update src/llm_quantize/models/__init__.py to export new model classes
- [ ] T007 Create tests/unit/test_advanced/ directory for advanced quantization tests

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure for importance analysis and quality estimation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T008 Implement src/llm_quantize/lib/analysis/__init__.py
- [ ] T009 Implement src/llm_quantize/lib/analysis/importance.py with compute_importance_matrix() using activation magnitude
- [ ] T010 [P] Implement src/llm_quantize/lib/analysis/quality.py with compute_perplexity() and generate_quality_report()
- [ ] T011 [P] Extend src/llm_quantize/lib/checkpoint.py with AnalysisCheckpoint for importance analysis operations
- [ ] T012 [P] Implement src/llm_quantize/lib/calibration.py with load_calibration_data() supporting WikiText-2 and custom datasets
- [ ] T013 Create src/llm_quantize/lib/quantizers/advanced/__init__.py
- [ ] T014 Create src/llm_quantize/lib/quantizers/advanced/profiles.py with preset profiles (attention-high, balanced, compression-max)
- [ ] T015 Implement src/llm_quantize/cli/analyze.py with analyze command skeleton
- [ ] T016 Write tests/unit/test_analysis/test_importance.py for importance matrix computation
- [ ] T017 [P] Write tests/unit/test_analysis/test_quality.py for perplexity and quality metrics
- [ ] T018 [P] Write tests/unit/test_calibration.py for calibration data loading

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Dynamic Layer-wise Quantization (Priority: P1) 🎯 MVP

**Goal**: Apply different quantization levels to different layers based on importance analysis

**Independent Test**: Quantize model with dynamic profile and verify per-layer bit-widths match configuration

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T019 [P] [US1] Write tests/contract/test_cli_quantize_dynamic.py for dynamic quantization CLI contract
- [ ] T020 [P] [US1] Write tests/integration/test_dynamic_quantization.py for end-to-end dynamic quantization

### Implementation for User Story 1

- [ ] T021 [US1] Implement src/llm_quantize/lib/quantizers/advanced/dynamic.py with DynamicQuantizer class
- [ ] T022 [US1] Add automatic layer importance computation using calibration data in dynamic.py
- [ ] T023 [US1] Implement profile-based layer assignment (attention-high, balanced, compression-max)
- [ ] T024 [US1] Extend src/llm_quantize/cli/quantize.py with --dynamic, --profile, --imatrix options
- [ ] T025 [US1] Add per-layer statistics reporting (bit-width, parameter count, compression ratio)
- [ ] T026 [US1] Write tests/unit/test_advanced/test_dynamic.py for dynamic quantizer unit tests

**Checkpoint**: Dynamic layer-wise quantization fully functional and testable independently

---

## Phase 4: User Story 2 - Ultra-Low-Bit Quantization (Priority: P2)

**Goal**: Quantize models to 1.5-2 bits per weight for extreme compression

**Independent Test**: Quantize model to IQ1_S and verify output produces coherent text

### Tests for User Story 2

- [ ] T027 [P] [US2] Write tests/contract/test_cli_quantize_ultra_low.py for ultra-low-bit CLI contract
- [ ] T028 [P] [US2] Write tests/integration/test_ultra_low_bit.py for IQ1_S/IQ1_M quantization

### Implementation for User Story 2

- [ ] T029 [US2] Implement src/llm_quantize/lib/quantizers/advanced/ultra_low_bit.py with UltraLowBitQuantizer
- [ ] T030 [US2] Add IQ1_S (1.5-bit) support using llama-cpp-python super-block encoding
- [ ] T031 [US2] Add IQ1_M (1.75-bit) support
- [ ] T032 [US2] Add ternary quantization (-1, 0, 1) support for compatible architectures
- [ ] T033 [US2] Implement coherence validation after ultra-low-bit quantization with warning system
- [ ] T034 [US2] Extend src/llm_quantize/cli/quantize.py with IQ1_S, IQ1_M, TERNARY quantization levels
- [ ] T035 [US2] Write tests/unit/test_advanced/test_ultra_low_bit.py for ultra-low-bit unit tests

**Checkpoint**: Dynamic and ultra-low-bit both functional and independently testable

---

## Phase 5: User Story 3 - SmoothQuant W8A8 (Priority: P3)

**Goal**: Enable INT8 inference with weight and activation quantization

**Independent Test**: Apply SmoothQuant and verify inference speedup on INT8 hardware

### Tests for User Story 3

- [ ] T036 [P] [US3] Write tests/contract/test_cli_smoothquant.py for SmoothQuant CLI contract
- [ ] T037 [P] [US3] Write tests/integration/test_smoothquant.py for W8A8 quantization

### Implementation for User Story 3

- [ ] T038 [US3] Implement src/llm_quantize/lib/quantizers/advanced/smoothquant.py with SmoothQuantQuantizer
- [ ] T039 [US3] Implement activation statistics collection during calibration
- [ ] T040 [US3] Implement smoothing factor optimization (alpha parameter)
- [ ] T041 [US3] Implement per-channel scale computation for weights and activations
- [ ] T042 [US3] Extend src/llm_quantize/cli/quantize.py with --smoothquant and --alpha options
- [ ] T043 [US3] Add W8A8 output format compatible with INT8 inference engines
- [ ] T044 [US3] Write tests/unit/test_advanced/test_smoothquant.py for SmoothQuant unit tests

**Checkpoint**: Dynamic, ultra-low-bit, and SmoothQuant all functional

---

## Phase 6: User Story 4 - Importance Analysis & Super Weight Protection (Priority: P4)

**Goal**: Identify and protect critical model parameters during quantization

**Independent Test**: Run importance analysis and verify super weights match expected patterns

### Tests for User Story 4

- [ ] T045 [P] [US4] Write tests/contract/test_cli_analyze.py for analyze command contract
- [ ] T046 [P] [US4] Write tests/integration/test_super_weight_protection.py for protected quantization

### Implementation for User Story 4

- [ ] T047 [US4] Implement src/llm_quantize/lib/analysis/super_weights.py with identify_super_weights()
- [ ] T048 [US4] Add gradient sensitivity computation (optional, requires backward pass)
- [ ] T049 [US4] Implement importance matrix save/load in .imatrix format (llama.cpp compatible)
- [ ] T050 [US4] Complete src/llm_quantize/cli/analyze.py with full analyze command
- [ ] T051 [US4] Extend quantize command with --protect-super-weights and --protection-coverage options
- [ ] T052 [US4] Implement mixed-precision storage for protected weights
- [ ] T053 [US4] Write tests/unit/test_analysis/test_super_weights.py for super weight identification

**Checkpoint**: All user stories complete and independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T054 Implement src/llm_quantize/cli/profile.py with profile list/show/create/validate commands
- [ ] T055 [P] Write tests/contract/test_cli_profile.py for profile command contract
- [ ] T056 Implement src/llm_quantize/cli/quality.py with quality report/compare commands
- [ ] T057 [P] Write tests/contract/test_cli_quality.py for quality command contract
- [ ] T058 Add comprehensive error handling with exit codes 10-14 for advanced quantization errors
- [ ] T059 [P] Validate all environment variables (LLM_QUANTIZE_IMATRIX_CACHE, etc.)
- [ ] T060 Run quickstart.md validation: test all documented commands work as specified
- [ ] T061 [P] Add type hints and run mypy on all new code
- [ ] T062 Run ruff and fix all linting issues
- [ ] T063 Verify test coverage meets 80% line, 70% branch requirement

---

## Dependencies & Execution Order

### Phase Dependencies

- **Prerequisite**: 001-multi-format-quantization complete
- **Setup (Phase 1)**: No additional dependencies
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1) - Dynamic**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2) - Ultra-Low-Bit**: Can start after Foundational - No dependencies on US1
- **User Story 3 (P3) - SmoothQuant**: Can start after Foundational - No dependencies on US1/US2
- **User Story 4 (P4) - Super Weights**: Can start after Foundational - Enhances US1 but not required

### Within Each User Story

1. Tests MUST be written and FAIL before implementation
2. Core algorithm implementation
3. CLI integration
4. Validation/reporting integration
5. Unit tests for algorithm

### Parallel Opportunities

**Setup Phase**:
- T002, T003, T004, T005 can run in parallel (models)

**Foundational Phase**:
- T010, T011, T012 can run in parallel (lib utilities)
- T016, T017, T018 can run in parallel (tests)

**User Story Phases**:
- All test tasks within a story can run in parallel
- Different user stories can be worked on by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write tests/contract/test_cli_quantize_dynamic.py"
Task: "Write tests/integration/test_dynamic_quantization.py"

# After tests fail, implement sequentially:
Task: "Implement DynamicQuantizer in src/llm_quantize/lib/quantizers/advanced/dynamic.py"
Task: "Add automatic layer importance computation"
Task: "Implement profile-based layer assignment"
Task: "Extend CLI with --dynamic, --profile options"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Verify 001-multi-format-quantization is complete
2. Complete Phase 1: Setup
3. Complete Phase 2: Foundational
4. Complete Phase 3: User Story 1 (Dynamic)
5. **STOP and VALIDATE**: Test dynamic quantization independently
6. Deploy - users can use dynamic layer-wise quantization

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 (Dynamic) → Test → Deploy (MVP!)
3. Add User Story 2 (Ultra-Low-Bit) → Test → Deploy
4. Add User Story 3 (SmoothQuant) → Test → Deploy
5. Add User Story 4 (Super Weights) → Test → Deploy
6. Polish → Final release

### Parallel Team Strategy

With multiple developers:

1. Verify 001 is complete
2. Team completes Setup + Foundational together
3. Once Foundational is done:
   - Developer A: User Story 1 (Dynamic)
   - Developer B: User Story 2 (Ultra-Low-Bit)
   - Developer C: User Story 3 (SmoothQuant)
   - Developer D: User Story 4 (Super Weights)
4. Team completes Polish together

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- This feature DEPENDS on 001-multi-format-quantization being complete
- Constitution requires TDD - tests are mandatory, not optional
