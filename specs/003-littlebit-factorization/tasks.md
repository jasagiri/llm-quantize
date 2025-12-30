# Tasks: LittleBit Ultra-Low-Bit Quantization

**Input**: Design documents from `/specs/003-littlebit-factorization/`
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

**Purpose**: Extension infrastructure for LittleBit factorization-based compression

**Prerequisite**: 001-multi-format-quantization must be complete

- [ ] T001 Create directory structure: src/llm_quantize/lib/factorization/, src/llm_quantize/lib/compensation/, src/llm_quantize/lib/formats/
- [ ] T002 [P] Create src/llm_quantize/models/factorization_config.py with FactorizationConfig dataclass (target_bpw, latent_rank, etc.)
- [ ] T003 [P] Create src/llm_quantize/models/factorized_weights.py with FactorizedWeights and BinaryFactor dataclasses
- [ ] T004 [P] Create src/llm_quantize/models/compensation_factors.py with CompensationFactors dataclass
- [ ] T005 [P] Create src/llm_quantize/models/compression_report.py with CompressionReport, LayerStats, QualityMetrics
- [ ] T006 Update src/llm_quantize/models/__init__.py to export new model classes
- [ ] T007 Create tests/unit/test_factorization/ directory for factorization tests

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core factorization and compensation algorithms

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T008 Implement src/llm_quantize/lib/factorization/__init__.py
- [ ] T009 Implement src/llm_quantize/lib/factorization/svd.py with factorize_weight() using torch.linalg.svd
- [ ] T010 [P] Implement src/llm_quantize/lib/factorization/binarization.py with binarize_factor() for sign-based binarization
- [ ] T011 [P] Implement src/llm_quantize/lib/factorization/rank_selection.py with select_rank_for_bpw() using binary search
- [ ] T012 Implement src/llm_quantize/lib/compensation/__init__.py
- [ ] T013 [P] Implement src/llm_quantize/lib/compensation/row.py with compute_row_compensation()
- [ ] T014 [P] Implement src/llm_quantize/lib/compensation/column.py with compute_column_compensation()
- [ ] T015 [P] Implement src/llm_quantize/lib/compensation/latent.py with compute_latent_compensation()
- [ ] T016 Implement src/llm_quantize/lib/compensation/residual.py with compute_residual_compensation()
- [ ] T017 Extend src/llm_quantize/lib/checkpoint.py with FactorizationCheckpoint for layer-level saves
- [ ] T018 Create src/llm_quantize/cli/littlebit.py with command group skeleton
- [ ] T019 Write tests/unit/test_factorization/test_svd.py for SVD factorization
- [ ] T020 [P] Write tests/unit/test_factorization/test_binarization.py for factor binarization
- [ ] T021 [P] Write tests/unit/test_factorization/test_rank_selection.py for BPW-based rank selection
- [ ] T022 [P] Write tests/unit/test_factorization/test_compensation.py for multi-scale compensation

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Extreme Model Compression (Priority: P1) 🎯 MVP

**Goal**: Compress models to 0.1 BPW (~31x compression) using latent matrix factorization

**Independent Test**: Compress 7B model to 0.1 BPW and verify it fits in ~450 MB with coherent outputs

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T023 [P] [US1] Write tests/contract/test_cli_littlebit_compress.py for compress CLI contract
- [ ] T024 [P] [US1] Write tests/integration/test_littlebit_compression.py for end-to-end compression

### Implementation for User Story 1

- [ ] T025 [US1] Implement src/llm_quantize/lib/factorization/dual_svid.py with Dual-SVID initialization
- [ ] T026 [US1] Create src/llm_quantize/lib/littlebit.py with LittleBitCompressor class orchestrating factorization pipeline
- [ ] T027 [US1] Implement layer-by-layer compression with progress reporting
- [ ] T028 [US1] Implement compression statistics computation (achieved BPW, memory reduction ratio)
- [ ] T029 [US1] Implement src/llm_quantize/cli/littlebit.py compress command with --target-bpw option
- [ ] T030 [US1] Add coherence validation with sample inference after compression
- [ ] T031 [US1] Write tests/unit/test_littlebit.py for LittleBitCompressor unit tests

**Checkpoint**: LittleBit compression at 0.1 BPW fully functional and testable

---

## Phase 4: User Story 2 - Configurable Compression Levels (Priority: P2)

**Goal**: Support different BPW targets (0.1, 0.3, 0.5, 0.7) for quality-compression tradeoff

**Independent Test**: Compress same model at different BPW levels and verify higher BPW = better quality

### Tests for User Story 2

- [ ] T032 [P] [US2] Write tests/contract/test_cli_littlebit_bpw.py for BPW configuration contract
- [ ] T033 [P] [US2] Write tests/integration/test_littlebit_bpw_levels.py for multi-BPW compression

### Implementation for User Story 2

- [ ] T034 [US2] Extend LittleBitCompressor to accept target_bpw range validation (0.1-2.0)
- [ ] T035 [US2] Implement automatic rank adjustment based on target BPW per layer
- [ ] T036 [US2] Add BPW validation with clear error messages for invalid ranges
- [ ] T037 [US2] Extend compress command with BPW validation and helpful error messages
- [ ] T038 [US2] Add perplexity comparison in output for different BPW levels
- [ ] T039 [US2] Write tests/unit/test_bpw_validation.py for BPW validation

**Checkpoint**: Configurable BPW compression functional alongside 0.1 BPW

---

## Phase 5: User Story 3 - Latent Rank Configuration (Priority: P3)

**Goal**: Allow manual rank specification for research experimentation

**Independent Test**: Compress with different rank settings and verify size/quality tradeoffs

### Tests for User Story 3

- [ ] T040 [P] [US3] Write tests/contract/test_cli_littlebit_rank.py for rank configuration contract
- [ ] T041 [P] [US3] Write tests/integration/test_littlebit_rank.py for manual rank compression

### Implementation for User Story 3

- [ ] T042 [US3] Extend LittleBitCompressor to accept manual rank parameter
- [ ] T043 [US3] Implement rank validation (must be <= min(m, n) for each layer)
- [ ] T044 [US3] Add per-layer rank statistics in compression report
- [ ] T045 [US3] Extend compress command with --rank option
- [ ] T046 [US3] Implement AUTO rank mode (default) vs MANUAL rank mode
- [ ] T047 [US3] Write tests/unit/test_rank_config.py for rank configuration

**Checkpoint**: Manual rank configuration functional for research use

---

## Phase 6: User Story 4 - Quality-Aware Compression (Priority: P4)

**Goal**: Automatically adjust compression to meet quality thresholds

**Independent Test**: Set quality threshold and verify system adjusts BPW to meet it

### Tests for User Story 4

- [ ] T048 [P] [US4] Write tests/contract/test_cli_littlebit_quality.py for quality threshold contract
- [ ] T049 [P] [US4] Write tests/integration/test_littlebit_quality_aware.py for quality-aware compression

### Implementation for User Story 4

- [ ] T050 [US4] Implement quality threshold enforcement in LittleBitCompressor
- [ ] T051 [US4] Add iterative BPW adjustment to meet quality target
- [ ] T052 [US4] Implement unachievable threshold detection with user warning
- [ ] T053 [US4] Extend compress command with --quality-threshold option
- [ ] T054 [US4] Add iteration report (attempts, final BPW, quality achieved)
- [ ] T055 [US4] Write tests/unit/test_quality_aware.py for quality-aware compression

**Checkpoint**: All user stories complete - quality-aware compression functional

---

## Phase 7: Output Formats (Required)

**Purpose**: Support native .littlebit, GGUF extension, and lossy GGUF formats

- [ ] T056 Implement src/llm_quantize/lib/formats/__init__.py
- [ ] T057 Implement src/llm_quantize/lib/formats/littlebit.py with native .littlebit format writer/reader
- [ ] T058 [P] Implement src/llm_quantize/lib/formats/gguf_ext.py with GGUF metadata extension
- [ ] T059 [P] Implement src/llm_quantize/lib/formats/gguf_lossy.py with lossy GGUF conversion
- [ ] T060 Write tests/unit/test_formats/test_littlebit_format.py for native format
- [ ] T061 [P] Write tests/unit/test_formats/test_gguf_ext.py for GGUF extension
- [ ] T062 [P] Write tests/unit/test_formats/test_gguf_lossy.py for lossy conversion
- [ ] T063 Extend compress command with --format option (littlebit, gguf-ext, gguf-lossy)

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T064 Implement src/llm_quantize/cli/littlebit.py info command for .littlebit file inspection
- [ ] T065 [P] Write tests/contract/test_cli_littlebit_info.py for info command contract
- [ ] T066 Implement src/llm_quantize/cli/littlebit.py convert command between LittleBit formats
- [ ] T067 [P] Write tests/contract/test_cli_littlebit_convert.py for convert command contract
- [ ] T068 Implement src/llm_quantize/cli/littlebit.py validate command for quality validation
- [ ] T069 [P] Write tests/contract/test_cli_littlebit_validate.py for validate command contract
- [ ] T070 Add comprehensive error handling with exit codes 20-24 for LittleBit errors
- [ ] T071 [P] Validate environment variables (LLM_QUANTIZE_SVD_PRECISION, etc.)
- [ ] T072 Run quickstart.md validation: test all documented commands work as specified
- [ ] T073 [P] Add type hints and run mypy on all new code
- [ ] T074 Run ruff and fix all linting issues
- [ ] T075 Verify test coverage meets 80% line, 70% branch requirement

---

## Dependencies & Execution Order

### Phase Dependencies

- **Prerequisite**: 001-multi-format-quantization complete
- **Setup (Phase 1)**: No additional dependencies
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
- **Output Formats (Phase 7)**: Depends on User Story 1 at minimum
- **Polish (Phase 8)**: Depends on all user stories and formats being complete

### User Story Dependencies

- **User Story 1 (P1) - Extreme Compression**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2) - Configurable BPW**: Can start after Foundational - Enhances US1 but independent
- **User Story 3 (P3) - Rank Config**: Can start after Foundational - Independent of US1/US2
- **User Story 4 (P4) - Quality-Aware**: Depends on US2 (BPW configuration) for iterative adjustment

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
- T010, T011 can run in parallel (factorization)
- T013, T014, T015 can run in parallel (compensation)
- T019, T020, T021, T022 can run in parallel (tests)

**Output Formats Phase**:
- T058, T059 can run in parallel (GGUF formats)
- T061, T062 can run in parallel (format tests)

**User Story Phases**:
- All test tasks within a story can run in parallel
- US1, US2, US3 can be worked on by different team members (US4 depends on US2)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write tests/contract/test_cli_littlebit_compress.py"
Task: "Write tests/integration/test_littlebit_compression.py"

# After tests fail, implement sequentially:
Task: "Implement Dual-SVID initialization"
Task: "Create LittleBitCompressor class"
Task: "Implement layer-by-layer compression with progress"
Task: "Implement compress command with --target-bpw"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Verify 001-multi-format-quantization is complete
2. Complete Phase 1: Setup
3. Complete Phase 2: Foundational
4. Complete Phase 3: User Story 1 (Extreme Compression)
5. Complete Phase 7: Output Formats (at least native .littlebit)
6. **STOP and VALIDATE**: Test 0.1 BPW compression independently
7. Deploy - users can compress models to extreme compression levels

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 (Extreme) + Formats → Test → Deploy (MVP!)
3. Add User Story 2 (BPW Config) → Test → Deploy
4. Add User Story 3 (Rank Config) → Test → Deploy
5. Add User Story 4 (Quality-Aware) → Test → Deploy
6. Polish → Final release

### Parallel Team Strategy

With multiple developers:

1. Verify 001 is complete
2. Team completes Setup + Foundational together
3. Once Foundational is done:
   - Developer A: User Story 1 (Extreme Compression)
   - Developer B: User Story 2 (BPW Config)
   - Developer C: User Story 3 (Rank Config)
   - Developer D: Output Formats (Phase 7)
4. After US2 complete:
   - Any developer: User Story 4 (Quality-Aware)
5. Team completes Polish together

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
- Output Formats phase is REQUIRED for usable compression (not optional)
