<!--
SYNC IMPACT REPORT
==================
Version change: 0.0.0 → 0.0.1
Bump rationale: PATCH - Initial constitution draft (pre-release)

Modified principles: N/A (initial version)
Added sections:
  - Core Principles (7 total)
  - Quality Standards
  - Development Workflow
  - Governance

Removed sections: N/A (initial version)

Templates requiring updates:
  - .specify/templates/plan-template.md ✅ (compatible - no changes needed)
  - .specify/templates/spec-template.md ✅ (compatible - no changes needed)
  - .specify/templates/tasks-template.md ✅ (compatible - no changes needed)

Follow-up TODOs: None
-->

# LLM-Quantize Constitution

## Core Principles

### I. Library-First Architecture

All functionality MUST be implemented as standalone, importable library modules before any CLI
exposure. Libraries MUST be:

- Self-contained with explicit dependencies
- Independently testable without CLI invocation
- Documented with public API surface clearly defined
- Free of circular dependencies

**Rationale**: Enables reuse, simplifies testing, and allows programmatic integration by other tools.

### II. CLI Interface Contract

Every library capability MUST be exposed via a CLI interface following these rules:

- Text-based I/O: arguments and stdin for input, stdout for results, stderr for errors/logs
- Support both JSON (machine-readable) and human-readable output formats via `--format` flag
- Exit codes MUST be meaningful: 0 for success, non-zero for specific error categories
- All operations MUST be idempotent where semantically possible

**Rationale**: CLI-first design ensures scriptability, pipeline integration, and testability.

### III. Test-Driven Development (NON-NEGOTIABLE)

TDD is mandatory for all feature development:

1. Write failing tests that define expected behavior
2. Get test approval/review before implementation
3. Verify tests fail for the right reason
4. Implement minimal code to pass tests
5. Refactor while maintaining green tests

Test categories required:
- **Unit tests**: All public functions and classes
- **Contract tests**: API boundaries and format specifications
- **Integration tests**: End-to-end quantization workflows

**Rationale**: TDD prevents regression, documents behavior, and forces clear API design.

### IV. Model Format Compliance

Quantized model outputs MUST comply with established format specifications:

- GGUF format MUST follow the official GGUF specification
- AWQ format MUST be compatible with AutoAWQ tooling
- GPTQ format MUST be compatible with AutoGPTQ tooling
- Custom formats MUST have documented specifications in `docs/formats/`

All format changes MUST include:
- Version bump in format metadata
- Migration path documentation
- Backward compatibility assessment

**Rationale**: Format compliance ensures interoperability with the broader LLM ecosystem.

### V. Observability and Diagnostics

All quantization operations MUST provide:

- Structured logging with configurable verbosity levels
- Progress reporting for long-running operations
- Memory usage tracking and reporting
- Quantization quality metrics (perplexity delta, accuracy benchmarks)
- Timing breakdowns for performance analysis

**Rationale**: Quantization is resource-intensive; visibility into operations is essential for
debugging and optimization.

### VI. Versioning and Breaking Changes

Version numbers MUST follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Backward-incompatible API changes or output format changes
- **MINOR**: New features, new quantization methods, new format support
- **PATCH**: Bug fixes, performance improvements, documentation updates

Breaking changes MUST:
- Be documented in CHANGELOG.md
- Include migration guides
- Have deprecation warnings in prior minor release when possible

**Rationale**: Predictable versioning enables safe upgrades and dependency management.

### VII. Simplicity and YAGNI

Implementation MUST prefer simplicity:

- Start with the minimal viable solution
- Add complexity only when proven necessary by real use cases
- Avoid speculative generalization
- Prefer composition over inheritance
- No feature flags for unreleased functionality

**Rationale**: Quantization algorithms are complex enough; infrastructure should be simple.

## Quality Standards

### Code Quality Gates

All code MUST pass before merge:

- Type checking (mypy strict mode or equivalent)
- Linting (ruff or equivalent with project configuration)
- Formatting (black/ruff format or equivalent)
- Test coverage minimum: 80% line coverage, 70% branch coverage
- No security vulnerabilities in dependencies (audited via pip-audit or equivalent)

### Documentation Requirements

- All public APIs MUST have docstrings with parameter descriptions
- All CLI commands MUST have `--help` output
- Complex algorithms MUST have inline comments explaining the approach
- README MUST include quickstart example

## Development Workflow

### Branch Strategy

- `main` branch is protected; direct pushes forbidden
- Feature branches: `feature/###-description`
- Bug fix branches: `fix/###-description`
- All changes via pull request with at least one approval

### Commit Standards

- Commits MUST follow Conventional Commits specification
- Commit messages MUST reference issue numbers when applicable
- Atomic commits: one logical change per commit

### Review Requirements

- All PRs MUST pass CI checks before review
- Constitution compliance MUST be verified in PR checklist
- Performance-sensitive changes MUST include benchmark results

## Governance

This constitution supersedes all other development practices for the llm-quantize project.

### Amendment Process

1. Propose amendment via pull request to this file
2. Document rationale and impact assessment
3. Obtain maintainer approval
4. Update version according to change scope
5. Propagate changes to dependent templates

### Compliance

- All PRs MUST include constitution compliance verification
- Reviewers MUST check for principle violations
- Complexity beyond principles MUST be justified in PR description

### Version Policy

- MAJOR: Principle removal, redefinition, or governance structure change
- MINOR: New principle added, existing principle expanded
- PATCH: Clarifications, typo fixes, non-semantic refinements

**Version**: 0.0.1 | **Ratified**: 2025-12-12 | **Last Amended**: 2025-12-12
