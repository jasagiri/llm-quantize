# Development Guide

This guide covers development setup, testing, and contributing to llm-quantize.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (recommended)

### Clone and Install

```bash
git clone https://github.com/your-org/llm-quantize.git
cd llm-quantize
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For GPU quantization (AWQ/GPTQ)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For GGUF conversion
pip install llama-cpp-python

# For full testing
pip install -e ".[dev,test]"
```

## Project Structure

```
llm-quantize/
├── src/llm_quantize/
│   ├── __init__.py
│   ├── cli/                    # CLI commands
│   │   ├── __init__.py
│   │   ├── main.py            # Entry point
│   │   ├── quantize.py        # quantize command
│   │   ├── convert.py         # convert command
│   │   └── analyze.py         # analyze command
│   ├── lib/                    # Core library
│   │   ├── __init__.py
│   │   ├── analysis/          # Analysis modules
│   │   │   ├── __init__.py
│   │   │   ├── importance.py  # Importance matrix
│   │   │   ├── quality.py     # Quality analysis
│   │   │   └── super_weights.py
│   │   ├── quantizers/        # Quantization backends
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # Base quantizer
│   │   │   ├── gguf.py        # GGUF quantizer
│   │   │   ├── awq.py         # AWQ quantizer
│   │   │   ├── gptq.py        # GPTQ quantizer
│   │   │   └── advanced/      # Advanced methods
│   │   │       ├── smoothquant.py
│   │   │       ├── dynamic.py
│   │   │       └── ultra_low_bit.py
│   │   ├── calibration.py     # Calibration data
│   │   ├── checkpoint.py      # Checkpointing
│   │   ├── converter.py       # Format conversion
│   │   ├── model_loader.py    # Model loading
│   │   ├── progress.py        # Progress reporting
│   │   └── validation.py      # Output validation
│   └── models/                 # Data models
│       ├── __init__.py
│       ├── source_model.py
│       ├── quantization_config.py
│       ├── quantized_model.py
│       └── ...
├── tests/
│   ├── conftest.py            # pytest fixtures
│   ├── unit/                   # Unit tests
│   ├── integration/           # Integration tests
│   └── contract/              # CLI contract tests
├── docs/                       # Documentation
├── pyproject.toml             # Project configuration
└── README.md
```

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=src/llm_quantize --cov-report=term-missing
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Contract tests (CLI)
pytest tests/contract/

# Specific file
pytest tests/unit/test_quantizers/test_gguf.py -v
```

### Test Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only GPU tests (if available)
pytest -m gpu
```

## Code Quality

### Formatting with Ruff

```bash
# Format code
ruff format src tests

# Check formatting
ruff format --check src tests
```

### Linting with Ruff

```bash
# Check for issues
ruff check src tests

# Auto-fix issues
ruff check --fix src tests
```

### Type Checking with MyPy

```bash
mypy src
```

## Adding a New Quantizer

1. Create a new file in `src/llm_quantize/lib/quantizers/`:

```python
# src/llm_quantize/lib/quantizers/my_quantizer.py
from .base import BaseQuantizer
from llm_quantize.models import QuantizedModel

class MyQuantizer(BaseQuantizer):
    """Custom quantizer implementation."""

    @classmethod
    def get_supported_levels(cls) -> list[str]:
        """Return supported quantization levels."""
        return ["4bit", "8bit"]

    def estimate_output_size(self) -> int:
        """Estimate output file size in bytes."""
        bits = 4 if "4bit" in self.config.quantization_level else 8
        return self.source_model.parameter_count * bits // 8

    def get_output_path(self) -> Path:
        """Get the output file path."""
        name = self.config.output_name or f"{self.source_model.model_path}-my"
        return Path(self.config.output_dir) / name

    def quantize(self) -> QuantizedModel:
        """Run quantization and return result."""
        # Implementation here
        pass
```

2. Register the quantizer in `__init__.py`:

```python
from .my_quantizer import MyQuantizer
from llm_quantize.models import OutputFormat

register_quantizer(OutputFormat.MY_FORMAT, MyQuantizer)
```

3. Add tests:

```python
# tests/unit/test_quantizers/test_my_quantizer.py
def test_my_quantizer_basic():
    quantizer = MyQuantizer(source_model, config)
    result = quantizer.quantize()
    assert result.format == "my_format"
```

## Adding a New CLI Command

1. Create command in `src/llm_quantize/cli/`:

```python
# src/llm_quantize/cli/my_command.py
import click

@click.command()
@click.argument("input_path")
@click.option("-o", "--output", help="Output path")
def my_command(input_path: str, output: str):
    """My custom command description."""
    # Implementation
    click.echo(f"Processing {input_path}")
```

2. Register in `main.py`:

```python
from .my_command import my_command

cli.add_command(my_command)
```

3. Add contract tests:

```python
# tests/contract/test_my_command.py
def test_my_command_basic():
    runner = CliRunner()
    result = runner.invoke(cli, ["my-command", "input.txt"])
    assert result.exit_code == 0
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### CLI Debug Mode

```bash
llm-quantize --verbosity debug quantize model gguf -q Q4_K_M
```

### Memory Profiling

```python
from llm_quantize.lib.progress import ProgressReporter

reporter = ProgressReporter()
# Memory is tracked automatically
print(f"Peak memory: {reporter.peak_memory / 1e9:.2f} GB")
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create and push tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

4. GitHub Actions will automatically:
   - Run tests
   - Build package
   - Create GitHub release
   - Publish to PyPI

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Document public APIs with docstrings

### Docstring Format

```python
def quantize(self, model: str, format: str) -> QuantizedModel:
    """Quantize a model to the specified format.

    Args:
        model: Path or HuggingFace Hub ID of the model.
        format: Target format (gguf, awq, gptq).

    Returns:
        QuantizedModel with output path and metadata.

    Raises:
        ValueError: If format is not supported.
        RuntimeError: If quantization fails.
    """
```

### Commit Messages

Follow conventional commits:

```
feat: add ultra-low-bit quantization support
fix: handle empty calibration data gracefully
docs: update API reference for v0.2.0
test: add integration tests for AWQ quantizer
refactor: extract common quantization logic
```

## License

Apache-2.0 - See LICENSE file for details.
