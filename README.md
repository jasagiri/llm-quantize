# llm-quantize

A unified CLI tool for LLM model quantization supporting GGUF, AWQ, GPTQ, and advanced quantization formats.

## Features

- **Multi-format support**: GGUF, AWQ, GPTQ, SmoothQuant, Dynamic Quantization, Ultra-low-bit (IQ)
- **Quality analysis**: Perplexity measurement, coherence testing, quality reports
- **Advanced quantization**: Importance-based dynamic bit allocation, layer-wise optimization
- **Format conversion**: Convert between different quantization formats
- **Checkpoint support**: Resume interrupted quantization jobs

## Installation

```bash
pip install llm-quantize
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Quantize a Model

```bash
# Quantize to GGUF Q4_K_M (recommended for most use cases)
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# Quantize to AWQ 4-bit
llm-quantize quantize ./my-model awq -q 4bit

# Quantize to GPTQ
llm-quantize quantize ./my-model gptq -q 4bit
```

### Advanced Quantization

```bash
# SmoothQuant (W8A8)
llm-quantize quantize ./my-model gguf -q W8A8 --method smoothquant

# Dynamic quantization with importance matrix
llm-quantize quantize ./my-model gguf -q dynamic --imatrix importance.dat

# Ultra-low-bit quantization (IQ2_XXS)
llm-quantize quantize ./my-model gguf -q IQ2_XXS
```

### Analyze Model Quality

```bash
# Analyze quantized model quality
llm-quantize analyze ./quantized-model --format gguf

# Generate detailed quality report
llm-quantize analyze ./quantized-model --format gguf --report quality_report.json
```

### Convert Between Formats

```bash
# Convert GGUF to AWQ
llm-quantize convert ./model.gguf awq -o ./model-awq
```

## Supported Quantization Levels

### GGUF Format
- `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`
- `Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`
- `Q5_0`, `Q5_1`, `Q5_K_S`, `Q5_K_M`
- `Q6_K`, `Q8_0`
- Ultra-low-bit: `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`

### AWQ Format
- `4bit` (default)

### GPTQ Format
- `2bit`, `3bit`, `4bit` (default), `8bit`

### SmoothQuant
- `W8A8` (8-bit weights, 8-bit activations)

## CLI Commands

| Command | Description |
|---------|-------------|
| `quantize` | Quantize a model to a target format |
| `convert` | Convert between quantization formats |
| `analyze` | Analyze quantized model quality |
| `info` | Display model information |

## Configuration

### Calibration Data

For quality quantization, provide calibration data:

```bash
llm-quantize quantize ./model gguf -q Q4_K_M --calibration-data ./calibration.txt
```

### Checkpoint and Resume

Large quantization jobs can be resumed:

```bash
# Enable checkpoints
llm-quantize quantize ./model gguf -q Q4_K_M --enable-checkpoints

# Resume from checkpoint
llm-quantize quantize ./model gguf -q Q4_K_M --resume-from ./checkpoints
```

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=src/llm_quantize --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_quantizers/test_gguf.py -v
```

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

## Architecture

```
llm-quantize/
├── src/llm_quantize/
│   ├── cli/           # CLI commands
│   ├── lib/           # Core library
│   │   ├── analysis/  # Quality analysis
│   │   ├── quantizers/
│   │   │   ├── advanced/  # SmoothQuant, Dynamic, Ultra-low-bit
│   │   │   ├── awq.py
│   │   │   ├── gguf.py
│   │   │   └── gptq.py
│   │   ├── calibration.py
│   │   ├── converter.py
│   │   └── validation.py
│   └── models/        # Data models
└── tests/             # Test suite
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.
