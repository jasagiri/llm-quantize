# llm-quantize Documentation

A unified CLI tool for LLM model quantization supporting GGUF, AWQ, and GPTQ formats.

## Overview

llm-quantize provides a simple, unified interface for quantizing large language models into efficient formats for deployment. It supports multiple quantization methods and output formats, with features like checkpointing, importance analysis, and quality validation.

## Key Features

- **Multiple Formats**: Support for GGUF (llama.cpp), AWQ (GPU inference), and GPTQ formats
- **Unified CLI**: Single command-line interface for all quantization needs
- **Importance Analysis**: Compute layer importance matrices for optimized quantization
- **Quality Validation**: Automatic validation of quantized outputs
- **Checkpointing**: Resume interrupted quantization from checkpoints
- **Progress Reporting**: Real-time progress with memory usage tracking

## Quick Start

```bash
# Install
pip install llm-quantize

# Get model information
llm-quantize info meta-llama/Llama-2-7b-hf

# Quantize to GGUF Q4_K_M (recommended for CPU inference)
llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

# Quantize to AWQ 4-bit (for GPU inference with vLLM)
llm-quantize quantize ./my-model awq -q 4bit
```

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [CLI Reference](cli-reference.md) - Complete command reference
- [Quantization Formats](quantization-formats.md) - GGUF, AWQ, and GPTQ details
- [Advanced Features](advanced-features.md) - Importance analysis, SmoothQuant, etc.
- [Python API](api-reference.md) - Using llm-quantize as a library
- [Development Guide](development.md) - Contributing and development setup

## Supported Models

llm-quantize supports any HuggingFace-compatible transformer model, including:

- LLaMA / LLaMA 2 / LLaMA 3
- Mistral / Mixtral
- Qwen / Qwen2
- Phi-2 / Phi-3
- GPT-2 / GPT-J / GPT-NeoX
- Falcon
- And many more

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers library
- Sufficient disk space for model weights

## License

Apache-2.0 License
