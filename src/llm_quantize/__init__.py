"""LLM-Quantize: A unified CLI tool for LLM model quantization.

Supports multiple quantization formats:
- GGUF: For llama.cpp and compatible inference engines
- AWQ: For vLLM and AutoAWQ-compatible frameworks
- GPTQ: For AutoGPTQ and ExLlama inference
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-quantize")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
