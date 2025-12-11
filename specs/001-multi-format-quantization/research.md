# Research: Multi-Format Quantization Tool

**Feature**: 001-multi-format-quantization
**Date**: 2025-12-12

## Technology Decisions

### 1. GGUF Quantization Library

**Decision**: Use `llama-cpp-python` with custom Python bindings for GGUF conversion

**Rationale**:
- llama.cpp is the de facto standard for GGUF format and quantization
- `llama-cpp-python` provides Python bindings with good performance
- Direct integration with HF models via `convert.py` scripts from llama.cpp
- Supports all GGUF quantization levels (Q2_K through Q8_0)

**Alternatives Considered**:
- Pure Python GGUF writer: Rejected - would need to reimplement quantization algorithms
- Subprocess to llama.cpp: Considered as fallback - less integrated but more robust
- ctransformers: Rejected - less mature GGUF support

### 2. AWQ Quantization Library

**Decision**: Use `autoawq` library

**Rationale**:
- Official implementation of AWQ paper
- Well-integrated with HuggingFace transformers
- Supports calibration with custom datasets
- Active maintenance and community support

**Alternatives Considered**:
- Custom implementation: Rejected - AWQ algorithm is complex, autoawq is proven
- llm-awq (original): Rejected - less maintained than autoawq fork

### 3. GPTQ Quantization Library

**Decision**: Use `auto-gptq` library

**Rationale**:
- Most mature GPTQ implementation
- Direct HuggingFace integration
- Configurable group size and bit-width
- ExLlama kernel support for fast inference

**Alternatives Considered**:
- GPTQ-for-LLaMa: Rejected - specialized for LLaMA, less general
- Custom implementation: Rejected - GPTQ requires careful numerical handling

### 4. CLI Framework

**Decision**: Use `click` with `rich` for output formatting

**Rationale**:
- Click provides clean command structure and argument parsing
- Rich enables progress bars, tables, and formatted output
- Both are well-maintained with good documentation
- Click's decorators align with library-first design

**Alternatives Considered**:
- argparse: Rejected - more verbose, less ergonomic for complex CLIs
- typer: Considered - built on click, but adds unnecessary dependency
- fire: Rejected - magic-based, less explicit

### 5. Model Loading Strategy

**Decision**: Use `transformers` with `huggingface_hub` for unified model access

**Rationale**:
- Single API for local and Hub models
- Built-in support for HF_TOKEN authentication
- Automatic model architecture detection
- Safetensors/PyTorch weight handling

**Alternatives Considered**:
- Direct file loading: Rejected - would need to handle all model formats
- Custom Hub client: Rejected - reinventing the wheel

### 6. Checkpoint Format

**Decision**: JSON metadata + pickled tensor state per layer

**Rationale**:
- JSON for human-readable checkpoint info (layer index, config, progress)
- Pickle for efficient tensor serialization (intermediate quantized weights)
- Layer-granular saves enable resume without full model reload
- Compatible with torch.save/load patterns

**Alternatives Considered**:
- Full model checkpoints: Rejected - too large for frequent saves
- SQLite: Rejected - overhead for simple key-value checkpoint data
- Safetensors for checkpoints: Considered - cleaner but pickle is faster for temporary data

### 7. Memory Management

**Decision**: Layer-by-layer processing with explicit garbage collection

**Rationale**:
- Process one layer at a time to bound peak memory
- Explicit `torch.cuda.empty_cache()` and `gc.collect()` between layers
- Memory tracking via `torch.cuda.memory_allocated()` for reporting
- Enables 7B models on 16GB GPU, larger models with CPU offload

**Alternatives Considered**:
- Full model in memory: Rejected - fails for larger models
- Disk offloading: Considered as extension - adds complexity

## Best Practices

### GGUF Format Compliance

- Follow GGUF v3 specification for file structure
- Include all required metadata (architecture, tokenizer, quantization info)
- Validate output files can be loaded by llama.cpp before reporting success
- Use standard quantization type names (Q4_K_M, not custom)

### AWQ Calibration

- Default to WikiText-2 for calibration (128-512 samples)
- Support custom calibration datasets via file path
- Warn users about calibration dataset impact on quality
- Cache calibration statistics for repeated runs

### GPTQ Best Practices

- Default group size of 128 (balance of speed/quality)
- Support act_order for better quality (with performance tradeoff)
- Validate quantized model loads correctly with AutoGPTQ

### Error Handling

- Clear error messages with suggested fixes
- Non-zero exit codes for all failure modes
- Graceful handling of OOM with checkpoint save
- Validation errors before starting long operations

## Integration Patterns

### HuggingFace Hub Authentication

```python
# Pattern: Environment variable with fallback
from huggingface_hub import HfFolder
token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
```

### Progress Reporting

```python
# Pattern: Rich progress with structured updates
from rich.progress import Progress, TaskID
with Progress() as progress:
    task = progress.add_task("Quantizing...", total=num_layers)
    for layer in layers:
        # ... quantize layer ...
        progress.update(task, advance=1)
```

### JSON Output Mode

```python
# Pattern: Structured output for scripting
if output_format == "json":
    print(json.dumps(result, indent=2))
else:
    console.print(result_table)
```

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Which GGUF quantization types to support? | All standard types (Q2_K through Q8_0) per FR-002 |
| Default calibration dataset? | WikiText-2, 256 samples default |
| Checkpoint storage location? | Same directory as output, `.checkpoint/` subdirectory |
| Memory reporting granularity? | Per-layer peak and cumulative |
