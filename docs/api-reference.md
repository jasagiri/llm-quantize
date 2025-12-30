# Python API Reference

Use llm-quantize as a Python library for programmatic quantization.

## Installation

```bash
pip install llm-quantize
```

## Quick Start

```python
from llm_quantize.lib.model_loader import create_source_model
from llm_quantize.lib.quantizers import GGUFQuantizer
from llm_quantize.models import QuantizationConfig, OutputFormat

# Load model information
source_model = create_source_model("meta-llama/Llama-2-7b-hf")

# Configure quantization
config = QuantizationConfig(
    target_format=OutputFormat.GGUF,
    quantization_level="Q4_K_M",
    output_dir="./output",
)

# Create quantizer and run
quantizer = GGUFQuantizer(source_model, config)
result = quantizer.quantize()

print(f"Output: {result.output_path}")
print(f"Size: {result.file_size:,} bytes")
print(f"Compression: {result.compression_ratio:.2%}")
```

---

## Models

### SourceModel

Represents a source model to be quantized.

```python
from llm_quantize.models import SourceModel, ModelType

model = SourceModel(
    model_path="meta-llama/Llama-2-7b-hf",
    model_type=ModelType.HF_HUB,
    architecture="LlamaForCausalLM",
    parameter_count=7_000_000_000,
    dtype="float16",
    num_layers=32,
    hidden_size=4096,
    num_heads=32,
    vocab_size=32000,
    hf_token=None,  # Optional HuggingFace token
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_path` | str | Path or HuggingFace Hub ID |
| `model_type` | ModelType | LOCAL or HF_HUB |
| `architecture` | str | Model architecture name |
| `parameter_count` | int | Total parameters |
| `dtype` | str | Original data type |
| `num_layers` | int | Number of transformer layers |
| `hidden_size` | int | Hidden dimension |
| `num_heads` | int | Attention heads |
| `vocab_size` | int | Vocabulary size |
| `hf_token` | Optional[str] | HuggingFace token |

### QuantizationConfig

Configuration for quantization process.

```python
from llm_quantize.models import QuantizationConfig, OutputFormat

config = QuantizationConfig(
    target_format=OutputFormat.GGUF,
    quantization_level="Q4_K_M",
    output_dir="./output",
    output_name="my-model",  # Optional
    calibration_data_path="./calibration.json",  # Optional
    calibration_samples=256,
    group_size=128,  # For GPTQ
    enable_checkpoints=True,
    checkpoint_dir="./.checkpoint",
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_format` | OutputFormat | - | GGUF, AWQ, or GPTQ |
| `quantization_level` | str | - | Format-specific level |
| `output_dir` | str | "." | Output directory |
| `output_name` | Optional[str] | None | Custom output name |
| `calibration_data_path` | Optional[str] | None | Calibration file path |
| `calibration_samples` | int | 256 | Number of samples |
| `group_size` | int | 128 | GPTQ group size |
| `enable_checkpoints` | bool | True | Enable checkpointing |
| `checkpoint_dir` | Optional[str] | None | Checkpoint directory |

### QuantizedModel

Result of quantization.

```python
from llm_quantize.models import QuantizedModel, ValidationStatus

result = QuantizedModel(
    output_path="/path/to/output",
    format="gguf",
    quantization_level="Q4_K_M",
    file_size=4_000_000_000,
    compression_ratio=0.25,
    duration_seconds=120.5,
    peak_memory_bytes=16_000_000_000,
    source_model_path="meta-llama/Llama-2-7b-hf",
    validation_status=ValidationStatus.VALID,
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `output_path` | str | Path to output file/directory |
| `format` | str | Output format |
| `quantization_level` | str | Applied quantization level |
| `file_size` | int | Output size in bytes |
| `compression_ratio` | float | Size relative to original |
| `duration_seconds` | float | Quantization time |
| `peak_memory_bytes` | int | Peak memory usage |
| `source_model_path` | str | Original model path |
| `validation_status` | ValidationStatus | Validation result |

---

## Quantizers

### GGUFQuantizer

Quantize models to GGUF format.

```python
from llm_quantize.lib.quantizers import GGUFQuantizer
from llm_quantize.lib.progress import ProgressReporter

# With progress reporting
reporter = ProgressReporter()
quantizer = GGUFQuantizer(
    source_model=source_model,
    config=config,
    progress_reporter=reporter,
    enable_checkpoints=True,
    resume_from=None,  # Path to resume checkpoint
)

# Get supported levels
levels = GGUFQuantizer.get_supported_levels()
print(levels)  # ['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q4_0', 'Q4_K_S', 'Q4_K_M', ...]

# Estimate output size
estimated_size = quantizer.estimate_output_size()

# Run quantization
result = quantizer.quantize()
```

### AWQQuantizer

Quantize models to AWQ format.

```python
from llm_quantize.lib.quantizers import AWQQuantizer

quantizer = AWQQuantizer(
    source_model=source_model,
    config=config,
    progress_reporter=reporter,
)

result = quantizer.quantize()
```

### GPTQQuantizer

Quantize models to GPTQ format.

```python
from llm_quantize.lib.quantizers import GPTQQuantizer

quantizer = GPTQQuantizer(
    source_model=source_model,
    config=config,
    progress_reporter=reporter,
)

result = quantizer.quantize()
```

### get_quantizer

Factory function to get appropriate quantizer.

```python
from llm_quantize.lib.quantizers import get_quantizer
from llm_quantize.models import OutputFormat

QuantizerClass = get_quantizer(OutputFormat.GGUF)
quantizer = QuantizerClass(source_model, config)
```

---

## Analysis

### Importance Matrix

Compute layer importance for optimized quantization.

```python
from llm_quantize.lib.analysis.importance import compute_importance_matrix
from llm_quantize.lib.model_loader import load_model
from llm_quantize.models import ImportanceMethod

# Load model
model, tokenizer = load_model("meta-llama/Llama-2-7b-hf")

# Compute importance
importance_matrix = compute_importance_matrix(
    model=model,
    tokenizer=tokenizer,
    calibration_data=["Sample text 1", "Sample text 2"],
    method=ImportanceMethod.ACTIVATION_MAGNITUDE,
)

# Access results
for layer in importance_matrix.layer_scores:
    print(f"{layer.layer_name}: {layer.importance_score:.4f}")

# Save to file
importance_matrix.save("importance.json")

# Load from file
from llm_quantize.models import ImportanceMatrix
loaded = ImportanceMatrix.load("importance.json")
```

### Super Weights

Identify and protect super weights.

```python
from llm_quantize.lib.analysis.super_weights import (
    identify_super_weights,
    compute_super_weight_statistics,
    create_protection_mask,
)

# Identify super weights (top 0.01%)
super_weights = identify_super_weights(
    model=model,
    coverage=0.0001,
    method="activation_magnitude",
)

# Get statistics
stats = compute_super_weight_statistics(model, super_weights)
print(f"Total super weights: {stats['total_super_weights']}")

# Create protection mask
masks = create_protection_mask(model, super_weights, protection_bits=8)
```

### Quality Analysis

Analyze quantization quality.

```python
from llm_quantize.lib.analysis.quality import (
    compute_perplexity,
    test_coherence,
    generate_quality_report,
)

# Compute perplexity
perplexity = compute_perplexity(
    model=quantized_model,
    tokenizer=tokenizer,
    texts=["Test text 1", "Test text 2"],
)

# Test coherence
coherence_results = test_coherence(
    model=quantized_model,
    tokenizer=tokenizer,
    prompts=["The capital of France is"],
)

# Generate full report
report = generate_quality_report(
    model_name="llama-7b-Q4_K_M",
    quantization_format="gguf",
    quantization_level="Q4_K_M",
    perplexity_original=5.2,
    perplexity_quantized=5.8,
    coherence_results=coherence_results,
)

print(f"Quality Grade: {report.quality_grade}")
```

---

## Utilities

### Model Loader

Load models and create source model info.

```python
from llm_quantize.lib.model_loader import (
    create_source_model,
    load_model,
)

# Create source model info (without loading weights)
source_model = create_source_model("meta-llama/Llama-2-7b-hf")

# Load full model (loads weights into memory)
model, tokenizer = load_model(
    "meta-llama/Llama-2-7b-hf",
    hf_token="your_token",  # Optional
)
```

### Progress Reporter

Track quantization progress.

```python
from llm_quantize.lib.progress import ProgressReporter
from llm_quantize.models import Verbosity

reporter = ProgressReporter(
    verbosity=Verbosity.VERBOSE,
)

# Manual progress updates
reporter.start_task("Loading model", total=1)
reporter.update_task(1)
reporter.complete_task()

# Log messages
reporter.log_info("Processing layer 1...")
reporter.log_verbose("Detailed information...")
reporter.log_debug("Debug output...")
```

### Validation

Validate quantized outputs.

```python
from llm_quantize.lib.validation import (
    validate_output,
    get_file_size,
    ValidationResult,
)
from llm_quantize.models import OutputFormat

# Validate GGUF output
result = validate_output("./model.gguf", OutputFormat.GGUF)
if result.is_valid:
    print("Validation passed")
else:
    print(f"Validation failed: {result.error_message}")

# Get file/directory size
size = get_file_size(Path("./output"))
```

### Calibration Data

Load and validate calibration data.

```python
from llm_quantize.lib.calibration import (
    load_calibration_data,
    validate_calibration_data,
    get_default_calibration_samples,
)

# Load from file
samples = load_calibration_data(
    "./calibration.json",
    num_samples=256,
    max_length=512,
)

# Validate data quality
is_valid, warnings = validate_calibration_data(samples)
for warning in warnings:
    print(f"Warning: {warning}")

# Get default samples
default_samples = get_default_calibration_samples(count=100)
```

### Checkpointing

Manage quantization checkpoints.

```python
from llm_quantize.lib.checkpoint import Checkpoint
from pathlib import Path

# Create checkpoint
checkpoint = Checkpoint(Path("./checkpoint"), config)
checkpoint.initialize(num_layers=32, config=config)

# Save layer progress
checkpoint.save_layer(0, {"weights": layer_weights})

# Check if resumable
if Checkpoint.can_resume(Path("./checkpoint")):
    checkpoint, start_layer = Checkpoint.from_resume(
        Path("./checkpoint"),
        config,
    )
    print(f"Resuming from layer {start_layer}")

# Clean up on completion
checkpoint.cleanup()
```

---

## Enums

### OutputFormat

```python
from llm_quantize.models import OutputFormat

OutputFormat.GGUF  # GGUF format
OutputFormat.AWQ   # AWQ format
OutputFormat.GPTQ  # GPTQ format
```

### Verbosity

```python
from llm_quantize.models import Verbosity

Verbosity.QUIET    # Minimal output
Verbosity.NORMAL   # Standard output
Verbosity.VERBOSE  # Detailed output
Verbosity.DEBUG    # Debug output
```

### ValidationStatus

```python
from llm_quantize.models import ValidationStatus

ValidationStatus.PENDING   # Not yet validated
ValidationStatus.VALID     # Validation passed
ValidationStatus.INVALID   # Validation failed
ValidationStatus.SKIPPED   # Validation skipped
```

### QualityGrade

```python
from llm_quantize.models import QualityGrade

QualityGrade.EXCELLENT   # <5% quality loss
QualityGrade.GOOD        # 5-15% quality loss
QualityGrade.ACCEPTABLE  # 15-30% quality loss
QualityGrade.DEGRADED    # 30-50% quality loss
QualityGrade.FAILED      # >50% quality loss
```

---

## Constants

### Quantization Types

```python
from llm_quantize.models import (
    GGUF_QUANT_TYPES,
    AWQ_QUANT_TYPES,
    GPTQ_QUANT_TYPES,
)

# Available GGUF quantization types
print(list(GGUF_QUANT_TYPES.keys()))
# ['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_0', 'Q4_K_S', 'Q4_K_M', ...]

# Get info about a quantization type
q4_info = GGUF_QUANT_TYPES["Q4_K_M"]
print(q4_info)  # {'bits': 4, 'description': '4-bit k-quant medium'}
```
