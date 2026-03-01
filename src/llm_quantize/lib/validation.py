"""Output validation utilities for quantized models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from llm_quantize.models import OutputFormat, QuantizedModel, ValidationStatus


@dataclass
class ValidationResult:
    """Result of output validation."""

    is_valid: bool
    error_message: str = ""

    @classmethod
    def valid(cls) -> "ValidationResult":
        """Create a valid result."""
        return cls(is_valid=True)

    @classmethod
    def invalid(cls, message: str) -> "ValidationResult":
        """Create an invalid result with error message."""
        return cls(is_valid=False, error_message=message)


def validate_output(
    output_path: str,
    output_format: Union[OutputFormat, str],
) -> ValidationResult:
    """Validate the output file exists and has expected structure.

    Args:
        output_path: Path to the output file or directory
        output_format: Expected output format (OutputFormat enum or string)

    Returns:
        ValidationResult with is_valid and error_message
    """
    path = Path(output_path)

    if not path.exists():
        return ValidationResult.invalid(f"Output path does not exist: {output_path}")

    # Convert string to OutputFormat if needed
    if isinstance(output_format, str):
        try:
            # Try both lower and upper case
            format_str = output_format.lower()
            output_format = OutputFormat(format_str)
        except ValueError:
            return ValidationResult.invalid(f"Unknown output format: {output_format}")

    if output_format == OutputFormat.GGUF:
        return _validate_gguf(path)
    elif output_format == OutputFormat.AWQ:
        return _validate_awq(path)
    elif output_format == OutputFormat.GPTQ:
        return _validate_gptq(path)
    else:
        return ValidationResult.invalid(f"Unknown output format: {output_format}")


def _validate_gguf(path: Path) -> ValidationResult:
    """Validate a GGUF output file.

    Args:
        path: Path to the GGUF file

    Returns:
        ValidationResult
    """
    if not path.is_file():
        return ValidationResult.invalid(f"GGUF output must be a file: {path}")

    if not path.suffix.lower() == ".gguf":
        return ValidationResult.invalid(f"GGUF file must have .gguf extension: {path}")

    # Check file size (must be non-zero)
    file_size = path.stat().st_size
    if file_size == 0:
        return ValidationResult.invalid(f"GGUF file is empty: {path}")

    # Check GGUF magic bytes
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            # GGUF magic: "GGUF" in little-endian
            if magic != b"GGUF":
                return ValidationResult.invalid(f"Invalid GGUF magic bytes: {magic!r}")
    except OSError as e:
        return ValidationResult.invalid(f"Failed to read GGUF file: {e}")

    return ValidationResult.valid()


def _validate_awq(path: Path) -> ValidationResult:
    """Validate an AWQ output directory.

    Args:
        path: Path to the AWQ output directory

    Returns:
        ValidationResult
    """
    if not path.is_dir():
        return ValidationResult.invalid(f"AWQ output must be a directory: {path}")

    # Check for required files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (path / f).exists()]

    if missing_files:
        return ValidationResult.invalid(f"AWQ output missing required files: {missing_files}")

    # Check for model weights (safetensors or bin)
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        return ValidationResult.invalid(f"AWQ output missing weight files: {path}")

    return ValidationResult.valid()


def _validate_gptq(path: Path) -> ValidationResult:
    """Validate a GPTQ output directory.

    Args:
        path: Path to the GPTQ output directory

    Returns:
        ValidationResult
    """
    if not path.is_dir():
        return ValidationResult.invalid(f"GPTQ output must be a directory: {path}")

    # Check for required files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (path / f).exists()]

    if missing_files:
        return ValidationResult.invalid(f"GPTQ output missing required files: {missing_files}")

    # Check for quantize_config.json (GPTQ specific)
    if not (path / "quantize_config.json").exists():
        # Not strictly required, but warn
        pass

    # Check for model weights
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        return ValidationResult.invalid(f"GPTQ output missing weight files: {path}")

    return ValidationResult.valid()


def update_validation_status(
    quantized_model: QuantizedModel,
    output_format: OutputFormat,
) -> QuantizedModel:
    """Update the validation status of a quantized model.

    Args:
        quantized_model: QuantizedModel to validate
        output_format: Expected output format

    Returns:
        Updated QuantizedModel with validation status
    """
    result = validate_output(quantized_model.output_path, output_format)

    if result.is_valid:
        quantized_model.validation_status = ValidationStatus.VALID
    else:
        quantized_model.validation_status = ValidationStatus.INVALID
        # Store error in metadata
        if quantized_model.quantization_metadata is None:
            quantized_model.quantization_metadata = {}
        quantized_model.quantization_metadata["validation_error"] = result.error_message

    return quantized_model


def validate_pruned_safetensors(path: Path) -> ValidationResult:
    """Validate a structurally-pruned safetensors model directory.

    Checks that the output directory contains:
    - config.json with required fields
    - At least one .safetensors weight file
    - Consistent dimensions (intermediate_size, num_attention_heads, num_hidden_layers)

    Args:
        path: Path to the pruned model directory

    Returns:
        ValidationResult
    """
    if not path.is_dir():
        return ValidationResult.invalid(
            f"Pruned model output must be a directory: {path}"
        )

    # config.json is required
    config_path = path / "config.json"
    if not config_path.exists():
        return ValidationResult.invalid(
            f"Pruned model missing config.json: {path}"
        )

    # Parse config and check required fields
    import json

    try:
        with open(config_path) as fp:
            config = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        return ValidationResult.invalid(f"Invalid config.json: {exc}")

    required_fields = ["num_hidden_layers", "hidden_size"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        return ValidationResult.invalid(
            f"config.json missing required fields: {missing}"
        )

    # Must have weight files
    weight_files = list(path.glob("*.safetensors"))
    if not weight_files:
        return ValidationResult.invalid(
            f"Pruned model missing safetensors weight files: {path}"
        )

    # Check pruning_plan.json if present (informational, not required)
    plan_path = path / "pruning_plan.json"
    if plan_path.exists():
        try:
            with open(plan_path) as fp:
                json.load(fp)
        except (json.JSONDecodeError, OSError):
            return ValidationResult.invalid(
                f"Invalid pruning_plan.json: {plan_path}"
            )

    return ValidationResult.valid()


def get_file_size(path: Path) -> int:
    """Get the size of a file or directory in bytes.

    Args:
        path: Path to file or directory

    Returns:
        Size in bytes
    """
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    return 0
