"""Quantize command for llm-quantize CLI."""

import json
import sys
from pathlib import Path

import click

from llm_quantize.cli.main import (
    EXIT_CHECKPOINT_ERROR,
    EXIT_GENERAL_ERROR,
    EXIT_INVALID_ARGUMENTS,
    EXIT_MODEL_NOT_FOUND,
    EXIT_OUT_OF_MEMORY,
    EXIT_SUCCESS,
    EXIT_VALIDATION_FAILED,
    CliContext,
    pass_context,
)
from llm_quantize.lib.model_loader import create_source_model
from llm_quantize.lib.progress import ProgressReporter
from llm_quantize.lib.quantizers import GGUFQuantizer, get_quantizer  # noqa: F401
from llm_quantize.lib.validation import validate_output
from llm_quantize.models import (
    AWQ_QUANT_TYPES,
    GGUF_QUANT_TYPES,
    GPTQ_QUANT_TYPES,
    OutputFormat,
    OutputMode,
    QuantizationConfig,
    ValidationStatus,
)


def validate_quant_level(format_str: str, level: str) -> bool:
    """Validate quantization level for the given format.

    Args:
        format_str: Output format string
        level: Quantization level string

    Returns:
        True if valid, False otherwise
    """
    format_levels = {
        "gguf": GGUF_QUANT_TYPES,
        "awq": AWQ_QUANT_TYPES,
        "gptq": GPTQ_QUANT_TYPES,
    }
    valid_levels = format_levels.get(format_str.lower(), {})
    return level in valid_levels


@click.command()
@click.argument("model")
@click.argument("output_format", type=click.Choice(["gguf", "awq", "gptq"], case_sensitive=False))
@click.option(
    "-q", "--quant-level",
    required=True,
    help="Quantization level (e.g., Q4_K_M for GGUF, 4bit for AWQ/GPTQ)",
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=".",
    help="Output directory",
)
@click.option(
    "-n", "--output-name",
    help="Output filename (auto-generated if not set)",
)
@click.option(
    "--calibration-data",
    type=click.Path(exists=True),
    help="Custom calibration dataset (for AWQ/GPTQ)",
)
@click.option(
    "--calibration-samples",
    type=int,
    default=256,
    help="Number of calibration samples",
)
@click.option(
    "--group-size",
    type=int,
    default=128,
    help="GPTQ group size",
)
@click.option(
    "--no-checkpoints",
    is_flag=True,
    default=False,
    help="Disable layer-level checkpoints",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=False),
    help="Checkpoint directory",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    help="Resume from checkpoint",
)
@pass_context
def quantize(
    ctx: CliContext,
    model: str,
    output_format: str,
    quant_level: str,
    output_dir: str,
    output_name: str | None,
    calibration_data: str | None,
    calibration_samples: int,
    group_size: int,
    no_checkpoints: bool,
    checkpoint_dir: str | None,
    resume: str | None,
) -> None:
    """Quantize a model to a specified format.

    MODEL is a HuggingFace Hub identifier or local directory path.
    OUTPUT_FORMAT is the target format: gguf, awq, or gptq.
    """
    # Validate quantization level
    if not validate_quant_level(output_format, quant_level):
        error_msg = f"Invalid quantization level '{quant_level}' for {output_format.upper()}"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_INVALID_ARGUMENTS,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_INVALID_ARGUMENTS)
        sys.exit(EXIT_INVALID_ARGUMENTS)

    # Load source model
    try:
        source_model = create_source_model(model)
    except ValueError as e:
        error_msg = str(e)
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_MODEL_NOT_FOUND,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_MODEL_NOT_FOUND)
        sys.exit(EXIT_MODEL_NOT_FOUND)

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create quantization config
    try:
        target_format = OutputFormat(output_format.upper())
    except ValueError:
        target_format = OutputFormat.GGUF  # Default

    config = QuantizationConfig(
        target_format=target_format,
        quantization_level=quant_level,
        output_dir=str(output_path),
        output_name=output_name,
        calibration_data_path=calibration_data,
        calibration_samples=calibration_samples,
        group_size=group_size,
        enable_checkpoints=not no_checkpoints,
        checkpoint_dir=checkpoint_dir,
    )

    # Create progress reporter
    progress_reporter = ProgressReporter(
        verbosity=ctx.verbosity,
        console=ctx.console,
    )

    # Get appropriate quantizer
    quantizer_class = get_quantizer(target_format)
    if quantizer_class is None:
        error_msg = f"No quantizer available for format: {output_format}"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_INVALID_ARGUMENTS,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_INVALID_ARGUMENTS)
        sys.exit(EXIT_INVALID_ARGUMENTS)

    # Create quantizer
    try:
        quantizer = quantizer_class(
            source_model=source_model,
            config=config,
            progress_reporter=progress_reporter,
            enable_checkpoints=not no_checkpoints,
            resume_from=resume,
        )
    except Exception as e:
        if "checkpoint" in str(e).lower():
            error_msg = f"Checkpoint error: {e}"
            code = EXIT_CHECKPOINT_ERROR
        else:
            error_msg = f"Failed to initialize quantizer: {e}"
            code = EXIT_GENERAL_ERROR

        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": code,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, code)
        sys.exit(code)

    # Run quantization
    try:
        result = quantizer.quantize()
    except MemoryError:
        error_msg = "Out of memory during quantization"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_OUT_OF_MEMORY,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_OUT_OF_MEMORY)
        sys.exit(EXIT_OUT_OF_MEMORY)
    except Exception as e:
        error_msg = f"Quantization failed: {e}"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_GENERAL_ERROR,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_GENERAL_ERROR)
        sys.exit(EXIT_GENERAL_ERROR)

    # Validate output
    validation_result = validate_output(result.output_path, output_format)
    result.validation_status = ValidationStatus.VALID if validation_result.is_valid else ValidationStatus.INVALID

    if not validation_result.is_valid:
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_VALIDATION_FAILED,
                "message": f"Output validation failed: {validation_result.error_message}",
                "output_path": result.output_path,
            }))
        else:
            ctx.error(f"Validation failed: {validation_result.error_message}", EXIT_VALIDATION_FAILED)
        sys.exit(EXIT_VALIDATION_FAILED)

    # Output result
    if ctx.output_format == OutputMode.JSON:
        output_data = {
            "status": "success",
            "output_path": result.output_path,
            "format": result.format,
            "quantization_level": result.quantization_level,
            "file_size": result.file_size,
            "compression_ratio": result.compression_ratio,
            "duration_seconds": result.duration_seconds,
            "peak_memory_bytes": result.peak_memory_bytes,
            "validation_status": result.validation_status.value,
            "metadata": result.quantization_metadata,
        }
        ctx.stdout_console.print_json(json.dumps(output_data))
    else:
        # Human-readable output
        progress_reporter.report_completion(
            output_path=result.output_path,
            file_size=result.file_size,
            duration=result.duration_seconds,
            compression_ratio=result.compression_ratio,
        )

    sys.exit(EXIT_SUCCESS)
