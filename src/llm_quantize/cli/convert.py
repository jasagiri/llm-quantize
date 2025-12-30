"""Convert command for llm-quantize CLI."""

import json
import sys
from pathlib import Path

import click

from llm_quantize.cli.main import (
    EXIT_GENERAL_ERROR,
    EXIT_INVALID_ARGUMENTS,
    EXIT_MODEL_NOT_FOUND,
    EXIT_SUCCESS,
    CliContext,
    pass_context,
)
from llm_quantize.lib.converter import (
    convert_format,
    detect_format,
    is_conversion_supported,
    is_lossy_conversion,
)
from llm_quantize.models import OutputMode


@click.command()
@click.argument("source")
@click.argument("target_format", type=click.Choice(["gguf", "awq", "gptq"], case_sensitive=False))
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
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing output",
)
@pass_context
def convert(
    ctx: CliContext,
    source: str,
    target_format: str,
    output_dir: str,
    output_name: str | None,
    force: bool,
) -> None:
    """Convert a quantized model to a different format.

    SOURCE is the path to the quantized model file or directory.
    TARGET_FORMAT is the target format: gguf, awq, or gptq.
    """
    source_path = Path(source)

    # Check source exists
    if not source_path.exists():
        error_msg = f"Source not found: {source}"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_MODEL_NOT_FOUND,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_MODEL_NOT_FOUND)
        sys.exit(EXIT_MODEL_NOT_FOUND)

    # Detect source format
    source_format = detect_format(source)
    if source_format is None:
        error_msg = f"Could not detect format of source: {source}"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_INVALID_ARGUMENTS,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_INVALID_ARGUMENTS)
        sys.exit(EXIT_INVALID_ARGUMENTS)

    # Check conversion is supported
    if not is_conversion_supported(source_format, target_format.lower()):
        error_msg = f"Conversion from {source_format} to {target_format} is not supported"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_INVALID_ARGUMENTS,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_INVALID_ARGUMENTS)
        sys.exit(EXIT_INVALID_ARGUMENTS)

    # Check for lossy conversion and warn
    if is_lossy_conversion(source_format, target_format.lower()):
        warning_msg = (
            f"Warning: Converting from {source_format} to {target_format} "
            "may result in quality degradation."
        )
        if ctx.output_format != OutputMode.JSON:
            ctx.console.print(f"[yellow]{warning_msg}[/yellow]")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Perform conversion
    try:
        result = convert_format(
            source_path=source,
            target_format=target_format.lower(),
            output_dir=output_dir,
            output_name=output_name,
            force=force,
        )
    except ValueError as e:
        error_msg = str(e)
        code = EXIT_INVALID_ARGUMENTS
        if "not found" in error_msg.lower():
            code = EXIT_MODEL_NOT_FOUND

        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": code,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, code)
        sys.exit(code)
    except Exception as e:
        error_msg = f"Conversion failed: {e}"
        if ctx.output_format == OutputMode.JSON:
            ctx.stdout_console.print_json(json.dumps({
                "status": "error",
                "error_code": EXIT_GENERAL_ERROR,
                "message": error_msg,
            }))
        else:
            ctx.error(error_msg, EXIT_GENERAL_ERROR)
        sys.exit(EXIT_GENERAL_ERROR)

    # Output result
    if ctx.output_format == OutputMode.JSON:
        output_data = {
            "status": "success",
            "output_path": result.output_path,
            "source_format": result.source_format,
            "target_format": result.target_format,
            "file_size": result.file_size,
            "is_lossy": result.is_lossy,
        }
        if result.warning_message:
            output_data["warning"] = result.warning_message
        ctx.stdout_console.print_json(json.dumps(output_data))
    else:
        ctx.console.print("[green]Conversion complete![/green]")
        ctx.console.print(f"  Source: {source} ({result.source_format})")
        ctx.console.print(f"  Output: {result.output_path} ({result.target_format})")
        ctx.console.print(f"  Size: {result.file_size:,} bytes")

        if result.is_lossy and result.warning_message:
            ctx.console.print(f"[yellow]{result.warning_message}[/yellow]")

    sys.exit(EXIT_SUCCESS)
