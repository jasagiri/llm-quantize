"""Main CLI entry point for llm-quantize."""

import json
import sys

import click
from rich.console import Console

from llm_quantize import __version__
from llm_quantize.models import OutputMode, Verbosity

# Exit codes per CLI contract
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_INVALID_ARGUMENTS = 2
EXIT_MODEL_NOT_FOUND = 3
EXIT_AUTH_REQUIRED = 4
EXIT_OUT_OF_MEMORY = 5
EXIT_VALIDATION_FAILED = 6
EXIT_CHECKPOINT_ERROR = 7


class CliContext:
    """Shared CLI context for all commands."""

    def __init__(
        self,
        output_format: OutputMode = OutputMode.HUMAN,
        verbosity: Verbosity = Verbosity.NORMAL,
    ) -> None:
        """Initialize CLI context.

        Args:
            output_format: Output format mode (human/json)
            verbosity: Logging verbosity level
        """
        self.output_format = output_format
        self.verbosity = verbosity
        self.console = Console(stderr=True)
        self.stdout_console = Console()

    def output(self, data: dict) -> None:
        """Output data in the configured format.

        Args:
            data: Data to output
        """
        if self.output_format == OutputMode.JSON:
            self.stdout_console.print_json(json.dumps(data))
        else:
            # Human-readable format handled by callers
            pass

    def error(self, message: str, code: int = EXIT_GENERAL_ERROR) -> None:
        """Output an error message.

        Args:
            message: Error message
            code: Exit code
        """
        if self.output_format == OutputMode.JSON:
            error_data = {
                "status": "error",
                "error_code": code,
                "message": message,
            }
            self.stdout_console.print_json(json.dumps(error_data))
        else:
            self.console.print(f"[red]Error: {message}[/red]")


pass_context = click.make_pass_decorator(CliContext, ensure=True)


def verbosity_callback(
    ctx: click.Context, param: click.Parameter, value: str
) -> Verbosity:
    """Convert verbosity string to enum."""
    try:
        return Verbosity(value)
    except ValueError:
        raise click.BadParameter(f"Invalid verbosity: {value}")


def format_callback(
    ctx: click.Context, param: click.Parameter, value: str
) -> OutputMode:
    """Convert format string to enum."""
    try:
        return OutputMode(value)
    except ValueError:
        raise click.BadParameter(f"Invalid format: {value}")


@click.group()
@click.version_option(version=__version__, prog_name="llm-quantize")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["human", "json"]),
    default="human",
    callback=format_callback,
    expose_value=True,
    is_eager=True,
    help="Output format: human (default), json",
)
@click.option(
    "--verbosity",
    type=click.Choice(["quiet", "normal", "verbose", "debug"]),
    default="normal",
    callback=verbosity_callback,
    expose_value=True,
    is_eager=True,
    help="Log level: quiet, normal (default), verbose, debug",
)
@click.pass_context
def cli(
    ctx: click.Context,
    output_format: OutputMode,
    verbosity: Verbosity,
) -> None:
    """LLM-Quantize: A unified CLI tool for LLM model quantization.

    Supports GGUF, AWQ, and GPTQ output formats for efficient LLM inference.

    Examples:

        # Quantize to GGUF Q4_K_M (recommended)
        llm-quantize quantize meta-llama/Llama-2-7b-hf gguf -q Q4_K_M

        # Quantize to AWQ 4-bit
        llm-quantize quantize ./my-model awq -q 4bit

        # Get model information
        llm-quantize info meta-llama/Llama-2-7b-hf
    """
    ctx.ensure_object(CliContext)
    ctx.obj = CliContext(output_format=output_format, verbosity=verbosity)


@cli.command()
@click.argument("model")
@pass_context
def info(ctx: CliContext, model: str) -> None:
    """Display model information.

    MODEL is a HuggingFace Hub identifier or local directory path.
    """
    from llm_quantize.lib.model_loader import create_source_model

    try:
        source_model = create_source_model(model)

        if ctx.output_format == OutputMode.JSON:
            data = {
                "model_name": source_model.model_path,
                "architecture": source_model.architecture,
                "parameter_count": source_model.parameter_count,
                "hidden_size": source_model.hidden_size,
                "num_layers": source_model.num_layers,
                "num_heads": source_model.num_heads,
                "vocab_size": source_model.vocab_size,
                "torch_dtype": source_model.dtype,
            }
            ctx.output(data)
        else:
            ctx.stdout_console.print(f"Model: {source_model.model_path}")
            ctx.stdout_console.print(f"Architecture: {source_model.architecture}")
            ctx.stdout_console.print(f"Parameters: {source_model.parameter_count:,}")
            ctx.stdout_console.print(f"Hidden Size: {source_model.hidden_size}")
            ctx.stdout_console.print(f"Layers: {source_model.num_layers}")
            ctx.stdout_console.print(f"Heads: {source_model.num_heads}")
            ctx.stdout_console.print(f"Vocab Size: {source_model.vocab_size}")
            ctx.stdout_console.print(f"Dtype: {source_model.dtype}")

    except ValueError as e:
        ctx.error(str(e), EXIT_MODEL_NOT_FOUND)
        sys.exit(EXIT_MODEL_NOT_FOUND)
    except Exception as e:
        ctx.error(f"Failed to load model info: {e}", EXIT_GENERAL_ERROR)
        sys.exit(EXIT_GENERAL_ERROR)


# Import and register subcommands
def register_commands() -> None:
    """Register all subcommands."""
    try:
        from llm_quantize.cli.quantize import quantize

        cli.add_command(quantize)
    except ImportError:
        pass

    try:
        from llm_quantize.cli.convert import convert

        cli.add_command(convert)
    except ImportError:
        pass

    try:
        from llm_quantize.cli.analyze import analyze

        cli.add_command(analyze)
    except ImportError:
        pass


register_commands()


if __name__ == "__main__":
    cli()
