"""Analyze command for importance analysis and quality metrics.

This module provides CLI commands for:
- Computing importance matrices
- Generating quality reports
- Identifying super weights
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from llm_quantize.lib.progress import ProgressReporter
from llm_quantize.models import ImportanceMethod, OutputMode, Verbosity

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_INVALID_ARGUMENTS = 2
EXIT_MODEL_NOT_FOUND = 3
EXIT_AUTH_REQUIRED = 4
EXIT_ANALYSIS_ERROR = 10


@click.group()
def analyze() -> None:
    """Analyze models for quantization optimization."""
    pass


@analyze.command("importance")
@click.argument("model", type=str)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path for importance matrix (default: <model>-imatrix.json)",
)
@click.option(
    "--calibration-data",
    type=click.Path(exists=True),
    help="Path to calibration data file",
)
@click.option(
    "--calibration-samples",
    type=int,
    default=256,
    help="Number of calibration samples",
)
@click.option(
    "--method",
    type=click.Choice(["activation", "gradient"]),
    default="activation",
    help="Importance computation method",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "imatrix"]),
    default="json",
    help="Output format",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output results as JSON",
)
def importance_command(
    model: str,
    output: Optional[str],
    calibration_data: Optional[str],
    calibration_samples: int,
    method: str,
    output_format: str,
    verbose: bool,
    json_output: bool,
) -> None:
    """Compute importance matrix for a model.

    MODEL: HuggingFace model ID or path to local model directory.

    The importance matrix identifies which layers and weights are most
    critical for model quality, enabling dynamic quantization and
    super weight protection.

    Examples:

        # Basic importance analysis
        llm-quantize analyze importance meta-llama/Llama-2-7b-hf

        # With custom calibration data
        llm-quantize analyze importance model -o imatrix.json --calibration-data data.json

        # Using gradient-based analysis (slower but more accurate)
        llm-quantize analyze importance model --method gradient
    """
    verbosity = Verbosity.VERBOSE if verbose else Verbosity.NORMAL
    reporter = ProgressReporter(verbosity)

    try:
        # Import dependencies
        from llm_quantize.lib.analysis.importance import compute_importance_matrix
        from llm_quantize.lib.calibration import load_calibration_data
        from llm_quantize.lib.model_loader import create_source_model, load_model

        reporter.log_info(f"Analyzing model: {model}")

        # Load source model info
        source_model = create_source_model(model)
        reporter.log_info(
            f"Architecture: {source_model.architecture} "
            f"({source_model.num_layers} layers, {source_model.parameter_count:,} params)"
        )

        # Load calibration data
        reporter.log_info("Loading calibration data...")
        calib_data = load_calibration_data(
            calibration_data, num_samples=calibration_samples
        )
        reporter.log_info(f"Loaded {len(calib_data)} calibration samples")

        # Load model
        reporter.log_info("Loading model (this may take a while)...")
        loaded_model = load_model(model)

        # Compute importance
        importance_method = (
            ImportanceMethod.GRADIENT_SENSITIVITY
            if method == "gradient"
            else ImportanceMethod.ACTIVATION_MAGNITUDE
        )

        reporter.log_info(f"Computing importance using {importance_method.value}...")

        def progress_callback(current: int, total: int) -> None:
            reporter.log_verbose(f"Processing sample {current}/{total}")

        imatrix = compute_importance_matrix(
            model=loaded_model,
            calibration_data=calib_data,
            method=importance_method,
            progress_callback=progress_callback,
        )

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            model_name = model.replace("/", "-")
            suffix = ".imatrix" if output_format == "imatrix" else ".json"
            output_path = Path(f"{model_name}-imatrix{suffix}")

        # Save results
        imatrix.save(output_path)
        reporter.log_info(f"Importance matrix saved to: {output_path}")

        # Output summary
        if json_output:
            result = {
                "status": "success",
                "output_path": str(output_path),
                "layers_analyzed": len(imatrix.layer_scores),
                "total_parameters": imatrix.total_parameters,
                "super_weights": imatrix.get_super_weight_count(),
            }
            click.echo(json.dumps(result, indent=2))
        else:
            reporter.log_info("\nImportance Analysis Summary:")
            reporter.log_info(f"  Layers analyzed: {len(imatrix.layer_scores)}")
            reporter.log_info(f"  Total parameters: {imatrix.total_parameters:,}")
            reporter.log_info(f"  Super weights identified: {imatrix.get_super_weight_count()}")

            # Show top layers by importance
            top_layers = imatrix.get_layer_ranking()[:5]
            if top_layers:
                reporter.log_info("\nTop 5 most important layers:")
                for layer in top_layers:
                    reporter.log_info(
                        f"  {layer.layer_name}: {layer.importance_score:.4f} "
                        f"(recommended: {layer.recommended_bits}-bit)"
                    )

        sys.exit(EXIT_SUCCESS)

    except FileNotFoundError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "message": str(e)}))
        else:
            reporter.log_error(f"Model not found: {e}")
        sys.exit(EXIT_MODEL_NOT_FOUND)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "message": str(e)}))
        else:
            reporter.log_error(f"Analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(EXIT_ANALYSIS_ERROR)


@analyze.command("quality")
@click.argument("model", type=str)
@click.option(
    "--original",
    type=str,
    help="Original model for comparison (if MODEL is quantized)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path for quality report",
)
@click.option(
    "--coherence-prompts",
    type=int,
    default=5,
    help="Number of coherence test prompts",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output results as JSON",
)
def quality_command(
    model: str,
    original: Optional[str],
    output: Optional[str],
    coherence_prompts: int,
    verbose: bool,
    json_output: bool,
) -> None:
    """Generate quality report for a model.

    MODEL: Path to quantized model or HuggingFace model ID.

    Computes perplexity, coherence tests, and generates a comprehensive
    quality assessment report.

    Examples:

        # Quality report for quantized model
        llm-quantize analyze quality ./model-Q4_K_M.gguf

        # Compare quantized vs original
        llm-quantize analyze quality ./quantized --original meta-llama/Llama-2-7b-hf
    """
    verbosity = Verbosity.VERBOSE if verbose else Verbosity.NORMAL
    reporter = ProgressReporter(verbosity)

    try:
        from llm_quantize.lib.analysis.quality import (
            compute_perplexity,
            generate_quality_report,
            test_coherence,
        )
        from llm_quantize.lib.calibration import load_calibration_data

        reporter.log_info(f"Analyzing quality: {model}")

        # This is a skeleton - full implementation would load models
        # and compute actual metrics

        reporter.log_info("Quality analysis requires model loading (not yet implemented)")
        reporter.log_info("Use --json-output for programmatic access")

        if json_output:
            result = {
                "status": "not_implemented",
                "message": "Full quality analysis not yet implemented",
            }
            click.echo(json.dumps(result, indent=2))

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "message": str(e)}))
        else:
            reporter.log_error(f"Quality analysis failed: {e}")
        sys.exit(EXIT_ANALYSIS_ERROR)


@analyze.command("profile")
@click.argument("model", type=str)
@click.option(
    "--imatrix",
    type=click.Path(exists=True),
    help="Pre-computed importance matrix",
)
@click.option(
    "--target-bits",
    type=float,
    default=4.0,
    help="Target average bits per weight",
)
@click.option(
    "--preset",
    type=click.Choice(["balanced", "attention-high", "compression-max", "quality-max"]),
    help="Use a preset profile instead of computing",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path for generated profile",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output results as JSON",
)
def profile_command(
    model: str,
    imatrix: Optional[str],
    target_bits: float,
    preset: Optional[str],
    output: Optional[str],
    verbose: bool,
    json_output: bool,
) -> None:
    """Generate or retrieve a quantization profile.

    MODEL: HuggingFace model ID or path to local model.

    Creates a dynamic quantization profile based on importance analysis
    or retrieves a preset profile.

    Examples:

        # Generate profile from importance matrix
        llm-quantize analyze profile model --imatrix imatrix.json --target-bits 3.5

        # Use preset profile
        llm-quantize analyze profile model --preset attention-high
    """
    verbosity = Verbosity.VERBOSE if verbose else Verbosity.NORMAL
    reporter = ProgressReporter(verbosity)

    try:
        from llm_quantize.lib.quantizers.advanced.profiles import (
            get_profile,
            save_profile,
            create_profile_from_importance,
        )
        from llm_quantize.models import ImportanceMatrix

        if preset:
            # Use preset profile
            profile = get_profile(preset)
            if profile is None:
                reporter.log_error(f"Unknown preset: {preset}")
                sys.exit(EXIT_INVALID_ARGUMENTS)

            reporter.log_info(f"Using preset profile: {preset}")
            reporter.log_info(f"  Description: {profile.description}")
            reporter.log_info(f"  Attention: {profile.attention_bits}-bit")
            reporter.log_info(f"  MLP: {profile.mlp_bits}-bit")
            reporter.log_info(f"  Embedding: {profile.embedding_bits}-bit")

        elif imatrix:
            # Generate from importance matrix
            reporter.log_info(f"Loading importance matrix: {imatrix}")
            im = ImportanceMatrix.load(Path(imatrix))

            layer_names = [ls.layer_name for ls in im.layer_scores]
            importance_scores = {ls.layer_name: ls.importance_score for ls in im.layer_scores}

            profile = create_profile_from_importance(
                model_architecture=im.model_name,
                layer_names=layer_names,
                importance_scores=importance_scores,
                target_avg_bits=target_bits,
            )

            reporter.log_info(f"Generated profile: {profile.profile_name}")

        else:
            reporter.log_error("Must specify either --preset or --imatrix")
            sys.exit(EXIT_INVALID_ARGUMENTS)

        # Save if output specified
        if output:
            save_profile(profile, output)
            reporter.log_info(f"Profile saved to: {output}")

        if json_output:
            click.echo(json.dumps(profile.to_dict(), indent=2))

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "message": str(e)}))
        else:
            reporter.log_error(f"Profile generation failed: {e}")
        sys.exit(EXIT_ANALYSIS_ERROR)
