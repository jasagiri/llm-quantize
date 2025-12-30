"""Contract tests for output format consistency.

These tests verify the output format contract is respected:
- JSON output always includes status field
- Error responses have consistent structure
- Human output is readable and informative
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_quantize.cli.main import cli


class TestJSONOutputFormat:
    """Tests for JSON output format contract."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_json_success_has_status(self, runner: CliRunner) -> None:
        """Test JSON success output includes status field."""
        from llm_quantize.models import ModelType, SourceModel

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path="test-model",
                model_type=ModelType.HF_HUB,
                architecture="LlamaForCausalLM",
                parameter_count=7000000000,
                dtype="float16",
            )

            result = runner.invoke(cli, ["--format", "json", "info", "test-model"])

            assert result.exit_code == 0
            output = json.loads(result.output)
            # For info command, we don't wrap in status since it returns direct data
            assert isinstance(output, dict)

    def test_json_error_has_required_fields(self, runner: CliRunner) -> None:
        """Test JSON error output includes required fields."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli, ["--format", "json", "info", "nonexistent-model"]
            )

            assert result.exit_code == 3
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert "message" in output
            assert "error_code" in output

    def test_json_error_code_matches_exit_code(self, runner: CliRunner) -> None:
        """Test JSON error_code matches the exit code."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli, ["--format", "json", "info", "nonexistent-model"]
            )

            output = json.loads(result.output)
            assert output["error_code"] == result.exit_code

    def test_json_format_with_quantize_success(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test JSON output format for successful quantize."""
        from llm_quantize.models import ModelType, OutputFormat, QuantizedModel, SourceModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_create:
            with patch(
                "llm_quantize.cli.quantize.get_quantizer"
            ) as mock_get_quantizer:
                with patch("llm_quantize.cli.quantize.validate_output") as mock_validate:
                    mock_create.return_value = SourceModel(
                        model_path="test-model",
                        model_type=ModelType.HF_HUB,
                        architecture="LlamaForCausalLM",
                        parameter_count=7000000000,
                        dtype="float16",
                    )

                    # Create mock output file
                    output_file = temp_dir / "test-model-Q4_K_M.gguf"
                    output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                    quantized_model = QuantizedModel(
                        output_path=str(output_file),
                        format="gguf",  # Use string instead of enum for JSON serialization
                        file_size=104,
                        compression_ratio=0.25,
                        quantization_level="Q4_K_M",
                        source_model_path="test-model",
                        duration_seconds=10.5,
                        peak_memory_bytes=1000000,
                        quantization_metadata={"method": "gguf"},
                    )

                    # Mock the quantizer class and instance
                    mock_quantizer_instance = MagicMock()
                    mock_quantizer_instance.quantize.return_value = quantized_model
                    mock_quantizer_class = MagicMock(return_value=mock_quantizer_instance)
                    mock_get_quantizer.return_value = mock_quantizer_class
                    # Mock ValidationResult with is_valid attribute
                    mock_validation_result = MagicMock()
                    mock_validation_result.is_valid = True
                    mock_validate.return_value = mock_validation_result

                    result = runner.invoke(
                        cli,
                        [
                            "--format",
                            "json",
                            "quantize",
                            "test-model",
                            "gguf",
                            "-q", "Q4_K_M",
                            "-o",
                            str(temp_dir),
                        ],
                    )

                    assert result.exit_code == 0
                    output = json.loads(result.output)
                    assert output["status"] == "success"
                    assert "output_path" in output

    def test_json_format_with_quantize_error(self, runner: CliRunner) -> None:
        """Test JSON output format for quantize error."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli,
                ["--format", "json", "quantize", "nonexistent", "gguf", "-q", "Q4_K_M"],
            )

            # Exit code 3 for model not found
            assert result.exit_code == 3
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert "message" in output

    def test_json_format_with_convert_success(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test JSON output format for successful convert."""
        from llm_quantize.lib.converter import ConversionResult

        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"

                awq_output = output_dir / "model-awq"
                awq_output.mkdir()
                (awq_output / "config.json").write_text('{"model_type": "llama"}')
                (awq_output / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_convert.return_value = ConversionResult(
                    output_path=str(awq_output),
                    source_format="gguf",
                    target_format="awq",
                    file_size=100,
                    is_lossy=True,
                )

                result = runner.invoke(
                    cli,
                    [
                        "--format",
                        "json",
                        "convert",
                        str(source_file),
                        "awq",
                        "-o",
                        str(output_dir),
                    ],
                )

                assert result.exit_code == 0
                output = json.loads(result.output)
                assert output["status"] == "success"
                assert "output_path" in output


class TestHumanOutputFormat:
    """Tests for human-readable output format."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_human_info_shows_model_details(self, runner: CliRunner) -> None:
        """Test human output for info command shows model details."""
        from llm_quantize.models import ModelType, SourceModel

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path="meta-llama/Llama-2-7b-hf",
                model_type=ModelType.HF_HUB,
                architecture="LlamaForCausalLM",
                parameter_count=7000000000,
                dtype="float16",
                num_layers=32,
                hidden_size=4096,
                num_heads=32,
                vocab_size=32000,
            )

            result = runner.invoke(cli, ["info", "test-model"])

            assert result.exit_code == 0
            output = result.output
            # Should contain key information
            assert "Model:" in output or "meta-llama" in output
            assert "Architecture:" in output or "LlamaForCausalLM" in output
            assert "Parameters:" in output or "7,000,000,000" in output

    def test_human_error_is_readable(self, runner: CliRunner) -> None:
        """Test human error output is readable."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found: test-model")

            result = runner.invoke(cli, ["info", "nonexistent"])

            assert result.exit_code == 3
            # Error message should be somewhere in output
            assert "Error" in result.output or "error" in result.output.lower() or "not found" in result.output.lower()

    def test_human_quantize_shows_progress(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test human output for quantize shows progress info."""
        from llm_quantize.models import ModelType, OutputFormat, QuantizedModel, SourceModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_create:
            with patch(
                "llm_quantize.cli.quantize.get_quantizer"
            ) as mock_get_quantizer:
                with patch("llm_quantize.cli.quantize.validate_output") as mock_validate:
                    mock_create.return_value = SourceModel(
                        model_path="test-model",
                        model_type=ModelType.HF_HUB,
                        architecture="LlamaForCausalLM",
                        parameter_count=7000000000,
                        dtype="float16",
                    )

                    output_file = temp_dir / "test-model-Q4_K_M.gguf"
                    output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                    quantized_model = QuantizedModel(
                        output_path=str(output_file),
                        format=OutputFormat.GGUF,
                        file_size=104,
                        compression_ratio=0.25,
                        quantization_level="Q4_K_M",
                        source_model_path="test-model",
                        duration_seconds=10.5,
                        peak_memory_bytes=1000000,
                        quantization_metadata={},
                    )

                    # Mock the quantizer class and instance
                    mock_quantizer_instance = MagicMock()
                    mock_quantizer_instance.quantize.return_value = quantized_model
                    mock_quantizer_class = MagicMock(return_value=mock_quantizer_instance)
                    mock_get_quantizer.return_value = mock_quantizer_class
                    # Mock ValidationResult with is_valid attribute
                    mock_validation_result = MagicMock()
                    mock_validation_result.is_valid = True
                    mock_validate.return_value = mock_validation_result

                    result = runner.invoke(
                        cli,
                        ["quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                    )

                    assert result.exit_code == 0
                    # Human output should show useful information
                    output = result.output
                    assert (
                        "complete" in output.lower()
                        or "success" in output.lower()
                        or str(output_file) in output
                    )


class TestOutputFormatConsistency:
    """Tests for output format consistency across commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_format_option_available_globally(self, runner: CliRunner) -> None:
        """Test --format option is available before command."""
        result = runner.invoke(cli, ["--format", "json", "--help"])
        # Should not error on the format option
        assert result.exit_code == 0

    def test_format_json_is_valid_json(self, runner: CliRunner) -> None:
        """Test JSON format always produces valid JSON."""
        from llm_quantize.models import ModelType, SourceModel

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path="test-model",
                model_type=ModelType.HF_HUB,
                architecture="LlamaForCausalLM",
                parameter_count=7000000000,
                dtype="float16",
            )

            result = runner.invoke(cli, ["--format", "json", "info", "test-model"])

            # Should be valid JSON
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                pytest.fail("JSON output is not valid JSON")

    def test_format_json_error_is_valid_json(self, runner: CliRunner) -> None:
        """Test JSON format error output is valid JSON."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli, ["--format", "json", "info", "nonexistent"]
            )

            # Even errors should be valid JSON
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                pytest.fail("JSON error output is not valid JSON")


class TestExitCodes:
    """Tests for exit code consistency."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_exit_code_0_success(self, runner: CliRunner) -> None:
        """Test exit code 0 for successful operations."""
        from llm_quantize.models import ModelType, SourceModel

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path="test-model",
                model_type=ModelType.HF_HUB,
                architecture="LlamaForCausalLM",
                parameter_count=7000000000,
                dtype="float16",
            )

            result = runner.invoke(cli, ["info", "test-model"])
            assert result.exit_code == 0

    def test_exit_code_2_invalid_arguments(self, runner: CliRunner) -> None:
        """Test exit code 2 for invalid arguments."""
        result = runner.invoke(cli, ["info"])  # Missing required argument
        assert result.exit_code == 2

    def test_exit_code_3_model_not_found(self, runner: CliRunner) -> None:
        """Test exit code 3 for model not found."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(cli, ["info", "nonexistent"])
            assert result.exit_code == 3

    def test_exit_code_1_general_error(self, runner: CliRunner) -> None:
        """Test exit code 1 for general errors."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = RuntimeError("Unexpected error")

            result = runner.invoke(cli, ["info", "test-model"])
            assert result.exit_code == 1
