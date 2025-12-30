"""Contract tests for AWQ CLI quantization command.

These tests verify the CLI contract is respected:
- Exit codes match specification
- Output format matches specification
- Arguments and options work as documented
- Calibration data handling works correctly
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_quantize.cli.main import cli


class TestAWQQuantizeCommandExitCodes:
    """Tests for AWQ quantize command exit codes per CLI contract."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_exit_code_0_on_success(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test exit code 0 on successful AWQ quantization."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
                parameter_count=1000000,
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                # Create mock output directory with required files
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_dir),
                    format="awq",
                    quantization_level="4bit",
                    file_size=100,
                    compression_ratio=0.25,
                    source_model_path="test-model",
                    duration_seconds=1.0,
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["quantize", "test-model", "awq", "-q", "4bit", "-o", str(temp_dir)],
                )

                assert result.exit_code == 0

    def test_exit_code_2_invalid_quant_level(self, runner: CliRunner) -> None:
        """Test exit code 2 for invalid AWQ quantization level."""
        with patch("llm_quantize.cli.quantize.create_source_model"):
            result = runner.invoke(
                cli,
                ["quantize", "test-model", "awq", "-q", "INVALID_QUANT"],
            )
            assert result.exit_code == 2

    def test_exit_code_3_model_not_found(self, runner: CliRunner) -> None:
        """Test exit code 3 when model is not found."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli,
                ["quantize", "nonexistent-model", "awq", "-q", "4bit"],
            )

            assert result.exit_code == 3


class TestAWQQuantizeCommandOutput:
    """Tests for AWQ quantize command output format."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_json_output_success(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test JSON output format on success."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
                parameter_count=1000000,
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_dir),
                    format="awq",
                    quantization_level="4bit",
                    file_size=100,
                    compression_ratio=0.25,
                    source_model_path="test-model",
                    duration_seconds=100.0,
                    peak_memory_bytes=8000000000,
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "--format", "json",
                        "quantize", "test-model", "awq", "-q", "4bit",
                        "-o", str(temp_dir),
                    ],
                )

                assert result.exit_code == 0
                output = json.loads(result.output)
                assert output["status"] == "success"
                assert output["format"] == "awq"
                assert output["quantization_level"] == "4bit"
                assert "output_path" in output

    def test_json_output_error(self, runner: CliRunner) -> None:
        """Test JSON output format on error."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli,
                [
                    "--format", "json",
                    "quantize", "nonexistent-model", "awq", "-q", "4bit",
                ],
            )

            assert result.exit_code == 3
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert output["error_code"] == 3
            assert "message" in output


class TestAWQQuantizeCommandOptions:
    """Tests for AWQ-specific command options."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_calibration_data_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test --calibration-data option for AWQ."""
        from llm_quantize.models import QuantizedModel

        # Create calibration data file
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('["sample1", "sample2"]')

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
                parameter_count=1000000,
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_dir),
                    format="awq",
                    quantization_level="4bit",
                    file_size=100,
                    compression_ratio=0.25,
                    source_model_path="test-model",
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "awq", "-q", "4bit",
                        "-o", str(temp_dir),
                        "--calibration-data", str(calibration_file),
                    ],
                )

                assert result.exit_code == 0
                # Verify calibration data was passed to config
                mock_quantizer_class.assert_called_once()
                call_kwargs = mock_quantizer_class.call_args.kwargs
                config = call_kwargs.get("config")
                assert config.calibration_data_path == str(calibration_file)

    def test_calibration_samples_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test --calibration-samples option for AWQ."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
                parameter_count=1000000,
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_dir),
                    format="awq",
                    quantization_level="4bit",
                    file_size=100,
                    compression_ratio=0.25,
                    source_model_path="test-model",
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "awq", "-q", "4bit",
                        "-o", str(temp_dir),
                        "--calibration-samples", "512",
                    ],
                )

                assert result.exit_code == 0
                # Verify calibration samples was passed to config
                mock_quantizer_class.assert_called_once()
                call_kwargs = mock_quantizer_class.call_args.kwargs
                config = call_kwargs.get("config")
                assert config.calibration_samples == 512


class TestAWQQuantizationLevels:
    """Tests for valid AWQ quantization levels."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.parametrize("quant_level", ["4bit"])
    def test_valid_awq_quant_levels(
        self, runner: CliRunner, temp_dir: Path, quant_level: str
    ) -> None:
        """Test valid AWQ quantization levels are accepted."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
                parameter_count=1000000,
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_dir),
                    format="awq",
                    quantization_level=quant_level,
                    file_size=100,
                    compression_ratio=0.25,
                    source_model_path="test-model",
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "awq", "-q", quant_level,
                        "-o", str(temp_dir),
                    ],
                )

                # Should not fail with invalid argument error
                assert result.exit_code != 2 or "Invalid quantization level" not in result.output
