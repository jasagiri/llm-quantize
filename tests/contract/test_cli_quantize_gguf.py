"""Contract tests for GGUF CLI quantization command.

These tests verify the CLI contract is respected:
- Exit codes match specification
- Output format matches specification
- Arguments and options work as documented
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_quantize.cli.main import cli


class TestGGUFQuantizeCommandExitCodes:
    """Tests for GGUF quantize command exit codes per CLI contract."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_exit_code_0_on_success(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test exit code 0 on successful quantization."""
        from llm_quantize.models import QuantizedModel, ValidationStatus

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                # Create mock output file
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_file),
                    format="gguf",
                    quantization_level="Q4_K_M",
                    file_size=104,
                    compression_ratio=0.28,
                    source_model_path="test-model",
                    duration_seconds=1.0,
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 0

    def test_exit_code_2_invalid_arguments(self, runner: CliRunner) -> None:
        """Test exit code 2 for invalid arguments."""
        # Missing required arguments
        result = runner.invoke(cli, ["quantize"])
        assert result.exit_code == 2

        # Missing quantization level
        result = runner.invoke(cli, ["quantize", "test-model", "gguf"])
        assert result.exit_code == 2

    def test_exit_code_2_invalid_format(self, runner: CliRunner) -> None:
        """Test exit code 2 for invalid output format."""
        result = runner.invoke(
            cli,
            ["quantize", "test-model", "invalid-format", "-q", "Q4_K_M"],
        )
        assert result.exit_code == 2

    def test_exit_code_2_invalid_quant_level(self, runner: CliRunner) -> None:
        """Test exit code 2 for invalid quantization level."""
        with patch("llm_quantize.cli.quantize.create_source_model"):
            result = runner.invoke(
                cli,
                ["quantize", "test-model", "gguf", "-q", "INVALID_QUANT"],
            )
            assert result.exit_code == 2

    def test_exit_code_3_model_not_found(self, runner: CliRunner) -> None:
        """Test exit code 3 when model is not found."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli,
                ["quantize", "nonexistent-model", "gguf", "-q", "Q4_K_M"],
            )

            assert result.exit_code == 3


class TestGGUFQuantizeCommandOutput:
    """Tests for GGUF quantize command output format."""

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
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_file),
                    format="gguf",
                    quantization_level="Q4_K_M",
                    file_size=104,
                    compression_ratio=0.28,
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
                        "quantize", "test-model", "gguf", "-q", "Q4_K_M",
                        "-o", str(temp_dir),
                    ],
                )

                assert result.exit_code == 0
                output = json.loads(result.output)
                assert output["status"] == "success"
                assert output["format"] == "gguf"
                assert output["quantization_level"] == "Q4_K_M"
                assert "output_path" in output

    def test_json_output_error(self, runner: CliRunner) -> None:
        """Test JSON output format on error."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli,
                [
                    "--format", "json",
                    "quantize", "nonexistent-model", "gguf", "-q", "Q4_K_M",
                ],
            )

            assert result.exit_code == 3
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert output["error_code"] == 3
            assert "message" in output


class TestGGUFQuantizeCommandOptions:
    """Tests for GGUF quantize command options."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_output_dir_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test -o/--output-dir option."""
        from llm_quantize.models import QuantizedModel

        output_dir = temp_dir / "custom_output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_file = output_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_file),
                    format="gguf",
                    quantization_level="Q4_K_M",
                    file_size=104,
                    compression_ratio=0.28,
                    source_model_path="test-model",
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "gguf", "-q", "Q4_K_M",
                        "-o", str(output_dir),
                    ],
                )

                assert result.exit_code == 0

    def test_output_name_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test -n/--output-name option."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_file = temp_dir / "custom-name.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_file),
                    format="gguf",
                    quantization_level="Q4_K_M",
                    file_size=104,
                    compression_ratio=0.28,
                    source_model_path="test-model",
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "gguf", "-q", "Q4_K_M",
                        "-o", str(temp_dir), "-n", "custom-name.gguf",
                    ],
                )

                assert result.exit_code == 0

    def test_no_checkpoints_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test --no-checkpoints option disables checkpointing."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)

                mock_result = QuantizedModel(
                    output_path=str(output_file),
                    format="gguf",
                    quantization_level="Q4_K_M",
                    file_size=104,
                    compression_ratio=0.28,
                    source_model_path="test-model",
                )

                mock_quantizer = MagicMock()
                mock_quantizer.quantize.return_value = mock_result

                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "gguf", "-q", "Q4_K_M",
                        "-o", str(temp_dir), "--no-checkpoints",
                    ],
                )

                assert result.exit_code == 0
                # Verify checkpoint was not enabled
                mock_quantizer_class.assert_called_once()
                call_kwargs = mock_quantizer_class.call_args.kwargs
                assert call_kwargs.get("enable_checkpoints") is False


class TestGGUFQuantizeErrorHandling:
    """Tests for error handling in GGUF quantize command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_quant_level_json_mode(self, runner: CliRunner) -> None:
        """Test JSON mode for invalid quantization level."""
        with patch("llm_quantize.cli.quantize.create_source_model"):
            result = runner.invoke(
                cli,
                ["--format", "json", "quantize", "test-model", "gguf", "-q", "INVALID"],
            )
            assert result.exit_code == 2
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert "Invalid quantization level" in output["message"]

    def test_quantizer_not_found_human_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test when quantizer is not available (human mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_get.return_value = None

                result = runner.invoke(
                    cli,
                    ["quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 2

    def test_quantizer_not_found_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test when quantizer is not available (JSON mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_get.return_value = None

                result = runner.invoke(
                    cli,
                    ["--format", "json", "quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 2
                output = json.loads(result.output)
                assert output["status"] == "error"
                assert "No quantizer available" in output["message"]

    def test_checkpoint_error_human_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test checkpoint error during quantizer init (human mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_quantizer_class = MagicMock()
                mock_quantizer_class.side_effect = ValueError("checkpoint corrupt")
                mock_get.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 7  # EXIT_CHECKPOINT_ERROR

    def test_checkpoint_error_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test checkpoint error during quantizer init (JSON mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_quantizer_class = MagicMock()
                mock_quantizer_class.side_effect = ValueError("checkpoint corrupt")
                mock_get.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["--format", "json", "quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 7
                output = json.loads(result.output)
                assert output["status"] == "error"
                assert "Checkpoint error" in output["message"]

    def test_general_init_error_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test general error during quantizer init (JSON mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_quantizer_class = MagicMock()
                mock_quantizer_class.side_effect = RuntimeError("Unexpected error")
                mock_get.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["--format", "json", "quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 1
                output = json.loads(result.output)
                assert output["status"] == "error"

    def test_memory_error_human_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test MemoryError during quantization (human mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_quantizer = MagicMock()
                mock_quantizer.quantize.side_effect = MemoryError()
                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 5  # EXIT_OUT_OF_MEMORY

    def test_memory_error_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test MemoryError during quantization (JSON mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_quantizer = MagicMock()
                mock_quantizer.quantize.side_effect = MemoryError()
                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["--format", "json", "quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 5
                output = json.loads(result.output)
                assert output["status"] == "error"
                assert "Out of memory" in output["message"]

    def test_general_quantization_error_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test general error during quantization (JSON mode)."""
        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                mock_quantizer = MagicMock()
                mock_quantizer.quantize.side_effect = RuntimeError("Quantization failed")
                mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                mock_get.return_value = mock_quantizer_class

                result = runner.invoke(
                    cli,
                    ["--format", "json", "quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                )

                assert result.exit_code == 1
                output = json.loads(result.output)
                assert output["status"] == "error"
                assert "Quantization failed" in output["message"]

    def test_validation_failure_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test validation failure (JSON mode)."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                with patch("llm_quantize.cli.quantize.validate_output") as mock_validate:
                    output_file = temp_dir / "model.gguf"
                    output_file.write_bytes(b"INVALID")

                    mock_result = QuantizedModel(
                        output_path=str(output_file),
                        format="gguf",
                        quantization_level="Q4_K_M",
                        file_size=7,
                        compression_ratio=0.28,
                        source_model_path="test-model",
                    )

                    mock_quantizer = MagicMock()
                    mock_quantizer.quantize.return_value = mock_result
                    mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                    mock_get.return_value = mock_quantizer_class

                    mock_validation = MagicMock()
                    mock_validation.is_valid = False
                    mock_validation.error_message = "Invalid GGUF header"
                    mock_validate.return_value = mock_validation

                    result = runner.invoke(
                        cli,
                        ["--format", "json", "quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                    )

                    assert result.exit_code == 6  # EXIT_VALIDATION_FAILED
                    output = json.loads(result.output)
                    assert output["status"] == "error"
                    assert "validation failed" in output["message"].lower()

    def test_validation_failure_human_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test validation failure (human mode)."""
        from llm_quantize.models import QuantizedModel

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
            mock_source.return_value = MagicMock(
                model_path="test-model",
                architecture="LlamaForCausalLM",
                num_layers=2,
                dtype="float16",
            )

            with patch("llm_quantize.cli.quantize.get_quantizer") as mock_get:
                with patch("llm_quantize.cli.quantize.validate_output") as mock_validate:
                    output_file = temp_dir / "model.gguf"
                    output_file.write_bytes(b"INVALID")

                    mock_result = QuantizedModel(
                        output_path=str(output_file),
                        format="gguf",
                        quantization_level="Q4_K_M",
                        file_size=7,
                        compression_ratio=0.28,
                        source_model_path="test-model",
                    )

                    mock_quantizer = MagicMock()
                    mock_quantizer.quantize.return_value = mock_result
                    mock_quantizer_class = MagicMock(return_value=mock_quantizer)
                    mock_get.return_value = mock_quantizer_class

                    mock_validation = MagicMock()
                    mock_validation.is_valid = False
                    mock_validation.error_message = "Invalid GGUF header"
                    mock_validate.return_value = mock_validation

                    result = runner.invoke(
                        cli,
                        ["quantize", "test-model", "gguf", "-q", "Q4_K_M", "-o", str(temp_dir)],
                    )

                    assert result.exit_code == 6  # EXIT_VALIDATION_FAILED


class TestGGUFQuantizationLevels:
    """Tests for valid GGUF quantization levels."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.parametrize(
        "quant_level",
        ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
         "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
    )
    def test_valid_gguf_quant_levels(
        self, runner: CliRunner, temp_dir: Path, quant_level: str
    ) -> None:
        """Test all valid GGUF quantization levels are accepted."""
        with patch("llm_quantize.cli.quantize.GGUFQuantizer") as mock_quantizer_class:
            mock_quantizer = MagicMock()
            mock_quantizer.quantize.return_value = MagicMock(
                output_path=str(temp_dir / f"model-{quant_level}.gguf"),
                format="gguf",
            )
            mock_quantizer_class.return_value = mock_quantizer

            with patch("llm_quantize.cli.quantize.create_source_model") as mock_source:
                mock_source.return_value = MagicMock(
                    model_path="test-model",
                    architecture="LlamaForCausalLM",
                    num_layers=2,
                )

                result = runner.invoke(
                    cli,
                    [
                        "quantize", "test-model", "gguf", "-q", quant_level,
                        "-o", str(temp_dir),
                    ],
                )

                # Should not fail with invalid argument error
                assert result.exit_code != 2 or "Invalid quantization level" not in result.output
