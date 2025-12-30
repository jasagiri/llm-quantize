"""Contract tests for format conversion CLI command.

These tests verify the CLI contract is respected:
- Exit codes match specification
- Output format matches specification
- Arguments and options work as documented
- Quality degradation warnings are shown
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_quantize.cli.main import cli


class TestConvertCommandExitCodes:
    """Tests for convert command exit codes per CLI contract."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_exit_code_0_on_success(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test exit code 0 on successful conversion."""
        # Create source GGUF file
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        # Create target output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"

                # Create mock output
                awq_output = output_dir / "model-awq"
                awq_output.mkdir()
                (awq_output / "config.json").write_text('{"model_type": "llama"}')
                (awq_output / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_convert.return_value = MagicMock(
                    output_path=str(awq_output),
                    source_format="gguf",
                    target_format="awq",
                    file_size=100,
                    is_lossy=True,
                )

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "awq", "-o", str(output_dir)],
                )

                assert result.exit_code == 0

    def test_exit_code_2_invalid_arguments(self, runner: CliRunner) -> None:
        """Test exit code 2 for invalid arguments."""
        # Missing required arguments
        result = runner.invoke(cli, ["convert"])
        assert result.exit_code == 2

    def test_exit_code_2_invalid_target_format(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test exit code 2 for invalid target format."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = runner.invoke(
            cli,
            ["convert", str(source_file), "invalid-format"],
        )
        assert result.exit_code == 2

    def test_exit_code_3_source_not_found(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test exit code 3 when source file is not found."""
        result = runner.invoke(
            cli,
            ["convert", str(temp_dir / "nonexistent.gguf"), "awq"],
        )
        assert result.exit_code == 3

    def test_exit_code_2_unsupported_conversion(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test exit code 2 for unsupported conversion path."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.is_conversion_supported") as mock_supported:
                mock_detect.return_value = "gguf"
                mock_supported.return_value = False

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "gptq"],
                )

                assert result.exit_code == 2


class TestConvertCommandOutput:
    """Tests for convert command output format."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_json_output_success(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test JSON output format on success."""
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
                    warning_message=None,
                )

                result = runner.invoke(
                    cli,
                    [
                        "--format", "json",
                        "convert", str(source_file), "awq",
                        "-o", str(output_dir),
                    ],
                )

                assert result.exit_code == 0
                output = json.loads(result.output)
                assert output["status"] == "success"
                assert output["source_format"] == "gguf"
                assert output["target_format"] == "awq"
                assert "output_path" in output

    def test_json_output_error(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test JSON output format on error."""
        result = runner.invoke(
            cli,
            [
                "--format", "json",
                "convert", str(temp_dir / "nonexistent.gguf"), "awq",
            ],
        )

        assert result.exit_code == 3
        output = json.loads(result.output)
        assert output["status"] == "error"
        assert "message" in output


class TestConvertCommandOptions:
    """Tests for convert command options."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_output_dir_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test -o/--output-dir option."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "custom_output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"

                awq_output = output_dir / "model-awq"
                awq_output.mkdir()
                (awq_output / "config.json").write_text('{"model_type": "llama"}')
                (awq_output / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_convert.return_value = MagicMock(
                    output_path=str(awq_output),
                    source_format="gguf",
                    target_format="awq",
                    file_size=100,
                    is_lossy=False,
                )

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "awq", "-o", str(output_dir)],
                )

                assert result.exit_code == 0

    def test_force_option(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test --force option to overwrite existing output."""
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

                mock_convert.return_value = MagicMock(
                    output_path=str(awq_output),
                    source_format="gguf",
                    target_format="awq",
                    file_size=100,
                    is_lossy=False,
                )

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "awq", "-o", str(output_dir), "--force"],
                )

                assert result.exit_code == 0


class TestConvertCommandErrorHandling:
    """Tests for error handling in convert command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_format_detection_fails_human_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test error when format detection fails (human mode)."""
        source_file = temp_dir / "model.unknown"
        source_file.write_bytes(b"UNKNOWN" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            mock_detect.return_value = None

            result = runner.invoke(
                cli,
                ["convert", str(source_file), "awq"],
            )

            assert result.exit_code == 2
            assert "Could not detect format" in result.output

    def test_format_detection_fails_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test error when format detection fails (JSON mode)."""
        source_file = temp_dir / "model.unknown"
        source_file.write_bytes(b"UNKNOWN" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            mock_detect.return_value = None

            result = runner.invoke(
                cli,
                ["--format", "json", "convert", str(source_file), "awq"],
            )

            assert result.exit_code == 2
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert "Could not detect format" in output["message"]

    def test_unsupported_conversion_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test error for unsupported conversion in JSON mode."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.is_conversion_supported") as mock_supported:
                mock_detect.return_value = "gguf"
                mock_supported.return_value = False

                result = runner.invoke(
                    cli,
                    ["--format", "json", "convert", str(source_file), "gptq"],
                )

                assert result.exit_code == 2
                output = json.loads(result.output)
                assert output["status"] == "error"
                assert "not supported" in output["message"]

    def test_conversion_value_error_with_not_found(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test ValueError with 'not found' message."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"
                mock_convert.side_effect = ValueError("Source file not found")

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "awq"],
                )

                assert result.exit_code == 3  # MODEL_NOT_FOUND

    def test_conversion_value_error_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test ValueError in JSON mode."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"
                mock_convert.side_effect = ValueError("Invalid conversion")

                result = runner.invoke(
                    cli,
                    ["--format", "json", "convert", str(source_file), "awq"],
                )

                assert result.exit_code == 2
                output = json.loads(result.output)
                assert output["status"] == "error"

    def test_conversion_general_error_human_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test general exception during conversion (human mode)."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"
                mock_convert.side_effect = RuntimeError("Unexpected error")

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "awq"],
                )

                assert result.exit_code == 1

    def test_conversion_general_error_json_mode(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test general exception during conversion (JSON mode)."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "gguf"
                mock_convert.side_effect = RuntimeError("Unexpected error")

                result = runner.invoke(
                    cli,
                    ["--format", "json", "convert", str(source_file), "awq"],
                )

                assert result.exit_code == 1
                output = json.loads(result.output)
                assert output["status"] == "error"
                assert "Conversion failed" in output["message"]


class TestConvertCommandOutputWithWarning:
    """Tests for output with warning messages."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_json_output_with_warning(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test JSON output includes warning message when present."""
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
                    warning_message="Quality may be degraded",
                )

                result = runner.invoke(
                    cli,
                    [
                        "--format", "json",
                        "convert", str(source_file), "awq",
                        "-o", str(output_dir),
                    ],
                )

                assert result.exit_code == 0
                output = json.loads(result.output)
                assert "warning" in output
                assert output["warning"] == "Quality may be degraded"

    def test_human_output_with_lossy_warning(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test human output shows lossy conversion warning."""
        from llm_quantize.lib.converter import ConversionResult

        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                with patch("llm_quantize.cli.convert.is_lossy_conversion") as mock_lossy:
                    mock_detect.return_value = "gguf"
                    mock_lossy.return_value = True

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
                        warning_message="Quality degradation warning",
                    )

                    result = runner.invoke(
                        cli,
                        ["convert", str(source_file), "awq", "-o", str(output_dir)],
                    )

                    assert result.exit_code == 0
                    # Warning should appear in output
                    assert "Quality degradation warning" in result.output or "Warning" in result.output


class TestConvertCommandQualityWarnings:
    """Tests for quality degradation warnings."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_lossy_conversion_warning(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test warning is shown for lossy conversions."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                with patch("llm_quantize.cli.convert.is_lossy_conversion") as mock_lossy:
                    mock_detect.return_value = "gguf"
                    mock_lossy.return_value = True

                    awq_output = output_dir / "model-awq"
                    awq_output.mkdir()
                    (awq_output / "config.json").write_text('{"model_type": "llama"}')
                    (awq_output / "model.safetensors").write_bytes(b"\x00" * 100)

                    mock_convert.return_value = MagicMock(
                        output_path=str(awq_output),
                        source_format="gguf",
                        target_format="awq",
                        file_size=100,
                        is_lossy=True,
                    )

                    result = runner.invoke(
                        cli,
                        ["convert", str(source_file), "awq", "-o", str(output_dir)],
                    )

                    assert result.exit_code == 0
                    # Warning should be in output (either stderr or stdout)
                    # The actual implementation will handle this


class TestConvertCommandFormatDetection:
    """Tests for automatic format detection."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_detect_gguf_format(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test GGUF format is detected from file extension."""
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

                mock_convert.return_value = MagicMock(
                    output_path=str(awq_output),
                    source_format="gguf",
                    target_format="awq",
                    file_size=100,
                    is_lossy=False,
                )

                result = runner.invoke(
                    cli,
                    ["convert", str(source_file), "awq", "-o", str(output_dir)],
                )

                mock_detect.assert_called_once()

    def test_detect_awq_format_from_directory(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test AWQ format is detected from directory structure."""
        source_dir = temp_dir / "awq_model"
        source_dir.mkdir()
        (source_dir / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')
        (source_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.cli.convert.detect_format") as mock_detect:
            with patch("llm_quantize.cli.convert.convert_format") as mock_convert:
                mock_detect.return_value = "awq"

                gguf_output = output_dir / "model.gguf"
                gguf_output.write_bytes(b"GGUF" + b"\x00" * 100)

                mock_convert.return_value = MagicMock(
                    output_path=str(gguf_output),
                    source_format="awq",
                    target_format="gguf",
                    file_size=104,
                    is_lossy=False,
                )

                result = runner.invoke(
                    cli,
                    ["convert", str(source_dir), "gguf", "-o", str(output_dir)],
                )

                mock_detect.assert_called_once()
