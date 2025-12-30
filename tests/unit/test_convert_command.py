"""Tests for CLI convert command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_quantize.lib.converter import ConversionResult


class TestConvertCommand:
    """Tests for convert command."""

    def create_gguf_file(self, tmpdir):
        """Create a mock GGUF file."""
        gguf_path = Path(tmpdir) / "model.gguf"
        # Write GGUF magic bytes
        gguf_path.write_bytes(b"GGUF" + b"\x00" * 100)
        return gguf_path

    def test_convert_source_not_found(self):
        """Test convert with source not found."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        result = runner.invoke(convert, ["nonexistent/model", "gguf"])

        assert result.exit_code == 3  # EXIT_MODEL_NOT_FOUND

    @patch("llm_quantize.cli.convert.detect_format")
    def test_convert_unknown_format(self, mock_detect):
        """Test convert with unknown source format."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.unknown"
            source_path.touch()

            result = runner.invoke(convert, [str(source_path), "gguf"])

        assert result.exit_code == 2  # EXIT_INVALID_ARGUMENTS

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    def test_convert_unsupported_conversion(self, mock_supported, mock_detect):
        """Test convert with unsupported conversion."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(convert, [str(source_path), "awq"])

        assert result.exit_code == 2

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_success(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test successful conversion."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = False
        mock_convert.return_value = ConversionResult(
            output_path="/tmp/output.awq",
            source_format="gguf",
            target_format="awq",
            file_size=1000,
            is_lossy=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(convert, [str(source_path), "awq", "-o", tmpdir])

        assert result.exit_code == 0
        assert "Conversion complete" in result.output

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_lossy_warning(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test conversion with lossy warning."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = True
        mock_convert.return_value = ConversionResult(
            output_path="/tmp/output.awq",
            source_format="gguf",
            target_format="awq",
            file_size=1000,
            is_lossy=True,
            warning_message="Quality may degrade",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(convert, [str(source_path), "awq", "-o", tmpdir])

        assert result.exit_code == 0
        # Warning should be printed
        assert "Warning" in result.output or "warning" in result.output.lower() or "Quality" in result.output

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_value_error(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test conversion with ValueError."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = False
        mock_convert.side_effect = ValueError("Invalid format")

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(convert, [str(source_path), "awq", "-o", tmpdir])

        assert result.exit_code == 2

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_general_error(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test conversion with general error."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = False
        mock_convert.side_effect = RuntimeError("Conversion failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(convert, [str(source_path), "awq", "-o", tmpdir])

        assert result.exit_code == 1

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_with_output_name(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test conversion with custom output name."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = False
        mock_convert.return_value = ConversionResult(
            output_path="/tmp/custom.awq",
            source_format="gguf",
            target_format="awq",
            file_size=1000,
            is_lossy=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(
                convert,
                [str(source_path), "awq", "-o", tmpdir, "-n", "custom.awq"],
            )

        assert result.exit_code == 0

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_value_error_not_found(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test conversion with ValueError containing 'not found'."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = False
        mock_convert.side_effect = ValueError("Model not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(convert, [str(source_path), "awq", "-o", tmpdir])

        assert result.exit_code == 3  # EXIT_MODEL_NOT_FOUND

    @patch("llm_quantize.cli.convert.detect_format")
    @patch("llm_quantize.cli.convert.is_conversion_supported")
    @patch("llm_quantize.cli.convert.is_lossy_conversion")
    @patch("llm_quantize.cli.convert.convert_format")
    def test_convert_with_force(
        self, mock_convert, mock_lossy, mock_supported, mock_detect
    ):
        """Test conversion with force flag."""
        from llm_quantize.cli.convert import convert

        runner = CliRunner()

        mock_detect.return_value = "gguf"
        mock_supported.return_value = True
        mock_lossy.return_value = False
        mock_convert.return_value = ConversionResult(
            output_path="/tmp/output.awq",
            source_format="gguf",
            target_format="awq",
            file_size=1000,
            is_lossy=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "model.gguf"
            source_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = runner.invoke(
                convert,
                [str(source_path), "awq", "-o", tmpdir, "--force"],
            )

        assert result.exit_code == 0
        # Verify force was passed
        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["force"] is True
