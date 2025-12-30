"""Unit tests for CLI main module."""

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from llm_quantize.cli.main import (
    CliContext,
    cli,
    format_callback,
    verbosity_callback,
)
from llm_quantize.models import OutputMode, Verbosity


class TestCliContext:
    """Tests for CliContext class."""

    def test_output_json_mode(self) -> None:
        """Test output in JSON mode."""
        ctx = CliContext(OutputMode.JSON, Verbosity.NORMAL)
        ctx.stdout_console = MagicMock(spec=Console)

        ctx.output({"key": "value"})

        ctx.stdout_console.print_json.assert_called_once()

    def test_output_human_mode(self) -> None:
        """Test output in human mode does nothing."""
        ctx = CliContext(OutputMode.HUMAN, Verbosity.NORMAL)
        ctx.stdout_console = MagicMock(spec=Console)

        # Should not raise and do nothing
        ctx.output({"key": "value"})

        ctx.stdout_console.print_json.assert_not_called()

    def test_error_json_mode(self) -> None:
        """Test error output in JSON mode."""
        ctx = CliContext(OutputMode.JSON, Verbosity.NORMAL)
        ctx.stdout_console = MagicMock(spec=Console)

        ctx.error("Test error", 2)

        ctx.stdout_console.print_json.assert_called_once()

    def test_error_human_mode(self) -> None:
        """Test error output in human mode."""
        ctx = CliContext(OutputMode.HUMAN, Verbosity.NORMAL)
        ctx.console = MagicMock(spec=Console)

        ctx.error("Test error", 1)

        ctx.console.print.assert_called_once()
        call_args = ctx.console.print.call_args[0][0]
        assert "Test error" in call_args


class TestVerbosityCallback:
    """Tests for verbosity_callback function."""

    def test_valid_verbosity_quiet(self) -> None:
        """Test valid quiet verbosity."""
        ctx = MagicMock()
        param = MagicMock()
        result = verbosity_callback(ctx, param, "quiet")
        assert result == Verbosity.QUIET

    def test_valid_verbosity_normal(self) -> None:
        """Test valid normal verbosity."""
        ctx = MagicMock()
        param = MagicMock()
        result = verbosity_callback(ctx, param, "normal")
        assert result == Verbosity.NORMAL

    def test_valid_verbosity_verbose(self) -> None:
        """Test valid verbose verbosity."""
        ctx = MagicMock()
        param = MagicMock()
        result = verbosity_callback(ctx, param, "verbose")
        assert result == Verbosity.VERBOSE

    def test_invalid_verbosity_raises(self) -> None:
        """Test invalid verbosity raises BadParameter."""
        ctx = MagicMock()
        param = MagicMock()
        with pytest.raises(click.BadParameter, match="Invalid verbosity"):
            verbosity_callback(ctx, param, "invalid")


class TestFormatCallback:
    """Tests for format_callback function."""

    def test_valid_format_human(self) -> None:
        """Test valid human format."""
        ctx = MagicMock()
        param = MagicMock()
        result = format_callback(ctx, param, "human")
        assert result == OutputMode.HUMAN

    def test_valid_format_json(self) -> None:
        """Test valid json format."""
        ctx = MagicMock()
        param = MagicMock()
        result = format_callback(ctx, param, "json")
        assert result == OutputMode.JSON

    def test_invalid_format_raises(self) -> None:
        """Test invalid format raises BadParameter."""
        ctx = MagicMock()
        param = MagicMock()
        with pytest.raises(click.BadParameter, match="Invalid format"):
            format_callback(ctx, param, "invalid")


class TestCliCommand:
    """Tests for main CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_version_option(self, runner: CliRunner) -> None:
        """Test --version option."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "llm-quantize" in result.output

    def test_help_option(self, runner: CliRunner) -> None:
        """Test --help option."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_format_option_json(self, runner: CliRunner) -> None:
        """Test --format json option."""
        result = runner.invoke(cli, ["--format", "json", "--help"])
        assert result.exit_code == 0

    def test_verbosity_option_verbose(self, runner: CliRunner) -> None:
        """Test --verbosity verbose option."""
        result = runner.invoke(cli, ["--verbosity", "verbose", "--help"])
        assert result.exit_code == 0

    def test_info_command_general_error(self, runner: CliRunner) -> None:
        """Test info command with general error."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock:
            mock.side_effect = RuntimeError("Unexpected error")

            result = runner.invoke(cli, ["info", "test-model"])
            assert result.exit_code == 1

    def test_info_command_value_error(self, runner: CliRunner) -> None:
        """Test info command with ValueError (model not found)."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock:
            mock.side_effect = ValueError("Model not found")

            result = runner.invoke(cli, ["info", "test-model"])
            assert result.exit_code == 3  # EXIT_MODEL_NOT_FOUND

    def test_info_command_success_human(self, runner: CliRunner) -> None:
        """Test info command success with human output."""
        from llm_quantize.models import SourceModel, ModelType

        mock_source = SourceModel(
            model_path="test/model",
            model_type=ModelType.HF_HUB,
            architecture="llama",
            parameter_count=1000000,
            dtype="float16",
            num_layers=4,
            hidden_size=768,
            num_heads=12,
            vocab_size=32000,
        )

        with patch("llm_quantize.lib.model_loader.create_source_model", return_value=mock_source):
            result = runner.invoke(cli, ["info", "test-model"])
            assert result.exit_code == 0
            assert "Architecture: llama" in result.output

    def test_info_command_success_json(self, runner: CliRunner) -> None:
        """Test info command success with JSON output."""
        from llm_quantize.models import SourceModel, ModelType

        mock_source = SourceModel(
            model_path="test/model",
            model_type=ModelType.HF_HUB,
            architecture="llama",
            parameter_count=1000000,
            dtype="float16",
            num_layers=4,
            hidden_size=768,
            num_heads=12,
            vocab_size=32000,
        )

        with patch("llm_quantize.lib.model_loader.create_source_model", return_value=mock_source):
            result = runner.invoke(cli, ["--format", "json", "info", "test-model"])
            assert result.exit_code == 0
            assert "llama" in result.output

    def test_verbosity_option_debug(self, runner: CliRunner) -> None:
        """Test --verbosity debug option."""
        result = runner.invoke(cli, ["--verbosity", "debug", "--help"])
        assert result.exit_code == 0

    def test_verbosity_option_quiet(self, runner: CliRunner) -> None:
        """Test --verbosity quiet option."""
        result = runner.invoke(cli, ["--verbosity", "quiet", "--help"])
        assert result.exit_code == 0
