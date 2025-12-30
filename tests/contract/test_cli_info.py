"""Contract tests for info CLI command.

These tests verify the CLI contract is respected:
- Exit codes match specification
- Output format matches specification
- Model information is correctly displayed
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_quantize.cli.main import cli


class TestInfoCommandExitCodes:
    """Tests for info command exit codes per CLI contract."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_exit_code_0_on_success(self, runner: CliRunner) -> None:
        """Test exit code 0 on successful model info retrieval."""
        from llm_quantize.models import ModelType, SourceModel

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path="test-model",
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

    def test_exit_code_3_model_not_found(self, runner: CliRunner) -> None:
        """Test exit code 3 when model is not found."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(cli, ["info", "nonexistent-model"])

            assert result.exit_code == 3

    def test_exit_code_1_general_error(self, runner: CliRunner) -> None:
        """Test exit code 1 on general error."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = RuntimeError("Unexpected error")

            result = runner.invoke(cli, ["info", "test-model"])

            assert result.exit_code == 1


class TestInfoCommandOutput:
    """Tests for info command output format."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_json_output_format(self, runner: CliRunner) -> None:
        """Test JSON output format contains required fields."""
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

            result = runner.invoke(cli, ["--format", "json", "info", "test-model"])

            assert result.exit_code == 0
            output = json.loads(result.output)

            # Required fields per contract
            assert "model_name" in output
            assert "architecture" in output
            assert "parameter_count" in output
            assert "hidden_size" in output
            assert "num_layers" in output
            assert "num_heads" in output
            assert "vocab_size" in output
            assert "torch_dtype" in output

    def test_json_output_values(self, runner: CliRunner) -> None:
        """Test JSON output contains correct values."""
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

            result = runner.invoke(cli, ["--format", "json", "info", "test-model"])

            output = json.loads(result.output)
            assert output["model_name"] == "meta-llama/Llama-2-7b-hf"
            assert output["architecture"] == "LlamaForCausalLM"
            assert output["parameter_count"] == 7000000000
            assert output["num_layers"] == 32

    def test_human_output_format(self, runner: CliRunner) -> None:
        """Test human-readable output contains key information."""
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
            # Check key info is present in output
            assert "meta-llama/Llama-2-7b-hf" in result.output or "Model:" in result.output
            assert "LlamaForCausalLM" in result.output or "Architecture:" in result.output

    def test_json_error_format(self, runner: CliRunner) -> None:
        """Test JSON error output format."""
        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(
                cli, ["--format", "json", "info", "nonexistent-model"]
            )

            assert result.exit_code == 3
            output = json.loads(result.output)
            assert output["status"] == "error"
            assert "message" in output


class TestInfoCommandArguments:
    """Tests for info command arguments."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_requires_model_argument(self, runner: CliRunner) -> None:
        """Test info command requires model argument."""
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 2  # Missing argument

    def test_accepts_huggingface_model_id(self, runner: CliRunner) -> None:
        """Test info command accepts HuggingFace model ID."""
        from llm_quantize.models import ModelType, SourceModel

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path="meta-llama/Llama-2-7b-hf",
                model_type=ModelType.HF_HUB,
                architecture="LlamaForCausalLM",
                parameter_count=7000000000,
                dtype="float16",
            )

            result = runner.invoke(cli, ["info", "meta-llama/Llama-2-7b-hf"])

            assert result.exit_code == 0
            mock_create.assert_called_once_with("meta-llama/Llama-2-7b-hf")

    def test_accepts_local_path(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test info command accepts local directory path."""
        from llm_quantize.models import ModelType, SourceModel

        model_dir = temp_dir / "local-model"
        model_dir.mkdir()

        with patch("llm_quantize.lib.model_loader.create_source_model") as mock_create:
            mock_create.return_value = SourceModel(
                model_path=str(model_dir),
                model_type=ModelType.LOCAL_DIR,
                architecture="LlamaForCausalLM",
                parameter_count=7000000000,
                dtype="float16",
            )

            result = runner.invoke(cli, ["info", str(model_dir)])

            assert result.exit_code == 0
