"""Unit tests for BaseQuantizer abstract class."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.checkpoint import Checkpoint
from llm_quantize.lib.progress import ProgressReporter
from llm_quantize.lib.quantizers.base import BaseQuantizer
from llm_quantize.models import (
    ModelType,
    OutputFormat,
    QuantizationConfig,
    QuantizedModel,
    SourceModel,
)


class ConcreteQuantizer(BaseQuantizer):
    """Concrete implementation for testing."""

    def quantize(self) -> QuantizedModel:
        """Mock quantize implementation."""
        return QuantizedModel(
            output_path="/tmp/test.gguf",
            format=OutputFormat.GGUF,
            file_size=1000,
            compression_ratio=0.25,
            quantization_level="Q4_K_M",
            source_model_path=self.source_model.model_path,
        )

    @classmethod
    def get_supported_levels(cls) -> list[str]:
        """Return supported levels."""
        return ["Q4_K_M", "Q5_K_M", "Q8_0"]

    def estimate_output_size(self) -> int:
        """Estimate output size."""
        return 1000


@pytest.fixture
def source_model() -> SourceModel:
    """Create test source model."""
    return SourceModel(
        model_path="test-org/test-model",
        model_type=ModelType.HF_HUB,
        architecture="LlamaForCausalLM",
        parameter_count=7000000000,
        dtype="float16",
        num_layers=32,
        hidden_size=4096,
    )


@pytest.fixture
def config(temp_dir: Path) -> QuantizationConfig:
    """Create test config."""
    return QuantizationConfig(
        target_format=OutputFormat.GGUF,
        quantization_level="Q4_K_M",
        output_dir=str(temp_dir),
    )


@pytest.fixture
def quantizer(source_model: SourceModel, config: QuantizationConfig) -> ConcreteQuantizer:
    """Create test quantizer."""
    return ConcreteQuantizer(
        source_model=source_model,
        config=config,
        enable_checkpoints=True,
    )


class TestBaseQuantizerInit:
    """Tests for BaseQuantizer initialization."""

    def test_init_with_all_params(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test initialization with all parameters."""
        progress = ProgressReporter(verbosity=1)
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=progress,
            enable_checkpoints=True,
        )
        assert quantizer.source_model == source_model
        assert quantizer.config == config
        assert quantizer.progress_reporter == progress
        assert quantizer.enable_checkpoints is True

    def test_init_with_defaults(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test initialization with default values."""
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
        )
        assert quantizer.progress_reporter is not None
        assert quantizer.enable_checkpoints is True
        assert quantizer._checkpoint is None

    def test_init_without_checkpoints(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test initialization with checkpoints disabled."""
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )
        assert quantizer.enable_checkpoints is False


class TestValidateConfig:
    """Tests for validate_config method."""

    def test_validate_config_valid(self, quantizer: ConcreteQuantizer) -> None:
        """Test validation with valid config."""
        errors = quantizer.validate_config()
        assert errors == []

    def test_validate_config_invalid_level(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test validation with invalid quantization level."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="INVALID_LEVEL",
            output_dir=str(temp_dir),
        )
        quantizer = ConcreteQuantizer(source_model=source_model, config=config)
        errors = quantizer.validate_config()
        assert len(errors) == 1
        assert "Unsupported quantization level" in errors[0]
        assert "INVALID_LEVEL" in errors[0]
        assert "Q4_K_M" in errors[0]


class TestSetupCheckpoint:
    """Tests for setup_checkpoint method."""

    def test_setup_checkpoint_creates_checkpoint(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test that setup_checkpoint creates a checkpoint."""
        quantizer.config.checkpoint_dir = str(temp_dir / "checkpoints")
        checkpoint = quantizer.setup_checkpoint(total_layers=32)

        assert checkpoint is not None
        assert quantizer._checkpoint == checkpoint
        assert isinstance(checkpoint, Checkpoint)

    def test_setup_checkpoint_default_dir(
        self, quantizer: ConcreteQuantizer
    ) -> None:
        """Test setup_checkpoint with default directory."""
        quantizer.config.checkpoint_dir = None
        checkpoint = quantizer.setup_checkpoint(total_layers=32)

        assert checkpoint is not None


class TestTryResume:
    """Tests for try_resume method."""

    def test_try_resume_no_checkpoint(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test try_resume when no checkpoint exists."""
        quantizer.config.checkpoint_dir = str(temp_dir / "nonexistent")
        checkpoint, start_layer = quantizer.try_resume()

        assert checkpoint is None
        assert start_layer == 0

    def test_try_resume_with_existing_checkpoint(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test try_resume with an existing checkpoint."""
        # First set up a checkpoint
        checkpoint_dir = temp_dir / "checkpoints"
        quantizer.config.checkpoint_dir = str(checkpoint_dir)

        # Create and save a checkpoint - save_layer automatically saves metadata
        first_checkpoint = quantizer.setup_checkpoint(total_layers=32)
        first_checkpoint.save_layer(0, {"data": "layer0"})
        first_checkpoint.save_layer(1, {"data": "layer1"})

        # Create new quantizer and try to resume
        new_quantizer = ConcreteQuantizer(
            source_model=quantizer.source_model,
            config=quantizer.config,
        )
        checkpoint, start_layer = new_quantizer.try_resume()

        assert checkpoint is not None
        assert start_layer == 2

    def test_try_resume_with_invalid_checkpoint(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test try_resume with invalid checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create invalid checkpoint file
        (checkpoint_dir / "checkpoint.json").write_text("invalid json")

        quantizer.config.checkpoint_dir = str(checkpoint_dir)
        checkpoint, start_layer = quantizer.try_resume()

        assert checkpoint is None
        assert start_layer == 0


class TestCleanupCheckpoint:
    """Tests for cleanup_checkpoint method."""

    def test_cleanup_checkpoint_removes_checkpoint(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test that cleanup removes checkpoint files."""
        checkpoint_dir = temp_dir / "checkpoints"
        quantizer.config.checkpoint_dir = str(checkpoint_dir)

        # Set up checkpoint
        quantizer.setup_checkpoint(total_layers=32)
        assert quantizer._checkpoint is not None

        # Clean up
        quantizer.cleanup_checkpoint()
        assert quantizer._checkpoint is None

    def test_cleanup_checkpoint_no_checkpoint(
        self, quantizer: ConcreteQuantizer
    ) -> None:
        """Test cleanup when no checkpoint exists."""
        assert quantizer._checkpoint is None
        # Should not raise
        quantizer.cleanup_checkpoint()
        assert quantizer._checkpoint is None


class TestGetOutputPath:
    """Tests for get_output_path method."""

    def test_get_output_path_auto_generated(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test auto-generated output path."""
        quantizer.config.output_name = None
        path = quantizer.get_output_path()

        assert path.parent == temp_dir
        assert "test-org-test-model" in str(path) or "test-model" in str(path)
        assert "Q4_K_M" in str(path)
        assert str(path).endswith(".gguf")

    def test_get_output_path_custom_name(
        self, quantizer: ConcreteQuantizer, temp_dir: Path
    ) -> None:
        """Test custom output name."""
        quantizer.config.output_name = "custom-output.gguf"
        path = quantizer.get_output_path()

        assert path == temp_dir / "custom-output.gguf"

    def test_get_output_path_local_model(
        self, temp_dir: Path
    ) -> None:
        """Test output path with local model path."""
        source_model = SourceModel(
            model_path="/path/to/local/model",
            model_type=ModelType.LOCAL_DIR,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )
        quantizer = ConcreteQuantizer(source_model=source_model, config=config)
        path = quantizer.get_output_path()

        assert "model" in str(path).lower() or "local" in str(path).lower()


class TestLogStart:
    """Tests for log_start method."""

    def test_log_start_logs_info(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test that log_start logs model information."""
        mock_reporter = MagicMock(spec=ProgressReporter)
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=mock_reporter,
        )

        quantizer.log_start()

        # Check that log_info was called
        assert mock_reporter.log_info.call_count >= 3

        # Check logged content
        calls = [str(call) for call in mock_reporter.log_info.call_args_list]
        call_str = " ".join(calls)
        assert "test-org/test-model" in call_str
        assert "LlamaForCausalLM" in call_str
        assert "gguf" in call_str.lower()

    def test_log_start_no_reporter(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test log_start when progress reporter returns None."""
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=None,
        )
        # progress_reporter defaults to a new ProgressReporter, but let's test None
        quantizer.progress_reporter = None
        # Should not raise
        quantizer.log_start()


class TestLogComplete:
    """Tests for log_complete method."""

    def test_log_complete_reports_completion(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test that log_complete reports completion."""
        mock_reporter = MagicMock(spec=ProgressReporter)
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=mock_reporter,
        )

        result = QuantizedModel(
            output_path="/tmp/output.gguf",
            format=OutputFormat.GGUF,
            file_size=5000000,
            compression_ratio=0.25,
            quantization_level="Q4_K_M",
            source_model_path="test-model",
            duration_seconds=120.5,
        )

        quantizer.log_complete(result)

        mock_reporter.report_completion.assert_called_once_with(
            output_path="/tmp/output.gguf",
            file_size=5000000,
            duration=120.5,
            compression_ratio=0.25,
        )

    def test_log_complete_no_reporter(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test log_complete when no progress reporter."""
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
        )
        quantizer.progress_reporter = None

        result = QuantizedModel(
            output_path="/tmp/output.gguf",
            format=OutputFormat.GGUF,
            file_size=5000000,
            compression_ratio=0.25,
            quantization_level="Q4_K_M",
            source_model_path="test-model",
        )

        # Should not raise
        quantizer.log_complete(result)


class TestAbstractMethods:
    """Tests for abstract method interface."""

    def test_quantize_returns_quantized_model(
        self, quantizer: ConcreteQuantizer
    ) -> None:
        """Test that quantize returns a QuantizedModel."""
        result = quantizer.quantize()
        assert isinstance(result, QuantizedModel)

    def test_get_supported_levels_returns_list(self) -> None:
        """Test that get_supported_levels returns a list."""
        levels = ConcreteQuantizer.get_supported_levels()
        assert isinstance(levels, list)
        assert len(levels) > 0
        assert all(isinstance(l, str) for l in levels)

    def test_estimate_output_size_returns_int(
        self, quantizer: ConcreteQuantizer
    ) -> None:
        """Test that estimate_output_size returns an int."""
        size = quantizer.estimate_output_size()
        assert isinstance(size, int)
        assert size > 0


class TestTryResumeWithProgress:
    """Tests for try_resume with progress reporting."""

    def test_try_resume_logs_progress(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test that try_resume logs progress on successful resume."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        mock_reporter = MagicMock(spec=ProgressReporter)

        # First create and populate a checkpoint
        first_quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
        )
        first_checkpoint = first_quantizer.setup_checkpoint(total_layers=32)
        first_checkpoint.save_layer(0, {"data": "layer0"})
        first_checkpoint.save_layer(1, {"data": "layer1"})

        # Now try to resume with a new quantizer
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=mock_reporter,
        )
        checkpoint, start_layer = quantizer.try_resume()

        # Should log resume info
        mock_reporter.log_info.assert_called()
        call_str = str(mock_reporter.log_info.call_args_list[-1])
        assert "Resuming" in call_str
        assert "2" in call_str

    def test_try_resume_logs_warning_on_config_mismatch(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test that try_resume logs warning on config mismatch."""
        # Create initial checkpoint with one config
        config1 = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        first_quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config1,
        )
        first_checkpoint = first_quantizer.setup_checkpoint(total_layers=32)
        first_checkpoint.save_layer(0, {"data": "layer0"})

        # Try to resume with different quantization level
        config2 = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q8_0",  # Different level
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        mock_reporter = MagicMock(spec=ProgressReporter)
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config2,
            progress_reporter=mock_reporter,
        )

        checkpoint, start_layer = quantizer.try_resume()

        # Should return None and log warning
        assert checkpoint is None
        assert start_layer == 0
        mock_reporter.log_warning.assert_called()

    def test_try_resume_without_progress_reporter(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test try_resume without progress reporter doesn't crash."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        # First create a checkpoint
        first_quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
        )
        first_checkpoint = first_quantizer.setup_checkpoint(total_layers=32)
        first_checkpoint.save_layer(0, {"data": "layer0"})

        # Try to resume with progress_reporter set to None
        quantizer = ConcreteQuantizer(
            source_model=source_model,
            config=config,
        )
        quantizer.progress_reporter = None

        # Should not raise
        checkpoint, start_layer = quantizer.try_resume()
        assert checkpoint is not None
        assert start_layer == 1
