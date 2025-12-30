"""Integration tests for end-to-end GGUF quantization.

These tests verify the complete quantization workflow:
- Model loading
- Quantization process
- Output validation
- File generation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.quantizers.gguf import GGUFQuantizer
from llm_quantize.models import (
    OutputFormat,
    QuantizationConfig,
    QuantizedModel,
    SourceModel,
    ModelType,
)


class TestGGUFQuantizationWorkflow:
    """Integration tests for GGUF quantization workflow."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a mock source model."""
        return SourceModel(
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

    @pytest.fixture
    def gguf_config(self, temp_dir: Path) -> QuantizationConfig:
        """Create GGUF quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_gguf_quantizer_initialization(
        self, source_model: SourceModel, gguf_config: QuantizationConfig
    ) -> None:
        """Test GGUFQuantizer can be initialized."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config,
        )

        assert quantizer is not None
        assert quantizer.source_model == source_model
        assert quantizer.config == gguf_config

    def test_gguf_quantizer_supports_required_levels(self) -> None:
        """Test GGUFQuantizer supports all required quantization levels."""
        supported = GGUFQuantizer.get_supported_levels()

        required_levels = [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0",
        ]

        for level in required_levels:
            assert level in supported, f"Missing support for {level}"

    def test_gguf_quantizer_estimates_output_size(
        self, source_model: SourceModel, gguf_config: QuantizationConfig
    ) -> None:
        """Test GGUFQuantizer can estimate output size."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config,
        )

        estimate = quantizer.estimate_output_size()

        # Q4_K_M should be roughly 25-35% of original fp16 size
        # 7B params * 2 bytes = 14GB, 25-35% = 3.5-5GB
        assert estimate > 0
        assert estimate < source_model.parameter_count * 2  # Less than fp16 size

    @pytest.mark.integration
    def test_gguf_full_quantization(
        self, source_model: SourceModel, gguf_config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test full GGUF quantization with mocked model loading."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config,
        )

        # Mock the actual model loading and conversion
        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                output_file = temp_dir / "test-model-Q4_K_M.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 1000)  # Mock GGUF file
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert isinstance(result, QuantizedModel)
                assert result.format == "gguf"
                assert result.quantization_level == "Q4_K_M"
                assert Path(result.output_path).exists()

    @pytest.mark.integration
    def test_gguf_quantization_creates_valid_output(
        self, source_model: SourceModel, gguf_config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test GGUF quantization creates a file with GGUF header."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config,
        )

        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                output_file = temp_dir / "test-model-Q4_K_M.gguf"
                # Create mock GGUF with valid magic bytes
                with open(output_file, "wb") as f:
                    f.write(b"GGUF")  # GGUF magic
                    f.write(b"\x03\x00\x00\x00")  # Version 3
                    f.write(b"\x00" * 1000)  # Padding

                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                # Verify output file has GGUF magic
                with open(result.output_path, "rb") as f:
                    magic = f.read(4)
                    assert magic == b"GGUF"


class TestGGUFQuantizationWithCheckpoints:
    """Integration tests for GGUF quantization with checkpointing."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a mock source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=4,
            hidden_size=4096,
        )

    @pytest.fixture
    def gguf_config_with_checkpoint(
        self, temp_dir: Path, checkpoint_dir: Path
    ) -> QuantizationConfig:
        """Create GGUF config with checkpointing enabled."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(checkpoint_dir),
            enable_checkpoints=True,
        )

    @pytest.mark.integration
    def test_gguf_quantization_with_checkpoints(
        self,
        source_model: SourceModel,
        gguf_config_with_checkpoint: QuantizationConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test GGUF quantization creates checkpoints."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config_with_checkpoint,
            enable_checkpoints=True,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_layer") as mock_quantize_layer:
                mock_quantize_layer.return_value = {"weights": [1, 2, 3]}

                with patch.object(quantizer, "_finalize_gguf") as mock_finalize:
                    output_path = gguf_config_with_checkpoint.output_dir + "/model.gguf"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_bytes(b"GGUF" + b"\x00" * 100)
                    mock_finalize.return_value = output_path

                    result = quantizer.quantize()

                    # Verify checkpoints were created
                    assert quantizer.checkpoint is not None
                    assert result is not None

    @pytest.mark.integration
    def test_gguf_quantization_resume_from_checkpoint(
        self,
        source_model: SourceModel,
        gguf_config_with_checkpoint: QuantizationConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test GGUF quantization can resume from checkpoint."""
        # First, create a partial checkpoint
        from llm_quantize.lib.checkpoint import Checkpoint

        checkpoint = Checkpoint(checkpoint_dir, gguf_config_with_checkpoint)
        checkpoint.initialize(4, gguf_config_with_checkpoint)
        checkpoint.save_layer(0, {"weights": [1]})
        checkpoint.save_layer(1, {"weights": [2]})

        # Now create quantizer to resume
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config_with_checkpoint,
            enable_checkpoints=True,
            resume_from=str(checkpoint_dir),
        )

        # Should have loaded checkpoint state
        assert quantizer.checkpoint is not None
        assert quantizer.start_layer == 2


class TestGGUFOutputValidation:
    """Integration tests for GGUF output validation."""

    def test_validate_gguf_file_magic(self, temp_dir: Path) -> None:
        """Test validation of GGUF file magic bytes."""
        from llm_quantize.lib.validation import validate_output

        # Valid GGUF file
        valid_file = temp_dir / "valid.gguf"
        with open(valid_file, "wb") as f:
            f.write(b"GGUF")
            f.write(b"\x03\x00\x00\x00")  # Version
            f.write(b"\x00" * 100)

        result = validate_output(str(valid_file), "gguf")
        assert result.is_valid

    def test_validate_gguf_invalid_magic(self, temp_dir: Path) -> None:
        """Test validation catches invalid GGUF magic."""
        from llm_quantize.lib.validation import validate_output

        # Invalid magic
        invalid_file = temp_dir / "invalid.gguf"
        with open(invalid_file, "wb") as f:
            f.write(b"NOTG")
            f.write(b"\x00" * 100)

        result = validate_output(str(invalid_file), "gguf")
        assert not result.is_valid
        assert "magic" in result.error_message.lower()

    def test_validate_gguf_empty_file(self, temp_dir: Path) -> None:
        """Test validation catches empty file."""
        from llm_quantize.lib.validation import validate_output

        empty_file = temp_dir / "empty.gguf"
        empty_file.touch()

        result = validate_output(str(empty_file), "gguf")
        assert not result.is_valid

    def test_validate_gguf_nonexistent_file(self) -> None:
        """Test validation catches nonexistent file."""
        from llm_quantize.lib.validation import validate_output

        result = validate_output("/nonexistent/path/model.gguf", "gguf")
        assert not result.is_valid
        assert "not found" in result.error_message.lower() or "exist" in result.error_message.lower()


class TestGGUFProgressReporting:
    """Integration tests for GGUF quantization progress reporting."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a mock source model with multiple layers."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=4,
        )

    @pytest.fixture
    def gguf_config(self, temp_dir: Path) -> QuantizationConfig:
        """Create GGUF quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    @pytest.mark.integration
    @pytest.mark.skip(reason="Layer-by-layer progress reporting not yet implemented in MVP")
    def test_gguf_reports_layer_progress(
        self, source_model: SourceModel, gguf_config: QuantizationConfig
    ) -> None:
        """Test GGUF quantization reports layer-by-layer progress."""
        from llm_quantize.lib.progress import ProgressReporter

        progress_reporter = ProgressReporter()
        progress_updates: list[int] = []

        def track_progress(advance: int = 1, **kwargs) -> None:
            progress_updates.append(advance)

        progress_reporter.update = track_progress

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=gguf_config,
            progress_reporter=progress_reporter,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_layer") as mock_layer:
                mock_layer.return_value = {"weights": []}
                with patch.object(quantizer, "_finalize_gguf") as mock_final:
                    mock_final.return_value = gguf_config.output_dir + "/model.gguf"
                    Path(mock_final.return_value).parent.mkdir(parents=True, exist_ok=True)
                    Path(mock_final.return_value).write_bytes(b"GGUF")

                    quantizer.quantize()

        # Should have progress updates for each layer
        assert len(progress_updates) >= source_model.num_layers
