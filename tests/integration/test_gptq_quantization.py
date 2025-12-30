"""Integration tests for GPTQ quantization workflow.

These tests verify the complete GPTQ quantization pipeline:
- Model loading → GPTQ quantization → Output validation
- Group size configuration
- Calibration data handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.quantizers import get_quantizer
from llm_quantize.lib.quantizers.gptq import GPTQQuantizer
from llm_quantize.lib.validation import validate_output
from llm_quantize.models import (
    GPTQ_QUANT_TYPES,
    ModelType,
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestGPTQQuantizationWorkflow:
    """Integration tests for GPTQ quantization workflow."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
        )

    @pytest.fixture
    def gptq_config(self, temp_dir: Path) -> QuantizationConfig:
        """Create GPTQ quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            group_size=128,
            calibration_samples=128,
        )

    @pytest.mark.integration
    def test_gptq_quantizer_initialization(
        self, source_model: SourceModel, gptq_config: QuantizationConfig
    ) -> None:
        """Test GPTQQuantizer can be initialized."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=gptq_config,
        )

        assert quantizer.source_model == source_model
        assert quantizer.config == gptq_config

    @pytest.mark.integration
    def test_gptq_quantizer_supports_required_levels(self) -> None:
        """Test GPTQQuantizer supports all required quantization levels."""
        levels = GPTQQuantizer.get_supported_levels()

        # GPTQ supports 2, 3, 4, and 8-bit quantization
        assert "2bit" in levels
        assert "3bit" in levels
        assert "4bit" in levels
        assert "8bit" in levels

    @pytest.mark.integration
    def test_gptq_quantizer_estimates_output_size(
        self, source_model: SourceModel, gptq_config: QuantizationConfig
    ) -> None:
        """Test GPTQQuantizer can estimate output size."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=gptq_config,
        )

        estimated_size = quantizer.estimate_output_size()

        # Should return positive estimate
        assert estimated_size > 0

        # GPTQ 4-bit should be ~25% of fp16 size (4/16 bits)
        original_size = source_model.parameter_count * 2  # fp16
        assert estimated_size < original_size

    @pytest.mark.integration
    def test_gptq_full_quantization(
        self, source_model: SourceModel, gptq_config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test full GPTQ quantization workflow."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=gptq_config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                mock_load.return_value = MagicMock()

                # Create mock output directory
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result.format == "gptq"
                assert result.quantization_level == "4bit"
                assert result.output_path == str(output_dir)

    @pytest.mark.integration
    def test_gptq_quantization_creates_valid_output(
        self, source_model: SourceModel, gptq_config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test GPTQ quantization creates valid output structure."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=gptq_config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                # Create proper GPTQ output structure
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                # Validate output
                validation = validate_output(result.output_path, "gptq")
                assert validation.is_valid


class TestGPTQGroupSize:
    """Tests for GPTQ group size configuration."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("group_size", [32, 64, 128, 256])
    def test_gptq_with_different_group_sizes(
        self, source_model: SourceModel, temp_dir: Path, group_size: int
    ) -> None:
        """Test GPTQ quantization with different group sizes."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            group_size=group_size,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                output_dir = temp_dir / f"gptq_output_{group_size}"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text(f'{{"bits": 4, "group_size": {group_size}}}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result.format == "gptq"


class TestGPTQCalibrationData:
    """Tests for GPTQ calibration data handling."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
        )

    @pytest.mark.integration
    def test_gptq_with_custom_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test GPTQ quantization with custom calibration data."""
        # Create calibration data file
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('["sample1", "sample2", "sample3"]')

        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=128,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_load_calibration_data") as mock_calib:
                with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                    mock_calib.return_value = ["sample1", "sample2", "sample3"]

                    output_dir = temp_dir / "gptq_output"
                    output_dir.mkdir()
                    (output_dir / "config.json").write_text('{"model_type": "llama"}')
                    (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                    (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                    mock_quantize.return_value = str(output_dir)

                    result = quantizer.quantize()

                    mock_calib.assert_called_once()
                    assert result.format == "gptq"


class TestGPTQOutputValidation:
    """Tests for GPTQ output validation."""

    @pytest.mark.integration
    def test_validate_gptq_directory_structure(self, temp_dir: Path) -> None:
        """Test GPTQ output directory validation."""
        # Create valid GPTQ output structure
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')
        (gptq_dir / "quantize_config.json").write_text('{"bits": 4}')
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_output(str(gptq_dir), "gptq")
        assert result.is_valid

    @pytest.mark.integration
    def test_validate_gptq_missing_config(self, temp_dir: Path) -> None:
        """Test GPTQ validation fails without config.json."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)
        # Missing config.json

        result = validate_output(str(gptq_dir), "gptq")
        assert not result.is_valid
        assert "config.json" in result.error_message

    @pytest.mark.integration
    def test_validate_gptq_missing_weights(self, temp_dir: Path) -> None:
        """Test GPTQ validation fails without weight files."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')
        # Missing weight files

        result = validate_output(str(gptq_dir), "gptq")
        assert not result.is_valid
        assert "weight" in result.error_message.lower()

    @pytest.mark.integration
    def test_validate_gptq_not_directory(self, temp_dir: Path) -> None:
        """Test GPTQ validation fails for file instead of directory."""
        gptq_file = temp_dir / "not_a_dir.gptq"
        gptq_file.write_bytes(b"\x00" * 100)

        result = validate_output(str(gptq_file), "gptq")
        assert not result.is_valid
        assert "directory" in result.error_message.lower()


class TestGPTQQuantizerRegistry:
    """Tests for GPTQ quantizer registration."""

    @pytest.mark.integration
    def test_gptq_quantizer_registered(self) -> None:
        """Test GPTQQuantizer is registered in quantizer registry."""
        quantizer_class = get_quantizer(OutputFormat.GPTQ)
        assert quantizer_class is not None
        assert quantizer_class == GPTQQuantizer

    @pytest.mark.integration
    def test_gptq_quantizer_from_string(self) -> None:
        """Test getting GPTQ quantizer from string format."""
        quantizer_class = get_quantizer(OutputFormat.GPTQ)
        assert quantizer_class is not None
