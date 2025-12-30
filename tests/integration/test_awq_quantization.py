"""Integration tests for AWQ quantization workflow.

These tests verify the complete AWQ quantization pipeline:
- Model loading → AWQ quantization → Output validation
- Calibration data handling
- Checkpoint/resume functionality
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.quantizers import get_quantizer
from llm_quantize.lib.quantizers.awq import AWQQuantizer
from llm_quantize.lib.validation import validate_output
from llm_quantize.models import (
    AWQ_QUANT_TYPES,
    ModelType,
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestAWQQuantizationWorkflow:
    """Integration tests for AWQ quantization workflow."""

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
    def awq_config(self, temp_dir: Path) -> QuantizationConfig:
        """Create AWQ quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_samples=128,
        )

    @pytest.mark.integration
    def test_awq_quantizer_initialization(
        self, source_model: SourceModel, awq_config: QuantizationConfig
    ) -> None:
        """Test AWQQuantizer can be initialized."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=awq_config,
        )

        assert quantizer.source_model == source_model
        assert quantizer.config == awq_config

    @pytest.mark.integration
    def test_awq_quantizer_supports_required_levels(self) -> None:
        """Test AWQQuantizer supports all required quantization levels."""
        levels = AWQQuantizer.get_supported_levels()

        # AWQ typically supports 4-bit quantization
        assert "4bit" in levels

    @pytest.mark.integration
    def test_awq_quantizer_estimates_output_size(
        self, source_model: SourceModel, awq_config: QuantizationConfig
    ) -> None:
        """Test AWQQuantizer can estimate output size."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=awq_config,
        )

        estimated_size = quantizer.estimate_output_size()

        # Should return positive estimate
        assert estimated_size > 0

        # AWQ 4-bit should be ~25% of fp16 size (4/16 bits)
        original_size = source_model.parameter_count * 2  # fp16
        assert estimated_size < original_size

    @pytest.mark.integration
    def test_awq_full_quantization(
        self, source_model: SourceModel, awq_config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test full AWQ quantization workflow."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=awq_config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                mock_load.return_value = MagicMock()

                # Create mock output directory
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result.format == "awq"
                assert result.quantization_level == "4bit"
                assert result.output_path == str(output_dir)

    @pytest.mark.integration
    def test_awq_quantization_creates_valid_output(
        self, source_model: SourceModel, awq_config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test AWQ quantization creates valid output structure."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=awq_config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                # Create proper AWQ output structure
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                # Validate output
                validation = validate_output(result.output_path, "awq")
                assert validation.is_valid


class TestAWQCalibrationData:
    """Tests for AWQ calibration data handling."""

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
    def test_awq_with_custom_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test AWQ quantization with custom calibration data."""
        # Create calibration data file
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('["sample1", "sample2", "sample3"]')

        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=128,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_load_calibration_data") as mock_calib:
                with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                    mock_calib.return_value = ["sample1", "sample2", "sample3"]

                    output_dir = temp_dir / "awq_output"
                    output_dir.mkdir()
                    (output_dir / "config.json").write_text('{"model_type": "llama"}')
                    (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                    mock_quantize.return_value = str(output_dir)

                    result = quantizer.quantize()

                    # Verify calibration data was loaded
                    mock_calib.assert_called_once()
                    assert result.format == "awq"

    @pytest.mark.integration
    def test_awq_uses_default_calibration_without_custom_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test AWQ uses default calibration when no custom data provided."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_samples=64,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_get_default_calibration_data") as mock_default:
                with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                    mock_default.return_value = ["default sample"] * 64

                    output_dir = temp_dir / "awq_output"
                    output_dir.mkdir()
                    (output_dir / "config.json").write_text('{"model_type": "llama"}')
                    (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                    mock_quantize.return_value = str(output_dir)

                    result = quantizer.quantize()

                    # Verify default calibration was used
                    mock_default.assert_called_once()


class TestAWQOutputValidation:
    """Tests for AWQ output validation."""

    @pytest.mark.integration
    def test_validate_awq_directory_structure(self, temp_dir: Path) -> None:
        """Test AWQ output directory validation."""
        # Create valid AWQ output structure
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "config.json").write_text('{"model_type": "llama"}')
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_output(str(awq_dir), "awq")
        assert result.is_valid

    @pytest.mark.integration
    def test_validate_awq_missing_config(self, temp_dir: Path) -> None:
        """Test AWQ validation fails without config.json."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)
        # Missing config.json

        result = validate_output(str(awq_dir), "awq")
        assert not result.is_valid
        assert "config.json" in result.error_message

    @pytest.mark.integration
    def test_validate_awq_missing_weights(self, temp_dir: Path) -> None:
        """Test AWQ validation fails without weight files."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "config.json").write_text('{"model_type": "llama"}')
        # Missing weight files

        result = validate_output(str(awq_dir), "awq")
        assert not result.is_valid
        assert "weight" in result.error_message.lower()

    @pytest.mark.integration
    def test_validate_awq_not_directory(self, temp_dir: Path) -> None:
        """Test AWQ validation fails for file instead of directory."""
        awq_file = temp_dir / "not_a_dir.awq"
        awq_file.write_bytes(b"\x00" * 100)

        result = validate_output(str(awq_file), "awq")
        assert not result.is_valid
        assert "directory" in result.error_message.lower()


class TestAWQQuantizerRegistry:
    """Tests for AWQ quantizer registration."""

    @pytest.mark.integration
    def test_awq_quantizer_registered(self) -> None:
        """Test AWQQuantizer is registered in quantizer registry."""
        quantizer_class = get_quantizer(OutputFormat.AWQ)
        assert quantizer_class is not None
        assert quantizer_class == AWQQuantizer

    @pytest.mark.integration
    def test_awq_quantizer_from_string(self) -> None:
        """Test getting AWQ quantizer from string format."""
        quantizer_class = get_quantizer(OutputFormat.AWQ)
        assert quantizer_class is not None
