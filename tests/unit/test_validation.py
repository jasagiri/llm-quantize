"""Unit tests for validation utilities."""

from pathlib import Path

import pytest

from llm_quantize.lib.validation import (
    ValidationResult,
    get_file_size,
    update_validation_status,
    validate_output,
    validate_pruned_safetensors,
)
from llm_quantize.models import OutputFormat, QuantizedModel, ValidationStatus


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult.valid()
        assert result.is_valid is True
        assert result.error_message == ""

    def test_invalid_result(self) -> None:
        """Test creating an invalid result."""
        result = ValidationResult.invalid("Test error")
        assert result.is_valid is False
        assert result.error_message == "Test error"


class TestValidateOutput:
    """Tests for validate_output function."""

    def test_path_does_not_exist(self, temp_dir: Path) -> None:
        """Test validation when path does not exist."""
        result = validate_output(str(temp_dir / "nonexistent"), "gguf")
        assert result.is_valid is False
        assert "does not exist" in result.error_message

    def test_unknown_format_string(self, temp_dir: Path) -> None:
        """Test validation with unknown format string."""
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"\x00" * 100)

        result = validate_output(str(test_file), "unknown_format")
        assert result.is_valid is False
        assert "Unknown output format" in result.error_message

    def test_format_string_conversion(self, temp_dir: Path) -> None:
        """Test format string is converted to OutputFormat."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = validate_output(str(test_file), "gguf")
        assert result.is_valid is True

    def test_format_enum(self, temp_dir: Path) -> None:
        """Test validation with OutputFormat enum."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = validate_output(str(test_file), OutputFormat.GGUF)
        assert result.is_valid is True


class TestValidateGGUF:
    """Tests for GGUF validation."""

    def test_valid_gguf(self, temp_dir: Path) -> None:
        """Test valid GGUF file passes validation."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = validate_output(str(test_file), "gguf")
        assert result.is_valid is True

    def test_gguf_must_be_file(self, temp_dir: Path) -> None:
        """Test GGUF output must be a file, not directory."""
        result = validate_output(str(temp_dir), "gguf")
        assert result.is_valid is False
        assert "must be a file" in result.error_message

    def test_gguf_wrong_extension(self, temp_dir: Path) -> None:
        """Test GGUF file must have .gguf extension."""
        test_file = temp_dir / "model.bin"
        test_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = validate_output(str(test_file), "gguf")
        assert result.is_valid is False
        assert ".gguf extension" in result.error_message

    def test_gguf_empty_file(self, temp_dir: Path) -> None:
        """Test empty GGUF file fails validation."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"")

        result = validate_output(str(test_file), "gguf")
        assert result.is_valid is False
        assert "empty" in result.error_message

    def test_gguf_invalid_magic(self, temp_dir: Path) -> None:
        """Test GGUF file with invalid magic bytes fails."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"XXXX" + b"\x00" * 100)

        result = validate_output(str(test_file), "gguf")
        assert result.is_valid is False
        assert "Invalid GGUF magic" in result.error_message


class TestValidateAWQ:
    """Tests for AWQ validation."""

    def test_valid_awq(self, temp_dir: Path) -> None:
        """Test valid AWQ directory passes validation."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "config.json").write_text('{"model_type": "llama"}')
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_output(str(awq_dir), "awq")
        assert result.is_valid is True

    def test_awq_must_be_directory(self, temp_dir: Path) -> None:
        """Test AWQ output must be a directory."""
        test_file = temp_dir / "model.bin"
        test_file.write_bytes(b"\x00" * 100)

        result = validate_output(str(test_file), "awq")
        assert result.is_valid is False
        assert "must be a directory" in result.error_message

    def test_awq_missing_config(self, temp_dir: Path) -> None:
        """Test AWQ without config.json fails."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_output(str(awq_dir), "awq")
        assert result.is_valid is False
        assert "missing required files" in result.error_message

    def test_awq_missing_weights(self, temp_dir: Path) -> None:
        """Test AWQ without weight files fails."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "config.json").write_text('{"model_type": "llama"}')

        result = validate_output(str(awq_dir), "awq")
        assert result.is_valid is False
        assert "missing weight files" in result.error_message


class TestValidateGPTQ:
    """Tests for GPTQ validation."""

    def test_valid_gptq(self, temp_dir: Path) -> None:
        """Test valid GPTQ directory passes validation."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')
        (gptq_dir / "quantize_config.json").write_text('{"bits": 4}')
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_output(str(gptq_dir), "gptq")
        assert result.is_valid is True

    def test_gptq_must_be_directory(self, temp_dir: Path) -> None:
        """Test GPTQ output must be a directory."""
        test_file = temp_dir / "model.bin"
        test_file.write_bytes(b"\x00" * 100)

        result = validate_output(str(test_file), "gptq")
        assert result.is_valid is False
        assert "must be a directory" in result.error_message

    def test_gptq_without_quantize_config(self, temp_dir: Path) -> None:
        """Test GPTQ without quantize_config.json still passes."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_output(str(gptq_dir), "gptq")
        assert result.is_valid is True

    def test_gptq_missing_weights(self, temp_dir: Path) -> None:
        """Test GPTQ without weight files fails."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')

        result = validate_output(str(gptq_dir), "gptq")
        assert result.is_valid is False
        assert "missing weight files" in result.error_message


class TestUpdateValidationStatus:
    """Tests for update_validation_status function."""

    def test_update_valid_status(self, temp_dir: Path) -> None:
        """Test updating with valid status."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(test_file),
            format="gguf",
            quantization_level="Q4_K_M",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
        )

        updated = update_validation_status(model, OutputFormat.GGUF)
        assert updated.validation_status == ValidationStatus.VALID

    def test_update_invalid_status(self, temp_dir: Path) -> None:
        """Test updating with invalid status."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"XXXX" + b"\x00" * 100)  # Invalid magic

        model = QuantizedModel(
            output_path=str(test_file),
            format="gguf",
            quantization_level="Q4_K_M",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
        )

        updated = update_validation_status(model, OutputFormat.GGUF)
        assert updated.validation_status == ValidationStatus.INVALID
        assert "validation_error" in updated.quantization_metadata

    def test_update_invalid_with_no_metadata(self, temp_dir: Path) -> None:
        """Test updating invalid when metadata is None."""
        test_file = temp_dir / "model.gguf"
        test_file.write_bytes(b"XXXX" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(test_file),
            format="gguf",
            quantization_level="Q4_K_M",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
            quantization_metadata=None,
        )

        updated = update_validation_status(model, OutputFormat.GGUF)
        assert updated.validation_status == ValidationStatus.INVALID
        assert updated.quantization_metadata is not None
        assert "validation_error" in updated.quantization_metadata


class TestValidatePrunedSafetensors:
    """Tests for validate_pruned_safetensors function."""

    def test_valid_pruned_model(self, temp_dir: Path) -> None:
        """Test valid pruned model directory passes validation."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"num_hidden_layers": 56, "hidden_size": 5120}'
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is True

    def test_must_be_directory(self, temp_dir: Path) -> None:
        """Test pruned model must be a directory."""
        f = temp_dir / "not_a_dir.bin"
        f.write_bytes(b"\x00")

        result = validate_pruned_safetensors(f)
        assert result.is_valid is False
        assert "must be a directory" in result.error_message

    def test_missing_config(self, temp_dir: Path) -> None:
        """Test missing config.json fails."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is False
        assert "config.json" in result.error_message

    def test_missing_required_config_fields(self, temp_dir: Path) -> None:
        """Test config.json without required fields fails."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"vocab_size": 1000}')
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is False
        assert "missing required fields" in result.error_message

    def test_missing_weight_files(self, temp_dir: Path) -> None:
        """Test directory without safetensors files fails."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"num_hidden_layers": 56, "hidden_size": 5120}'
        )

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is False
        assert "missing safetensors" in result.error_message

    def test_invalid_config_json(self, temp_dir: Path) -> None:
        """Test malformed config.json fails."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{not valid json")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is False
        assert "Invalid config.json" in result.error_message

    def test_with_pruning_plan(self, temp_dir: Path) -> None:
        """Test valid model with pruning_plan.json passes."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"num_hidden_layers": 56, "hidden_size": 5120}'
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)
        (model_dir / "pruning_plan.json").write_text(
            '{"mlp_new_intermediate_size": 12224}'
        )

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is True

    def test_invalid_pruning_plan(self, temp_dir: Path) -> None:
        """Test invalid pruning_plan.json fails."""
        model_dir = temp_dir / "pruned_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"num_hidden_layers": 56, "hidden_size": 5120}'
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)
        (model_dir / "pruning_plan.json").write_text("{broken json")

        result = validate_pruned_safetensors(model_dir)
        assert result.is_valid is False
        assert "pruning_plan.json" in result.error_message


class TestGetFileSize:
    """Tests for get_file_size function."""

    def test_file_size(self, temp_dir: Path) -> None:
        """Test getting size of a file."""
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"\x00" * 1000)

        size = get_file_size(test_file)
        assert size == 1000

    def test_directory_size(self, temp_dir: Path) -> None:
        """Test getting size of a directory."""
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.bin").write_bytes(b"\x00" * 100)
        (test_dir / "file2.bin").write_bytes(b"\x00" * 200)

        size = get_file_size(test_dir)
        assert size == 300

    def test_nonexistent_path(self, temp_dir: Path) -> None:
        """Test getting size of nonexistent path."""
        size = get_file_size(temp_dir / "nonexistent")
        assert size == 0
