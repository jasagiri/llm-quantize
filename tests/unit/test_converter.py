"""Unit tests for format converter."""

import json
from pathlib import Path

import pytest

from llm_quantize.lib.converter import (
    ConversionResult,
    detect_format,
    get_supported_conversions,
    is_conversion_supported,
    is_lossy_conversion,
)


class TestFormatDetection:
    """Tests for format detection."""

    def test_detect_gguf_from_extension(self, temp_dir: Path) -> None:
        """Test GGUF detection from .gguf extension."""
        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = detect_format(str(gguf_file))
        assert result == "gguf"

    def test_detect_gguf_from_magic_bytes(self, temp_dir: Path) -> None:
        """Test GGUF detection from magic bytes without .gguf extension."""
        gguf_file = temp_dir / "model.bin"
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)

        result = detect_format(str(gguf_file))
        assert result == "gguf"

    def test_detect_awq_from_config(self, temp_dir: Path) -> None:
        """Test AWQ detection from config.json."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        config = {"quantization_config": {"quant_method": "awq"}}
        (awq_dir / "config.json").write_text(json.dumps(config))
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_format(str(awq_dir))
        assert result == "awq"

    def test_detect_gptq_from_quantize_config(self, temp_dir: Path) -> None:
        """Test GPTQ detection from quantize_config.json."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')
        (gptq_dir / "quantize_config.json").write_text('{"bits": 4}')
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_format(str(gptq_dir))
        assert result == "gptq"

    def test_detect_nonexistent_path(self, temp_dir: Path) -> None:
        """Test detection returns None for nonexistent path."""
        result = detect_format(str(temp_dir / "nonexistent"))
        assert result is None

    def test_detect_unknown_format(self, temp_dir: Path) -> None:
        """Test detection returns None for unknown format."""
        unknown_file = temp_dir / "unknown.bin"
        unknown_file.write_bytes(b"\x00" * 100)

        result = detect_format(str(unknown_file))
        assert result is None


class TestConversionSupport:
    """Tests for conversion support checking."""

    def test_get_supported_conversions_returns_list(self) -> None:
        """Test get_supported_conversions returns a list."""
        conversions = get_supported_conversions()
        assert isinstance(conversions, list)

    def test_get_supported_conversions_has_entries(self) -> None:
        """Test get_supported_conversions has entries."""
        conversions = get_supported_conversions()
        assert len(conversions) > 0

    def test_get_supported_conversions_format(self) -> None:
        """Test each conversion is a (source, target) tuple."""
        conversions = get_supported_conversions()
        for conversion in conversions:
            assert len(conversion) == 2
            assert isinstance(conversion[0], str)
            assert isinstance(conversion[1], str)

    def test_gguf_to_awq_supported(self) -> None:
        """Test GGUF to AWQ is supported."""
        assert is_conversion_supported("gguf", "awq") is True

    def test_awq_to_gguf_supported(self) -> None:
        """Test AWQ to GGUF is supported."""
        assert is_conversion_supported("awq", "gguf") is True

    def test_gguf_to_gptq_supported(self) -> None:
        """Test GGUF to GPTQ is supported."""
        assert is_conversion_supported("gguf", "gptq") is True

    def test_gptq_to_gguf_supported(self) -> None:
        """Test GPTQ to GGUF is supported."""
        assert is_conversion_supported("gptq", "gguf") is True

    def test_awq_to_gptq_supported(self) -> None:
        """Test AWQ to GPTQ is supported."""
        assert is_conversion_supported("awq", "gptq") is True

    def test_gptq_to_awq_supported(self) -> None:
        """Test GPTQ to AWQ is supported."""
        assert is_conversion_supported("gptq", "awq") is True

    def test_same_format_not_supported(self) -> None:
        """Test same format conversion is not supported."""
        assert is_conversion_supported("gguf", "gguf") is False
        assert is_conversion_supported("awq", "awq") is False
        assert is_conversion_supported("gptq", "gptq") is False

    def test_case_insensitive(self) -> None:
        """Test conversion support check is case insensitive."""
        assert is_conversion_supported("GGUF", "AWQ") is True
        assert is_conversion_supported("Gguf", "Awq") is True


class TestLossyConversion:
    """Tests for lossy conversion detection."""

    def test_all_cross_format_conversions_are_lossy(self) -> None:
        """Test that converting between different formats is lossy."""
        formats = ["gguf", "awq", "gptq"]

        for source in formats:
            for target in formats:
                if source != target:
                    assert is_lossy_conversion(source, target) is True

    def test_gguf_to_awq_is_lossy(self) -> None:
        """Test GGUF to AWQ is lossy."""
        assert is_lossy_conversion("gguf", "awq") is True

    def test_awq_to_gguf_is_lossy(self) -> None:
        """Test AWQ to GGUF is lossy."""
        assert is_lossy_conversion("awq", "gguf") is True


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_conversion_result_creation(self) -> None:
        """Test ConversionResult can be created."""
        result = ConversionResult(
            output_path="/path/to/output",
            source_format="gguf",
            target_format="awq",
            file_size=1000,
            is_lossy=True,
        )

        assert result.output_path == "/path/to/output"
        assert result.source_format == "gguf"
        assert result.target_format == "awq"
        assert result.file_size == 1000
        assert result.is_lossy is True
        assert result.warning_message is None

    def test_conversion_result_with_warning(self) -> None:
        """Test ConversionResult with warning message."""
        result = ConversionResult(
            output_path="/path/to/output",
            source_format="gguf",
            target_format="awq",
            file_size=1000,
            is_lossy=True,
            warning_message="Quality may be degraded",
        )

        assert result.warning_message == "Quality may be degraded"


class TestConvertFormat:
    """Tests for convert_format function."""

    def test_convert_gguf_to_awq(self, temp_dir: Path) -> None:
        """Test converting GGUF to AWQ."""
        from llm_quantize.lib.converter import convert_format

        # Create source GGUF
        source = temp_dir / "model.gguf"
        source.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="awq",
            output_dir=str(output_dir),
        )

        assert isinstance(result, ConversionResult)
        assert result.source_format == "gguf"
        assert result.target_format == "awq"
        assert Path(result.output_path).exists()

    def test_convert_awq_to_gguf(self, temp_dir: Path) -> None:
        """Test converting AWQ to GGUF."""
        from llm_quantize.lib.converter import convert_format

        # Create source AWQ
        source = temp_dir / "awq_model"
        source.mkdir()
        (source / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')
        (source / "model.safetensors").write_bytes(b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="gguf",
            output_dir=str(output_dir),
        )

        assert result.source_format == "awq"
        assert result.target_format == "gguf"
        assert Path(result.output_path).exists()

    def test_convert_with_custom_name(self, temp_dir: Path) -> None:
        """Test converting with custom output name."""
        from llm_quantize.lib.converter import convert_format

        source = temp_dir / "model.gguf"
        source.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="awq",
            output_dir=str(output_dir),
            output_name="custom_name",
        )

        assert "custom_name" in result.output_path

    def test_convert_nonexistent_source_raises(self, temp_dir: Path) -> None:
        """Test error when source doesn't exist."""
        from llm_quantize.lib.converter import convert_format

        with pytest.raises(ValueError, match="not found"):
            convert_format(
                source_path=str(temp_dir / "nonexistent"),
                target_format="awq",
                output_dir=str(temp_dir),
            )

    def test_convert_unknown_format_raises(self, temp_dir: Path) -> None:
        """Test error when source format is unknown."""
        from llm_quantize.lib.converter import convert_format

        source = temp_dir / "unknown.bin"
        source.write_bytes(b"\x00" * 100)

        with pytest.raises(ValueError, match="detect|format"):
            convert_format(
                source_path=str(source),
                target_format="awq",
                output_dir=str(temp_dir),
            )

    def test_convert_existing_output_without_force_raises(self, temp_dir: Path) -> None:
        """Test error when output exists without --force."""
        from llm_quantize.lib.converter import convert_format

        source = temp_dir / "model.gguf"
        source.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create existing output
        existing = output_dir / "model-awq"
        existing.mkdir()
        (existing / "config.json").write_text("{}")

        with pytest.raises(ValueError, match="exists|force"):
            convert_format(
                source_path=str(source),
                target_format="awq",
                output_dir=str(output_dir),
                force=False,
            )

    def test_convert_with_force_overwrites(self, temp_dir: Path) -> None:
        """Test --force overwrites existing output."""
        from llm_quantize.lib.converter import convert_format

        source = temp_dir / "model.gguf"
        source.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create existing output
        existing = output_dir / "model-awq"
        existing.mkdir()
        (existing / "old_file.txt").write_text("old content")

        result = convert_format(
            source_path=str(source),
            target_format="awq",
            output_dir=str(output_dir),
            force=True,
        )

        assert Path(result.output_path).exists()

    def test_convert_gptq_to_gguf(self, temp_dir: Path) -> None:
        """Test converting GPTQ to GGUF."""
        from llm_quantize.lib.converter import convert_format

        # Create source GPTQ
        source = temp_dir / "gptq_model"
        source.mkdir()
        (source / "config.json").write_text('{"model_type": "llama"}')
        (source / "quantize_config.json").write_text('{"bits": 4}')
        (source / "model.safetensors").write_bytes(b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="gguf",
            output_dir=str(output_dir),
        )

        assert result.source_format == "gptq"
        assert result.target_format == "gguf"
        assert Path(result.output_path).exists()
        assert result.output_path.endswith(".gguf")

    def test_convert_gguf_to_gptq(self, temp_dir: Path) -> None:
        """Test converting GGUF to GPTQ."""
        from llm_quantize.lib.converter import convert_format

        source = temp_dir / "model.gguf"
        source.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="gptq",
            output_dir=str(output_dir),
        )

        assert result.source_format == "gguf"
        assert result.target_format == "gptq"
        assert Path(result.output_path).exists()

    def test_convert_unsupported_path_raises(self, temp_dir: Path) -> None:
        """Test error for unsupported conversion path."""
        from llm_quantize.lib.converter import convert_format
        from unittest.mock import patch

        source = temp_dir / "model.gguf"
        source.write_bytes(b"GGUF" + b"\x00" * 100)

        # Mock is_conversion_supported to return False
        with patch("llm_quantize.lib.converter.is_conversion_supported") as mock_supported:
            mock_supported.return_value = False

            with pytest.raises(ValueError, match="not supported"):
                convert_format(
                    source_path=str(source),
                    target_format="awq",
                    output_dir=str(temp_dir),
                )

    def test_convert_awq_to_gptq(self, temp_dir: Path) -> None:
        """Test converting AWQ to GPTQ."""
        from llm_quantize.lib.converter import convert_format

        # Create source AWQ
        source = temp_dir / "awq_model"
        source.mkdir()
        (source / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')
        (source / "model.safetensors").write_bytes(b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="gptq",
            output_dir=str(output_dir),
        )

        assert result.source_format == "awq"
        assert result.target_format == "gptq"
        assert Path(result.output_path).exists()

    def test_convert_gptq_to_awq(self, temp_dir: Path) -> None:
        """Test converting GPTQ to AWQ."""
        from llm_quantize.lib.converter import convert_format

        # Create source GPTQ
        source = temp_dir / "gptq_model"
        source.mkdir()
        (source / "config.json").write_text('{"model_type": "llama"}')
        (source / "quantize_config.json").write_text('{"bits": 4}')
        (source / "model.safetensors").write_bytes(b"\x00" * 100)

        output_dir = temp_dir / "output"

        result = convert_format(
            source_path=str(source),
            target_format="awq",
            output_dir=str(output_dir),
        )

        assert result.source_format == "gptq"
        assert result.target_format == "awq"
        assert Path(result.output_path).exists()


class TestFormatDetectionEdgeCases:
    """Additional edge case tests for format detection."""

    def test_detect_gptq_from_config_quant_method(self, temp_dir: Path) -> None:
        """Test GPTQ detection from quant_method in config."""
        gptq_dir = temp_dir / "gptq_config_method"
        gptq_dir.mkdir()
        # Config with quant_method = gptq but no quantize_config.json
        config = {"quantization_config": {"quant_method": "gptq"}}
        (gptq_dir / "config.json").write_text(json.dumps(config))
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_format(str(gptq_dir))
        assert result == "gptq"

    def test_detect_awq_from_text_match(self, temp_dir: Path) -> None:
        """Test AWQ detection from text matching in config."""
        awq_dir = temp_dir / "awq_text_match"
        awq_dir.mkdir()
        # Config with awq mentioned but no explicit quant_method
        config = {"model_type": "llama", "awq_config": {"bits": 4}}
        (awq_dir / "config.json").write_text(json.dumps(config))
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_format(str(awq_dir))
        assert result == "awq"

    def test_detect_default_to_awq_for_hf_style(self, temp_dir: Path) -> None:
        """Test default to AWQ for HF-style directories without quantization config."""
        model_dir = temp_dir / "hf_style"
        model_dir.mkdir()
        # Plain HF config without explicit quantization method
        config = {"model_type": "llama", "vocab_size": 32000}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_format(str(model_dir))
        assert result == "awq"

    def test_detect_format_with_bin_file(self, temp_dir: Path) -> None:
        """Test format detection with .bin weight files."""
        model_dir = temp_dir / "bin_model"
        model_dir.mkdir()
        config = {"model_type": "llama"}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 100)

        result = detect_format(str(model_dir))
        assert result == "awq"

    def test_detect_format_config_without_weights(self, temp_dir: Path) -> None:
        """Test format detection when config exists but no weight files."""
        model_dir = temp_dir / "config_only"
        model_dir.mkdir()
        config = {"model_type": "llama"}
        (model_dir / "config.json").write_text(json.dumps(config))
        # No weight files

        result = detect_format(str(model_dir))
        assert result is None

    def test_detect_format_invalid_json_config(self, temp_dir: Path) -> None:
        """Test format detection with invalid JSON config but weight files."""
        model_dir = temp_dir / "invalid_json"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{ invalid json }")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_format(str(model_dir))
        # Falls back to awq since config.json exists and has weight files
        assert result == "awq"

    def test_detect_format_invalid_json_no_weights(self, temp_dir: Path) -> None:
        """Test format detection with invalid JSON config and no weight files."""
        model_dir = temp_dir / "invalid_json_no_weights"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{ invalid json }")
        # No weight files

        result = detect_format(str(model_dir))
        # Should return None since config is invalid and no weights
        assert result is None
