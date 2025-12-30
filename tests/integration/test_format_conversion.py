"""Integration tests for format conversion.

These tests verify the complete conversion pipeline:
- Format detection
- Conversion between formats
- Quality degradation handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.converter import (
    ConversionResult,
    convert_format,
    detect_format,
    get_supported_conversions,
    is_conversion_supported,
    is_lossy_conversion,
)


class TestFormatDetection:
    """Tests for automatic format detection."""

    @pytest.mark.integration
    def test_detect_gguf_from_extension(self, temp_dir: Path) -> None:
        """Test GGUF format detection from file extension."""
        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)

        detected = detect_format(str(gguf_file))
        assert detected == "gguf"

    @pytest.mark.integration
    def test_detect_gguf_from_magic_bytes(self, temp_dir: Path) -> None:
        """Test GGUF format detection from magic bytes."""
        gguf_file = temp_dir / "model.bin"  # Wrong extension
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)

        detected = detect_format(str(gguf_file))
        assert detected == "gguf"

    @pytest.mark.integration
    def test_detect_awq_from_config(self, temp_dir: Path) -> None:
        """Test AWQ format detection from config.json."""
        awq_dir = temp_dir / "awq_model"
        awq_dir.mkdir()
        (awq_dir / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')
        (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        detected = detect_format(str(awq_dir))
        assert detected == "awq"

    @pytest.mark.integration
    def test_detect_gptq_from_config(self, temp_dir: Path) -> None:
        """Test GPTQ format detection from quantize_config.json."""
        gptq_dir = temp_dir / "gptq_model"
        gptq_dir.mkdir()
        (gptq_dir / "config.json").write_text('{"model_type": "llama"}')
        (gptq_dir / "quantize_config.json").write_text('{"bits": 4, "group_size": 128}')
        (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        detected = detect_format(str(gptq_dir))
        assert detected == "gptq"

    @pytest.mark.integration
    def test_detect_unknown_format(self, temp_dir: Path) -> None:
        """Test unknown format detection returns None."""
        unknown_file = temp_dir / "unknown.bin"
        unknown_file.write_bytes(b"\x00" * 100)

        detected = detect_format(str(unknown_file))
        assert detected is None


class TestConversionSupport:
    """Tests for conversion support checking."""

    @pytest.mark.integration
    def test_get_supported_conversions(self) -> None:
        """Test getting list of supported conversions."""
        conversions = get_supported_conversions()

        assert isinstance(conversions, list)
        assert len(conversions) > 0

        # Each conversion should be a tuple of (source, target)
        for conversion in conversions:
            assert len(conversion) == 2
            assert isinstance(conversion[0], str)
            assert isinstance(conversion[1], str)

    @pytest.mark.integration
    def test_gguf_to_awq_supported(self) -> None:
        """Test GGUF to AWQ conversion is supported."""
        assert is_conversion_supported("gguf", "awq")

    @pytest.mark.integration
    def test_awq_to_gguf_supported(self) -> None:
        """Test AWQ to GGUF conversion is supported."""
        assert is_conversion_supported("awq", "gguf")

    @pytest.mark.integration
    def test_same_format_not_supported(self) -> None:
        """Test conversion to same format is not supported."""
        assert not is_conversion_supported("gguf", "gguf")
        assert not is_conversion_supported("awq", "awq")
        assert not is_conversion_supported("gptq", "gptq")


class TestLossyConversionDetection:
    """Tests for lossy conversion detection."""

    @pytest.mark.integration
    def test_gguf_to_awq_is_lossy(self) -> None:
        """Test GGUF to AWQ is a lossy conversion."""
        # Converting from one quantization to another typically loses quality
        is_lossy = is_lossy_conversion("gguf", "awq")
        assert is_lossy is True

    @pytest.mark.integration
    def test_awq_to_gguf_is_lossy(self) -> None:
        """Test AWQ to GGUF is a lossy conversion."""
        is_lossy = is_lossy_conversion("awq", "gguf")
        assert is_lossy is True


class TestFormatConversion:
    """Tests for actual format conversion."""

    @pytest.mark.integration
    def test_convert_gguf_to_awq(self, temp_dir: Path) -> None:
        """Test converting GGUF to AWQ format."""
        # Create source GGUF file
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.lib.converter._convert_gguf_to_awq") as mock_convert:
            awq_output = output_dir / "model-awq"
            awq_output.mkdir()
            (awq_output / "config.json").write_text('{"model_type": "llama"}')
            (awq_output / "model.safetensors").write_bytes(b"\x00" * 100)

            mock_convert.return_value = str(awq_output)

            result = convert_format(
                source_path=str(source_file),
                target_format="awq",
                output_dir=str(output_dir),
                force=True,  # Need force since mock creates output first
            )

            assert isinstance(result, ConversionResult)
            assert result.source_format == "gguf"
            assert result.target_format == "awq"
            assert result.output_path == str(awq_output)

    @pytest.mark.integration
    def test_convert_awq_to_gguf(self, temp_dir: Path) -> None:
        """Test converting AWQ to GGUF format."""
        # Create source AWQ directory
        source_dir = temp_dir / "awq_model"
        source_dir.mkdir()
        (source_dir / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')
        (source_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.lib.converter._convert_awq_to_gguf") as mock_convert:
            gguf_output = output_dir / "model.gguf"
            gguf_output.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_convert.return_value = str(gguf_output)

            result = convert_format(
                source_path=str(source_dir),
                target_format="gguf",
                output_dir=str(output_dir),
            )

            assert isinstance(result, ConversionResult)
            assert result.source_format == "awq"
            assert result.target_format == "gguf"

    @pytest.mark.integration
    def test_conversion_result_includes_metadata(self, temp_dir: Path) -> None:
        """Test conversion result includes relevant metadata."""
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        with patch("llm_quantize.lib.converter._convert_gguf_to_awq") as mock_convert:
            awq_output = output_dir / "model-awq"
            awq_output.mkdir()
            (awq_output / "config.json").write_text('{"model_type": "llama"}')
            (awq_output / "model.safetensors").write_bytes(b"\x00" * 100)

            mock_convert.return_value = str(awq_output)

            result = convert_format(
                source_path=str(source_file),
                target_format="awq",
                output_dir=str(output_dir),
                force=True,  # Need force since mock creates output first
            )

            assert hasattr(result, "source_format")
            assert hasattr(result, "target_format")
            assert hasattr(result, "output_path")
            assert hasattr(result, "file_size")
            assert hasattr(result, "is_lossy")


class TestConversionErrors:
    """Tests for conversion error handling."""

    @pytest.mark.integration
    def test_convert_nonexistent_source(self, temp_dir: Path) -> None:
        """Test error when source file doesn't exist."""
        with pytest.raises(ValueError, match="not found|does not exist"):
            convert_format(
                source_path=str(temp_dir / "nonexistent.gguf"),
                target_format="awq",
                output_dir=str(temp_dir),
            )

    @pytest.mark.integration
    def test_convert_unknown_source_format(self, temp_dir: Path) -> None:
        """Test error when source format cannot be detected."""
        unknown_file = temp_dir / "unknown.bin"
        unknown_file.write_bytes(b"\x00" * 100)

        with pytest.raises(ValueError, match="detect|unknown|format"):
            convert_format(
                source_path=str(unknown_file),
                target_format="awq",
                output_dir=str(temp_dir),
            )

    @pytest.mark.integration
    def test_convert_unsupported_path(self, temp_dir: Path) -> None:
        """Test error for unsupported conversion path."""
        # Create a file that will be detected as GGUF
        source_file = temp_dir / "model.gguf"
        source_file.write_bytes(b"GGUF" + b"\x00" * 100)

        with patch("llm_quantize.lib.converter.is_conversion_supported") as mock_supported:
            mock_supported.return_value = False

            with pytest.raises(ValueError, match="not supported|unsupported"):
                convert_format(
                    source_path=str(source_file),
                    target_format="unknown_format",
                    output_dir=str(temp_dir),
                )
