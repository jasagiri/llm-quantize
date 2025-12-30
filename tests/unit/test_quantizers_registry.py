"""Tests for quantizer registry functions."""

from llm_quantize.lib.quantizers import (
    BaseQuantizer,
    get_available_formats,
    get_quantizer,
    register_quantizer,
)
from llm_quantize.lib.quantizers.gguf import GGUFQuantizer
from llm_quantize.models import OutputFormat


class TestGetQuantizer:
    """Tests for get_quantizer function."""

    def test_get_quantizer_gguf(self) -> None:
        """Test getting GGUF quantizer."""
        quantizer = get_quantizer(OutputFormat.GGUF)
        assert quantizer == GGUFQuantizer

    def test_get_quantizer_awq(self) -> None:
        """Test getting AWQ quantizer."""
        from llm_quantize.lib.quantizers.awq import AWQQuantizer

        quantizer = get_quantizer(OutputFormat.AWQ)
        assert quantizer == AWQQuantizer

    def test_get_quantizer_gptq(self) -> None:
        """Test getting GPTQ quantizer."""
        from llm_quantize.lib.quantizers.gptq import GPTQQuantizer

        quantizer = get_quantizer(OutputFormat.GPTQ)
        assert quantizer == GPTQQuantizer


class TestGetAvailableFormats:
    """Tests for get_available_formats function."""

    def test_get_available_formats_returns_list(self) -> None:
        """Test get_available_formats returns a list."""
        formats = get_available_formats()
        assert isinstance(formats, list)

    def test_get_available_formats_contains_all_formats(self) -> None:
        """Test get_available_formats contains all expected formats."""
        formats = get_available_formats()
        assert OutputFormat.GGUF in formats
        assert OutputFormat.AWQ in formats
        assert OutputFormat.GPTQ in formats


class TestRegisterQuantizer:
    """Tests for register_quantizer function."""

    def test_register_quantizer(self) -> None:
        """Test registering a new quantizer."""
        from llm_quantize.lib.quantizers import _QUANTIZERS

        # Create a mock quantizer class
        class MockQuantizer(BaseQuantizer):
            def quantize(self):
                pass

            @classmethod
            def get_supported_levels(cls):
                return []

            def estimate_output_size(self):
                return 0

        # Store original to restore later
        original = _QUANTIZERS.get(OutputFormat.GGUF)

        try:
            # Register the mock
            register_quantizer(OutputFormat.GGUF, MockQuantizer)

            # Verify it was registered
            assert get_quantizer(OutputFormat.GGUF) == MockQuantizer
        finally:
            # Restore original
            if original:
                _QUANTIZERS[OutputFormat.GGUF] = original
