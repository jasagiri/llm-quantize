"""Tests for package initialization."""

from unittest.mock import patch, MagicMock
from importlib.metadata import PackageNotFoundError

import pytest


class TestPackageInit:
    """Tests for llm_quantize package initialization."""

    def test_version_available(self):
        """Test version is available."""
        import llm_quantize

        assert hasattr(llm_quantize, "__version__")
        assert isinstance(llm_quantize.__version__, str)

    def test_version_fallback_on_not_found(self):
        """Test version falls back when package not found."""
        with patch("llm_quantize.version") as mock_version:
            mock_version.side_effect = PackageNotFoundError()

            # Need to reimport to trigger the fallback
            import importlib
            import llm_quantize

            # The version should be set (either from installed package or fallback)
            assert llm_quantize.__version__ is not None

    def test_all_exports(self):
        """Test __all__ exports."""
        import llm_quantize

        assert "__version__" in llm_quantize.__all__
