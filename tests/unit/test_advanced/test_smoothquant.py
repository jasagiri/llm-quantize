"""Tests for SmoothQuant W8A8 quantization."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from llm_quantize.models import (
    ActivationStatistics,
    OutputFormat,
    QuantizationConfig,
    RECOMMENDED_ALPHA,
    SmoothQuantConfig,
    SourceModel,
)


class TestSmoothQuantQuantizer:
    """Tests for SmoothQuantQuantizer class."""

    def create_mock_source_model(self):
        """Create a mock source model."""
        from llm_quantize.models import ModelType
        return SourceModel(
            model_path="test/model",
            model_type=ModelType.HF_HUB,
            architecture="llama",
            parameter_count=1000000,
            dtype="float16",
            num_layers=4,
            hidden_size=768,
            num_heads=12,
            vocab_size=32000,
        )

    def create_mock_config(self, tmpdir):
        """Create a mock quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="W8A8",
            output_dir=str(tmpdir),
        )

    def test_get_supported_levels(self):
        """Test supported quantization levels."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        levels = SmoothQuantQuantizer.get_supported_levels()

        assert "W8A8" in levels
        assert "smoothquant" in levels
        assert "int8" in levels

    def test_init_with_config(self):
        """Test initialization with SmoothQuant config."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            sq_config = SmoothQuantConfig(alpha=0.6)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
                smoothquant_config=sq_config,
            )

            assert quantizer.smoothquant_config.alpha == 0.6

    def test_init_default_config(self):
        """Test initialization with default config."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Should use recommended alpha for llama
            assert quantizer.smoothquant_config.alpha == 0.5

    def test_estimate_output_size(self):
        """Test output size estimation."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer.estimate_output_size()

            # W8A8 should be about 50% of FP16
            original_size = source_model.parameter_count * 2
            assert size < original_size
            assert size > original_size * 0.3  # Plus overhead

    def test_get_recommended_alpha_llama(self):
        """Test recommended alpha for llama."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            alpha = quantizer._get_recommended_alpha()

            assert alpha == 0.5

    def test_get_recommended_alpha_bloom(self):
        """Test recommended alpha for bloom."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            from llm_quantize.models import ModelType
            source_model = SourceModel(
                model_path="test/model",
                model_type=ModelType.HF_HUB,
                architecture="bloom",
                parameter_count=1000000,
                dtype="float16",
                num_layers=4,
                hidden_size=768,
                num_heads=12,
                vocab_size=32000,
            )
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            alpha = quantizer._get_recommended_alpha()

            assert alpha == 0.75

    def test_get_output_path(self):
        """Test output path generation."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()

            assert "smoothquant" in str(path)
            assert "w8a8" in str(path)

    def test_get_output_path_custom_name(self):
        """Test output path with custom name."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
                output_name="custom-model",
            )

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()

            assert "custom-model" in str(path)

    def test_generate_synthetic_statistics(self):
        """Test synthetic statistics generation."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._generate_synthetic_statistics()

            assert len(quantizer._activation_stats) > 0

    def test_create_smoothquant_plan(self):
        """Test SmoothQuant plan creation."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            output_path = Path(tmpdir) / "output"
            result = quantizer._create_smoothquant_plan(output_path)

            assert result.exists()
            with open(result) as f:
                plan = json.load(f)
            assert "SmoothQuant" in plan["quantization"]
            assert plan["status"] == "plan_only"


class TestSmoothQuantConfig:
    """Tests for SmoothQuantConfig model."""

    def test_compute_smoothing_scale(self):
        """Test smoothing scale computation."""
        config = SmoothQuantConfig(alpha=0.5)

        activation_max = np.array([1.0, 2.0, 4.0])
        weight_max = np.array([4.0, 2.0, 1.0])

        scales = config.compute_smoothing_scale("layer1", activation_max, weight_max)

        # scale = act^alpha / weight^(1-alpha)
        # For alpha=0.5: scale = sqrt(act) / sqrt(weight) = sqrt(act/weight)
        expected = np.sqrt(activation_max / weight_max)
        np.testing.assert_array_almost_equal(scales, expected)

        # Verify stored
        assert "layer1" in config.smoothing_scales

    def test_compute_smoothing_scale_with_zeros(self):
        """Test smoothing scale with zero values."""
        config = SmoothQuantConfig(alpha=0.5)

        activation_max = np.array([0.0, 1.0])
        weight_max = np.array([1.0, 0.0])

        scales = config.compute_smoothing_scale("layer1", activation_max, weight_max)

        # Should not have inf or nan
        assert not np.any(np.isnan(scales))
        assert not np.any(np.isinf(scales))

    def test_get_layer_statistics(self):
        """Test getting layer statistics."""
        config = SmoothQuantConfig()
        stats = ActivationStatistics(
            layer_name="layer1",
            mean_value=0.5,
        )
        config.layer_statistics.append(stats)

        result = config.get_layer_statistics("layer1")
        assert result == stats

        result = config.get_layer_statistics("nonexistent")
        assert result is None

    def test_get_problematic_layers(self):
        """Test identifying problematic layers."""
        config = SmoothQuantConfig()
        config.layer_statistics = [
            ActivationStatistics("layer1", outlier_ratio=0.05),
            ActivationStatistics("layer2", outlier_ratio=0.2),
            ActivationStatistics("layer3", outlier_ratio=0.01),
        ]

        problematic = config.get_problematic_layers(outlier_threshold=0.1)

        assert "layer2" in problematic
        assert "layer1" not in problematic
        assert "layer3" not in problematic


class TestCreateSmoothQuantQuantizer:
    """Tests for create_smoothquant_quantizer factory function."""

    def test_create_with_alpha(self):
        """Test creating quantizer with specific alpha."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import create_smoothquant_quantizer
        from llm_quantize.models import ModelType

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = SourceModel(
                model_path="test/model",
                model_type=ModelType.HF_HUB,
                architecture="llama",
                parameter_count=1000000,
                dtype="float16",
                num_layers=4,
                hidden_size=768,
                num_heads=12,
                vocab_size=32000,
            )

            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
            )

            quantizer = create_smoothquant_quantizer(
                source_model=source_model,
                config=config,
                alpha=0.7,
            )

            assert quantizer.smoothquant_config.alpha == 0.7

    def test_create_with_per_channel_false(self):
        """Test creating quantizer with per_channel=False."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import create_smoothquant_quantizer
        from llm_quantize.models import ModelType

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = SourceModel(
                model_path="test/model",
                model_type=ModelType.HF_HUB,
                architecture="llama",
                parameter_count=1000000,
                dtype="float16",
                num_layers=4,
                hidden_size=768,
                num_heads=12,
                vocab_size=32000,
            )

            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
            )

            quantizer = create_smoothquant_quantizer(
                source_model=source_model,
                config=config,
                per_channel=False,
            )

            assert quantizer.smoothquant_config.per_channel is False

    def test_recommended_alpha_values(self):
        """Test recommended alpha values dict."""
        assert RECOMMENDED_ALPHA["llama"] == 0.5
        assert RECOMMENDED_ALPHA["bloom"] == 0.75
        assert RECOMMENDED_ALPHA["default"] == 0.5
