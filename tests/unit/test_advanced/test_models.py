"""Tests for advanced quantization data models."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_quantize.models import (
    ActivationStatistics,
    CalibrationInfo,
    CoherenceTestResult,
    DynamicQuantizationProfile,
    ImportanceMatrix,
    ImportanceMethod,
    LayerError,
    LayerImportance,
    LayerQuantizationConfig,
    LayerType,
    PRESET_PROFILES,
    QualityGrade,
    QuantizationMethod,
    QuantizationQualityReport,
    RECOMMENDED_ALPHA,
    SmoothQuantConfig,
)


class TestImportanceMatrix:
    """Tests for ImportanceMatrix model."""

    def test_create_importance_matrix(self):
        """Test creating an importance matrix."""
        calib_info = CalibrationInfo(
            dataset_name="test",
            num_samples=100,
            sequence_length=512,
        )

        layer_scores = [
            LayerImportance(
                layer_name="layer_0",
                layer_index=0,
                importance_score=0.8,
                parameter_count=1000,
                recommended_bits=4,
            ),
            LayerImportance(
                layer_name="layer_1",
                layer_index=1,
                importance_score=0.3,
                parameter_count=1000,
                recommended_bits=2,
            ),
        ]

        matrix = ImportanceMatrix(
            model_name="test-model",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=calib_info,
            layer_scores=layer_scores,
            total_parameters=2000,
        )

        assert matrix.model_name == "test-model"
        assert len(matrix.layer_scores) == 2
        assert matrix.get_layer_importance("layer_0").importance_score == 0.8
        assert matrix.get_layer_by_index(1).layer_name == "layer_1"

    def test_layer_ranking(self):
        """Test layer ranking by importance."""
        calib_info = CalibrationInfo(dataset_name="test", num_samples=10)

        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=calib_info,
            layer_scores=[
                LayerImportance("a", 0, 0.3, 100),
                LayerImportance("b", 1, 0.9, 100),
                LayerImportance("c", 2, 0.5, 100),
            ],
        )

        ranking = matrix.get_layer_ranking()
        assert ranking[0].layer_name == "b"
        assert ranking[1].layer_name == "c"
        assert ranking[2].layer_name == "a"

    def test_save_and_load(self):
        """Test saving and loading importance matrix."""
        calib_info = CalibrationInfo(dataset_name="test", num_samples=10)

        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=calib_info,
            layer_scores=[
                LayerImportance("layer_0", 0, 0.5, 100, [1, 2, 3]),
            ],
            total_parameters=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "imatrix.json"
            matrix.save(path)

            loaded = ImportanceMatrix.load(path)

            assert loaded.model_name == matrix.model_name
            assert len(loaded.layer_scores) == 1
            assert loaded.layer_scores[0].super_weight_indices == [1, 2, 3]

    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        calib_info = CalibrationInfo(dataset_name="test", num_samples=10)

        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.GRADIENT_SENSITIVITY,
            calibration_info=calib_info,
        )

        data = matrix.to_dict()
        loaded = ImportanceMatrix.from_dict(data)

        assert loaded.computation_method == ImportanceMethod.GRADIENT_SENSITIVITY


class TestLayerQuantizationConfig:
    """Tests for LayerQuantizationConfig model."""

    def test_create_config(self):
        """Test creating layer config."""
        config = LayerQuantizationConfig(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            layer_type=LayerType.ATTENTION,
            bit_width=4,
            quantization_method=QuantizationMethod.STANDARD,
            importance_score=0.7,
        )

        assert config.layer_type == LayerType.ATTENTION
        assert config.bit_width == 4

    def test_invalid_bit_width(self):
        """Test validation of bit width."""
        with pytest.raises(ValueError):
            LayerQuantizationConfig(
                layer_name="test",
                layer_index=0,
                bit_width=20,  # Invalid
            )

    def test_effective_bits_per_weight(self):
        """Test effective bits calculation."""
        config = LayerQuantizationConfig(
            layer_name="test",
            layer_index=0,
            bit_width=4,
            group_size=128,
        )

        effective = config.get_effective_bits_per_weight()
        # 4 bits + 32/128 = 4.25 bits
        assert effective == pytest.approx(4.25, 0.01)


class TestDynamicQuantizationProfile:
    """Tests for DynamicQuantizationProfile model."""

    def test_preset_profiles_exist(self):
        """Test that preset profiles are defined."""
        assert "balanced" in PRESET_PROFILES
        assert "attention-high" in PRESET_PROFILES
        assert "compression-max" in PRESET_PROFILES
        assert "quality-max" in PRESET_PROFILES

    def test_get_layer_bits(self):
        """Test getting layer bits from profile."""
        profile = DynamicQuantizationProfile(
            profile_name="test",
            attention_bits=6,
            mlp_bits=2,
            embedding_bits=8,
        )

        assert profile.get_layer_bits("test", LayerType.ATTENTION) == 6
        assert profile.get_layer_bits("test", LayerType.MLP) == 2
        assert profile.get_layer_bits("test", LayerType.EMBEDDING) == 8

    def test_calculate_compression(self):
        """Test compression ratio calculation."""
        profile = DynamicQuantizationProfile(
            profile_name="test",
            layer_configs=[
                LayerQuantizationConfig("a", 0, bit_width=4),
                LayerQuantizationConfig("b", 1, bit_width=2),
            ],
        )

        layer_params = {"a": 1000, "b": 1000}
        ratio = profile.calculate_actual_compression(layer_params)

        # Original: 2000 * 16 = 32000 bits
        # Quantized: 1000 * 4 + 1000 * 2 = 6000 bits
        # Ratio: 32000 / 6000 = 5.33
        assert ratio == pytest.approx(5.33, 0.1)


class TestSmoothQuantConfig:
    """Tests for SmoothQuantConfig model."""

    def test_create_config(self):
        """Test creating SmoothQuant config."""
        config = SmoothQuantConfig(
            alpha=0.5,
            per_channel=True,
            weight_bits=8,
            activation_bits=8,
        )

        assert config.alpha == 0.5
        assert config.weight_bits == 8

    def test_invalid_alpha(self):
        """Test validation of alpha."""
        with pytest.raises(ValueError):
            SmoothQuantConfig(alpha=1.5)  # Invalid

    def test_recommended_alpha(self):
        """Test recommended alpha values."""
        assert RECOMMENDED_ALPHA["llama"] == 0.5
        assert RECOMMENDED_ALPHA["bloom"] == 0.75
        assert "default" in RECOMMENDED_ALPHA


class TestQuantizationQualityReport:
    """Tests for QuantizationQualityReport model."""

    def test_compute_quality_grade(self):
        """Test quality grade computation."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.05,
            perplexity_increase=0.01,
        )

        grade = report.compute_quality_grade()
        assert grade == QualityGrade.EXCELLENT

    def test_quality_grade_degraded(self):
        """Test degraded quality grade."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q2_K",
            perplexity_original=5.0,
            perplexity_quantized=5.75,
            perplexity_increase=0.15,
        )

        grade = report.compute_quality_grade()
        assert grade == QualityGrade.DEGRADED

    def test_get_worst_layers(self):
        """Test getting worst performing layers."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            layer_errors=[
                LayerError("a", 0, mse=0.1, max_error=0.5, relative_error=0.05, bit_width=4),
                LayerError("b", 1, mse=0.5, max_error=1.0, relative_error=0.1, bit_width=4),
                LayerError("c", 2, mse=0.2, max_error=0.6, relative_error=0.06, bit_width=4),
            ],
        )

        worst = report.get_worst_layers(2)
        assert len(worst) == 2
        assert worst[0].layer_name == "b"
        assert worst[1].layer_name == "c"

    def test_coherence_rate(self):
        """Test coherence rate calculation."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            coherence_tests=[
                CoherenceTestResult("prompt1", "output1", True),
                CoherenceTestResult("prompt2", "output2", True),
                CoherenceTestResult("prompt3", "output3", False),
            ],
        )

        rate = report.get_coherence_rate()
        assert rate == pytest.approx(0.67, 0.01)

    def test_generate_summary(self):
        """Test summary generation."""
        report = QuantizationQualityReport(
            model_name="test-model",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.25,
            perplexity_increase=0.05,
            compression_ratio=4.0,
            average_bits_per_weight=4.0,
        )

        summary = report.generate_summary()
        assert "test-model" in summary
        assert "GOOD" in summary
        assert "4.00x" in summary


class TestActivationStatistics:
    """Tests for ActivationStatistics model."""

    def test_dynamic_range(self):
        """Test dynamic range calculation."""
        stats = ActivationStatistics(
            layer_name="test",
            min_value=-5.0,
            max_value=10.0,
        )

        assert stats.get_dynamic_range() == 15.0

    def test_outlier_threshold(self):
        """Test outlier threshold calculation."""
        stats = ActivationStatistics(
            layer_name="test",
            mean_value=0.0,
            std_value=1.0,
        )

        threshold = stats.get_outlier_threshold(sigma=3.0)
        assert threshold == 3.0

    def test_activation_statistics_to_dict(self):
        """Test serialization to dict."""
        stats = ActivationStatistics(
            layer_name="test",
            min_value=-5.0,
            max_value=10.0,
            mean_value=2.5,
            std_value=3.0,
            outlier_ratio=0.05,
            max_channel_values=[1.0, 2.0, 3.0],
        )

        d = stats.to_dict()
        assert d["layer_name"] == "test"
        assert d["outlier_ratio"] == 0.05
        assert d["max_channel_values"] == [1.0, 2.0, 3.0]

    def test_activation_statistics_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "layer_name": "layer1",
            "min_value": -1.0,
            "max_value": 1.0,
            "mean_value": 0.0,
            "std_value": 0.5,
        }

        stats = ActivationStatistics.from_dict(data)
        assert stats.layer_name == "layer1"
        assert stats.mean_value == 0.0


class TestSmoothQuantConfigCoverage:
    """Additional tests for SmoothQuantConfig coverage."""

    def test_invalid_weight_bits(self):
        """Test validation of weight_bits."""
        with pytest.raises(ValueError, match="weight_bits"):
            SmoothQuantConfig(weight_bits=20)  # Invalid

    def test_invalid_activation_bits(self):
        """Test validation of activation_bits."""
        with pytest.raises(ValueError, match="activation_bits"):
            SmoothQuantConfig(activation_bits=0)  # Invalid

    def test_get_layer_statistics_found(self):
        """Test getting layer statistics when found."""
        stats = ActivationStatistics(layer_name="layer1", mean_value=0.5)
        config = SmoothQuantConfig(
            layer_statistics=[stats]
        )

        result = config.get_layer_statistics("layer1")
        assert result is not None
        assert result.mean_value == 0.5

    def test_get_layer_statistics_not_found(self):
        """Test getting layer statistics when not found."""
        config = SmoothQuantConfig()

        result = config.get_layer_statistics("nonexistent")
        assert result is None

    def test_get_smoothing_scale_found(self):
        """Test getting smoothing scale when found."""
        config = SmoothQuantConfig(
            smoothing_scales={"layer1": [1.0, 2.0, 3.0]}
        )

        result = config.get_smoothing_scale("layer1")
        assert result == [1.0, 2.0, 3.0]

    def test_get_smoothing_scale_not_found(self):
        """Test getting smoothing scale when not found."""
        config = SmoothQuantConfig()

        result = config.get_smoothing_scale("nonexistent")
        assert result is None

    def test_get_total_layers_analyzed(self):
        """Test getting total layers analyzed."""
        stats1 = ActivationStatistics(layer_name="layer1")
        stats2 = ActivationStatistics(layer_name="layer2")
        config = SmoothQuantConfig(
            layer_statistics=[stats1, stats2]
        )

        assert config.get_total_layers_analyzed() == 2

    def test_get_problematic_layers(self):
        """Test getting problematic layers."""
        config = SmoothQuantConfig(
            layer_statistics=[
                ActivationStatistics(layer_name="good", outlier_ratio=0.01),
                ActivationStatistics(layer_name="bad", outlier_ratio=0.2),
                ActivationStatistics(layer_name="ok", outlier_ratio=0.05),
            ]
        )

        problematic = config.get_problematic_layers(outlier_threshold=0.1)
        assert "bad" in problematic
        assert "good" not in problematic

    def test_to_dict(self):
        """Test serialization to dict."""
        config = SmoothQuantConfig(
            alpha=0.6,
            per_channel=False,
            weight_bits=4,
            activation_bits=8,
            symmetric=False,
            smoothing_scales={"layer1": [1.0, 2.0]},
        )

        d = config.to_dict()
        assert d["alpha"] == 0.6
        assert d["per_channel"] is False
        assert d["smoothing_scales"] == {"layer1": [1.0, 2.0]}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "alpha": 0.7,
            "per_channel": True,
            "weight_bits": 4,
            "activation_bits": 4,
            "symmetric": True,
            "layer_statistics": [
                {"layer_name": "layer1", "mean_value": 0.5}
            ],
            "smoothing_scales": {"layer1": [1.0]},
        }

        config = SmoothQuantConfig.from_dict(data)
        assert config.alpha == 0.7
        assert len(config.layer_statistics) == 1
        assert config.smoothing_scales == {"layer1": [1.0]}

    def test_compute_smoothing_scale(self):
        """Test computing smoothing scale."""
        import numpy as np

        config = SmoothQuantConfig(alpha=0.5)

        activation_max = np.array([1.0, 2.0, 3.0])
        weight_max = np.array([0.5, 1.0, 1.5])

        scales = config.compute_smoothing_scale("layer1", activation_max, weight_max)

        assert len(scales) == 3
        assert "layer1" in config.smoothing_scales


class TestLayerQuantizationConfigCoverage:
    """Additional tests for LayerQuantizationConfig coverage."""

    def test_default_values(self):
        """Test default values."""
        config = LayerQuantizationConfig(
            layer_name="test",
            layer_index=0,
        )

        assert config.bit_width == 4  # Default
        assert config.group_size is None  # Default is None

    def test_invalid_bit_width_low(self):
        """Test validation of bit width too low."""
        with pytest.raises(ValueError):
            LayerQuantizationConfig(
                layer_name="test",
                layer_index=0,
                bit_width=0,  # Invalid
            )

    def test_to_dict(self):
        """Test serialization to dict."""
        config = LayerQuantizationConfig(
            layer_name="test_layer",
            layer_index=5,
            layer_type=LayerType.MLP,
            bit_width=2,
            quantization_method=QuantizationMethod.PROTECTED,
            importance_score=0.8,
            group_size=64,
        )

        d = config.to_dict()
        assert d["layer_name"] == "test_layer"
        assert d["layer_index"] == 5
        assert d["group_size"] == 64

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "layer_name": "layer1",
            "layer_index": 0,
            "layer_type": "attention",
            "bit_width": 8,
            "group_size": 64,
            "quantization_method": "standard",
            "importance_score": 0.5,
        }

        config = LayerQuantizationConfig.from_dict(data)
        assert config.layer_name == "layer1"
        assert config.bit_width == 8
        assert config.group_size == 64

    def test_get_effective_bits_no_group_size(self):
        """Test effective bits when group_size is None."""
        config = LayerQuantizationConfig(
            layer_name="test",
            layer_index=0,
            bit_width=4,
            group_size=None,
        )

        effective = config.get_effective_bits_per_weight()
        assert effective == 4.0

    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {
            "layer_name": "layer1",
            "layer_index": 0,
        }

        config = LayerQuantizationConfig.from_dict(data)
        assert config.layer_name == "layer1"
        assert config.bit_width == 4  # Default
