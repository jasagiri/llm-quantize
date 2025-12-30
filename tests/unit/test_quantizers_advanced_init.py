"""Tests for advanced quantizers __init__ module."""

import pytest


class TestAdvancedQuantizersInit:
    """Tests for advanced quantizers initialization."""

    def test_import_dynamic_quantizer(self):
        """Test importing DynamicQuantizer."""
        from llm_quantize.lib.quantizers.advanced import DynamicQuantizer
        assert DynamicQuantizer is not None

    def test_import_ultra_low_bit_quantizer(self):
        """Test importing UltraLowBitQuantizer."""
        from llm_quantize.lib.quantizers.advanced import UltraLowBitQuantizer
        assert UltraLowBitQuantizer is not None

    def test_import_smoothquant_quantizer(self):
        """Test importing SmoothQuantQuantizer."""
        from llm_quantize.lib.quantizers.advanced import SmoothQuantQuantizer
        assert SmoothQuantQuantizer is not None

    def test_import_get_profile(self):
        """Test importing get_profile."""
        from llm_quantize.lib.quantizers.advanced import get_profile
        assert get_profile is not None

    def test_import_get_available_profiles(self):
        """Test importing get_available_profiles."""
        from llm_quantize.lib.quantizers.advanced import get_available_profiles
        assert get_available_profiles is not None


class TestQuantizersRegistryInit:
    """Tests for quantizers registry."""

    def test_get_quantizer_valid_format(self):
        """Test getting quantizer for valid format."""
        from llm_quantize.lib.quantizers import get_quantizer
        from llm_quantize.models import OutputFormat

        # Should return a quantizer class
        quantizer_class = get_quantizer(OutputFormat.GGUF)
        assert quantizer_class is not None

    def test_get_available_formats_returns_enums(self):
        """Test that available formats returns list of OutputFormat enums."""
        from llm_quantize.lib.quantizers import get_available_formats
        from llm_quantize.models import OutputFormat

        formats = get_available_formats()
        assert all(isinstance(f, OutputFormat) for f in formats)


class TestModelsCoverage:
    """Tests to improve models coverage."""

    def test_source_model_repr(self):
        """Test SourceModel repr."""
        from llm_quantize.models import SourceModel, ModelType

        model = SourceModel(
            model_path="test/path",
            model_type=ModelType.HF_HUB,
            architecture="llama",
            parameter_count=1000,
            dtype="float16",
            num_layers=4,
            hidden_size=768,
            num_heads=12,
            vocab_size=32000,
        )

        repr_str = repr(model)
        assert "test/path" in repr_str

    def test_quantization_config_basic(self):
        """Test QuantizationConfig basic creation."""
        from llm_quantize.models import QuantizationConfig, OutputFormat

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir="/tmp",
        )

        assert config.target_format == OutputFormat.GGUF
        assert config.quantization_level == "Q4_K_M"

    def test_quantized_model_basic(self):
        """Test QuantizedModel basic creation."""
        from llm_quantize.models import QuantizedModel

        model = QuantizedModel(
            output_path="/tmp/model.gguf",
            format="gguf",
            quantization_level="Q4_K_M",
            compression_ratio=4.0,
            file_size=1000000,
            source_model_path="/source/model",
        )

        assert model.format == "gguf"
        assert model.file_size == 1000000


class TestImportanceMatrixCoverage:
    """Tests to improve ImportanceMatrix coverage."""

    def test_layer_importance_recommended_bits_default(self):
        """Test LayerImportance default recommended bits."""
        from llm_quantize.models import LayerImportance

        layer = LayerImportance(
            layer_name="layer1",
            layer_index=0,
            importance_score=0.5,
            parameter_count=100,
        )

        # Default should be None or 4
        assert layer.recommended_bits is None or layer.recommended_bits == 4

    def test_importance_matrix_get_super_weight_count(self):
        """Test getting super weight count."""
        from llm_quantize.models import (
            ImportanceMatrix,
            ImportanceMethod,
            CalibrationInfo,
            LayerImportance,
        )

        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.5, 100, super_weight_indices=[1, 2, 3]),
                LayerImportance("layer2", 1, 0.5, 100, super_weight_indices=[4, 5]),
            ],
        )

        count = matrix.get_super_weight_count()
        assert count == 5


class TestQualityReportCoverage:
    """Tests to improve quality report coverage."""

    def test_quality_grade_values(self):
        """Test QualityGrade enum values."""
        from llm_quantize.models import QualityGrade

        assert QualityGrade.EXCELLENT.value == "excellent"
        assert QualityGrade.GOOD.value == "good"
        assert QualityGrade.ACCEPTABLE.value == "acceptable"
        assert QualityGrade.DEGRADED.value == "degraded"
        assert QualityGrade.POOR.value == "poor"

    def test_coherence_test_result_basic(self):
        """Test CoherenceTestResult basic creation."""
        from llm_quantize.models import CoherenceTestResult

        result = CoherenceTestResult(
            prompt="test prompt",
            output="test output",
            is_coherent=True,
        )

        assert result.prompt == "test prompt"
        assert result.is_coherent is True


class TestLayerConfigCoverage:
    """Tests to improve layer config coverage."""

    def test_layer_type_values(self):
        """Test LayerType enum values."""
        from llm_quantize.models import LayerType

        assert LayerType.ATTENTION.value == "attention"
        assert LayerType.MLP.value == "mlp"
        assert LayerType.EMBEDDING.value == "embedding"
        assert LayerType.NORM.value == "norm"
        assert LayerType.OUTPUT.value == "output"

    def test_layer_quantization_config_basic(self):
        """Test LayerQuantizationConfig basic creation."""
        from llm_quantize.models import LayerQuantizationConfig

        config = LayerQuantizationConfig(
            layer_name="layer1",
            layer_index=0,
            bit_width=4,
        )

        assert config.layer_name == "layer1"
        assert config.bit_width == 4
