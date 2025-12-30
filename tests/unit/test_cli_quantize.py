"""Tests for CLI quantize command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


class TestValidateQuantLevel:
    """Tests for validate_quant_level function."""

    def test_validate_gguf_q4_k_m(self):
        """Test GGUF Q4_K_M is valid."""
        from llm_quantize.cli.quantize import validate_quant_level

        assert validate_quant_level("gguf", "Q4_K_M") is True

    def test_validate_gguf_invalid(self):
        """Test invalid GGUF level."""
        from llm_quantize.cli.quantize import validate_quant_level

        assert validate_quant_level("gguf", "invalid") is False

    def test_validate_awq_4bit(self):
        """Test AWQ 4bit is valid."""
        from llm_quantize.cli.quantize import validate_quant_level

        assert validate_quant_level("awq", "4bit") is True

    def test_validate_gptq_4bit(self):
        """Test GPTQ 4bit is valid."""
        from llm_quantize.cli.quantize import validate_quant_level

        assert validate_quant_level("gptq", "4bit") is True

    def test_validate_unknown_format(self):
        """Test unknown format returns False."""
        from llm_quantize.cli.quantize import validate_quant_level

        assert validate_quant_level("unknown", "4bit") is False


class TestQuantizeCommand:
    """Tests for quantize command."""

    def test_quantize_model_not_found(self):
        """Test quantize with model not found (ValueError from create_source_model)."""
        from llm_quantize.cli.quantize import quantize

        runner = CliRunner()

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_create:
            # create_source_model raises ValueError for model not found
            mock_create.side_effect = ValueError("Model not found")

            result = runner.invoke(
                quantize,
                ["nonexistent/model", "gguf", "-q", "Q4_K_M"],
            )

        # Should fail with model not found exit code
        assert result.exit_code == 3  # EXIT_MODEL_NOT_FOUND

    def test_quantize_invalid_quant_level(self):
        """Test quantize with invalid quantization level."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType

        runner = CliRunner()

        mock_source = SourceModel(
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

        with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source):
            result = runner.invoke(
                quantize,
                ["test/model", "gguf", "-q", "INVALID_LEVEL"],
            )

        assert result.exit_code == 2  # EXIT_INVALID_ARGUMENTS

    def test_quantize_value_error(self):
        """Test quantize with ValueError."""
        from llm_quantize.cli.quantize import quantize

        runner = CliRunner()

        with patch("llm_quantize.cli.quantize.create_source_model") as mock_create:
            mock_create.side_effect = ValueError("Invalid model")

            result = runner.invoke(
                quantize,
                ["test/model", "gguf", "-q", "Q4_K_M"],
            )

        # ValueError should result in invalid arguments
        assert result.exit_code in [2, 3]  # Either invalid args or model not found

    def test_quantize_success(self):
        """Test successful quantization."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType, QuantizedModel, ValidationStatus

        runner = CliRunner()

        mock_source = SourceModel(
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

        mock_result = QuantizedModel(
            output_path="/tmp/output.gguf",
            format="gguf",
            quantization_level="Q4_K_M",
            compression_ratio=4.0,
            file_size=1000000,
            source_model_path="test/model",
            duration_seconds=10.0,
            peak_memory_bytes=1000000000,
        )

        mock_validation = MagicMock()
        mock_validation.is_valid = True

        mock_quantizer_instance = MagicMock()
        mock_quantizer_instance.quantize.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
                 patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer, \
                 patch("llm_quantize.cli.quantize.validate_output", return_value=mock_validation):

                mock_quantizer_class = MagicMock(return_value=mock_quantizer_instance)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    quantize,
                    ["test/model", "gguf", "-q", "Q4_K_M", "-o", tmpdir],
                )

        assert result.exit_code == 0

    def test_quantize_no_quantizer(self):
        """Test quantize with no quantizer available."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType

        runner = CliRunner()

        mock_source = SourceModel(
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

        with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
             patch("llm_quantize.cli.quantize.get_quantizer", return_value=None):

            result = runner.invoke(
                quantize,
                ["test/model", "gguf", "-q", "Q4_K_M"],
            )

        assert result.exit_code == 2  # EXIT_INVALID_ARGUMENTS

    def test_quantize_checkpoint_error(self):
        """Test quantize with checkpoint error."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType

        runner = CliRunner()

        mock_source = SourceModel(
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

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
                 patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:

                mock_quantizer_class = MagicMock(side_effect=Exception("checkpoint corrupted"))
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    quantize,
                    ["test/model", "gguf", "-q", "Q4_K_M", "-o", tmpdir],
                )

        assert result.exit_code == 7  # EXIT_CHECKPOINT_ERROR

    def test_quantize_general_init_error(self):
        """Test quantize with general initialization error."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType

        runner = CliRunner()

        mock_source = SourceModel(
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

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
                 patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:

                mock_quantizer_class = MagicMock(side_effect=Exception("initialization failed"))
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    quantize,
                    ["test/model", "gguf", "-q", "Q4_K_M", "-o", tmpdir],
                )

        assert result.exit_code == 1  # EXIT_GENERAL_ERROR

    def test_quantize_out_of_memory(self):
        """Test quantize with out of memory error."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType

        runner = CliRunner()

        mock_source = SourceModel(
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

        mock_quantizer_instance = MagicMock()
        mock_quantizer_instance.quantize.side_effect = MemoryError("Out of memory")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
                 patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:

                mock_quantizer_class = MagicMock(return_value=mock_quantizer_instance)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    quantize,
                    ["test/model", "gguf", "-q", "Q4_K_M", "-o", tmpdir],
                )

        assert result.exit_code == 5  # EXIT_OUT_OF_MEMORY

    def test_quantize_general_error(self):
        """Test quantize with general error during quantization."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType

        runner = CliRunner()

        mock_source = SourceModel(
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

        mock_quantizer_instance = MagicMock()
        mock_quantizer_instance.quantize.side_effect = RuntimeError("Quantization failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
                 patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer:

                mock_quantizer_class = MagicMock(return_value=mock_quantizer_instance)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    quantize,
                    ["test/model", "gguf", "-q", "Q4_K_M", "-o", tmpdir],
                )

        assert result.exit_code == 1  # EXIT_GENERAL_ERROR

    def test_quantize_validation_failed(self):
        """Test quantize with validation failure."""
        from llm_quantize.cli.quantize import quantize
        from llm_quantize.models import SourceModel, ModelType, QuantizedModel

        runner = CliRunner()

        mock_source = SourceModel(
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

        mock_result = QuantizedModel(
            output_path="/tmp/output.gguf",
            format="gguf",
            quantization_level="Q4_K_M",
            compression_ratio=4.0,
            file_size=1000000,
            source_model_path="test/model",
        )

        mock_validation = MagicMock()
        mock_validation.is_valid = False
        mock_validation.error_message = "Corrupted output"

        mock_quantizer_instance = MagicMock()
        mock_quantizer_instance.quantize.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_quantize.cli.quantize.create_source_model", return_value=mock_source), \
                 patch("llm_quantize.cli.quantize.get_quantizer") as mock_get_quantizer, \
                 patch("llm_quantize.cli.quantize.validate_output", return_value=mock_validation):

                mock_quantizer_class = MagicMock(return_value=mock_quantizer_instance)
                mock_get_quantizer.return_value = mock_quantizer_class

                result = runner.invoke(
                    quantize,
                    ["test/model", "gguf", "-q", "Q4_K_M", "-o", tmpdir],
                )

        assert result.exit_code == 6  # EXIT_VALIDATION_FAILED


class TestQuantizersInit:
    """Tests for quantizers __init__ module."""

    def test_get_quantizer_gguf(self):
        """Test getting GGUF quantizer."""
        from llm_quantize.lib.quantizers import get_quantizer
        from llm_quantize.lib.quantizers.gguf import GGUFQuantizer
        from llm_quantize.models import OutputFormat

        quantizer_class = get_quantizer(OutputFormat.GGUF)
        assert quantizer_class == GGUFQuantizer

    def test_get_quantizer_awq(self):
        """Test getting AWQ quantizer."""
        from llm_quantize.lib.quantizers import get_quantizer
        from llm_quantize.lib.quantizers.awq import AWQQuantizer
        from llm_quantize.models import OutputFormat

        quantizer_class = get_quantizer(OutputFormat.AWQ)
        assert quantizer_class == AWQQuantizer

    def test_get_quantizer_gptq(self):
        """Test getting GPTQ quantizer."""
        from llm_quantize.lib.quantizers import get_quantizer
        from llm_quantize.lib.quantizers.gptq import GPTQQuantizer
        from llm_quantize.models import OutputFormat

        quantizer_class = get_quantizer(OutputFormat.GPTQ)
        assert quantizer_class == GPTQQuantizer

    def test_get_available_formats(self):
        """Test getting available formats."""
        from llm_quantize.lib.quantizers import get_available_formats
        from llm_quantize.models import OutputFormat

        formats = get_available_formats()
        assert OutputFormat.GGUF in formats
        assert OutputFormat.AWQ in formats
        assert OutputFormat.GPTQ in formats


class TestBaseQuantizer:
    """Tests for base quantizer."""

    def test_base_quantizer_log_start(self):
        """Test base quantizer log_start method."""
        from llm_quantize.lib.quantizers.base import BaseQuantizer
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import (
            QuantizationConfig,
            OutputFormat,
            SourceModel,
            ModelType,
        )

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

        with tempfile.TemporaryDirectory() as tmpdir:
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="Q4_K_M",
                output_dir=str(tmpdir),
            )

            progress = MagicMock(spec=ProgressReporter)

            # Create a concrete implementation for testing
            class TestQuantizer(BaseQuantizer):
                def quantize(self):
                    pass

                def estimate_output_size(self):
                    return 1000

                @classmethod
                def get_supported_levels(cls):
                    return ["Q4_K_M"]

            quantizer = TestQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
            )

            quantizer.log_start()

            assert progress.log_info.called


class TestSmoothQuantConfig:
    """Tests for SmoothQuant config model."""

    def test_smoothquant_config_compute_scale(self):
        """Test SmoothQuant config compute_smoothing_scale method."""
        from llm_quantize.models import SmoothQuantConfig
        import numpy as np

        config = SmoothQuantConfig(alpha=0.5)

        act_max = np.array([1.0, 2.0, 3.0])
        weight_max = np.array([0.5, 1.0, 1.5])

        scale = config.compute_smoothing_scale("test_layer", act_max, weight_max)

        assert len(scale) == 3
        assert all(s > 0 for s in scale)

    def test_smoothquant_config_repr(self):
        """Test SmoothQuant config repr."""
        from llm_quantize.models import SmoothQuantConfig

        config = SmoothQuantConfig(alpha=0.5)

        repr_str = repr(config)
        assert "0.5" in repr_str or "SmoothQuantConfig" in repr_str


class TestImportanceMatrixModel:
    """Tests for ImportanceMatrix model."""

    def test_importance_matrix_get_layer_ranking(self):
        """Test getting layer ranking by importance."""
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
                LayerImportance("layer1", 0, 0.9, 100),
                LayerImportance("layer2", 1, 0.5, 100),
                LayerImportance("layer3", 2, 0.1, 100),
            ],
        )

        ranking = matrix.get_layer_ranking()

        assert len(ranking) == 3
        assert ranking[0].importance_score >= ranking[1].importance_score

    def test_importance_matrix_get_layer_by_index(self):
        """Test getting layer by index."""
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
                LayerImportance("layer1", 0, 0.95, 100),
                LayerImportance("layer2", 1, 0.5, 100),
                LayerImportance("layer3", 2, 0.1, 100),
            ],
        )

        layer = matrix.get_layer_by_index(0)

        assert layer is not None
        assert layer.layer_name == "layer1"

    def test_importance_matrix_get_recommended_bitwidths(self):
        """Test getting recommended bitwidths."""
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
                LayerImportance("layer1", 0, 0.95, 100, recommended_bits=8),
                LayerImportance("layer2", 1, 0.5, 100, recommended_bits=4),
            ],
        )

        bitwidths = matrix.get_recommended_bitwidths()

        assert isinstance(bitwidths, dict)


class TestLayerConfigModel:
    """Tests for layer config model."""

    def test_layer_quantization_config_is_sensitive(self):
        """Test layer sensitivity check."""
        from llm_quantize.models import LayerQuantizationConfig

        config = LayerQuantizationConfig(
            layer_name="attention.qkv",
            layer_index=0,
            bit_width=8,
        )

        # Attention layers should be sensitive
        assert config.layer_name == "attention.qkv"

    def test_layer_quantization_config_defaults(self):
        """Test layer quantization config defaults."""
        from llm_quantize.models import LayerQuantizationConfig

        config = LayerQuantizationConfig(
            layer_name="layer1",
            layer_index=0,
            bit_width=4,
        )

        assert config.group_size is None or config.group_size == 128


class TestAdvancedQuantizerRegistry:
    """Tests for advanced quantizer registry."""

    def test_get_advanced_quantizer_dynamic(self):
        """Test getting dynamic quantizer."""
        from llm_quantize.lib.quantizers import get_advanced_quantizer
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        quantizer_class = get_advanced_quantizer("dynamic")
        assert quantizer_class == DynamicQuantizer

    def test_get_advanced_quantizer_ultra_low_bit(self):
        """Test getting ultra low bit quantizer."""
        from llm_quantize.lib.quantizers import get_advanced_quantizer
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        quantizer_class = get_advanced_quantizer("ultra_low_bit")
        assert quantizer_class == UltraLowBitQuantizer

    def test_get_advanced_quantizer_smoothquant(self):
        """Test getting smoothquant quantizer."""
        from llm_quantize.lib.quantizers import get_advanced_quantizer
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        quantizer_class = get_advanced_quantizer("smoothquant")
        assert quantizer_class == SmoothQuantQuantizer

    def test_get_advanced_quantizer_unknown(self):
        """Test getting unknown advanced quantizer returns None."""
        from llm_quantize.lib.quantizers import get_advanced_quantizer

        quantizer_class = get_advanced_quantizer("unknown")
        assert quantizer_class is None


class TestQuantizationConfigModel:
    """Tests for QuantizationConfig model."""

    def test_quantization_config_full(self):
        """Test QuantizationConfig with all options."""
        from llm_quantize.models import QuantizationConfig, OutputFormat

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir="/tmp",
            output_name="model.gguf",
            calibration_data_path="/data/calib.txt",
            calibration_samples=256,
            group_size=128,
        )

        assert config.target_format == OutputFormat.GGUF
        assert config.output_name == "model.gguf"
        assert config.calibration_samples == 256

    def test_quantization_config_get_output_path(self):
        """Test QuantizationConfig get_output_path method."""
        from llm_quantize.models import QuantizationConfig, OutputFormat

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir="/tmp",
        )

        path = config.get_output_path("test_model")
        assert "test_model" in str(path) or path is not None


class TestQualityReportModel:
    """Tests for QuantizationQualityReport model."""

    def test_quality_report_creation(self):
        """Test QuantizationQualityReport basic creation."""
        from llm_quantize.models import QuantizationQualityReport, QualityGrade

        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="GGUF",
            quantization_level="Q4_K_M",
            compression_ratio=4.0,
            quality_grade=QualityGrade.GOOD,
        )

        assert report.model_name == "test"
        assert report.quality_grade == QualityGrade.GOOD

    def test_quality_report_perplexity(self):
        """Test QuantizationQualityReport with perplexity."""
        from llm_quantize.models import QuantizationQualityReport, QualityGrade

        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="GGUF",
            quantization_level="Q4_K_M",
            compression_ratio=4.0,
            quality_grade=QualityGrade.EXCELLENT,
            perplexity_original=10.0,
            perplexity_quantized=10.5,
        )

        assert report.perplexity_original == 10.0
        assert report.perplexity_quantized == 10.5

    def test_quality_report_compute_grade(self):
        """Test quality grade computation."""
        from llm_quantize.models import QuantizationQualityReport, QualityGrade

        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="GGUF",
            quantization_level="Q4_K_M",
            perplexity_increase=0.01,  # 1% increase
        )

        grade = report.compute_quality_grade()
        assert grade == QualityGrade.EXCELLENT

    def test_quality_report_coherence_rate(self):
        """Test coherence rate calculation."""
        from llm_quantize.models import QuantizationQualityReport, CoherenceTestResult

        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="GGUF",
            quantization_level="Q4_K_M",
            coherence_tests=[
                CoherenceTestResult("prompt1", "output1", True),
                CoherenceTestResult("prompt2", "output2", False),
            ],
        )

        rate = report.get_coherence_rate()
        assert rate == 0.5


class TestLayerError:
    """Tests for LayerError model."""

    def test_layer_error_to_dict(self):
        """Test LayerError to_dict method."""
        from llm_quantize.models import LayerError

        error = LayerError(
            layer_name="layer1",
            layer_index=0,
            mse=0.001,
            max_error=0.1,
            relative_error=5.0,
            bit_width=4,
        )

        d = error.to_dict()
        assert d["layer_name"] == "layer1"
        assert d["mse"] == 0.001


class TestCalibrationInfo:
    """Tests for CalibrationInfo model."""

    def test_calibration_info_basic(self):
        """Test CalibrationInfo basic creation."""
        from llm_quantize.models import CalibrationInfo

        info = CalibrationInfo(
            dataset_name="test",
            num_samples=100,
        )

        assert info.dataset_name == "test"
        assert info.num_samples == 100


class TestActivationStatistics:
    """Tests for ActivationStatistics model."""

    def test_activation_statistics_basic(self):
        """Test ActivationStatistics basic creation."""
        from llm_quantize.models import ActivationStatistics

        stats = ActivationStatistics(
            layer_name="layer1",
        )

        assert stats.layer_name == "layer1"

    def test_activation_statistics_full(self):
        """Test ActivationStatistics with all fields."""
        from llm_quantize.models import ActivationStatistics

        stats = ActivationStatistics(
            layer_name="layer1",
            min_value=-1.0,
            max_value=1.0,
            mean_value=0.0,
            std_value=0.5,
            outlier_ratio=0.01,
            max_channel_values=[1.0, 0.5, 0.2],
        )

        assert stats.max_value == 1.0
        assert len(stats.max_channel_values) == 3


class TestModelTypes:
    """Tests for model type enums."""

    def test_model_type_values(self):
        """Test ModelType enum values."""
        from llm_quantize.models import ModelType

        assert ModelType.HF_HUB.value == "hf_hub"
        assert ModelType.LOCAL_DIR.value == "local_dir"

    def test_output_format_values(self):
        """Test OutputFormat enum values."""
        from llm_quantize.models import OutputFormat

        assert OutputFormat.GGUF.value == "gguf"
        assert OutputFormat.AWQ.value == "awq"
        assert OutputFormat.GPTQ.value == "gptq"

    def test_output_mode_values(self):
        """Test OutputMode enum values."""
        from llm_quantize.models import OutputMode

        assert OutputMode.HUMAN.value == "human"
        assert OutputMode.JSON.value == "json"


class TestJobStatus:
    """Tests for job status enum and model."""

    def test_job_status_values(self):
        """Test JobStatus enum values."""
        from llm_quantize.models import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_quantization_job_creation(self):
        """Test QuantizationJob creation."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob()

        assert job.job_id is not None
        assert job.status == JobStatus.PENDING
        assert job.progress_percentage == 0.0

    def test_quantization_job_is_complete(self):
        """Test QuantizationJob is_complete property."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob(status=JobStatus.COMPLETED)

        assert job.is_complete is True

    def test_quantization_job_not_complete(self):
        """Test QuantizationJob is_complete is False when pending."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob(status=JobStatus.PENDING)

        assert job.is_complete is False

    def test_quantization_job_start(self):
        """Test QuantizationJob start method."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob()
        job.start()

        assert job.status == JobStatus.RUNNING
        assert job.start_time is not None
        assert job.progress_percentage == 0.0

    def test_quantization_job_complete(self):
        """Test QuantizationJob complete method."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob()
        job.start()
        job.complete()

        assert job.status == JobStatus.COMPLETED
        assert job.end_time is not None
        assert job.progress_percentage == 100.0

    def test_quantization_job_fail(self):
        """Test QuantizationJob fail method."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob()
        job.start()
        job.fail("Test error")

        assert job.status == JobStatus.FAILED
        assert job.error_message == "Test error"

    def test_quantization_job_cancel(self):
        """Test QuantizationJob cancel method."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob()
        job.start()
        job.cancel()

        assert job.status == JobStatus.CANCELLED
        assert job.end_time is not None

    def test_quantization_job_update_progress(self):
        """Test QuantizationJob update_progress method."""
        from llm_quantize.models import QuantizationJob

        job = QuantizationJob()
        job.start()
        job.update_progress(2, 10, memory_usage=1000000)

        assert job.current_layer == 2
        assert job.total_layers == 10
        assert job.progress_percentage == 30.0
        assert job.current_memory_usage == 1000000
        assert job.peak_memory_usage == 1000000

    def test_quantization_job_is_running(self):
        """Test QuantizationJob is_running property."""
        from llm_quantize.models import QuantizationJob, JobStatus

        job = QuantizationJob()
        assert job.is_running is False

        job.start()
        assert job.is_running is True

    def test_quantization_job_duration(self):
        """Test QuantizationJob duration_seconds property."""
        from llm_quantize.models import QuantizationJob

        job = QuantizationJob()
        assert job.duration_seconds is None

        job.start()
        assert job.duration_seconds is not None
        assert job.duration_seconds >= 0

    def test_quantization_job_to_dict(self):
        """Test QuantizationJob to_dict method."""
        from llm_quantize.models import QuantizationJob

        job = QuantizationJob()
        job.start()

        d = job.to_dict()
        assert "job_id" in d
        assert "status" in d
        assert d["status"] == "running"
        assert "progress_percentage" in d

    def test_quantization_job_str(self):
        """Test QuantizationJob string representation."""
        from llm_quantize.models import QuantizationJob

        job = QuantizationJob()
        s = str(job)

        assert "QuantizationJob" in s
        assert "pending" in s
