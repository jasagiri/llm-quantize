"""Tests for dynamic layer-wise quantization."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.models import (
    CalibrationInfo,
    DynamicQuantizationProfile,
    ImportanceMatrix,
    ImportanceMethod,
    LayerImportance,
    LayerQuantizationConfig,
    LayerType,
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestDynamicQuantizer:
    """Tests for DynamicQuantizer class."""

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
            quantization_level="dynamic",
            output_dir=str(tmpdir),
        )

    def test_get_supported_levels(self):
        """Test supported quantization levels."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        levels = DynamicQuantizer.get_supported_levels()

        assert "dynamic" in levels
        assert "auto" in levels

    def test_init_with_profile(self):
        """Test initialization with profile."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                attention_bits=6,
                mlp_bits=2,
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            assert quantizer.profile == profile

    def test_init_with_importance_matrix(self):
        """Test initialization with importance matrix."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            imatrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                importance_matrix=imatrix,
            )

            assert quantizer.importance_matrix == imatrix

    def test_estimate_output_size_with_profile(self):
        """Test output size estimation with profile."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                target_avg_bits=3.0,
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            size = quantizer.estimate_output_size()

            assert size > 0
            # 3-bit should be smaller than FP16
            original_size = source_model.parameter_count * 2
            assert size < original_size

    def test_estimate_output_size_no_profile(self):
        """Test output size estimation without profile."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer.estimate_output_size()

            assert size > 0

    def test_prepare_layer_configs_from_profile(self):
        """Test layer config preparation from profile."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                layer_configs=[
                    LayerQuantizationConfig("layer1", 0, bit_width=4),
                    LayerQuantizationConfig("layer2", 1, bit_width=2),
                ],
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            quantizer._prepare_layer_configs()

            assert "layer1" in quantizer._layer_configs
            assert "layer2" in quantizer._layer_configs
            assert quantizer._layer_configs["layer1"].bit_width == 4

    def test_prepare_layer_configs_from_importance(self):
        """Test layer config preparation from importance matrix."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            imatrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
                layer_scores=[
                    LayerImportance("layer1", 0, 0.9, 100, recommended_bits=6),
                    LayerImportance("layer2", 1, 0.3, 100, recommended_bits=2),
                ],
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                importance_matrix=imatrix,
            )

            quantizer._prepare_layer_configs()

            assert "layer1" in quantizer._layer_configs
            assert quantizer._layer_configs["layer1"].bit_width == 6

    def test_calculate_average_bits(self):
        """Test average bits calculation."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "layer1": LayerQuantizationConfig("layer1", 0, bit_width=4),
                "layer2": LayerQuantizationConfig("layer2", 1, bit_width=2),
            }

            avg = quantizer._calculate_average_bits()

            assert avg == 3.0

    def test_get_base_quant_type(self):
        """Test base quantization type selection."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            # Test various bit-widths
            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=2),
            }
            assert quantizer._get_base_quant_type() == "Q2_K"

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=4),
            }
            assert quantizer._get_base_quant_type() == "Q4_K_M"

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=8),
            }
            assert quantizer._get_base_quant_type() == "Q8_0"

    def test_get_output_path(self):
        """Test output path generation."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                target_avg_bits=3.5,
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            path = quantizer.get_output_path()

            assert "dynamic" in str(path)
            assert ".gguf" in str(path)


class TestCreateDynamicQuantizer:
    """Tests for create_dynamic_quantizer factory function."""

    def test_create_with_profile_name(self):
        """Test creating quantizer with profile name."""
        from llm_quantize.lib.quantizers.advanced.dynamic import create_dynamic_quantizer
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
                quantization_level="dynamic",
                output_dir=str(tmpdir),
            )

            quantizer = create_dynamic_quantizer(
                source_model=source_model,
                config=config,
                profile_name="balanced",
            )

            assert quantizer.profile is not None
            assert quantizer.profile.profile_name == "balanced"

    def test_create_with_unknown_profile(self):
        """Test error on unknown profile name."""
        from llm_quantize.lib.quantizers.advanced.dynamic import create_dynamic_quantizer
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
                quantization_level="dynamic",
                output_dir=str(tmpdir),
            )

            with pytest.raises(ValueError, match="Unknown profile"):
                create_dynamic_quantizer(
                    source_model=source_model,
                    config=config,
                    profile_name="nonexistent",
                )

    def test_create_with_imatrix(self):
        """Test creating quantizer with importance matrix path."""
        from llm_quantize.lib.quantizers.advanced.dynamic import create_dynamic_quantizer
        from llm_quantize.models import ModelType

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create imatrix file
            imatrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
                layer_scores=[
                    LayerImportance("layer1", 0, 0.5, 100),
                ],
            )
            imatrix_path = Path(tmpdir) / "imatrix.json"
            imatrix.save(imatrix_path)

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
                quantization_level="dynamic",
                output_dir=str(tmpdir),
            )

            quantizer = create_dynamic_quantizer(
                source_model=source_model,
                config=config,
                imatrix_path=str(imatrix_path),
                target_bits=3.5,
            )

            assert quantizer.importance_matrix is not None
            assert quantizer.profile is not None

    def test_create_default_uses_balanced(self):
        """Test that default profile is balanced."""
        from llm_quantize.lib.quantizers.advanced.dynamic import create_dynamic_quantizer
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
                quantization_level="dynamic",
                output_dir=str(tmpdir),
            )

            quantizer = create_dynamic_quantizer(
                source_model=source_model,
                config=config,
            )

            assert quantizer.profile.profile_name == "balanced"


class TestDynamicQuantizerAdvanced:
    """Advanced tests for DynamicQuantizer to improve coverage."""

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
            quantization_level="dynamic",
            output_dir=str(tmpdir),
        )

    def test_estimate_output_size_with_layer_configs(self):
        """Test output size estimation with layer configs."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                layer_configs=[
                    LayerQuantizationConfig("layer1", 0, bit_width=4),
                    LayerQuantizationConfig("layer2", 1, bit_width=8),
                ],
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            size = quantizer.estimate_output_size()
            assert size > 0

    def test_generate_default_configs(self):
        """Test generating default layer configs."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._generate_default_configs()

            # Should have generated configs for layers
            assert len(quantizer._layer_configs) > 0

    def test_create_quantization_plan(self):
        """Test creating quantization plan."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                enable_checkpoints=False,
            )

            quantizer._layer_configs = {
                "layer1": LayerQuantizationConfig("layer1", 0, bit_width=4),
            }

            output_path = Path(tmpdir) / "output.gguf"
            result = quantizer._create_quantization_plan(output_path)

            assert result.exists()
            with open(result) as f:
                plan = json.load(f)
            assert plan["dynamic_quantization"] is True

    def test_save_layer_config(self):
        """Test saving layer config to file."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "layer1": LayerQuantizationConfig("layer1", 0, bit_width=4),
            }

            config_path = Path(tmpdir) / "quantize_config.json"
            quantizer._save_layer_config(config_path)

            assert config_path.exists()
            with open(config_path) as f:
                data = json.load(f)
            assert data["dynamic_quantization"] is True

    def test_save_llama_cpp_imatrix(self):
        """Test saving importance matrix in llama.cpp format."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            imatrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
                layer_scores=[
                    LayerImportance("layer1", 0, 0.5, 100),
                ],
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                importance_matrix=imatrix,
            )

            imatrix_path = Path(tmpdir) / "importance.dat"
            quantizer._save_llama_cpp_imatrix(imatrix_path)

            assert imatrix_path.exists()

    def test_save_llama_cpp_imatrix_no_matrix(self):
        """Test saving importance matrix when no matrix is set."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            imatrix_path = Path(tmpdir) / "importance.dat"
            quantizer._save_llama_cpp_imatrix(imatrix_path)

            # Should create empty file
            assert imatrix_path.exists()

    def test_log_config_summary(self):
        """Test logging configuration summary."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.lib.progress import ProgressReporter
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)
            progress = MagicMock(spec=ProgressReporter)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
            )

            quantizer._layer_configs = {
                "layer1": LayerQuantizationConfig("layer1", 0, bit_width=4),
                "layer2": LayerQuantizationConfig("layer2", 1, bit_width=2),
            }

            quantizer._log_config_summary()

            assert progress.log_info.called

    def test_log_config_summary_no_progress(self):
        """Test logging configuration summary without progress reporter."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "layer1": LayerQuantizationConfig("layer1", 0, bit_width=4),
            }

            # Should not raise
            quantizer._log_config_summary()

    def test_get_output_size_file(self):
        """Test getting output size from file."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"x" * 1000)

            size = quantizer._get_output_size(test_file)
            assert size == 1000

    def test_get_output_size_directory(self):
        """Test getting output size from directory."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            (test_dir / "file1.bin").write_bytes(b"x" * 500)
            (test_dir / "file2.bin").write_bytes(b"x" * 500)

            size = quantizer._get_output_size(test_dir)
            assert size == 1000

    def test_get_output_size_nonexistent(self):
        """Test getting output size for nonexistent path."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer._get_output_size(Path(tmpdir) / "nonexistent")
            assert size == 0

    def test_calculate_average_bits_empty(self):
        """Test average bits calculation with empty configs."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                target_avg_bits=3.5,
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            # Empty layer configs
            quantizer._layer_configs = {}

            avg = quantizer._calculate_average_bits()
            assert avg == 3.5  # Falls back to profile target

    def test_get_output_path_with_custom_name(self):
        """Test output path with custom name."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="dynamic",
                output_dir=str(tmpdir),
                output_name="custom_name.gguf",
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()
            assert "custom_name.gguf" in str(path)

    def test_get_output_path_non_gguf(self):
        """Test output path for non-GGUF format."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.AWQ,
                quantization_level="dynamic",
                output_dir=str(tmpdir),
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()
            assert "dynamic" in str(path)
            assert ".gguf" not in str(path)

    def test_quantize_with_progress_reporter(self):
        """Test quantize method logs progress."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.lib.progress import ProgressReporter
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)
            progress = MagicMock(spec=ProgressReporter)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
                enable_checkpoints=False,
            )

            result = quantizer.quantize()

            assert result is not None
            assert progress.log_info.called

    def test_generate_configs_from_importance_empty(self):
        """Test generating configs from empty importance matrix."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                importance_matrix=None,
            )

            # Should not crash
            quantizer._generate_configs_from_importance()
            assert len(quantizer._layer_configs) == 0

    def test_generate_default_configs_with_profile(self):
        """Test generating default configs with profile bits."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            profile = DynamicQuantizationProfile(
                profile_name="test",
                attention_bits=6,
                mlp_bits=2,
                embedding_bits=8,
            )

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                profile=profile,
            )

            quantizer._generate_default_configs()

            # Should have configs with profile bits
            assert len(quantizer._layer_configs) > 0
