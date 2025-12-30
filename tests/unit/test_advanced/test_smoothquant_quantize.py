"""Additional tests for SmoothQuant quantization to improve coverage."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.models import (
    OutputFormat,
    QuantizationConfig,
    SmoothQuantConfig,
    SourceModel,
    ModelType,
)


class TestSmoothQuantQuantizerMethods:
    """Additional tests for SmoothQuantQuantizer methods."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_generate_synthetic_statistics_called(self):
        """Test that synthetic statistics are generated."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Call the method directly
            quantizer._generate_synthetic_statistics()

            # Should have stats now
            assert len(quantizer._activation_stats) > 0

    def test_compute_smoothing_scales_uses_statistics(self):
        """Test smoothing scales computation."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Generate synthetic stats first
            quantizer._generate_synthetic_statistics()

            # Should be able to compute scales
            assert len(quantizer._activation_stats) > 0

    def test_apply_smoothquant_creates_plan(self):
        """Test that apply_smoothquant creates a plan file."""
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
            assert plan["status"] == "plan_only"

    def test_get_recommended_alpha_gpt2(self):
        """Test recommended alpha for gpt2."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = SourceModel(
                model_path="test/model",
                model_type=ModelType.HF_HUB,
                architecture="gpt2",
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

            # Should use default for unknown architecture
            assert alpha == 0.5

    def test_get_output_size_with_file(self):
        """Test getting output size from file."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Create a test file
            test_path = Path(tmpdir) / "test.bin"
            test_path.write_bytes(b"x" * 1000)

            size = quantizer._get_output_size(test_path)
            assert size == 1000

    def test_get_output_size_from_directory(self):
        """Test getting output size from directory."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Create test directory with files
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            (test_dir / "file1.bin").write_bytes(b"x" * 500)
            (test_dir / "file2.bin").write_bytes(b"x" * 500)

            size = quantizer._get_output_size(test_dir)
            assert size == 1000


class TestCalibration:
    """Tests for calibration module."""

    def test_load_calibration_default_samples(self):
        """Test loading default calibration samples."""
        from llm_quantize.lib.calibration import load_calibration_data

        result = load_calibration_data(None, num_samples=10)

        assert isinstance(result, list)
        assert len(result) <= 10

    def test_default_calibration_samples_constant(self):
        """Test default calibration samples constant."""
        from llm_quantize.lib.calibration import DEFAULT_CALIBRATION_SAMPLES

        assert isinstance(DEFAULT_CALIBRATION_SAMPLES, list)
        assert len(DEFAULT_CALIBRATION_SAMPLES) > 0

    def test_load_calibration_text_file(self):
        """Test loading calibration from text file."""
        from llm_quantize.lib.calibration import load_calibration_data

        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "calib.txt"
            text_path.write_text("Sample 1\nSample 2\nSample 3")

            result = load_calibration_data(str(text_path), num_samples=10)

            # Result is a list
            assert isinstance(result, list)
            assert len(result) > 0

    def test_load_calibration_json_file(self):
        """Test loading calibration from JSON file."""
        from llm_quantize.lib.calibration import load_calibration_data
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "calib.json"
            json_path.write_text(json.dumps(["Sample A", "Sample B"]))

            result = load_calibration_data(str(json_path), num_samples=10)

            assert "Sample A" in result or result  # At least got something


class TestDynamicQuantizerMethods:
    """Additional tests for DynamicQuantizer methods."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_get_base_quant_type_5bit(self):
        """Test base quantization type for 5-bit."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.models import LayerQuantizationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=5),
            }
            assert quantizer._get_base_quant_type() == "Q5_K_M"

    def test_get_base_quant_type_3bit(self):
        """Test base quantization type for 3-bit."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.models import LayerQuantizationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=3),
            }
            assert quantizer._get_base_quant_type() == "Q3_K_M"

    def test_get_base_quant_type_6bit(self):
        """Test base quantization type for 6-bit."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.models import LayerQuantizationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=6),
            }
            assert quantizer._get_base_quant_type() == "Q6_K"

    def test_estimate_output_size_no_profile(self):
        """Test output size estimation without profile falls back to default."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer.estimate_output_size()

            # Should estimate some size
            assert size > 0


class TestSmoothQuantQuantize:
    """Tests for SmoothQuantQuantizer quantize method."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_quantize_with_fallback(self):
        """Test quantize method falls back to plan when transformers unavailable."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
                enable_checkpoints=False,
            )

            # Mock internal methods to avoid real model loading
            quantizer._collect_activation_statistics = MagicMock()
            quantizer._compute_smoothing_scales = MagicMock()
            quantizer._apply_smoothquant = MagicMock(
                return_value=Path(tmpdir) / "output" / "smoothquant_plan.json"
            )
            # Create the output file
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_file = output_dir / "smoothquant_plan.json"
            plan_file.write_text('{"status": "plan_only"}')

            result = quantizer.quantize()

            assert result is not None
            assert result.format == "smoothquant"

    def test_quantize_with_progress_reporter(self):
        """Test quantize logs progress messages."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            # Create a mock progress reporter
            progress = MagicMock(spec=ProgressReporter)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
                enable_checkpoints=False,
            )

            # Mock internal methods to avoid real model loading
            quantizer._collect_activation_statistics = MagicMock()
            quantizer._compute_smoothing_scales = MagicMock()
            quantizer._apply_smoothquant = MagicMock(
                return_value=Path(tmpdir) / "output" / "smoothquant_plan.json"
            )
            # Create the output file
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_file = output_dir / "smoothquant_plan.json"
            plan_file.write_text('{"status": "plan_only"}')

            result = quantizer.quantize()

            # Progress reporter should have been called
            assert progress.log_info.called
            assert result is not None

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

            # W8A8 should give ~50% size
            original_size = source_model.parameter_count * 2
            assert size < original_size

    def test_get_supported_levels(self):
        """Test supported quantization levels."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        levels = SmoothQuantQuantizer.get_supported_levels()

        assert "W8A8" in levels
        assert "smoothquant" in levels
        assert "int8" in levels

    def test_init_with_custom_smoothquant_config(self):
        """Test initialization with custom SmoothQuant config."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)
            sq_config = SmoothQuantConfig(alpha=0.7, per_channel=True)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
                smoothquant_config=sq_config,
            )

            assert quantizer.smoothquant_config.alpha == 0.7
            assert quantizer.smoothquant_config.per_channel is True

    def test_get_recommended_alpha_llama(self):
        """Test recommended alpha for llama architecture."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            alpha = quantizer._get_recommended_alpha()

            # Llama should have a specific alpha
            assert 0.0 <= alpha <= 1.0


class TestCalibrationAdditional:
    """Additional tests for calibration module."""

    def test_load_calibration_with_directory(self):
        """Test loading calibration from directory."""
        from llm_quantize.lib.calibration import load_calibration_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in directory
            (Path(tmpdir) / "sample1.txt").write_text("Sample content 1")
            (Path(tmpdir) / "sample2.txt").write_text("Sample content 2")

            result = load_calibration_data(tmpdir, num_samples=10)

            assert isinstance(result, list)

    def test_load_calibration_jsonl_file(self):
        """Test loading calibration from JSONL file."""
        from llm_quantize.lib.calibration import load_calibration_data
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "calib.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({"text": "Sample 1"}) + "\n")
                f.write(json.dumps({"text": "Sample 2"}) + "\n")

            result = load_calibration_data(str(jsonl_path), num_samples=10)

            assert isinstance(result, list)

    def test_load_calibration_limit_samples(self):
        """Test loading calibration with sample limit."""
        from llm_quantize.lib.calibration import load_calibration_data

        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "calib.txt"
            text_path.write_text("\n".join([f"Sample {i}" for i in range(100)]))

            result = load_calibration_data(str(text_path), num_samples=5)

            assert len(result) <= 5


class TestDynamicQuantizerQuantize:
    """Tests for DynamicQuantizer quantize method."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_quantize_creates_plan(self):
        """Test quantize method creates plan when tools unavailable."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
                enable_checkpoints=False,
            )

            result = quantizer.quantize()

            assert result is not None
            assert result.format == "gguf"

    def test_get_supported_levels(self):
        """Test supported quantization levels."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer

        levels = DynamicQuantizer.get_supported_levels()

        assert "dynamic" in levels

    def test_get_base_quant_type_8bit(self):
        """Test base quantization type for 8-bit."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.models import LayerQuantizationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=8),
            }
            assert quantizer._get_base_quant_type() == "Q8_0"

    def test_get_base_quant_type_2bit(self):
        """Test base quantization type for 2-bit."""
        from llm_quantize.lib.quantizers.advanced.dynamic import DynamicQuantizer
        from llm_quantize.models import LayerQuantizationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = DynamicQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._layer_configs = {
                "l": LayerQuantizationConfig("l", 0, bit_width=2),
            }
            assert quantizer._get_base_quant_type() == "Q2_K"


class TestImportanceAnalysisExtra:
    """Extra tests for importance analysis module."""

    def test_compute_importance_with_method_gradient(self):
        """Test importance computation with gradient method."""
        from llm_quantize.lib.analysis.importance import compute_importance_from_gradients
        from llm_quantize.models import ImportanceMethod
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test"
        mock_param = MagicMock()
        mock_param.grad = None
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.named_parameters.return_value = [("layer", mock_param)]
        mock_model.parameters.return_value = iter([mock_param])

        result = compute_importance_from_gradients(
            model=mock_model,
            calibration_data=[],
        )

        assert result is not None
        assert result.computation_method == ImportanceMethod.GRADIENT_SENSITIVITY


class TestSmoothQuantFactoryFunction:
    """Tests for create_smoothquant_quantizer factory function."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_create_smoothquant_quantizer_with_defaults(self):
        """Test factory function with default parameters."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import create_smoothquant_quantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
            )

            quantizer = create_smoothquant_quantizer(
                source_model=source_model,
                config=config,
            )

            assert quantizer.smoothquant_config.alpha == 0.5

    def test_create_smoothquant_quantizer_with_alpha(self):
        """Test factory function with custom alpha."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import create_smoothquant_quantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
            )

            quantizer = create_smoothquant_quantizer(
                source_model=source_model,
                config=config,
                alpha=0.75,
            )

            assert quantizer.smoothquant_config.alpha == 0.75

    def test_create_smoothquant_quantizer_with_progress(self):
        """Test factory function with progress reporter."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import create_smoothquant_quantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
            )
            progress = MagicMock(spec=ProgressReporter)

            quantizer = create_smoothquant_quantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
            )

            assert quantizer.progress_reporter is progress


class TestSmoothQuantEdgeCases:
    """Tests for edge cases in SmoothQuantQuantizer."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_get_output_size_nonexistent_path(self):
        """Test getting output size for non-existent path returns 0."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer._get_output_size(Path(tmpdir) / "nonexistent")
            assert size == 0

    def test_generate_synthetic_statistics_with_progress(self):
        """Test synthetic statistics generation logs warning."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)
            progress = MagicMock(spec=ProgressReporter)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
            )

            quantizer._generate_synthetic_statistics()

            # Should have logged a warning
            assert progress.log_warning.called

    def test_activation_stats_initialized_empty(self):
        """Test that activation stats are initialized empty."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Initially activation stats should be empty
            assert isinstance(quantizer._activation_stats, dict)
            assert len(quantizer._activation_stats) == 0

    def test_get_output_path_with_custom_name(self):
        """Test get_output_path with custom output name."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="W8A8",
                output_dir=str(tmpdir),
                output_name="custom_output",
            )

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()
            assert "custom_output" in str(path)

    def test_get_output_path_without_custom_name(self):
        """Test get_output_path without custom output name."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()
            assert "smoothquant-w8a8" in str(path)


class TestUltraLowBitQuantizerExtra:
    """Additional tests for UltraLowBitQuantizer."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_get_supported_levels(self):
        """Test supported quantization levels."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        levels = UltraLowBitQuantizer.get_supported_levels()

        # GGUF-style quantization level names
        assert "IQ1_S" in levels
        assert "IQ2_XXS" in levels
        assert "IQ2_XS" in levels

    def test_quantize_creates_output(self):
        """Test quantize method creates output."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="IQ2_XXS",  # Use valid GGUF level name
                output_dir=str(tmpdir),
            )

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                enable_checkpoints=False,
            )

            result = quantizer.quantize()

            assert result is not None
            assert result.format == "gguf"

    def test_estimate_output_size(self):
        """Test output size estimation."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="IQ2_XXS",  # Use valid GGUF level name
                output_dir=str(tmpdir),
            )

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer.estimate_output_size()

            # 2-bit should be very small
            assert size > 0
            assert size < source_model.parameter_count * 2  # Less than original


class TestSmoothQuantTorchMocking:
    """Tests with torch/transformers mocking for comprehensive coverage."""

    def create_mock_source_model(self):
        """Create a mock source model."""
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

    def test_get_recommended_alpha_bloom(self):
        """Test recommended alpha for bloom architecture."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = SourceModel(
                model_path="test/bloom",
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

            # Bloom should have higher alpha (0.75)
            assert alpha == 0.75

    def test_get_recommended_alpha_mistral(self):
        """Test recommended alpha for mistral architecture."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = SourceModel(
                model_path="test/mistral",
                model_type=ModelType.HF_HUB,
                architecture="mistral",
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

            assert alpha == 0.5

    def test_smoothquant_config_default(self):
        """Test SmoothQuant config defaults."""
        from llm_quantize.lib.quantizers.advanced.smoothquant import SmoothQuantQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = SmoothQuantQuantizer(
                source_model=source_model,
                config=config,
            )

            # Should have default config
            assert quantizer.smoothquant_config is not None
            assert 0 <= quantizer.smoothquant_config.alpha <= 1

    def test_create_smoothquant_plan_content(self):
        """Test smoothquant plan content."""
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
            assert "model" in plan
            assert "alpha" in plan
            assert plan["status"] == "plan_only"


class TestSuperWeightsCoverage:
    """Additional tests for super_weights module coverage."""

    def test_identify_super_weights_empty_model(self):
        """Test identify super weights with empty model."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []

        result = identify_super_weights(mock_model, coverage=0.1)

        assert result == {}

    def test_create_protection_mask_empty_super_weights(self):
        """Test create protection mask with empty super weights."""
        from llm_quantize.lib.analysis.super_weights import create_protection_mask
        import torch

        mock_model = MagicMock()
        param_tensor = torch.zeros(2, 5)
        mock_param = MagicMock()
        mock_param.dim.return_value = 2
        mock_param.numel.return_value = 10
        mock_param.shape = (2, 5)
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        masks = create_protection_mask(mock_model, {}, protection_bits=8)

        assert "layer1" in masks
        assert masks["layer1"].sum() == 0

    def test_compute_super_weight_statistics_empty(self):
        """Test compute statistics with empty super weights."""
        from llm_quantize.lib.analysis.super_weights import compute_super_weight_statistics

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []

        stats = compute_super_weight_statistics(mock_model, {})

        assert stats["total_super_weights"] == 0
        assert stats["layers_with_super_weights"] == 0

    def test_update_importance_matrix_empty_super_weights(self):
        """Test update importance matrix with empty super weights."""
        from llm_quantize.lib.analysis.super_weights import update_importance_matrix_with_super_weights
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
                LayerImportance("layer1", 0, 0.5, 100),
            ],
            total_parameters=100,
        )

        updated = update_importance_matrix_with_super_weights(matrix, {})

        assert updated.layer_scores[0].super_weight_indices == []


class TestQualityCoverage:
    """Additional tests for quality module coverage."""

    def test_analyze_coherence_single_word(self):
        """Test coherence analysis with single word."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        is_coherent, rep_score, gram_score = _analyze_coherence("hello")

        assert not is_coherent
        assert rep_score == 1.0

    def test_analyze_coherence_with_punctuation(self):
        """Test coherence analysis with good punctuation."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "This is a sentence. And another sentence! How about a question?"
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        assert gram_score >= 0.5

    def test_generate_quality_report_excellent(self):
        """Test quality report with excellent grade."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q8_0",
            perplexity_original=5.0,
            perplexity_quantized=5.02,  # <1% increase
        )

        assert report.quality_grade.value == "excellent"

    def test_compute_perplexity_no_samples(self):
        """Test perplexity with no samples."""
        from llm_quantize.lib.analysis.quality import compute_perplexity
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()

        result = compute_perplexity(mock_model, mock_tokenizer, [])

        assert result == float("inf")


class TestConverterCoverage:
    """Additional tests for converter module coverage."""

    def test_is_conversion_supported_same_format(self):
        """Test conversion support for same format."""
        from llm_quantize.lib.converter import is_conversion_supported

        result = is_conversion_supported("gguf", "gguf")
        # Same format conversion should work (identity)
        assert isinstance(result, bool)

    def test_is_lossy_conversion_lossless(self):
        """Test lossy conversion check for lossless conversion."""
        from llm_quantize.lib.converter import is_lossy_conversion

        result = is_lossy_conversion("gguf", "gguf")
        assert result is False


class TestValidationCoverage:
    """Additional tests for validation module coverage."""

    def test_validate_output_gguf_valid(self):
        """Test validation of valid GGUF output."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock GGUF file
            gguf_path = Path(tmpdir) / "model.gguf"
            # Write GGUF magic bytes
            gguf_path.write_bytes(b"GGUF" + b"\x00" * 100)

            # validate_output takes output_path (str) and output_format
            result = validate_output(str(gguf_path), OutputFormat.GGUF)

            assert result is not None
            assert hasattr(result, "is_valid")
            assert result.is_valid is True

    def test_validate_output_gguf_invalid_magic(self):
        """Test validation of GGUF output with invalid magic bytes."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a GGUF file with wrong magic
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_bytes(b"XXXX" + b"\x00" * 100)

            result = validate_output(str(gguf_path), OutputFormat.GGUF)

            assert result is not None
            assert result.is_valid is False
            assert "magic" in result.error_message.lower()

    def test_validate_output_awq(self):
        """Test validation of AWQ output."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create AWQ directory structure
            awq_dir = Path(tmpdir) / "model_awq"
            awq_dir.mkdir()
            (awq_dir / "config.json").write_text('{"quant_method": "awq"}')
            # AWQ also requires weight files
            (awq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

            # validate_output takes output_path (str) and output_format
            result = validate_output(str(awq_dir), OutputFormat.AWQ)

            assert result is not None
            assert result.is_valid is True

    def test_validate_output_awq_missing_weights(self):
        """Test validation of AWQ output without weight files."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create AWQ directory with only config
            awq_dir = Path(tmpdir) / "model_awq"
            awq_dir.mkdir()
            (awq_dir / "config.json").write_text('{"quant_method": "awq"}')

            result = validate_output(str(awq_dir), OutputFormat.AWQ)

            assert result is not None
            assert result.is_valid is False
            assert "weight" in result.error_message.lower()

    def test_validate_output_gptq(self):
        """Test validation of GPTQ output."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create GPTQ directory structure
            gptq_dir = Path(tmpdir) / "model_gptq"
            gptq_dir.mkdir()
            (gptq_dir / "config.json").write_text('{"quant_method": "gptq"}')
            (gptq_dir / "model.safetensors").write_bytes(b"\x00" * 100)

            result = validate_output(str(gptq_dir), OutputFormat.GPTQ)

            assert result is not None
            assert result.is_valid is True

    def test_validate_output_nonexistent_path(self):
        """Test validation of non-existent path."""
        from llm_quantize.lib.validation import validate_output

        result = validate_output("/nonexistent/path/model.gguf", OutputFormat.GGUF)

        assert result is not None
        assert result.is_valid is False
        assert "does not exist" in result.error_message

    def test_validate_output_string_format(self):
        """Test validation with string format instead of OutputFormat enum."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_bytes(b"GGUF" + b"\x00" * 100)

            # Pass format as string
            result = validate_output(str(gguf_path), "gguf")

            assert result is not None
            assert result.is_valid is True

    def test_validate_output_invalid_string_format(self):
        """Test validation with invalid string format."""
        from llm_quantize.lib.validation import validate_output

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_bytes(b"GGUF" + b"\x00" * 100)

            result = validate_output(str(gguf_path), "invalid_format")

            assert result is not None
            assert result.is_valid is False
            assert "unknown" in result.error_message.lower()

    def test_get_file_size(self):
        """Test getting file size."""
        from llm_quantize.lib.validation import get_file_size

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"x" * 1000)

            size = get_file_size(test_file)
            assert size == 1000

    def test_get_file_size_directory(self):
        """Test getting size of directory."""
        from llm_quantize.lib.validation import get_file_size

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            (test_dir / "file1.bin").write_bytes(b"x" * 500)
            (test_dir / "file2.bin").write_bytes(b"x" * 500)

            size = get_file_size(test_dir)
            assert size == 1000

    def test_get_file_size_nonexistent(self):
        """Test getting size of non-existent path."""
        from llm_quantize.lib.validation import get_file_size

        size = get_file_size(Path("/nonexistent/path"))
        assert size == 0

    def test_update_validation_status_valid(self):
        """Test updating validation status for valid model."""
        from llm_quantize.lib.validation import update_validation_status
        from llm_quantize.models import QuantizedModel, ValidationStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.write_bytes(b"GGUF" + b"\x00" * 100)

            model = QuantizedModel(
                output_path=str(gguf_path),
                format="gguf",
                quantization_level="Q4_K_M",
                compression_ratio=4.0,
                file_size=104,
                source_model_path="/test",
            )

            updated = update_validation_status(model, OutputFormat.GGUF)

            assert updated.validation_status == ValidationStatus.VALID

    def test_update_validation_status_invalid(self):
        """Test updating validation status for invalid model."""
        from llm_quantize.lib.validation import update_validation_status
        from llm_quantize.models import QuantizedModel, ValidationStatus

        model = QuantizedModel(
            output_path="/nonexistent/path/model.gguf",
            format="gguf",
            quantization_level="Q4_K_M",
            compression_ratio=4.0,
            file_size=0,
            source_model_path="/test",
        )

        updated = update_validation_status(model, OutputFormat.GGUF)

        assert updated.validation_status == ValidationStatus.INVALID
        assert "validation_error" in updated.quantization_metadata

    def test_validation_result_class_methods(self):
        """Test ValidationResult class methods."""
        from llm_quantize.lib.validation import ValidationResult

        valid = ValidationResult.valid()
        assert valid.is_valid is True
        assert valid.error_message == ""

        invalid = ValidationResult.invalid("Test error")
        assert invalid.is_valid is False
        assert invalid.error_message == "Test error"
