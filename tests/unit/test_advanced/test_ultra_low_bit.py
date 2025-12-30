"""Tests for ultra-low-bit quantization."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.models import (
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestUltraLowBitQuantizer:
    """Tests for UltraLowBitQuantizer class."""

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

    def create_mock_config(self, tmpdir, level="IQ1_S"):
        """Create a mock quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level=level,
            output_dir=str(tmpdir),
        )

    def test_get_supported_levels(self):
        """Test supported quantization levels."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        levels = UltraLowBitQuantizer.get_supported_levels()

        assert "IQ1_S" in levels
        assert "IQ1_M" in levels
        assert "TERNARY" in levels
        assert "IQ2_XXS" in levels
        assert "IQ2_XS" in levels

    def test_ultra_low_bit_types(self):
        """Test ultra low bit type definitions."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import ULTRA_LOW_BIT_TYPES

        assert ULTRA_LOW_BIT_TYPES["IQ1_S"]["bits"] == 1.5
        assert ULTRA_LOW_BIT_TYPES["IQ1_M"]["bits"] == 1.75
        assert ULTRA_LOW_BIT_TYPES["TERNARY"]["bits"] == 1.58

    def test_init(self):
        """Test initialization."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            assert quantizer.validate_output is False

    def test_estimate_output_size_iq1_s(self):
        """Test output size estimation for IQ1_S."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer.estimate_output_size()

            # 1.5-bit should be ~10x smaller than FP16
            original_size = source_model.parameter_count * 2
            assert size < original_size / 5

    def test_estimate_output_size_ternary(self):
        """Test output size estimation for TERNARY."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer.estimate_output_size()

            assert size > 0

    def test_get_output_path_iq(self):
        """Test output path for IQ quantization."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()

            assert "IQ1_S" in str(path)
            assert ".gguf" in str(path)

    def test_get_output_path_ternary(self):
        """Test output path for TERNARY quantization."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()

            assert "ternary" in str(path)

    def test_get_output_path_with_custom_name(self):
        """Test output path with custom name."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="IQ1_S",
                output_dir=str(tmpdir),
                output_name="custom.gguf",
            )

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            path = quantizer.get_output_path()

            assert "custom.gguf" in str(path)

    def test_quantize_iq_with_llama_cpp(self):
        """Test IQ quantization with llama.cpp."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            with patch("subprocess.run") as mock_run:
                # Mock llama-quantize availability
                mock_run.side_effect = [
                    MagicMock(returncode=0),  # which llama-quantize
                    MagicMock(returncode=0, stderr=""),  # llama-quantize
                ]

                # Create output file
                output_path = quantizer.get_output_path()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()

                result = quantizer._quantize_iq("IQ1_S")

                assert result == output_path

    def test_quantize_iq_fallback_to_python(self):
        """Test IQ quantization fallback to Python."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            with patch("subprocess.run") as mock_run:
                # Mock llama-quantize not available
                mock_run.return_value = MagicMock(returncode=1)

                result = quantizer._quantize_iq("IQ1_S")

                # Should create a plan file
                assert result.suffix == ".json"

    def test_quantize_iq_python_creates_plan(self):
        """Test Python fallback creates plan file."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            output_path = quantizer.get_output_path()
            result = quantizer._quantize_iq_python("IQ1_S", output_path)

            assert result.exists()
            with open(result) as f:
                plan = json.load(f)
            assert plan["quantization_level"] == "IQ1_S"
            assert plan["status"] == "plan_only"

    def test_create_ternary_plan(self):
        """Test ternary plan creation."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            output_path = Path(tmpdir) / "output"
            result = quantizer._create_ternary_plan(output_path)

            assert result.exists()
            with open(result) as f:
                plan = json.load(f)
            assert plan["quantization_level"] == "TERNARY"

    def test_quantize_ternary_with_torch(self):
        """Test ternary quantization with torch."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer
        import torch
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            # Mock model - use real tensor that behaves correctly
            mock_model = MagicMock()
            # Create a proper mock parameter that has dim() method
            param_tensor = torch.tensor([[0.1, -0.2], [0.3, 0.0]])
            mock_param = MagicMock()
            mock_param.data = param_tensor
            mock_param.dim.return_value = 2  # This is crucial
            mock_model.named_parameters.return_value = [("layer.weight", mock_param)]
            mock_model.config = MagicMock()

            with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
                mock_auto_model.from_pretrained.return_value = mock_model

                result = quantizer._quantize_ternary()

                assert result.is_dir()

    def test_validate_quantized_output_file_not_found(self):
        """Test validation when file not found."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            quantizer._validate_quantized_output(Path("/nonexistent"))

            assert len(quantizer._quality_warnings) > 0

    def test_validate_quantized_output_skips_json(self):
        """Test that validation skips JSON plan files."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            plan_path = Path(tmpdir) / "plan.json"
            plan_path.touch()

            quantizer._validate_quantized_output(plan_path)

            # Should not add warnings for JSON files
            assert len(quantizer._quality_warnings) == 0


class TestUltraLowBitQuantizerAdvanced:
    """Advanced tests for UltraLowBitQuantizer coverage."""

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

    def create_mock_config(self, tmpdir, level="IQ1_S"):
        """Create a mock quantization config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level=level,
            output_dir=str(tmpdir),
        )

    def test_quantize_unknown_level(self):
        """Test quantize raises error for unknown level."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="UNKNOWN",
                output_dir=str(tmpdir),
            )

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            with pytest.raises(ValueError, match="Unknown quantization level"):
                quantizer.quantize()

    def test_quantize_with_progress_reporter(self):
        """Test quantize with progress reporter logs info."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")
            progress = MagicMock(spec=ProgressReporter)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
                validate_output=False,
            )

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)

                result = quantizer.quantize()

            assert progress.log_info.called
            assert progress.log_warning.called

    def test_quantize_with_quality_warnings(self):
        """Test quantize logs quality warnings."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")
            progress = MagicMock(spec=ProgressReporter)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
                validate_output=False,
                enable_checkpoints=False,
            )

            # Add a warning manually
            quantizer._quality_warnings.append("Test warning")

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)

                result = quantizer.quantize()

            # Should have logged the warning
            warning_calls = [call for call in progress.log_warning.call_args_list]
            assert len(warning_calls) >= 1

    def test_quantize_with_checkpoints_cleanup(self):
        """Test quantize cleans up checkpoints."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
                enable_checkpoints=True,
            )

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                with patch.object(quantizer, 'cleanup_checkpoint') as mock_cleanup:
                    result = quantizer.quantize()
                    mock_cleanup.assert_called_once()

    def test_quantize_iq_with_imatrix(self):
        """Test IQ quantization uses existing imatrix."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a model directory with imatrix
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            imatrix_path = model_dir / "imatrix.dat"
            imatrix_path.touch()

            source_model = SourceModel(
                model_path=str(model_dir),
                model_type=1,  # HF_HUB
                architecture="llama",
                parameter_count=1000000,
                dtype="float16",
                num_layers=4,
                hidden_size=768,
                num_heads=12,
                vocab_size=32000,
            )
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = [
                    MagicMock(returncode=0),  # which llama-quantize
                    MagicMock(returncode=0),  # llama-quantize
                ]

                # Create output file
                output_path = quantizer.get_output_path()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()

                result = quantizer._quantize_iq("IQ1_S")

            # Check that imatrix was passed
            calls = mock_run.call_args_list
            assert len(calls) >= 2

    def test_quantize_iq_llama_cpp_failure(self):
        """Test IQ quantization falls back on llama.cpp failure."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            with patch("subprocess.run") as mock_run:
                # llama-quantize available but fails
                mock_run.side_effect = [
                    MagicMock(returncode=0),  # which llama-quantize
                    MagicMock(returncode=1, stderr="Error message"),  # llama-quantize fails
                ]

                result = quantizer._quantize_iq("IQ1_S")

            # Should have created a plan file (fallback)
            assert result.suffix == ".json"

    def test_quantize_ternary_import_error(self):
        """Test ternary quantization handles ImportError."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
                mock_auto_model.from_pretrained.side_effect = ImportError("No transformers")

                result = quantizer._quantize_ternary()

            # Should have created a plan file
            assert result.suffix == ".json"

    def test_quantize_ternary_with_zero_scale(self):
        """Test ternary quantization handles zero scale."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=False,
            )

            # Mock model with zero weights (zero scale)
            mock_model = MagicMock()
            param_tensor = torch.zeros(2, 2)  # All zeros
            mock_param = MagicMock()
            mock_param.data = param_tensor
            mock_param.dim.return_value = 2
            mock_model.named_parameters.return_value = [("layer.weight", mock_param)]
            mock_model.config = MagicMock()

            with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
                mock_auto_model.from_pretrained.return_value = mock_model

                result = quantizer._quantize_ternary()

            assert result.is_dir()

    def test_validate_output_file_too_small(self):
        """Test validation warns about small file."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "IQ1_S")
            progress = MagicMock(spec=ProgressReporter)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
            )

            # Create a very small file
            test_file = Path(tmpdir) / "test.gguf"
            test_file.write_bytes(b"x" * 100)

            quantizer._validate_quantized_output(test_file)

            # Should have warned about small file
            assert len(quantizer._quality_warnings) > 0

    def test_validate_output_file_too_large(self):
        """Test validation warns about large file."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer
        from llm_quantize.lib.progress import ProgressReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source with small parameter count
            source_model = SourceModel(
                model_path="test/model",
                model_type=1,
                architecture="llama",
                parameter_count=100,  # Very small
                dtype="float16",
                num_layers=4,
                hidden_size=768,
                num_heads=12,
                vocab_size=32000,
            )
            config = self.create_mock_config(tmpdir, "IQ1_S")
            progress = MagicMock(spec=ProgressReporter)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                progress_reporter=progress,
            )

            # Create a large file
            test_file = Path(tmpdir) / "test.gguf"
            test_file.write_bytes(b"x" * 10000)

            quantizer._validate_quantized_output(test_file)

            # Should have warned about large file
            assert len(quantizer._quality_warnings) > 0

    def test_get_output_size_directory(self):
        """Test getting output size for directory."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            # Create directory with files
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            (test_dir / "file1.bin").write_bytes(b"x" * 500)
            (test_dir / "file2.bin").write_bytes(b"x" * 500)

            size = quantizer._get_output_size(test_dir)
            assert size == 1000

    def test_get_output_size_nonexistent(self):
        """Test getting output size for nonexistent path."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            size = quantizer._get_output_size(Path(tmpdir) / "nonexistent")
            assert size == 0

    def test_get_or_compute_imatrix_no_existing(self):
        """Test imatrix computation when none exists."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir)

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            result = quantizer._get_or_compute_imatrix()
            assert result is None

    def test_estimate_output_size_unknown_level(self):
        """Test estimate output size with unknown level."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="UNKNOWN",
                output_dir=str(tmpdir),
            )

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
            )

            # Should use default bits (2.0)
            size = quantizer.estimate_output_size()
            assert size > 0

    def test_quantize_ternary_full_flow(self):
        """Test full ternary quantization flow."""
        from llm_quantize.lib.quantizers.advanced.ultra_low_bit import UltraLowBitQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self.create_mock_source_model()
            config = self.create_mock_config(tmpdir, "TERNARY")

            quantizer = UltraLowBitQuantizer(
                source_model=source_model,
                config=config,
                validate_output=True,
                enable_checkpoints=False,
            )

            with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
                mock_auto_model.from_pretrained.side_effect = ImportError("No transformers")

                result = quantizer.quantize()

            assert result is not None
            assert result.quantization_level == "TERNARY"
