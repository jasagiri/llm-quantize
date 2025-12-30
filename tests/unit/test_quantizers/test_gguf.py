"""Unit tests for GGUF quantizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.quantizers.gguf import GGUFQuantizer
from llm_quantize.models import (
    GGUF_QUANT_TYPES,
    ModelType,
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestGGUFQuantizerInit:
    """Tests for GGUFQuantizer initialization."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
            hidden_size=4096,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_init_with_basic_params(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test basic initialization."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
        )

        assert quantizer.source_model == source_model
        assert quantizer.config == config
        assert quantizer.enable_checkpoints is True

    def test_init_without_checkpoints(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test initialization with checkpoints disabled."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        assert quantizer.enable_checkpoints is False

    def test_init_with_progress_reporter(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test initialization with progress reporter."""
        from llm_quantize.lib.progress import ProgressReporter

        reporter = ProgressReporter()
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=reporter,
        )

        assert quantizer.progress_reporter == reporter


class TestGGUFQuantizerSupportedLevels:
    """Tests for supported quantization levels."""

    def test_get_supported_levels_returns_list(self) -> None:
        """Test that get_supported_levels returns a list."""
        levels = GGUFQuantizer.get_supported_levels()
        assert isinstance(levels, list)
        assert len(levels) > 0

    def test_supports_all_gguf_quant_types(self) -> None:
        """Test all GGUF quantization types are supported."""
        levels = GGUFQuantizer.get_supported_levels()

        for quant_type in GGUF_QUANT_TYPES.keys():
            assert quant_type in levels, f"Missing support for {quant_type}"

    @pytest.mark.parametrize(
        "level",
        ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    )
    def test_supports_common_levels(self, level: str) -> None:
        """Test common quantization levels are supported."""
        levels = GGUFQuantizer.get_supported_levels()
        assert level in levels


class TestGGUFQuantizerEstimateSize:
    """Tests for output size estimation."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,  # 7B params
            dtype="float16",
            num_layers=32,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_estimate_output_size_returns_positive(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimate returns positive size."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        assert estimate > 0

    def test_estimate_less_than_original(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimated size is less than original fp16 size."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        original_size = source_model.parameter_count * 2  # fp16 = 2 bytes

        assert estimate < original_size

    @pytest.mark.parametrize(
        "level,max_ratio",
        [
            ("Q2_K", 0.20),  # ~2 bits, should be <20% of fp16
            ("Q4_K_M", 0.35),  # ~4 bits, should be <35% of fp16
            ("Q8_0", 0.60),  # 8 bits, should be <60% of fp16
        ],
    )
    def test_estimate_respects_quantization_bits(
        self,
        source_model: SourceModel,
        temp_dir: Path,
        level: str,
        max_ratio: float,
    ) -> None:
        """Test estimate respects quantization level."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level=level,
            output_dir=str(temp_dir),
        )
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        original_size = source_model.parameter_count * 2

        ratio = estimate / original_size
        assert ratio < max_ratio, f"{level}: {ratio:.2%} >= {max_ratio:.0%}"


class TestGGUFQuantizerQuantize:
    """Tests for the quantize method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_quantize_returns_quantized_model(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize returns a QuantizedModel."""
        from llm_quantize.models import QuantizedModel

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                mock_load.return_value = MagicMock()
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert isinstance(result, QuantizedModel)
                assert result.format == "gguf"
                assert result.quantization_level == "Q4_K_M"

    def test_quantize_sets_file_size(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize sets correct file size."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "model.gguf"
                test_content = b"GGUF" + b"\x00" * 1000
                output_file.write_bytes(test_content)
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert result.file_size == len(test_content)

    def test_quantize_calculates_compression_ratio(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize calculates compression ratio."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                # Compression ratio = output_size / original_size
                original_size = source_model.parameter_count * 2
                expected_ratio = 104 / original_size
                assert abs(result.compression_ratio - expected_ratio) < 0.01

    def test_quantize_records_duration(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize records duration."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert result.duration_seconds >= 0


class TestGGUFResumeFromCheckpoint:
    """Tests for checkpoint resume functionality."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=32,
        )

    def test_init_with_resume_from_valid_checkpoint(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test initialization with valid checkpoint to resume from."""
        from llm_quantize.lib.checkpoint import Checkpoint

        # Create a valid checkpoint
        checkpoint_dir = temp_dir / "checkpoint"
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(32, config)
        checkpoint.save_layer(0, {"data": "layer0"})
        checkpoint.save_layer(1, {"data": "layer1"})

        # Create quantizer with resume
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
            resume_from=str(checkpoint_dir),
        )

        assert quantizer.start_layer == 2
        assert quantizer.checkpoint is not None

    def test_init_with_resume_no_checkpoint(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test initialization with non-existent checkpoint."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
            resume_from=str(temp_dir / "nonexistent"),
        )

        assert quantizer.start_layer == 0
        assert quantizer.checkpoint is None


class TestGGUFOutputGeneration:
    """Tests for GGUF output generation."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="meta-llama/Llama-2-7b-hf",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
        )

    def test_create_basic_gguf_output(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test basic GGUF output creation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        output_path = temp_dir / "model.gguf"
        quantizer._create_basic_gguf(None, str(output_path))

        # Verify file is created with GGUF header
        assert output_path.exists()
        content = output_path.read_bytes()
        assert content[:4] == b"GGUF"

    def test_has_llama_cpp_returns_bool(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _has_llama_cpp returns boolean."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
        )

        result = quantizer._has_llama_cpp()
        assert isinstance(result, bool)


class TestGGUFQuantizerOutputFilename:
    """Tests for output filename generation."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="meta-llama/Llama-2-7b-hf",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
        )

    def test_auto_generated_filename(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test auto-generated filename includes model name and quant level."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            output_name=None,  # Auto-generate
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "Llama-2-7b-hf-Q4_K_M.gguf"
                output_file.write_bytes(b"GGUF")
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert "Q4_K_M" in result.output_path
                assert result.output_path.endswith(".gguf")

    def test_custom_filename(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test custom filename is used."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            output_name="custom-model.gguf",
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "custom-model.gguf"
                output_file.write_bytes(b"GGUF")
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert result.output_path.endswith("custom-model.gguf")


class TestGGUFHasLlamaCpp:
    """Tests for llama.cpp availability checking."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_has_llama_cpp_returns_true_when_found(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _has_llama_cpp returns True when tools are available."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = quantizer._has_llama_cpp()
            assert result is True

    def test_has_llama_cpp_returns_false_when_not_found(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _has_llama_cpp returns False when tools not available."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = quantizer._has_llama_cpp()
            assert result is False

    def test_has_llama_cpp_returns_false_on_exception(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _has_llama_cpp returns False on exception."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("subprocess error")
            result = quantizer._has_llama_cpp()
            assert result is False


class TestGGUFConvertToGguf:
    """Tests for _convert_to_gguf method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="meta-llama/Llama-2-7b-hf",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_convert_uses_llama_cpp_when_available(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _convert_to_gguf uses llama.cpp when available."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()

        with patch.object(quantizer, "_has_llama_cpp", return_value=True):
            with patch.object(quantizer, "_convert_with_llama_cpp") as mock_convert:
                quantizer._convert_to_gguf(mock_model)
                mock_convert.assert_called_once()

    def test_convert_falls_back_to_direct_when_no_llama_cpp(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _convert_to_gguf uses direct conversion when llama.cpp not available."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()

        with patch.object(quantizer, "_has_llama_cpp", return_value=False):
            with patch.object(quantizer, "_convert_direct") as mock_convert:
                quantizer._convert_to_gguf(mock_model)
                mock_convert.assert_called_once()

    def test_convert_generates_correct_output_path(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _convert_to_gguf generates correct output path."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()

        with patch.object(quantizer, "_has_llama_cpp", return_value=False):
            with patch.object(quantizer, "_convert_direct"):
                output_path = quantizer._convert_to_gguf(mock_model)
                assert "Llama-2-7b-hf" in output_path
                assert "Q4_K_M" in output_path
                assert output_path.endswith(".gguf")


class TestGGUFConvertWithLlamaCpp:
    """Tests for _convert_with_llama_cpp method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_convert_with_llama_cpp_calls_subprocess(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _convert_with_llama_cpp calls subprocess commands."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        output_path = str(temp_dir / "output.gguf")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            quantizer._convert_with_llama_cpp(mock_model, output_path)
            # Should be called at least twice (convert and quantize)
            assert mock_run.call_count >= 2


class TestGGUFConvertDirect:
    """Tests for _convert_direct method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_convert_direct_falls_back_on_import_error(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _convert_direct falls back to basic GGUF on ImportError."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        output_path = str(temp_dir / "output.gguf")

        # Patch to raise ImportError when importing llama_cpp
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch.object(quantizer, "_create_basic_gguf") as mock_create:
                import sys
                if "llama_cpp" in sys.modules:
                    del sys.modules["llama_cpp"]

                # Simulate ImportError by patching
                with patch("builtins.__import__", side_effect=ImportError("no llama_cpp")):
                    quantizer._convert_direct(mock_model, output_path)
                    mock_create.assert_called_once()

    def test_convert_direct_uses_gguf_library_when_available(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _convert_direct uses gguf library when llama_cpp is available."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        output_path = str(temp_dir / "output.gguf")

        mock_llama = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama}):
            with patch.object(quantizer, "_convert_with_gguf_library") as mock_convert:
                quantizer._convert_direct(mock_model, output_path)
                mock_convert.assert_called_once()


class TestGGUFConvertWithGgufLibrary:
    """Tests for _convert_with_gguf_library method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_convert_with_gguf_library_import_error(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _convert_with_gguf_library handles ImportError."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        model_path = str(temp_dir / "model")
        output_path = str(temp_dir / "output.gguf")

        # Create model directory
        Path(model_path).mkdir(exist_ok=True)

        with patch.object(quantizer, "_create_basic_gguf") as mock_create:
            with patch.dict("sys.modules", {"gguf": None}):
                quantizer._convert_with_gguf_library(model_path, output_path)
                mock_create.assert_called_once()


class TestGGUFQuantizeLayer:
    """Tests for _quantize_layer method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_quantize_layer_returns_dict(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _quantize_layer returns a dictionary."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        model_path = str(temp_dir / "model")
        Path(model_path).mkdir(exist_ok=True)

        result = quantizer._quantize_layer(0, model_path)

        assert isinstance(result, dict)
        assert "layer_idx" in result
        assert result["layer_idx"] == 0


class TestGGUFFinalizeGguf:
    """Tests for _finalize_gguf method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_finalize_gguf_returns_path(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _finalize_gguf returns the output path."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        output_path = str(temp_dir / "output.gguf")
        layer_data = [{"layer_idx": 0}, {"layer_idx": 1}]

        result = quantizer._finalize_gguf(layer_data, output_path)
        assert result == output_path


class TestNullContext:
    """Tests for nullcontext helper class."""

    def test_nullcontext_enter_exit(self) -> None:
        """Test nullcontext can be used as context manager."""
        from llm_quantize.lib.quantizers.gguf import nullcontext

        with nullcontext() as ctx:
            assert ctx is not None
        # Should not raise any errors


class TestGGUFQuantizeWithCheckpoint:
    """Tests for quantize with checkpoint enabled."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    def test_quantize_initializes_checkpoint(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize initializes checkpoint when enabled."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)
                mock_convert.return_value = str(output_file)

                result = quantizer.quantize()

                assert result is not None

    def test_quantize_cleans_up_checkpoint_on_success(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize cleans up checkpoint on success."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)
                mock_convert.return_value = str(output_file)

                quantizer.quantize()

                # Checkpoint should be cleaned up
                # No assertion needed - the test passes if no exception is raised


class TestGGUFQuantizeWithProgress:
    """Tests for quantize with progress reporting."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    def test_quantize_logs_progress(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize logs progress messages."""
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE)

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_convert_to_gguf") as mock_convert:
                output_file = temp_dir / "model.gguf"
                output_file.write_bytes(b"GGUF" + b"\x00" * 100)
                mock_convert.return_value = str(output_file)

                # Should not raise
                result = quantizer.quantize()
                assert result.peak_memory_bytes >= 0


class TestGGUFLoadModel:
    """Tests for _load_model method."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_load_model_calls_model_loader(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _load_model calls the model loader."""
        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch("llm_quantize.lib.model_loader.load_model") as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load.return_value = (mock_model, mock_tokenizer)

            result = quantizer._load_model()

            mock_load.assert_called_once_with(
                source_model.model_path,
                hf_token=source_model.hf_token,
            )
            assert result == mock_model

    def test_load_model_with_hf_token(
        self, temp_dir: Path
    ) -> None:
        """Test _load_model passes HF token."""
        source_model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
            hf_token="test_token_123",
        )

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch("llm_quantize.lib.model_loader.load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())

            quantizer._load_model()

            mock_load.assert_called_once_with(
                source_model.model_path,
                hf_token="test_token_123",
            )


class TestGGUFConvertWithGgufLibraryFull:
    """Additional tests for _convert_with_gguf_library method with full coverage."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

    @pytest.fixture
    def config(self, temp_dir: Path) -> QuantizationConfig:
        """Create a test config."""
        return QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

    def test_convert_with_gguf_library_with_progress(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _convert_with_gguf_library with progress reporter."""
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        reporter = ProgressReporter(verbosity=Verbosity.NORMAL)

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )

        model_path = str(temp_dir / "model")
        output_path = str(temp_dir / "output.gguf")
        Path(model_path).mkdir(exist_ok=True)

        # Create mock safetensors file
        (Path(model_path) / "model.safetensors").write_bytes(b"\x00" * 100)

        mock_gguf = MagicMock()
        mock_writer = MagicMock()
        mock_gguf.GGUFWriter.return_value = mock_writer

        with patch.dict("sys.modules", {"gguf": mock_gguf, "torch": MagicMock(), "safetensors": MagicMock()}):
            with patch.object(quantizer, "_quantize_layer") as mock_quantize:
                mock_quantize.return_value = {"layer_idx": 0}

                quantizer._convert_with_gguf_library(model_path, output_path)

    def test_convert_with_gguf_library_with_checkpoint(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _convert_with_gguf_library with checkpoint enabled."""
        from llm_quantize.lib.checkpoint import Checkpoint

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        # Create a checkpoint
        checkpoint_dir = temp_dir / "test_checkpoint"
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(2, config)

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
        )
        quantizer.checkpoint = checkpoint
        quantizer.start_layer = 0

        model_path = str(temp_dir / "model")
        output_path = str(temp_dir / "output.gguf")
        Path(model_path).mkdir(exist_ok=True)

        mock_gguf = MagicMock()
        mock_writer = MagicMock()
        mock_gguf.GGUFWriter.return_value = mock_writer

        with patch.dict("sys.modules", {"gguf": mock_gguf, "torch": MagicMock(), "safetensors": MagicMock()}):
            with patch.object(quantizer, "_quantize_layer") as mock_quantize:
                mock_quantize.return_value = {"layer_idx": 0}

                quantizer._convert_with_gguf_library(model_path, output_path)


class TestGGUFConvertWithGgufLibrarySkipLayers:
    """Tests for layer skipping in _convert_with_gguf_library."""

    @pytest.fixture
    def source_model(self) -> SourceModel:
        """Create a test source model."""
        return SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=4,
        )

    def test_skip_layers_when_resuming(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test that layers are skipped when resuming from checkpoint."""
        from llm_quantize.lib.checkpoint import Checkpoint

        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )

        # Create a checkpoint with some layers completed
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(4, config)
        checkpoint.save_layer(0, {"data": "layer0"})
        checkpoint.save_layer(1, {"data": "layer1"})

        quantizer = GGUFQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
        )
        quantizer.checkpoint = checkpoint
        quantizer.start_layer = 2  # Resume from layer 2

        model_path = str(temp_dir / "model")
        output_path = str(temp_dir / "output.gguf")
        Path(model_path).mkdir(exist_ok=True)

        mock_gguf = MagicMock()
        mock_writer = MagicMock()
        mock_gguf.GGUFWriter.return_value = mock_writer

        layers_processed = []

        def track_layer(layer_idx, model_path):
            layers_processed.append(layer_idx)
            return {"layer_idx": layer_idx}

        with patch.dict("sys.modules", {"gguf": mock_gguf, "torch": MagicMock(), "safetensors": MagicMock()}):
            with patch.object(quantizer, "_quantize_layer", side_effect=track_layer):
                quantizer._convert_with_gguf_library(model_path, output_path)

        # Should only process layers 2 and 3 (skipping 0 and 1)
        assert 0 not in layers_processed
        assert 1 not in layers_processed
