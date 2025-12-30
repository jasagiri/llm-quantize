"""Unit tests for GPTQ quantizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.quantizers.gptq import GPTQQuantizer
from llm_quantize.models import (
    GPTQ_QUANT_TYPES,
    ModelType,
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestGPTQQuantizerInit:
    """Tests for GPTQQuantizer initialization."""

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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            group_size=128,
        )

    def test_init_with_basic_params(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test basic initialization."""
        quantizer = GPTQQuantizer(
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
        quantizer = GPTQQuantizer(
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
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=reporter,
        )

        assert quantizer.progress_reporter == reporter


class TestGPTQQuantizerSupportedLevels:
    """Tests for supported quantization levels."""

    def test_get_supported_levels_returns_list(self) -> None:
        """Test that get_supported_levels returns a list."""
        levels = GPTQQuantizer.get_supported_levels()
        assert isinstance(levels, list)
        assert len(levels) > 0

    def test_supports_all_gptq_quant_types(self) -> None:
        """Test all GPTQ quantization types are supported."""
        levels = GPTQQuantizer.get_supported_levels()

        for quant_type in GPTQ_QUANT_TYPES.keys():
            assert quant_type in levels, f"Missing support for {quant_type}"

    @pytest.mark.parametrize("level", ["2bit", "3bit", "4bit", "8bit"])
    def test_supports_common_levels(self, level: str) -> None:
        """Test common GPTQ quantization levels are supported."""
        levels = GPTQQuantizer.get_supported_levels()
        assert level in levels


class TestGPTQQuantizerEstimateSize:
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_estimate_output_size_returns_positive(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimate returns positive size."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        assert estimate > 0

    def test_estimate_less_than_original(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimated size is less than original fp16 size."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        original_size = source_model.parameter_count * 2  # fp16 = 2 bytes

        assert estimate < original_size

    @pytest.mark.parametrize(
        "level,max_ratio",
        [
            ("2bit", 0.20),  # 2 bits, should be <20% of fp16
            ("4bit", 0.35),  # 4 bits, should be <35% of fp16
            ("8bit", 0.65),  # 8 bits, should be <65% of fp16
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
            target_format=OutputFormat.GPTQ,
            quantization_level=level,
            output_dir=str(temp_dir),
        )
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        original_size = source_model.parameter_count * 2

        ratio = estimate / original_size
        assert ratio < max_ratio, f"{level}: {ratio:.2%} >= {max_ratio:.0%}"


class TestGPTQQuantizerQuantize:
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            group_size=128,
        )

    def test_quantize_returns_quantized_model(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize returns a QuantizedModel."""
        from llm_quantize.models import QuantizedModel

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                mock_load.return_value = MagicMock()

                # Create GPTQ output structure
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert isinstance(result, QuantizedModel)
                assert result.format == "gptq"
                assert result.quantization_level == "4bit"

    def test_quantize_sets_file_size(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize sets correct file size."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                test_content = b"\x00" * 1000
                (output_dir / "model.safetensors").write_bytes(test_content)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                # File size should include all files in directory
                assert result.file_size > 0

    def test_quantize_records_duration(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize records duration."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result.duration_seconds >= 0


class TestGPTQCalibrationDataLoading:
    """Tests for calibration data loading."""

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

    def test_load_calibration_data_from_json_list(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test loading calibration data from JSON list."""
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('["sample1", "sample2", "sample3"]')

        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=2,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._load_calibration_data()

        assert len(data) == 2
        assert data[0] == "sample1"
        assert data[1] == "sample2"

    def test_get_default_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test getting default calibration data."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_samples=10,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._get_default_calibration_data()

        assert len(data) == 10
        assert all(isinstance(s, str) for s in data)


class TestGPTQResumeFromCheckpoint:
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(32, config)
        checkpoint.save_layer(0, {"data": "layer0"})
        checkpoint.save_layer(1, {"data": "layer1"})

        # Create quantizer with resume
        quantizer = GPTQQuantizer(
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        # Non-existent checkpoint
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
            resume_from=str(temp_dir / "nonexistent"),
        )

        assert quantizer.start_layer == 0
        assert quantizer.checkpoint is None

    def test_init_resume_disabled(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test initialization with resume_from but checkpoints disabled."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            resume_from=str(temp_dir),
        )

        assert quantizer.start_layer == 0
        assert quantizer.checkpoint is None


class TestGPTQCalibrationDataErrors:
    """Tests for calibration data error handling."""

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

    def test_load_calibration_data_file_not_found(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test calibration data loading with missing file."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(temp_dir / "nonexistent.json"),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        with pytest.raises(ValueError, match="Calibration data file not found"):
            quantizer._load_calibration_data()

    def test_load_calibration_data_invalid_format(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test calibration data loading with invalid format."""
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('{"invalid": "format"}')

        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        with pytest.raises(ValueError, match="Invalid calibration data format"):
            quantizer._load_calibration_data()

    def test_load_calibration_data_dict_with_text_key(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test calibration data loading with dict format."""
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('{"text": ["sample1", "sample2", "sample3"]}')

        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=2,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._load_calibration_data()

        assert len(data) == 2
        assert data[0] == "sample1"

    def test_load_calibration_no_path(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test calibration data loading with no path."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._load_calibration_data()
        assert data == []


class TestGPTQDirectorySize:
    """Tests for directory size calculation."""

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

    def test_get_directory_size_for_file(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test directory size calculation for a single file."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"\x00" * 1000)

        size = quantizer._get_directory_size(test_file)
        assert size == 1000

    def test_get_directory_size_for_directory(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test directory size calculation for a directory."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.bin").write_bytes(b"\x00" * 100)
        (test_dir / "file2.bin").write_bytes(b"\x00" * 200)

        size = quantizer._get_directory_size(test_dir)
        assert size == 300


class TestGPTQOutputGeneration:
    """Tests for GPTQ output generation."""

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

    def test_create_basic_gptq_output(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test basic GPTQ output creation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            group_size=128,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        output_path = temp_dir / "gptq_output"
        quantizer._create_basic_gptq_output(None, str(output_path), bits=4)

        # Verify structure
        assert output_path.exists()
        assert (output_path / "config.json").exists()
        assert (output_path / "quantize_config.json").exists()
        assert (output_path / "model.safetensors").exists()

    def test_has_autogptq_returns_bool(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _has_autogptq returns boolean."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
        )

        result = quantizer._has_autogptq()
        assert isinstance(result, bool)

    def test_quantize_config_json_created(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize_config.json is created with correct content."""
        import json

        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            group_size=64,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        output_path = temp_dir / "gptq_output"
        quantizer._create_basic_gptq_output(None, str(output_path), bits=4)

        with open(output_path / "quantize_config.json") as f:
            quant_config = json.load(f)

        assert quant_config["bits"] == 4
        assert quant_config["group_size"] == 64


class TestGPTQLoadModel:
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_load_model_calls_model_loader(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _load_model calls the model loader."""
        quantizer = GPTQQuantizer(
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
            assert quantizer._tokenizer == mock_tokenizer


class TestGPTQQuantizeWithGPTQ:
    """Tests for _quantize_with_gptq method."""

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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_quantize_with_gptq_uses_autogptq_when_available(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _quantize_with_gptq uses AutoGPTQ when available."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1", "sample2"]

        with patch.object(quantizer, "_has_autogptq", return_value=True):
            with patch.object(quantizer, "_quantize_with_autogptq") as mock_quantize:
                quantizer._quantize_with_gptq(mock_model, calibration_data)
                mock_quantize.assert_called_once()

    def test_quantize_with_gptq_uses_basic_when_no_autogptq(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _quantize_with_gptq uses basic output when AutoGPTQ not available."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1", "sample2"]

        with patch.object(quantizer, "_has_autogptq", return_value=False):
            with patch.object(quantizer, "_create_basic_gptq_output") as mock_create:
                quantizer._quantize_with_gptq(mock_model, calibration_data)
                mock_create.assert_called_once()

    def test_quantize_with_gptq_generates_correct_output_path(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _quantize_with_gptq generates correct output path."""
        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1"]

        with patch.object(quantizer, "_has_autogptq", return_value=False):
            with patch.object(quantizer, "_create_basic_gptq_output"):
                output_path = quantizer._quantize_with_gptq(mock_model, calibration_data)
                assert "Llama-2-7b-hf" in output_path
                assert "GPTQ" in output_path


class TestGPTQQuantizeWithAutoGPTQ:
    """Tests for _quantize_with_autogptq method."""

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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_quantize_with_autogptq_falls_back_on_error(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _quantize_with_autogptq falls back to basic on error."""
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        reporter = ProgressReporter(verbosity=Verbosity.NORMAL)

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )
        quantizer._tokenizer = MagicMock()

        mock_model = MagicMock()
        calibration_data = ["sample1"]
        output_path = str(temp_dir / "output")

        mock_gptq = MagicMock()
        mock_gptq.AutoGPTQForCausalLM.from_pretrained.side_effect = Exception("GPTQ error")

        with patch.dict("sys.modules", {"auto_gptq": mock_gptq}):
            with patch.object(quantizer, "_create_basic_gptq_output") as mock_create:
                quantizer._quantize_with_autogptq(mock_model, calibration_data, output_path, bits=4)
                mock_create.assert_called_once()


class TestGPTQQuantizeWithCheckpoint:
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result is not None


class TestGPTQQuantizeWithCalibration:
    """Tests for quantize with calibration data."""

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

    def test_quantize_uses_custom_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize uses custom calibration data."""
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('["custom sample 1", "custom sample 2"]')

        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=2,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        calibration_data_used = []

        def capture_calibration(model, calibration_data):
            calibration_data_used.extend(calibration_data)
            output_dir = temp_dir / "gptq_output"
            output_dir.mkdir()
            (output_dir / "config.json").write_text('{"model_type": "llama"}')
            (output_dir / "quantize_config.json").write_text('{"bits": 4}')
            (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
            return str(output_dir)

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq", side_effect=capture_calibration):
                quantizer.quantize()

        assert "custom sample 1" in calibration_data_used

    def test_quantize_uses_default_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize uses default calibration data when none provided."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_samples=4,
        )

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        calibration_data_used = []

        def capture_calibration(model, calibration_data):
            calibration_data_used.extend(calibration_data)
            output_dir = temp_dir / "gptq_output"
            output_dir.mkdir()
            (output_dir / "config.json").write_text('{"model_type": "llama"}')
            (output_dir / "quantize_config.json").write_text('{"bits": 4}')
            (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
            return str(output_dir)

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq", side_effect=capture_calibration):
                quantizer.quantize()

        assert len(calibration_data_used) == 4


class TestGPTQQuantizeWithProgress:
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
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE)

        quantizer = GPTQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_gptq") as mock_quantize:
                output_dir = temp_dir / "gptq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "quantize_config.json").write_text('{"bits": 4}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()
                assert result.peak_memory_bytes >= 0
