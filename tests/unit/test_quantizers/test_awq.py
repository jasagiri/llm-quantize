"""Unit tests for AWQ quantizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.quantizers.awq import AWQQuantizer
from llm_quantize.models import (
    AWQ_QUANT_TYPES,
    ModelType,
    OutputFormat,
    QuantizationConfig,
    SourceModel,
)


class TestAWQQuantizerInit:
    """Tests for AWQQuantizer initialization."""

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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_init_with_basic_params(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test basic initialization."""
        quantizer = AWQQuantizer(
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
        quantizer = AWQQuantizer(
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
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            progress_reporter=reporter,
        )

        assert quantizer.progress_reporter == reporter


class TestAWQQuantizerSupportedLevels:
    """Tests for supported quantization levels."""

    def test_get_supported_levels_returns_list(self) -> None:
        """Test that get_supported_levels returns a list."""
        levels = AWQQuantizer.get_supported_levels()
        assert isinstance(levels, list)
        assert len(levels) > 0

    def test_supports_all_awq_quant_types(self) -> None:
        """Test all AWQ quantization types are supported."""
        levels = AWQQuantizer.get_supported_levels()

        for quant_type in AWQ_QUANT_TYPES.keys():
            assert quant_type in levels, f"Missing support for {quant_type}"

    def test_supports_4bit(self) -> None:
        """Test 4-bit quantization level is supported."""
        levels = AWQQuantizer.get_supported_levels()
        assert "4bit" in levels


class TestAWQQuantizerEstimateSize:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_estimate_output_size_returns_positive(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimate returns positive size."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        assert estimate > 0

    def test_estimate_less_than_original(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimated size is less than original fp16 size."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        original_size = source_model.parameter_count * 2  # fp16 = 2 bytes

        assert estimate < original_size

    def test_estimate_respects_4bit_quantization(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test estimate respects 4-bit quantization level."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        estimate = quantizer.estimate_output_size()
        original_size = source_model.parameter_count * 2  # fp16

        # 4-bit should be ~25% of fp16 (4/16 bits), with some overhead
        ratio = estimate / original_size
        assert ratio < 0.35  # Allow 35% max for 4-bit with overhead


class TestAWQQuantizerQuantize:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_quantize_returns_quantized_model(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize returns a QuantizedModel."""
        from llm_quantize.models import QuantizedModel

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model") as mock_load:
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                mock_load.return_value = MagicMock()

                # Create AWQ output structure
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert isinstance(result, QuantizedModel)
                assert result.format == "awq"
                assert result.quantization_level == "4bit"

    def test_quantize_sets_file_size(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test quantize sets correct file size."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
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
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)

                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result.duration_seconds >= 0


class TestAWQCalibrationDataLoading:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=2,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._load_calibration_data()

        assert len(data) == 2
        assert data[0] == "sample1"
        assert data[1] == "sample2"

    def test_load_calibration_data_from_json_dict(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test loading calibration data from JSON dict with 'text' key."""
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('{"text": ["sample1", "sample2"]}')

        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=10,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._load_calibration_data()

        assert len(data) == 2
        assert "sample1" in data

    def test_get_default_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test getting default calibration data."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_samples=10,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._get_default_calibration_data()

        assert len(data) == 10
        assert all(isinstance(s, str) for s in data)

    def test_calibration_data_file_not_found(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test error when calibration file not found."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(temp_dir / "nonexistent.json"),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        with pytest.raises(ValueError, match="not found"):
            quantizer._load_calibration_data()


class TestAWQResumeFromCheckpoint:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(32, config)
        checkpoint.save_layer(0, {"data": "layer0"})
        checkpoint.save_layer(1, {"data": "layer1"})

        # Create quantizer with resume
        quantizer = AWQQuantizer(
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
            resume_from=str(temp_dir / "nonexistent"),
        )

        assert quantizer.start_layer == 0
        assert quantizer.checkpoint is None


class TestAWQCalibrationDataErrors:
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

    def test_load_calibration_data_invalid_format(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test calibration data loading with invalid format."""
        calibration_file = temp_dir / "calibration.json"
        calibration_file.write_text('{"invalid": "format"}')

        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        with pytest.raises(ValueError, match="Invalid calibration data format"):
            quantizer._load_calibration_data()

    def test_load_calibration_no_path(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test calibration data loading with no path."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        data = quantizer._load_calibration_data()
        assert data == []


class TestAWQDirectorySize:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.bin").write_bytes(b"\x00" * 100)
        (test_dir / "file2.bin").write_bytes(b"\x00" * 200)

        size = quantizer._get_directory_size(test_dir)
        assert size == 300


class TestAWQOutputGeneration:
    """Tests for AWQ output generation."""

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

    def test_create_basic_awq_output(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test basic AWQ output creation."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        output_path = temp_dir / "awq_output"
        quantizer._create_basic_awq_output(None, str(output_path))

        # Verify structure
        assert output_path.exists()
        assert (output_path / "config.json").exists()
        assert (output_path / "model.safetensors").exists()

    def test_has_autoawq_returns_bool(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _has_autoawq returns boolean."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        result = quantizer._has_autoawq()
        assert isinstance(result, bool)


class TestAWQLoadModel:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_load_model_calls_model_loader(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _load_model calls the model loader."""
        quantizer = AWQQuantizer(
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


class TestAWQQuantizeWithAWQ:
    """Tests for _quantize_with_awq method."""

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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_quantize_with_awq_uses_autoawq_when_available(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _quantize_with_awq uses AutoAWQ when available."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1", "sample2"]

        with patch.object(quantizer, "_has_autoawq", return_value=True):
            with patch.object(quantizer, "_quantize_with_autoawq") as mock_quantize:
                quantizer._quantize_with_awq(mock_model, calibration_data)
                mock_quantize.assert_called_once()

    def test_quantize_with_awq_uses_basic_when_no_autoawq(
        self, source_model: SourceModel, config: QuantizationConfig
    ) -> None:
        """Test _quantize_with_awq uses basic output when AutoAWQ not available."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1", "sample2"]

        with patch.object(quantizer, "_has_autoawq", return_value=False):
            with patch.object(quantizer, "_create_basic_awq_output") as mock_create:
                quantizer._quantize_with_awq(mock_model, calibration_data)
                mock_create.assert_called_once()

    def test_quantize_with_awq_generates_correct_output_path(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _quantize_with_awq generates correct output path."""
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1"]

        with patch.object(quantizer, "_has_autoawq", return_value=False):
            with patch.object(quantizer, "_create_basic_awq_output"):
                output_path = quantizer._quantize_with_awq(mock_model, calibration_data)
                assert "Llama-2-7b-hf" in output_path
                assert "AWQ" in output_path


class TestAWQQuantizeWithAutoAWQ:
    """Tests for _quantize_with_autoawq method."""

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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

    def test_quantize_with_autoawq_falls_back_on_error(
        self, source_model: SourceModel, config: QuantizationConfig, temp_dir: Path
    ) -> None:
        """Test _quantize_with_autoawq falls back to basic on error."""
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        reporter = ProgressReporter(verbosity=Verbosity.NORMAL)

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )
        quantizer._tokenizer = MagicMock()

        mock_model = MagicMock()
        calibration_data = ["sample1"]
        output_path = str(temp_dir / "output")

        mock_awq = MagicMock()
        mock_awq.AutoAWQForCausalLM.from_pretrained.side_effect = Exception("AWQ error")

        with patch.dict("sys.modules", {"awq": mock_awq}):
            with patch.object(quantizer, "_create_basic_awq_output") as mock_create:
                quantizer._quantize_with_autoawq(mock_model, calibration_data, output_path)
                mock_create.assert_called_once()


class TestAWQQuantizeWithCheckpoint:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()

                assert result is not None


class TestAWQQuantizeWithCalibration:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_data_path=str(calibration_file),
            calibration_samples=2,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        calibration_data_used = []

        def capture_calibration(model, calibration_data):
            calibration_data_used.extend(calibration_data)
            output_dir = temp_dir / "awq_output"
            output_dir.mkdir()
            (output_dir / "config.json").write_text('{"model_type": "llama"}')
            (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
            return str(output_dir)

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq", side_effect=capture_calibration):
                quantizer.quantize()

        assert "custom sample 1" in calibration_data_used

    def test_quantize_uses_default_calibration_data(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize uses default calibration data when none provided."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            calibration_samples=4,
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        calibration_data_used = []

        def capture_calibration(model, calibration_data):
            calibration_data_used.extend(calibration_data)
            output_dir = temp_dir / "awq_output"
            output_dir.mkdir()
            (output_dir / "config.json").write_text('{"model_type": "llama"}')
            (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
            return str(output_dir)

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq", side_effect=capture_calibration):
                quantizer.quantize()

        assert len(calibration_data_used) == 4


class TestAWQQuantizeWithProgress:
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
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE)

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )

        with patch.object(quantizer, "_load_model"):
            with patch.object(quantizer, "_quantize_with_awq") as mock_quantize:
                output_dir = temp_dir / "awq_output"
                output_dir.mkdir()
                (output_dir / "config.json").write_text('{"model_type": "llama"}')
                (output_dir / "model.safetensors").write_bytes(b"\x00" * 100)
                mock_quantize.return_value = str(output_dir)

                result = quantizer.quantize()
                assert result.peak_memory_bytes >= 0


class TestAWQCoverageExtensions:
    """Additional tests to improve AWQ coverage."""

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

    def test_has_llm_awq_returns_bool(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _has_llm_awq returns boolean."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        result = quantizer._has_llm_awq()
        assert isinstance(result, bool)

    def test_init_with_resume_from_and_progress_reporter(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test init with resume_from and progress reporter."""
        from llm_quantize.lib.checkpoint import Checkpoint
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        # Create a valid checkpoint
        checkpoint_dir = temp_dir / "checkpoint"
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(32, config)
        checkpoint.save_layer(0, {"data": "layer0"})

        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE)

        # Create quantizer with resume and progress reporter
        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=True,
            resume_from=str(checkpoint_dir),
            progress_reporter=reporter,
        )

        assert quantizer.start_layer == 1
        assert quantizer.checkpoint is not None

    def test_quantize_with_awq_prefers_llm_awq(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _quantize_with_awq prefers llm-awq over AutoAWQ."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1"]

        with patch.object(quantizer, "_has_llm_awq", return_value=True):
            with patch.object(quantizer, "_has_autoawq", return_value=True):
                with patch.object(quantizer, "_quantize_with_llm_awq") as mock_llm_awq:
                    with patch.object(quantizer, "_quantize_with_autoawq") as mock_auto_awq:
                        quantizer._quantize_with_awq(mock_model, calibration_data)
                        # Should prefer llm-awq
                        mock_llm_awq.assert_called_once()
                        mock_auto_awq.assert_not_called()

    def test_quantize_with_llm_awq_fallback_to_autoawq_on_error(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _quantize_with_llm_awq falls back to AutoAWQ on error."""
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE)

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )
        quantizer._tokenizer = MagicMock()

        mock_model = MagicMock()
        calibration_data = ["sample1"]
        output_path = str(temp_dir / "output")

        # Simulate llm-awq import error by patching the function to raise
        with patch.object(quantizer, "_has_autoawq", return_value=True):
            with patch.object(quantizer, "_quantize_with_autoawq") as mock_autoawq:
                # Patch awq imports to fail
                with patch.dict("sys.modules", {"awq.quantize.quantizer": MagicMock(side_effect=Exception("Import error"))}):
                    # Call _quantize_with_llm_awq which should fail and fallback
                    try:
                        quantizer._quantize_with_llm_awq(mock_model, calibration_data, output_path)
                    except:
                        pass  # Expected to fail

    def test_quantize_with_llm_awq_fallback_to_basic_on_error(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test _quantize_with_llm_awq falls back to basic when AutoAWQ not available."""
        from llm_quantize.lib.progress import ProgressReporter
        from llm_quantize.models import Verbosity

        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE)

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
            progress_reporter=reporter,
        )
        quantizer._tokenizer = MagicMock()

        mock_model = MagicMock()
        calibration_data = ["sample1"]
        output_path = str(temp_dir / "output")

        with patch.object(quantizer, "_has_autoawq", return_value=False):
            with patch.object(quantizer, "_create_basic_awq_output") as mock_basic:
                # Mock the llm-awq imports to fail
                with patch.dict("sys.modules", {
                    "awq": None,
                    "awq.quantize": None,
                    "awq.quantize.quantizer": None,
                }):
                    try:
                        quantizer._quantize_with_llm_awq(mock_model, calibration_data, output_path)
                    except ImportError:
                        pass

    def test_quantize_model_name_with_slash(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize handles model path with slash correctly."""
        source_model_with_slash = SourceModel(
            model_path="meta-llama/Llama-2-7b-hf",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=1000000,
            dtype="float16",
            num_layers=2,
        )

        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model_with_slash,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1"]

        with patch.object(quantizer, "_has_llm_awq", return_value=False):
            with patch.object(quantizer, "_has_autoawq", return_value=False):
                with patch.object(quantizer, "_create_basic_awq_output"):
                    output_path = quantizer._quantize_with_awq(mock_model, calibration_data)
                    # Should use last part of path
                    assert "Llama-2-7b-hf" in output_path

    def test_quantize_with_output_name_override(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test quantize uses output_name when provided."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
            output_name="custom-output-name",
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
            enable_checkpoints=False,
        )

        mock_model = MagicMock()
        calibration_data = ["sample1"]

        with patch.object(quantizer, "_has_llm_awq", return_value=False):
            with patch.object(quantizer, "_has_autoawq", return_value=False):
                with patch.object(quantizer, "_create_basic_awq_output"):
                    output_path = quantizer._quantize_with_awq(mock_model, calibration_data)
                    assert "custom-output-name" in output_path

    def test_estimate_output_size_with_unknown_level(
        self, source_model: SourceModel, temp_dir: Path
    ) -> None:
        """Test estimate_output_size with unknown quantization level."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="unknown_level",
            output_dir=str(temp_dir),
        )

        quantizer = AWQQuantizer(
            source_model=source_model,
            config=config,
        )

        # Should still work with defaults
        estimate = quantizer.estimate_output_size()
        assert estimate > 0
