"""Unit tests for model dataclasses."""

import tempfile
from pathlib import Path

import pytest

from llm_quantize.models import (
    AWQ_QUANT_TYPES,
    GGUF_QUANT_TYPES,
    GPTQ_QUANT_TYPES,
    OutputFormat,
    OutputMode,
    QuantizationConfig,
    QuantizationJob,
    QuantizedModel,
    SourceModel,
    Verbosity,
    ValidationStatus,
    JobStatus,
    ModelType,
)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_gguf_value(self) -> None:
        """Test GGUF format value."""
        assert OutputFormat.GGUF.value == "gguf"

    def test_awq_value(self) -> None:
        """Test AWQ format value."""
        assert OutputFormat.AWQ.value == "awq"

    def test_gptq_value(self) -> None:
        """Test GPTQ format value."""
        assert OutputFormat.GPTQ.value == "gptq"


class TestVerbosity:
    """Tests for Verbosity enum."""

    def test_quiet_value(self) -> None:
        """Test quiet verbosity."""
        assert Verbosity.QUIET.value == "quiet"

    def test_normal_value(self) -> None:
        """Test normal verbosity."""
        assert Verbosity.NORMAL.value == "normal"

    def test_verbose_value(self) -> None:
        """Test verbose verbosity."""
        assert Verbosity.VERBOSE.value == "verbose"

    def test_debug_value(self) -> None:
        """Test debug verbosity."""
        assert Verbosity.DEBUG.value == "debug"


class TestOutputMode:
    """Tests for OutputMode enum."""

    def test_human_value(self) -> None:
        """Test human output mode."""
        assert OutputMode.HUMAN.value == "human"

    def test_json_value(self) -> None:
        """Test JSON output mode."""
        assert OutputMode.JSON.value == "json"


class TestQuantTypes:
    """Tests for quantization type dictionaries."""

    def test_gguf_quant_types_has_entries(self) -> None:
        """Test GGUF quant types dictionary."""
        assert len(GGUF_QUANT_TYPES) > 0

    def test_gguf_quant_types_has_bits(self) -> None:
        """Test GGUF quant types have bits info."""
        for name, info in GGUF_QUANT_TYPES.items():
            assert "bits" in info
            assert 2 <= info["bits"] <= 8

    def test_awq_quant_types_has_4bit(self) -> None:
        """Test AWQ quant types has 4bit."""
        assert "4bit" in AWQ_QUANT_TYPES
        assert AWQ_QUANT_TYPES["4bit"]["bits"] == 4

    def test_gptq_quant_types_has_entries(self) -> None:
        """Test GPTQ quant types dictionary."""
        assert "2bit" in GPTQ_QUANT_TYPES
        assert "3bit" in GPTQ_QUANT_TYPES
        assert "4bit" in GPTQ_QUANT_TYPES
        assert "8bit" in GPTQ_QUANT_TYPES


class TestQuantizationConfigValidation:
    """Tests for QuantizationConfig validation."""

    def test_valid_gguf_config(self, temp_dir: Path) -> None:
        """Test valid GGUF config passes validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )
        errors = config.validate()
        assert errors == []

    def test_valid_awq_config(self, temp_dir: Path) -> None:
        """Test valid AWQ config passes validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        errors = config.validate()
        assert errors == []

    def test_valid_gptq_config(self, temp_dir: Path) -> None:
        """Test valid GPTQ config passes validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        errors = config.validate()
        assert errors == []

    def test_invalid_gguf_level(self, temp_dir: Path) -> None:
        """Test invalid GGUF level fails validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="invalid",
            output_dir=str(temp_dir),
        )
        errors = config.validate()
        assert len(errors) > 0
        assert "Invalid GGUF quantization level" in errors[0]

    def test_invalid_awq_level(self, temp_dir: Path) -> None:
        """Test invalid AWQ level fails validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="invalid",
            output_dir=str(temp_dir),
        )
        errors = config.validate()
        assert len(errors) > 0
        assert "Invalid AWQ quantization level" in errors[0]

    def test_invalid_gptq_level(self, temp_dir: Path) -> None:
        """Test invalid GPTQ level fails validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="invalid",
            output_dir=str(temp_dir),
        )
        errors = config.validate()
        assert len(errors) > 0
        assert "Invalid GPTQ quantization level" in errors[0]

    def test_invalid_calibration_samples_too_low(self, temp_dir: Path) -> None:
        """Test calibration samples too low fails validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            calibration_samples=0,
        )
        errors = config.validate()
        assert any("calibration_samples" in e for e in errors)

    def test_invalid_calibration_samples_too_high(self, temp_dir: Path) -> None:
        """Test calibration samples too high fails validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            calibration_samples=2000,
        )
        errors = config.validate()
        assert any("calibration_samples" in e for e in errors)

    def test_invalid_group_size(self, temp_dir: Path) -> None:
        """Test invalid group size fails validation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            group_size=100,
        )
        errors = config.validate()
        assert any("group_size" in e for e in errors)

    def test_valid_group_sizes(self, temp_dir: Path) -> None:
        """Test all valid group sizes pass validation."""
        for group_size in [32, 64, 128, 256]:
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="Q4_K_M",
                output_dir=str(temp_dir),
                group_size=group_size,
            )
            errors = config.validate()
            assert not any("group_size" in e for e in errors)

    def test_creates_output_dir_if_not_exists(self) -> None:
        """Test validation creates output dir if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp:
            new_dir = Path(temp) / "new_output"
            config = QuantizationConfig(
                target_format=OutputFormat.GGUF,
                quantization_level="Q4_K_M",
                output_dir=str(new_dir),
            )
            errors = config.validate()
            assert errors == []
            assert new_dir.exists()


class TestQuantizationConfigOutputPath:
    """Tests for QuantizationConfig output path generation."""

    def test_gguf_auto_filename(self, temp_dir: Path) -> None:
        """Test auto-generated GGUF filename."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )
        path = config.get_output_path("test-model")
        assert path == temp_dir / "test-model-Q4_K_M.gguf"

    def test_awq_auto_filename(self, temp_dir: Path) -> None:
        """Test auto-generated AWQ filename."""
        config = QuantizationConfig(
            target_format=OutputFormat.AWQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        path = config.get_output_path("test-model")
        assert path == temp_dir / "test-model-awq-4bit"

    def test_gptq_auto_filename(self, temp_dir: Path) -> None:
        """Test auto-generated GPTQ filename."""
        config = QuantizationConfig(
            target_format=OutputFormat.GPTQ,
            quantization_level="4bit",
            output_dir=str(temp_dir),
        )
        path = config.get_output_path("test-model")
        assert path == temp_dir / "test-model-gptq-4bit"

    def test_custom_filename(self, temp_dir: Path) -> None:
        """Test custom filename is used."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            output_name="custom-name.gguf",
        )
        path = config.get_output_path("test-model")
        assert path == temp_dir / "custom-name.gguf"


class TestQuantizationConfigCheckpointDir:
    """Tests for QuantizationConfig checkpoint directory."""

    def test_default_checkpoint_dir(self, temp_dir: Path) -> None:
        """Test default checkpoint directory."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )
        checkpoint_dir = config.get_checkpoint_dir()
        assert checkpoint_dir == temp_dir / ".checkpoint"

    def test_custom_checkpoint_dir(self, temp_dir: Path) -> None:
        """Test custom checkpoint directory."""
        custom_dir = temp_dir / "custom_checkpoints"
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
            checkpoint_dir=str(custom_dir),
        )
        checkpoint_dir = config.get_checkpoint_dir()
        assert checkpoint_dir == custom_dir


class TestQuantizationConfigStr:
    """Tests for QuantizationConfig string representation."""

    def test_str_representation(self, temp_dir: Path) -> None:
        """Test string representation."""
        config = QuantizationConfig(
            target_format=OutputFormat.GGUF,
            quantization_level="Q4_K_M",
            output_dir=str(temp_dir),
        )
        result = str(config)
        assert "QuantizationConfig" in result
        assert "gguf" in result
        assert "Q4_K_M" in result


class TestQuantizedModelValidation:
    """Tests for QuantizedModel validation."""

    def test_valid_model(self, temp_dir: Path) -> None:
        """Test valid model passes validation."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
        )
        errors = model.validate()
        assert errors == []

    def test_missing_output_path(self) -> None:
        """Test missing output path fails validation."""
        model = QuantizedModel(
            output_path="",
            format="gguf",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
        )
        errors = model.validate()
        assert any("output_path" in e for e in errors)

    def test_nonexistent_file(self, temp_dir: Path) -> None:
        """Test nonexistent file fails validation."""
        model = QuantizedModel(
            output_path=str(temp_dir / "nonexistent.gguf"),
            format="gguf",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
        )
        errors = model.validate()
        assert any("does not exist" in e for e in errors)

    def test_invalid_compression_ratio_zero(self, temp_dir: Path) -> None:
        """Test zero compression ratio fails validation."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=104,
            compression_ratio=0,
            source_model_path="test-model",
        )
        errors = model.validate()
        assert any("compression_ratio" in e for e in errors)

    def test_invalid_compression_ratio_one(self, temp_dir: Path) -> None:
        """Test compression ratio of 1 fails validation."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=104,
            compression_ratio=1.0,
            source_model_path="test-model",
        )
        errors = model.validate()
        assert any("compression_ratio" in e for e in errors)

    def test_invalid_file_size(self, temp_dir: Path) -> None:
        """Test invalid file size fails validation."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=0,
            compression_ratio=0.25,
            source_model_path="test-model",
        )
        errors = model.validate()
        assert any("file_size" in e for e in errors)


class TestQuantizedModelToDict:
    """Tests for QuantizedModel to_dict method."""

    def test_to_dict(self, temp_dir: Path) -> None:
        """Test to_dict returns correct dictionary."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
            quantization_level="Q4_K_M",
            duration_seconds=10.5,
            peak_memory_bytes=1000000,
        )
        result = model.to_dict()

        assert result["status"] == "success"
        assert result["output_path"] == str(output_file)
        assert result["format"] == "gguf"
        assert result["file_size"] == 104
        assert result["compression_ratio"] == 0.25
        assert result["quantization_level"] == "Q4_K_M"
        assert result["duration_seconds"] == 10.5
        assert result["peak_memory_bytes"] == 1000000


class TestQuantizedModelIsValid:
    """Tests for QuantizedModel is_valid property."""

    def test_is_valid_true(self, temp_dir: Path) -> None:
        """Test is_valid returns True when validation status is VALID."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
            validation_status=ValidationStatus.VALID,
        )
        assert model.is_valid is True

    def test_is_valid_false(self, temp_dir: Path) -> None:
        """Test is_valid returns False when validation status is not VALID."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=104,
            compression_ratio=0.25,
            source_model_path="test-model",
            validation_status=ValidationStatus.INVALID,
        )
        assert model.is_valid is False


class TestQuantizedModelStr:
    """Tests for QuantizedModel string representation."""

    def test_str_representation(self, temp_dir: Path) -> None:
        """Test string representation."""
        output_file = temp_dir / "model.gguf"
        output_file.write_bytes(b"GGUF" + b"\x00" * 100)

        model = QuantizedModel(
            output_path=str(output_file),
            format="gguf",
            file_size=1048576,  # 1 MB
            compression_ratio=0.25,
            source_model_path="test-model",
        )
        result = str(model)
        assert "QuantizedModel" in result
        assert str(output_file) in result


class TestQuantizationJob:
    """Tests for QuantizationJob dataclass."""

    def test_create_job(self) -> None:
        """Test creating a job."""
        job = QuantizationJob(job_id="job-123")
        assert job.job_id == "job-123"
        assert job.status == JobStatus.PENDING

    def test_create_job_default_id(self) -> None:
        """Test creating a job with default UUID."""
        job = QuantizationJob()
        assert job.job_id is not None
        assert len(job.job_id) > 0
        assert job.status == JobStatus.PENDING

    def test_job_start(self) -> None:
        """Test starting a job."""
        job = QuantizationJob()
        job.start()
        assert job.status == JobStatus.RUNNING
        assert job.start_time is not None
        assert job.progress_percentage == 0.0

    def test_job_complete(self) -> None:
        """Test completing a job."""
        job = QuantizationJob()
        job.start()
        job.complete()

        assert job.status == JobStatus.COMPLETED
        assert job.end_time is not None
        assert job.progress_percentage == 100.0

    def test_job_fail(self) -> None:
        """Test failing a job."""
        job = QuantizationJob()
        job.start()
        job.fail("Something went wrong")

        assert job.status == JobStatus.FAILED
        assert job.error_message == "Something went wrong"
        assert job.end_time is not None

    def test_job_cancel(self) -> None:
        """Test cancelling a job."""
        job = QuantizationJob()
        job.start()
        job.cancel()

        assert job.status == JobStatus.CANCELLED
        assert job.end_time is not None

    def test_job_to_dict(self) -> None:
        """Test job to_dict method."""
        job = QuantizationJob(job_id="job-123")
        result = job.to_dict()

        assert result["job_id"] == "job-123"
        assert result["status"] == "pending"
        assert result["progress_percentage"] == 0.0

    def test_job_duration_not_started(self) -> None:
        """Test job duration when not started."""
        job = QuantizationJob()
        assert job.duration_seconds is None

    def test_job_duration_running(self) -> None:
        """Test job duration while running."""
        job = QuantizationJob()
        job.start()
        duration = job.duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_job_duration_completed(self) -> None:
        """Test job duration when completed."""
        job = QuantizationJob()
        job.start()
        job.complete()
        duration = job.duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_job_progress_update(self) -> None:
        """Test updating job progress."""
        job = QuantizationJob()
        job.start()
        job.update_progress(49, 100)  # 0-based index

        assert job.current_layer == 49
        assert job.total_layers == 100
        assert job.progress_percentage == 50.0  # (49+1)/100 * 100

    def test_job_progress_update_with_memory(self) -> None:
        """Test updating job progress with memory tracking."""
        job = QuantizationJob()
        job.start()
        job.update_progress(10, 100, memory_usage=1000000)

        assert job.current_memory_usage == 1000000
        assert job.peak_memory_usage == 1000000

        # Update with higher memory
        job.update_progress(20, 100, memory_usage=2000000)
        assert job.current_memory_usage == 2000000
        assert job.peak_memory_usage == 2000000

        # Update with lower memory - peak should stay same
        job.update_progress(30, 100, memory_usage=1500000)
        assert job.current_memory_usage == 1500000
        assert job.peak_memory_usage == 2000000

    def test_job_is_running(self) -> None:
        """Test is_running property."""
        job = QuantizationJob()
        assert job.is_running is False

        job.start()
        assert job.is_running is True

        job.complete()
        assert job.is_running is False

    def test_job_is_complete(self) -> None:
        """Test is_complete property."""
        job = QuantizationJob()
        assert job.is_complete is False

        job.start()
        assert job.is_complete is False

        job.complete()
        assert job.is_complete is True

    def test_job_is_complete_failed(self) -> None:
        """Test is_complete property when failed."""
        job = QuantizationJob()
        job.start()
        job.fail("Error")
        assert job.is_complete is True

    def test_job_is_complete_cancelled(self) -> None:
        """Test is_complete property when cancelled."""
        job = QuantizationJob()
        job.start()
        job.cancel()
        assert job.is_complete is True

    def test_job_str(self) -> None:
        """Test job string representation."""
        job = QuantizationJob(job_id="job-12345678")
        result = str(job)
        assert "QuantizationJob" in result
        assert "job-1234" in result  # First 8 chars
        assert "pending" in result


class TestSourceModel:
    """Tests for SourceModel dataclass."""

    def test_create_source_model(self) -> None:
        """Test creating a source model."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        assert model.model_path == "test-model"
        assert model.model_type == ModelType.HF_HUB

    def test_is_local_false(self) -> None:
        """Test is_local returns False for HF Hub model."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        assert model.is_local is False

    def test_is_local_true(self) -> None:
        """Test is_local returns True for local model."""
        model = SourceModel(
            model_path="/path/to/model",
            model_type=ModelType.LOCAL_DIR,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        assert model.is_local is True

    def test_is_hub_true(self) -> None:
        """Test is_hub returns True for HF Hub model."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        assert model.is_hub is True

    def test_is_hub_false(self) -> None:
        """Test is_hub returns False for local model."""
        model = SourceModel(
            model_path="/path/to/model",
            model_type=ModelType.LOCAL_DIR,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        assert model.is_hub is False

    def test_source_model_with_optional_fields(self) -> None:
        """Test source model with all optional fields."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
            num_layers=32,
            hidden_size=4096,
            num_heads=32,
            vocab_size=32000,
        )
        assert model.num_layers == 32
        assert model.hidden_size == 4096
        assert model.num_heads == 32
        assert model.vocab_size == 32000

    def test_source_model_from_path_local(self, temp_dir: Path) -> None:
        """Test from_path with local directory."""
        model_dir = temp_dir / "test_model"
        model_dir.mkdir()

        model = SourceModel.from_path(str(model_dir))
        assert model.model_type == ModelType.LOCAL_DIR
        assert model.model_path == str(model_dir)

    def test_source_model_from_path_hub(self) -> None:
        """Test from_path with HF Hub identifier."""
        model = SourceModel.from_path("meta-llama/Llama-2-7b-hf")
        assert model.model_type == ModelType.HF_HUB
        assert model.model_path == "meta-llama/Llama-2-7b-hf"

    def test_source_model_validate_valid(self) -> None:
        """Test validate returns empty list for valid model."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        errors = model.validate()
        assert errors == []

    def test_source_model_validate_missing_path(self) -> None:
        """Test validate catches missing model_path."""
        model = SourceModel(
            model_path="",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        errors = model.validate()
        assert any("model_path" in e for e in errors)

    def test_source_model_validate_negative_params(self) -> None:
        """Test validate catches negative parameter_count."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=-1,
            dtype="float16",
        )
        errors = model.validate()
        assert any("parameter_count" in e for e in errors)

    def test_source_model_validate_unknown_architecture(self) -> None:
        """Test validate catches unknown architecture."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="Unknown",
            parameter_count=7000000000,
            dtype="float16",
        )
        errors = model.validate()
        assert any("architecture" in e for e in errors)

    def test_source_model_str(self) -> None:
        """Test source model string representation."""
        model = SourceModel(
            model_path="test-model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )
        result = str(model)
        assert "SourceModel" in result
        assert "test-model" in result
        assert "LlamaForCausalLM" in result


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_valid_value(self) -> None:
        """Test VALID value."""
        assert ValidationStatus.VALID.value == "valid"

    def test_invalid_value(self) -> None:
        """Test INVALID value."""
        assert ValidationStatus.INVALID.value == "invalid"

    def test_unchecked_value(self) -> None:
        """Test UNCHECKED value."""
        assert ValidationStatus.UNCHECKED.value == "unchecked"


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_pending_value(self) -> None:
        """Test PENDING value."""
        assert JobStatus.PENDING.value == "pending"

    def test_running_value(self) -> None:
        """Test RUNNING value."""
        assert JobStatus.RUNNING.value == "running"

    def test_completed_value(self) -> None:
        """Test COMPLETED value."""
        assert JobStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        """Test FAILED value."""
        assert JobStatus.FAILED.value == "failed"


class TestModelType:
    """Tests for ModelType enum."""

    def test_hf_hub_value(self) -> None:
        """Test HF_HUB value."""
        assert ModelType.HF_HUB.value == "hf_hub"

    def test_local_dir_value(self) -> None:
        """Test LOCAL_DIR value."""
        assert ModelType.LOCAL_DIR.value == "local_dir"
