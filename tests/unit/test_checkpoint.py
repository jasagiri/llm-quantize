"""Tests for checkpoint functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_quantize.lib.checkpoint import Checkpoint
from llm_quantize.models import OutputFormat, QuantizationConfig


@pytest.fixture
def config() -> QuantizationConfig:
    """Create a test quantization config."""
    return QuantizationConfig(
        target_format=OutputFormat.GGUF,
        quantization_level="Q4_K_M",
        output_dir="./output",
    )


class TestCheckpointInitialization:
    """Tests for checkpoint initialization."""

    def test_initialize_creates_directory(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test that initialization creates checkpoint directory."""
        # Remove the directory first
        import shutil

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(32, config)

        assert checkpoint_dir.exists()
        assert (checkpoint_dir / Checkpoint.METADATA_FILE).exists()

    def test_initialize_sets_metadata(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test that initialization sets metadata correctly."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(32, config)

        assert checkpoint.total_layers == 32
        assert checkpoint.completed_layers == 0


class TestCheckpointSaveLoad:
    """Tests for checkpoint save/load functionality."""

    def test_save_layer_updates_progress(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test that saving a layer updates progress."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)

        checkpoint.save_layer(0, {"test": "data"})

        assert checkpoint.completed_layers == 1
        assert checkpoint.progress_percentage == 10.0

    def test_save_multiple_layers(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test saving multiple layers."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)

        for i in range(3):
            checkpoint.save_layer(i, {"layer": i})

        assert checkpoint.completed_layers == 3
        assert checkpoint.progress_percentage == 60.0

    def test_load_saved_layer(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test loading a previously saved layer."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)

        test_data = {"weights": [1, 2, 3], "bias": [0.1, 0.2]}
        checkpoint.save_layer(2, test_data)

        # Create new checkpoint and load
        checkpoint2 = Checkpoint(checkpoint_dir, config)
        loaded = checkpoint2.load_layer(2)

        assert loaded == test_data

    def test_load_nonexistent_layer(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test loading a layer that doesn't exist returns None."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)

        loaded = checkpoint.load_layer(99)
        assert loaded is None


class TestCheckpointResume:
    """Tests for checkpoint resume functionality."""

    def test_can_resume_returns_true(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test can_resume returns True for valid checkpoint."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)
        checkpoint.save_layer(0, {"data": "test"})

        assert Checkpoint.can_resume(checkpoint_dir) is True

    def test_can_resume_returns_false_empty(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test can_resume returns False for empty checkpoint."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)
        # No layers saved

        assert Checkpoint.can_resume(checkpoint_dir) is False

    def test_can_resume_returns_false_no_checkpoint(
        self, temp_dir: Path
    ) -> None:
        """Test can_resume returns False when no checkpoint exists."""
        nonexistent = temp_dir / "no-checkpoint"
        assert Checkpoint.can_resume(nonexistent) is False

    def test_from_resume_returns_checkpoint(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test from_resume returns checkpoint and start layer."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)
        checkpoint.save_layer(0, {"data": "layer0"})
        checkpoint.save_layer(1, {"data": "layer1"})
        checkpoint.save_layer(2, {"data": "layer2"})

        resumed, start_layer = Checkpoint.from_resume(checkpoint_dir, config)

        assert start_layer == 3
        assert resumed.completed_layers == 3

    def test_from_resume_raises_on_config_mismatch(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test from_resume raises ValueError on config mismatch."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)
        checkpoint.save_layer(0, {"data": "test"})

        # Try to resume with different config
        different_config = QuantizationConfig(
            target_format=OutputFormat.AWQ,  # Different format
            quantization_level="4bit",
            output_dir="./output",
        )

        with pytest.raises(ValueError, match="Configuration mismatch"):
            Checkpoint.from_resume(checkpoint_dir, different_config)


class TestCheckpointCleanup:
    """Tests for checkpoint cleanup."""

    def test_cleanup_removes_files(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test cleanup removes all checkpoint files."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)
        checkpoint.save_layer(0, {"data": "test"})
        checkpoint.save_layer(1, {"data": "test"})

        checkpoint.cleanup()

        assert not (checkpoint_dir / Checkpoint.METADATA_FILE).exists()
        assert not list(checkpoint_dir.glob("layer_*.pkl"))

    def test_cleanup_handles_nonexistent(self, temp_dir: Path) -> None:
        """Test cleanup handles nonexistent directory gracefully."""
        checkpoint = Checkpoint(temp_dir / "nonexistent")
        checkpoint.cleanup()  # Should not raise


class TestCheckpointProperties:
    """Tests for checkpoint properties."""

    def test_progress_percentage_calculation(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test progress percentage calculation."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(4, config)

        assert checkpoint.progress_percentage == 0.0

        checkpoint.save_layer(0, {})
        assert checkpoint.progress_percentage == 25.0

        checkpoint.save_layer(1, {})
        assert checkpoint.progress_percentage == 50.0

    def test_str_representation(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test string representation."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)
        checkpoint.save_layer(0, {})

        str_repr = str(checkpoint)
        assert "1/10 layers" in str_repr

    def test_progress_percentage_zero_total(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test progress percentage when total is 0."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        # Don't initialize - metadata is empty
        assert checkpoint.progress_percentage == 0.0


class TestCheckpointEdgeCases:
    """Tests for edge cases in checkpoint functionality."""

    def test_load_layer_from_cache(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test loading layer from cache (already loaded)."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)

        test_data = {"weights": [1, 2, 3]}
        checkpoint.save_layer(0, test_data)

        # First load - from file
        loaded1 = checkpoint.load_layer(0)
        # Second load - from cache
        loaded2 = checkpoint.load_layer(0)

        assert loaded1 == test_data
        assert loaded2 == test_data

    def test_can_resume_handles_json_decode_error(
        self, checkpoint_dir: Path
    ) -> None:
        """Test can_resume handles corrupt metadata file."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = checkpoint_dir / Checkpoint.METADATA_FILE

        # Write invalid JSON
        metadata_file.write_text("{ invalid json")

        assert Checkpoint.can_resume(checkpoint_dir) is False

    def test_from_resume_no_valid_checkpoint(
        self, temp_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test from_resume raises when no valid checkpoint."""
        checkpoint_dir = temp_dir / "empty_checkpoint"
        checkpoint_dir.mkdir()

        with pytest.raises(ValueError, match="No valid checkpoint found"):
            Checkpoint.from_resume(checkpoint_dir, config)

    def test_from_resume_quantization_level_mismatch(
        self, checkpoint_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test from_resume raises on quantization level mismatch."""
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(10, config)
        checkpoint.save_layer(0, {"data": "test"})

        # Try to resume with different quantization level
        different_config = QuantizationConfig(
            target_format=OutputFormat.GGUF,  # Same format
            quantization_level="Q8_0",  # Different level
            output_dir="./output",
        )

        with pytest.raises(ValueError, match="Configuration mismatch.*quantization level"):
            Checkpoint.from_resume(checkpoint_dir, different_config)

    def test_cleanup_removes_directory_when_empty(
        self, temp_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test cleanup removes directory when it becomes empty."""
        checkpoint_dir = temp_dir / "cleanup_test"
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)
        checkpoint.save_layer(0, {"data": "test"})

        checkpoint.cleanup()

        # Directory should be removed since it's empty
        assert not checkpoint_dir.exists()

    def test_cleanup_keeps_directory_if_not_empty(
        self, temp_dir: Path, config: QuantizationConfig
    ) -> None:
        """Test cleanup keeps directory if other files exist."""
        checkpoint_dir = temp_dir / "cleanup_test2"
        checkpoint = Checkpoint(checkpoint_dir, config)
        checkpoint.initialize(5, config)
        checkpoint.save_layer(0, {"data": "test"})

        # Add another file that shouldn't be cleaned up
        (checkpoint_dir / "other_file.txt").write_text("keep me")

        checkpoint.cleanup()

        # Directory should still exist because of other_file.txt
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "other_file.txt").exists()

    def test_can_resume_handles_os_error(
        self, temp_dir: Path
    ) -> None:
        """Test can_resume handles OS errors gracefully."""
        # Non-existent directory
        nonexistent = temp_dir / "nonexistent"
        assert Checkpoint.can_resume(nonexistent) is False
