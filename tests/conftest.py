"""Shared pytest fixtures for llm-quantize tests."""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_model_path(temp_dir: Path) -> Path:
    """Create a mock model directory structure for testing."""
    model_dir = temp_dir / "test-model"
    model_dir.mkdir(parents=True)

    # Create minimal config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 1000,
        "torch_dtype": "float16",
    }
    import json

    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock HuggingFace model for testing."""
    model = MagicMock()
    model.config.architectures = ["LlamaForCausalLM"]
    model.config.hidden_size = 256
    model.config.num_hidden_layers = 2
    model.config.num_attention_heads = 4
    model.config.vocab_size = 1000
    model.config.torch_dtype = "float16"
    return model


@pytest.fixture
def checkpoint_dir(temp_dir: Path) -> Path:
    """Provide a temporary checkpoint directory."""
    checkpoint_path = temp_dir / ".checkpoint"
    checkpoint_path.mkdir(parents=True)
    return checkpoint_path


@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Provide a temporary output directory."""
    output_path = temp_dir / "output"
    output_path.mkdir(parents=True)
    return output_path


@pytest.fixture
def cli_runner() -> Generator:
    """Provide a Click CLI test runner."""
    from click.testing import CliRunner

    yield CliRunner()
