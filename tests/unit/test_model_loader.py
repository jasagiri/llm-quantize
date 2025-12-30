"""Tests for model loading functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.lib.model_loader import (
    count_parameters,
    create_source_model,
    get_hf_token,
    load_model_config,
)
from llm_quantize.models import ModelType


class TestGetHfToken:
    """Tests for HuggingFace token retrieval."""

    def test_returns_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that HF_TOKEN environment variable is used."""
        monkeypatch.setenv("HF_TOKEN", "test_token_123")
        assert get_hf_token() == "test_token_123"

    def test_returns_none_when_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that None is returned when no token is available."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("huggingface_hub.HfFolder") as mock_folder:
            mock_folder.get_token.return_value = None
            assert get_hf_token() is None

    def test_falls_back_to_huggingface_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test fallback to huggingface_hub token cache."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("huggingface_hub.HfFolder") as mock_folder:
            mock_folder.get_token.return_value = "cached_token"
            assert get_hf_token() == "cached_token"


class TestLoadModelConfig:
    """Tests for model configuration loading."""

    def test_loads_local_config(self, sample_model_path: Path) -> None:
        """Test loading configuration from local directory."""
        with patch("transformers.AutoConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 256,
            }
            mock_config_class.from_pretrained.return_value = mock_config

            config = load_model_config(str(sample_model_path))

            assert config["architectures"] == ["LlamaForCausalLM"]
            assert config["hidden_size"] == 256

    def test_raises_on_invalid_path(self) -> None:
        """Test that ValueError is raised for invalid path."""
        with patch("transformers.AutoConfig") as mock_config_class:
            mock_config_class.from_pretrained.side_effect = Exception("Model not found")

            with pytest.raises(ValueError, match="Failed to load model configuration"):
                load_model_config("nonexistent/model")


class TestCountParameters:
    """Tests for parameter counting."""

    def test_estimates_parameters(self) -> None:
        """Test parameter estimation from config."""
        with patch("transformers.AutoConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.hidden_size = 4096
            mock_config.num_hidden_layers = 32
            mock_config.vocab_size = 32000
            mock_config.intermediate_size = 11008
            mock_config.num_attention_heads = 32
            mock_config_class.from_pretrained.return_value = mock_config

            params = count_parameters("test-model")

            # Should return a positive number for valid config
            assert params > 0

    def test_returns_zero_on_error(self) -> None:
        """Test that 0 is returned on configuration error."""
        with patch("transformers.AutoConfig") as mock_config_class:
            mock_config_class.from_pretrained.side_effect = Exception("Error")

            params = count_parameters("invalid-model")
            assert params == 0


class TestCreateSourceModel:
    """Tests for SourceModel creation."""

    def test_creates_local_source_model(self, sample_model_path: Path) -> None:
        """Test creating SourceModel from local path."""
        with patch("llm_quantize.lib.model_loader.load_model_config") as mock_load:
            mock_load.return_value = {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "vocab_size": 1000,
                "torch_dtype": "float16",
            }
            with patch("llm_quantize.lib.model_loader.count_parameters") as mock_count:
                mock_count.return_value = 1000000

                source = create_source_model(str(sample_model_path))

                assert source.model_type == ModelType.LOCAL_DIR
                assert source.architecture == "LlamaForCausalLM"
                assert source.hidden_size == 256
                assert source.num_layers == 2
                assert source.parameter_count == 1000000

    def test_creates_hub_source_model(self) -> None:
        """Test creating SourceModel from HF Hub identifier."""
        with patch("llm_quantize.lib.model_loader.load_model_config") as mock_load:
            mock_load.return_value = {
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000,
                "torch_dtype": "bfloat16",
            }
            with patch("llm_quantize.lib.model_loader.count_parameters") as mock_count:
                mock_count.return_value = 7000000000

                source = create_source_model("mistralai/Mistral-7B-v0.1")

                assert source.model_type == ModelType.HF_HUB
                assert source.architecture == "MistralForCausalLM"
                assert "bfloat16" in source.dtype


class TestCountParametersMissingValues:
    """Tests for parameter counting with missing config values."""

    def test_returns_zero_when_hidden_size_missing(self) -> None:
        """Test that 0 is returned when hidden_size is missing."""
        with patch("transformers.AutoConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.hidden_size = 0  # Missing
            mock_config.num_hidden_layers = 32
            mock_config.vocab_size = 32000
            mock_config_class.from_pretrained.return_value = mock_config

            params = count_parameters("test-model")
            assert params == 0

    def test_returns_zero_when_num_layers_missing(self) -> None:
        """Test that 0 is returned when num_layers is missing."""
        with patch("transformers.AutoConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.hidden_size = 4096
            mock_config.num_hidden_layers = 0  # Missing
            mock_config.vocab_size = 32000
            mock_config_class.from_pretrained.return_value = mock_config

            params = count_parameters("test-model")
            assert params == 0


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_with_bfloat16(self) -> None:
        """Test loading model with bfloat16 dtype."""
        from llm_quantize.lib.model_loader import load_model

        with patch("transformers.AutoTokenizer") as mock_tokenizer:
            with patch("transformers.AutoModelForCausalLM") as mock_model:
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                model, tokenizer = load_model("test-model", torch_dtype="bfloat16")

                # Verify bfloat16 was used
                call_kwargs = mock_model.from_pretrained.call_args.kwargs
                import torch
                assert call_kwargs["torch_dtype"] == torch.bfloat16

    def test_load_model_with_float32(self) -> None:
        """Test loading model with float32 dtype."""
        from llm_quantize.lib.model_loader import load_model

        with patch("transformers.AutoTokenizer") as mock_tokenizer:
            with patch("transformers.AutoModelForCausalLM") as mock_model:
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                model, tokenizer = load_model("test-model", torch_dtype="float32")

                call_kwargs = mock_model.from_pretrained.call_args.kwargs
                import torch
                assert call_kwargs["torch_dtype"] == torch.float32

    def test_load_model_with_float16(self) -> None:
        """Test loading model with float16 dtype."""
        from llm_quantize.lib.model_loader import load_model

        with patch("transformers.AutoTokenizer") as mock_tokenizer:
            with patch("transformers.AutoModelForCausalLM") as mock_model:
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                model, tokenizer = load_model("test-model", torch_dtype="float16")

                call_kwargs = mock_model.from_pretrained.call_args.kwargs
                import torch
                assert call_kwargs["torch_dtype"] == torch.float16

    def test_load_model_with_unknown_dtype(self) -> None:
        """Test loading model with unknown dtype defaults to float16."""
        from llm_quantize.lib.model_loader import load_model

        with patch("transformers.AutoTokenizer") as mock_tokenizer:
            with patch("transformers.AutoModelForCausalLM") as mock_model:
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                model, tokenizer = load_model("test-model", torch_dtype="invalid")

                call_kwargs = mock_model.from_pretrained.call_args.kwargs
                import torch
                assert call_kwargs["torch_dtype"] == torch.float16

    def test_load_model_raises_on_error(self) -> None:
        """Test that ValueError is raised on load error."""
        from llm_quantize.lib.model_loader import load_model

        with patch("transformers.AutoTokenizer") as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Load failed")

            with pytest.raises(ValueError, match="Failed to load model"):
                load_model("test-model")


class TestSourceModelValidation:
    """Tests for SourceModel validation."""

    def test_validate_valid_model(self) -> None:
        """Test validation of valid SourceModel."""
        from llm_quantize.models import SourceModel, ModelType

        model = SourceModel(
            model_path="test/model",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )

        errors = model.validate()
        assert len(errors) == 0

    def test_validate_missing_path(self) -> None:
        """Test validation catches missing path."""
        from llm_quantize.models import SourceModel, ModelType

        model = SourceModel(
            model_path="",
            model_type=ModelType.HF_HUB,
            architecture="LlamaForCausalLM",
            parameter_count=7000000000,
            dtype="float16",
        )

        errors = model.validate()
        assert any("model_path" in e for e in errors)

    def test_validate_unknown_architecture(self) -> None:
        """Test validation catches unknown architecture."""
        from llm_quantize.models import SourceModel, ModelType

        model = SourceModel(
            model_path="test/model",
            model_type=ModelType.HF_HUB,
            architecture="Unknown",
            parameter_count=7000000000,
            dtype="float16",
        )

        errors = model.validate()
        assert any("architecture" in e for e in errors)
