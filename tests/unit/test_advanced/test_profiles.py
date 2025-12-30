"""Tests for quantization profiles."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_quantize.models import (
    DynamicQuantizationProfile,
    LayerQuantizationConfig,
    LayerType,
    PRESET_PROFILES,
)


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_profile_balanced(self):
        """Test getting balanced profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import get_profile

        profile = get_profile("balanced")

        assert profile is not None
        assert profile.profile_name == "balanced"
        assert profile.attention_bits == 4
        assert profile.mlp_bits == 4

    def test_get_profile_attention_high(self):
        """Test getting attention-high profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import get_profile

        profile = get_profile("attention-high")

        assert profile is not None
        assert profile.attention_bits > profile.mlp_bits

    def test_get_profile_compression_max(self):
        """Test getting compression-max profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import get_profile

        profile = get_profile("compression-max")

        assert profile is not None
        assert profile.target_avg_bits < 4

    def test_get_profile_nonexistent(self):
        """Test getting nonexistent profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import get_profile

        profile = get_profile("nonexistent")

        assert profile is None


class TestGetAvailableProfiles:
    """Tests for get_available_profiles function."""

    def test_get_available_profiles(self):
        """Test getting list of available profiles."""
        from llm_quantize.lib.quantizers.advanced.profiles import get_available_profiles

        profiles = get_available_profiles()

        assert isinstance(profiles, list)
        assert "balanced" in profiles
        assert "attention-high" in profiles
        assert "compression-max" in profiles
        assert "quality-max" in profiles


class TestCreateProfileFromImportance:
    """Tests for create_profile_from_importance function."""

    def test_create_profile_basic(self):
        """Test basic profile creation from importance."""
        from llm_quantize.lib.quantizers.advanced.profiles import create_profile_from_importance

        layer_names = ["layer1", "layer2", "layer3"]
        importance_scores = {
            "layer1": 0.9,  # High importance -> more bits
            "layer2": 0.5,  # Medium
            "layer3": 0.1,  # Low importance -> fewer bits
        }

        profile = create_profile_from_importance(
            model_architecture="test",
            layer_names=layer_names,
            importance_scores=importance_scores,
            target_avg_bits=4.0,
        )

        assert len(profile.layer_configs) == 3
        # High importance layer should get more bits
        assert profile.layer_configs[0].bit_width >= profile.layer_configs[2].bit_width

    def test_create_profile_respects_target_bits(self):
        """Test that profile respects target average bits."""
        from llm_quantize.lib.quantizers.advanced.profiles import create_profile_from_importance

        layer_names = ["layer1", "layer2"]
        importance_scores = {"layer1": 0.5, "layer2": 0.5}

        profile = create_profile_from_importance(
            model_architecture="test",
            layer_names=layer_names,
            importance_scores=importance_scores,
            target_avg_bits=3.0,
        )

        actual_avg = sum(lc.bit_width for lc in profile.layer_configs) / len(profile.layer_configs)
        # Should be close to target
        assert abs(actual_avg - 3.0) < 2.0


class TestCreateUnslothStyleProfile:
    """Tests for create_unsloth_style_profile function."""

    def test_create_unsloth_profile(self):
        """Test creating Unsloth-style profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import create_unsloth_style_profile

        layer_names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.gate_proj",
            "model.embed_tokens",
        ]

        profile = create_unsloth_style_profile(
            layer_names=layer_names,
            attention_bits=4,
            mlp_bits=2,
            embedding_bits=8,
        )

        assert profile.profile_name == "unsloth-style"

        # Check that layers got correct bits
        attn_config = next(
            (c for c in profile.layer_configs if "attn" in c.layer_name.lower()),
            None,
        )
        mlp_config = next(
            (c for c in profile.layer_configs if "mlp" in c.layer_name.lower()),
            None,
        )

        if attn_config:
            assert attn_config.bit_width == 4
        if mlp_config:
            assert mlp_config.bit_width == 2


class TestInferLayerType:
    """Tests for _infer_layer_type function."""

    def test_infer_attention(self):
        """Test inferring attention layer type."""
        from llm_quantize.lib.quantizers.advanced.profiles import _infer_layer_type

        assert _infer_layer_type("model.layers.0.self_attn.q_proj") == LayerType.ATTENTION
        assert _infer_layer_type("attention.query") == LayerType.ATTENTION
        assert _infer_layer_type("k_proj") == LayerType.ATTENTION

    def test_infer_mlp(self):
        """Test inferring MLP layer type."""
        from llm_quantize.lib.quantizers.advanced.profiles import _infer_layer_type

        assert _infer_layer_type("model.layers.0.mlp.gate_proj") == LayerType.MLP
        assert _infer_layer_type("feed_forward.fc1") == LayerType.MLP
        assert _infer_layer_type("ffn.up_proj") == LayerType.MLP

    def test_infer_embedding(self):
        """Test inferring embedding layer type."""
        from llm_quantize.lib.quantizers.advanced.profiles import _infer_layer_type

        assert _infer_layer_type("model.embed_tokens") == LayerType.EMBEDDING
        assert _infer_layer_type("wte") == LayerType.EMBEDDING

    def test_infer_norm(self):
        """Test inferring norm layer type."""
        from llm_quantize.lib.quantizers.advanced.profiles import _infer_layer_type

        assert _infer_layer_type("input_layernorm") == LayerType.NORM
        assert _infer_layer_type("ln_f") == LayerType.NORM

    def test_infer_output(self):
        """Test inferring output layer type."""
        from llm_quantize.lib.quantizers.advanced.profiles import _infer_layer_type

        assert _infer_layer_type("lm_head") == LayerType.OUTPUT

    def test_infer_unknown(self):
        """Test unknown layer type."""
        from llm_quantize.lib.quantizers.advanced.profiles import _infer_layer_type

        assert _infer_layer_type("some_random_layer") == LayerType.UNKNOWN


class TestValidateProfile:
    """Tests for validate_profile function."""

    def test_validate_valid_profile(self):
        """Test validating a valid profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import validate_profile

        profile = DynamicQuantizationProfile(
            profile_name="test",
            layer_configs=[
                LayerQuantizationConfig("layer1", 0, bit_width=4),
                LayerQuantizationConfig("layer2", 1, bit_width=4),
            ],
        )

        model_layers = ["layer1", "layer2"]

        is_valid, messages = validate_profile(profile, model_layers)

        assert is_valid

    def test_validate_profile_missing_layers(self):
        """Test validation with missing layers in profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import validate_profile

        profile = DynamicQuantizationProfile(
            profile_name="test",
            layer_configs=[
                LayerQuantizationConfig("layer1", 0, bit_width=4),
            ],
        )

        model_layers = ["layer1", "layer2", "layer3"]

        is_valid, messages = validate_profile(profile, model_layers)

        assert any("not in profile" in m for m in messages)

    def test_validate_profile_extra_layers(self):
        """Test validation with extra layers in profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import validate_profile

        profile = DynamicQuantizationProfile(
            profile_name="test",
            layer_configs=[
                LayerQuantizationConfig("layer1", 0, bit_width=4),
                LayerQuantizationConfig("layer2", 1, bit_width=4),
                LayerQuantizationConfig("extra", 2, bit_width=4),
            ],
        )

        model_layers = ["layer1", "layer2"]

        is_valid, messages = validate_profile(profile, model_layers)

        assert any("not in model" in m for m in messages)

    def test_validate_empty_profile(self):
        """Test validation of empty profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import validate_profile

        profile = DynamicQuantizationProfile(
            profile_name="test",
            layer_configs=[],
        )

        is_valid, messages = validate_profile(profile, ["layer1"])

        assert not is_valid
        assert any("no layer" in m.lower() for m in messages)


class TestSaveLoadProfile:
    """Tests for save_profile and load_profile functions."""

    def test_save_and_load_profile(self):
        """Test saving and loading profile."""
        from llm_quantize.lib.quantizers.advanced.profiles import save_profile, load_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"

            profile = DynamicQuantizationProfile(
                profile_name="test",
                description="Test profile",
                attention_bits=6,
                mlp_bits=2,
                layer_configs=[
                    LayerQuantizationConfig("layer1", 0, bit_width=6),
                ],
            )

            save_profile(profile, str(path))

            assert path.exists()

            loaded = load_profile(str(path))

            assert loaded.profile_name == "test"
            assert loaded.description == "Test profile"
            assert loaded.attention_bits == 6
            assert len(loaded.layer_configs) == 1

    def test_load_profile_preserves_configs(self):
        """Test that loaded profile preserves layer configs."""
        from llm_quantize.lib.quantizers.advanced.profiles import save_profile, load_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"

            profile = DynamicQuantizationProfile(
                profile_name="test",
                layer_configs=[
                    LayerQuantizationConfig(
                        "layer1", 0,
                        bit_width=4,
                        layer_type=LayerType.ATTENTION,
                        importance_score=0.8,
                    ),
                ],
            )

            save_profile(profile, str(path))
            loaded = load_profile(str(path))

            assert loaded.layer_configs[0].bit_width == 4
            assert loaded.layer_configs[0].layer_type == LayerType.ATTENTION
            assert loaded.layer_configs[0].importance_score == 0.8
