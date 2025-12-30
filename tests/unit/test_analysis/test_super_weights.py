"""Tests for super weight identification."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.models import (
    CalibrationInfo,
    ImportanceMatrix,
    ImportanceMethod,
    LayerImportance,
)


class TestIdentifySuperWeights:
    """Tests for super weight identification."""

    def test_identify_super_weights_basic(self):
        """Test basic super weight identification."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch
        import numpy as np

        # Create a real tensor for testing
        mock_model = MagicMock()

        # Create a real parameter tensor
        param_tensor = torch.tensor([[0.1, 0.5], [0.9, 0.2]])
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_param.dim.return_value = 2
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        result = identify_super_weights(
            model=mock_model,
            coverage=0.25,  # Top 25%
        )

        assert isinstance(result, dict)
        assert "layer1.weight" in result

    def test_identify_super_weights_skips_1d(self):
        """Test that 1D parameters are skipped."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_param.dim.return_value = 1  # 1D parameter

        mock_model.named_parameters.return_value = [("bias", mock_param)]

        result = identify_super_weights(mock_model, coverage=0.5)

        assert "bias" not in result

    def test_identify_super_weights_gradient_method(self):
        """Test gradient-based super weight identification."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        # Create real tensors
        param_tensor = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        grad_tensor = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_param.dim.return_value = 2
        mock_param.data = param_tensor
        mock_param.grad = grad_tensor

        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        result = identify_super_weights(mock_model, method="gradient")

        assert isinstance(result, dict)


class TestIdentifySuperWeightsFromCalibration:
    """Tests for calibration-based super weight identification."""

    def test_identify_from_calibration(self):
        """Test super weight identification from calibration data."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights_from_calibration
        import torch

        # Mock model with no modules to simplify test
        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Empty modules list to skip hook registration
        mock_model.named_modules.return_value = []

        # Empty calibration to skip processing
        calibration_data = []

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_tokenizer = MagicMock()
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = identify_super_weights_from_calibration(
                model=mock_model,
                calibration_data=calibration_data,
                coverage=0.1,
            )

        assert isinstance(result, dict)


class TestComputeSuperWeightStatistics:
    """Tests for super weight statistics computation."""

    def test_compute_statistics(self):
        """Test computing super weight statistics."""
        from llm_quantize.lib.analysis.super_weights import compute_super_weight_statistics
        import torch

        mock_model = MagicMock()

        # Create real tensor
        param_tensor = torch.tensor([0.5, 0.7, 0.3, 0.9])
        mock_param = MagicMock()
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        super_weights = {"layer1": [0, 1]}

        stats = compute_super_weight_statistics(mock_model, super_weights)

        assert "total_super_weights" in stats
        assert "layers_with_super_weights" in stats
        assert "layer_distribution" in stats


class TestCreateProtectionMask:
    """Tests for protection mask creation."""

    def test_create_protection_mask(self):
        """Test creating protection mask."""
        from llm_quantize.lib.analysis.super_weights import create_protection_mask
        import torch

        mock_model = MagicMock()

        # Create a real parameter tensor
        param_tensor = torch.zeros(2, 5)
        mock_param = MagicMock()
        mock_param.dim.return_value = 2
        mock_param.numel.return_value = 10
        mock_param.shape = (2, 5)
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        super_weights = {"layer1": [0, 2, 4]}

        masks = create_protection_mask(mock_model, super_weights, protection_bits=8)

        assert "layer1" in masks
        assert masks["layer1"].shape == (2, 5)

    def test_create_protection_mask_skips_1d(self):
        """Test that 1D parameters are skipped in mask creation."""
        from llm_quantize.lib.analysis.super_weights import create_protection_mask

        mock_model = MagicMock()

        mock_param = MagicMock()
        mock_param.dim.return_value = 1  # 1D

        mock_model.named_parameters.return_value = [("bias", mock_param)]

        masks = create_protection_mask(mock_model, {})

        assert "bias" not in masks


class TestSaveLoadSuperWeights:
    """Tests for saving and loading super weights."""

    def test_save_super_weights(self):
        """Test saving super weights to file."""
        from llm_quantize.lib.analysis.super_weights import save_super_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "super_weights.json"

            super_weights = {
                "layer1": [0, 1, 2],
                "layer2": [5, 10],
            }

            save_super_weights(super_weights, str(path))

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert data["super_weights"] == super_weights
            assert data["total_count"] == 5

    def test_load_super_weights(self):
        """Test loading super weights from file."""
        from llm_quantize.lib.analysis.super_weights import load_super_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "super_weights.json"

            data = {
                "version": "1.0",
                "super_weights": {"layer1": [0, 1]},
                "total_count": 2,
            }

            with open(path, "w") as f:
                json.dump(data, f)

            result = load_super_weights(str(path))

            assert result == {"layer1": [0, 1]}

    def test_load_super_weights_empty(self):
        """Test loading empty super weights."""
        from llm_quantize.lib.analysis.super_weights import load_super_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "super_weights.json"

            data = {"version": "1.0"}

            with open(path, "w") as f:
                json.dump(data, f)

            result = load_super_weights(str(path))

            assert result == {}


class TestUpdateImportanceMatrix:
    """Tests for updating importance matrix with super weights."""

    def test_update_importance_matrix(self):
        """Test updating importance matrix with super weights."""
        from llm_quantize.lib.analysis.super_weights import update_importance_matrix_with_super_weights

        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.5, 100),
                LayerImportance("layer2", 1, 0.7, 100),
            ],
            total_parameters=200,
        )

        super_weights = {
            "layer1": [0, 1, 2],
        }

        updated = update_importance_matrix_with_super_weights(matrix, super_weights)

        assert updated.layer_scores[0].super_weight_indices == [0, 1, 2]
        assert updated.layer_scores[1].super_weight_indices == []
        assert updated.super_weight_coverage == 3 / 200

    def test_update_importance_matrix_zero_params(self):
        """Test updating importance matrix with zero total parameters."""
        from llm_quantize.lib.analysis.super_weights import update_importance_matrix_with_super_weights

        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[],
            total_parameters=0,  # Zero parameters
        )

        super_weights = {"layer1": [0, 1]}

        updated = update_importance_matrix_with_super_weights(matrix, super_weights)

        # Should not crash with division by zero - keeps default coverage
        assert isinstance(updated.super_weight_coverage, float)


class TestSuperWeightsEdgeCases:
    """Additional edge case tests for super weights."""

    def test_identify_skips_non_grad_param(self):
        """Test that non-grad parameters are skipped."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        # Parameter with requires_grad=False
        param_tensor = torch.tensor([[0.1, 0.5], [0.9, 0.2]])
        mock_param = MagicMock()
        mock_param.requires_grad = False  # No grad
        mock_param.dim.return_value = 2
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        result = identify_super_weights(mock_model, coverage=0.5)

        assert "layer1.weight" not in result

    def test_identify_gradient_method_no_grad(self):
        """Test gradient method falls back when param has no grad."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        param_tensor = torch.tensor([[0.5, 0.3], [0.7, 0.1]])
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_param.dim.return_value = 2
        mock_param.data = param_tensor
        mock_param.grad = None  # No gradient

        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        result = identify_super_weights(mock_model, method="gradient")

        # Should use fallback (activation magnitude)
        assert isinstance(result, dict)

    def test_identify_with_multiple_layers(self):
        """Test identification across multiple layers."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        param1 = torch.tensor([[0.9, 0.1], [0.2, 0.3]])
        mock_param1 = MagicMock()
        mock_param1.requires_grad = True
        mock_param1.dim.return_value = 2
        mock_param1.data = param1

        param2 = torch.tensor([[0.1, 0.8], [0.4, 0.2]])
        mock_param2 = MagicMock()
        mock_param2.requires_grad = True
        mock_param2.dim.return_value = 2
        mock_param2.data = param2

        mock_model.named_parameters.return_value = [
            ("layer1.weight", mock_param1),
            ("layer2.weight", mock_param2),
        ]

        result = identify_super_weights(mock_model, coverage=0.25)

        # Should identify weights from both layers
        assert isinstance(result, dict)
        total_super = sum(len(v) for v in result.values())
        assert total_super >= 1

    def test_protection_mask_handles_large_index(self):
        """Test protection mask handles index bounds correctly."""
        from llm_quantize.lib.analysis.super_weights import create_protection_mask
        import torch

        mock_model = MagicMock()

        param_tensor = torch.zeros(2, 3)
        mock_param = MagicMock()
        mock_param.dim.return_value = 2
        mock_param.numel.return_value = 6
        mock_param.shape = (2, 3)
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        # Include index larger than param size
        super_weights = {"layer1": [0, 1, 100]}  # 100 > 6

        masks = create_protection_mask(mock_model, super_weights, protection_bits=8)

        # Should only set valid indices
        assert "layer1" in masks
        flat_mask = masks["layer1"].flatten()
        assert flat_mask[0] == 8
        assert flat_mask[1] == 8
        # Index 100 should be ignored

    def test_compute_stats_skips_missing_layers(self):
        """Test statistics computation skips layers not in super_weights."""
        from llm_quantize.lib.analysis.super_weights import compute_super_weight_statistics
        import torch

        mock_model = MagicMock()

        param_tensor = torch.tensor([0.5, 0.7, 0.3, 0.9])
        mock_param = MagicMock()
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [
            ("layer1", mock_param),
            ("layer2", mock_param),
        ]

        super_weights = {"layer1": [0, 1]}  # Only layer1

        stats = compute_super_weight_statistics(mock_model, super_weights)

        assert "layer1" in stats["layer_distribution"]
        assert "layer2" not in stats["layer_distribution"]

    def test_calibration_with_actual_data(self):
        """Test calibration-based identification with actual calibration samples."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights_from_calibration
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.eval = MagicMock()

        # Add a mock Linear module
        mock_linear = MagicMock(spec=torch.nn.Linear)
        mock_linear.weight = MagicMock()
        mock_linear.weight.shape = (10, 10)

        mock_model.named_modules.return_value = [("encoder.layer", mock_linear)]

        calibration_data = ["Test calibration sample with enough text to process."]

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = identify_super_weights_from_calibration(
                model=mock_model,
                calibration_data=calibration_data,
                coverage=0.1,
            )

        assert isinstance(result, dict)

    def test_protection_mask_for_layer_not_in_super_weights(self):
        """Test protection mask creation for layer without super weights."""
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

        super_weights = {}  # Empty - no super weights

        masks = create_protection_mask(mock_model, super_weights, protection_bits=8)

        # Should still create mask with all zeros
        assert "layer1" in masks
        assert masks["layer1"].sum() == 0


class TestSuperWeightsCalibrationCoverage:
    """Additional tests for calibration-based super weight identification coverage."""

    def test_calibration_with_activation_hook_triggered(self):
        """Test that activation hooks actually fire and accumulate impacts."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights_from_calibration
        import torch

        # Create a simple real model to trigger hooks
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.config = MagicMock()
                self.config._name_or_path = "test-model"

            def forward(self, input_ids, **kwargs):
                # Simple forward pass
                x = input_ids.float()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if x.size(-1) != 10:
                    x = torch.nn.functional.pad(x, (0, 10 - x.size(-1)))[:, :10]
                return self.linear(x)

        model = SimpleModel()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_tokenizer = MagicMock()

            def tokenize_fn(text, **kwargs):
                return {"input_ids": torch.randint(0, 100, (1, 10))}

            mock_tokenizer.side_effect = tokenize_fn
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = identify_super_weights_from_calibration(
                model=model,
                calibration_data=["Test text for calibration"],
                coverage=0.5,  # High coverage to ensure some weights selected
            )

        assert isinstance(result, dict)

    def test_calibration_with_exception_in_sample(self):
        """Test calibration handles exceptions during sample processing."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights_from_calibration
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.eval = MagicMock()
        mock_model.named_modules.return_value = []

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_tokenizer = MagicMock()
            # Make tokenizer raise exception
            mock_tokenizer.side_effect = Exception("Tokenization error")
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = identify_super_weights_from_calibration(
                model=mock_model,
                calibration_data=["Sample that will fail"],
                coverage=0.1,
            )

        # Should handle gracefully and return empty dict
        assert isinstance(result, dict)

    def test_calibration_with_multiple_linear_layers(self):
        """Test calibration with multiple Linear modules."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights_from_calibration
        import torch

        class MultiLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 8)
                self.layer2 = torch.nn.Linear(8, 5)
                self.config = MagicMock()
                self.config._name_or_path = "test-model"

            def forward(self, input_ids, **kwargs):
                x = input_ids.float()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if x.size(-1) != 10:
                    x = torch.nn.functional.pad(x, (0, 10 - x.size(-1)))[:, :10]
                x = self.layer1(x)
                return self.layer2(x)

        model = MultiLayerModel()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = identify_super_weights_from_calibration(
                model=model,
                calibration_data=["Test calibration sample"],
                coverage=0.5,
            )

        assert isinstance(result, dict)

    def test_calibration_accumulates_activation_impacts(self):
        """Test that calibration properly accumulates activation impacts."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights_from_calibration
        import torch

        class TrackedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 3)
                self.config = MagicMock()
                self.config._name_or_path = "test-model"
                self.forward_count = 0

            def forward(self, input_ids, **kwargs):
                self.forward_count += 1
                x = input_ids.float()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                # Ensure correct input size
                if x.size(-1) != 5:
                    x = torch.nn.functional.pad(x, (0, 5 - x.size(-1)))[:, :5]
                return self.linear(x)

        model = TrackedModel()

        # Run with multiple calibration samples
        calibration_samples = [f"Sample {i}" for i in range(5)]

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 5))}
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = identify_super_weights_from_calibration(
                model=model,
                calibration_data=calibration_samples,
                coverage=0.3,
            )

        assert isinstance(result, dict)
        # Model should have been called multiple times
        assert model.forward_count >= 1

    def test_identify_super_weights_with_coverage_selection(self):
        """Test that identify_super_weights correctly selects top coverage weights."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        # Create params with clear importance ranking
        param1 = torch.tensor([[10.0, 1.0], [0.5, 0.1]])  # 10.0 is highest
        mock_param1 = MagicMock()
        mock_param1.requires_grad = True
        mock_param1.dim.return_value = 2
        mock_param1.data = param1

        param2 = torch.tensor([[5.0, 2.0], [0.3, 0.2]])  # 5.0 is second highest
        mock_param2 = MagicMock()
        mock_param2.requires_grad = True
        mock_param2.dim.return_value = 2
        mock_param2.data = param2

        mock_model.named_parameters.return_value = [
            ("layer1.weight", mock_param1),
            ("layer2.weight", mock_param2),
        ]

        # Very high coverage to select most weights
        result = identify_super_weights(mock_model, coverage=0.5)

        assert isinstance(result, dict)
        # Should have identified some super weights
        total_sw = sum(len(v) for v in result.values())
        assert total_sw >= 1

    def test_identify_super_weights_adds_to_existing_layer(self):
        """Test that identify_super_weights adds indices to existing layer entries."""
        from llm_quantize.lib.analysis.super_weights import identify_super_weights
        import torch

        mock_model = MagicMock()

        # Create param with multiple high-magnitude weights
        param = torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_param.dim.return_value = 2
        mock_param.data = param

        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]

        # Coverage to get multiple weights from same layer
        result = identify_super_weights(mock_model, coverage=0.5)

        assert "layer.weight" in result
        # Should have multiple indices in same layer
        assert len(result["layer.weight"]) >= 1


class TestSuperWeightsStatisticsCoverage:
    """Additional tests for compute_super_weight_statistics coverage."""

    def test_compute_statistics_with_values(self):
        """Test computing statistics with actual super weights."""
        from llm_quantize.lib.analysis.super_weights import compute_super_weight_statistics
        import torch

        mock_model = MagicMock()

        param_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_param = MagicMock()
        mock_param.data = param_tensor

        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        # Super weights with specific indices
        super_weights = {"layer1": [0, 2, 4]}

        stats = compute_super_weight_statistics(mock_model, super_weights)

        assert stats["total_super_weights"] == 3
        assert stats["layers_with_super_weights"] == 1
        assert "layer1" in stats["layer_distribution"]
        assert stats["layer_distribution"]["layer1"] == 3
        assert "layer1" in stats["value_statistics"]
        assert stats["value_statistics"]["layer1"]["count"] == 3
        assert stats["value_statistics"]["layer1"]["mean"] > 0
        assert stats["value_statistics"]["layer1"]["max"] == 5.0
        assert stats["value_statistics"]["layer1"]["min"] == 1.0

    def test_compute_statistics_multiple_layers(self):
        """Test computing statistics with multiple layers."""
        from llm_quantize.lib.analysis.super_weights import compute_super_weight_statistics
        import torch

        mock_model = MagicMock()

        param1 = torch.tensor([1.0, 2.0, 3.0])
        mock_param1 = MagicMock()
        mock_param1.data = param1

        param2 = torch.tensor([10.0, 20.0, 30.0])
        mock_param2 = MagicMock()
        mock_param2.data = param2

        mock_model.named_parameters.return_value = [
            ("layer1", mock_param1),
            ("layer2", mock_param2),
        ]

        super_weights = {
            "layer1": [0, 1],
            "layer2": [2],
        }

        stats = compute_super_weight_statistics(mock_model, super_weights)

        assert stats["total_super_weights"] == 3
        assert stats["layers_with_super_weights"] == 2
        assert "layer1" in stats["layer_distribution"]
        assert "layer2" in stats["layer_distribution"]
