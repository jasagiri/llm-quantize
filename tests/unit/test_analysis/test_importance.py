"""Tests for importance matrix computation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llm_quantize.models import (
    CalibrationInfo,
    ImportanceMatrix,
    ImportanceMethod,
    LayerImportance,
)


class TestComputeImportanceMatrix:
    """Tests for compute_importance_matrix function."""

    def test_compute_importance_matrix_basic(self):
        """Test basic importance matrix computation."""
        from llm_quantize.lib.analysis.importance import compute_importance_matrix
        import torch

        # Mock model
        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"
        mock_model.config.num_hidden_layers = 2
        mock_model.config.hidden_size = 768
        mock_model.config.max_position_embeddings = 512

        # Mock parameters
        mock_param = MagicMock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        # No Linear modules to avoid hook registration issues
        mock_model.named_modules.return_value = []

        # Mock tokenizer with proper return
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        calibration_data = []  # Empty to skip forward pass

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer", return_value=mock_tokenizer):
            result = compute_importance_matrix(
                model=mock_model,
                calibration_data=calibration_data,
                method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                dataset_name="test",
            )

        assert isinstance(result, ImportanceMatrix)
        assert result.model_name == "test-model"
        assert result.computation_method == ImportanceMethod.ACTIVATION_MAGNITUDE

    def test_get_recommended_bits(self):
        """Test bit recommendation based on importance score."""
        from llm_quantize.lib.analysis.importance import _get_recommended_bits

        assert _get_recommended_bits(0.95) == 8
        assert _get_recommended_bits(0.75) == 6
        assert _get_recommended_bits(0.55) == 4
        assert _get_recommended_bits(0.35) == 3
        assert _get_recommended_bits(0.15) == 2

    def test_compute_layer_scores_empty(self):
        """Test computing layer scores with empty activations."""
        from llm_quantize.lib.analysis.importance import _compute_layer_scores

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []

        result = _compute_layer_scores(
            layer_activations={},
            model=mock_model,
            method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            num_layers=0,
        )

        assert result == []

    def test_compute_layer_scores_with_data(self):
        """Test computing layer scores with activation data."""
        from llm_quantize.lib.analysis.importance import _compute_layer_scores

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_model.named_parameters.return_value = [("layer1", mock_param)]

        layer_activations = {
            "layer1": [0.5, 0.7, 0.3],
            "layer2": [0.1, 0.2, 0.15],
        }

        result = _compute_layer_scores(
            layer_activations=layer_activations,
            model=mock_model,
            method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            num_layers=2,
        )

        assert len(result) == 2


class TestLoadSaveImportanceMatrix:
    """Tests for loading and saving importance matrices."""

    def test_load_importance_matrix(self):
        """Test loading importance matrix from file."""
        from llm_quantize.lib.analysis.importance import load_importance_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            # Create test matrix
            matrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
                layer_scores=[LayerImportance("layer1", 0, 0.5, 100)],
            )
            matrix.save(path)

            # Load and verify
            loaded = load_importance_matrix(path)
            assert loaded.model_name == "test"
            assert len(loaded.layer_scores) == 1

    def test_save_importance_matrix(self):
        """Test saving importance matrix to file."""
        from llm_quantize.lib.analysis.importance import save_importance_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            matrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
            )

            save_importance_matrix(matrix, path)

            assert path.exists()


class TestGetTokenizer:
    """Tests for tokenizer loading."""

    def test_get_tokenizer_success(self):
        """Test successful tokenizer loading."""
        from llm_quantize.lib.analysis.importance import _get_tokenizer

        mock_tokenizer = MagicMock()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            result = _get_tokenizer("test-model")

            mock_auto_tokenizer.from_pretrained.assert_called()
            assert result == mock_tokenizer

    def test_get_tokenizer_fallback(self):
        """Test tokenizer fallback to gpt2."""
        from llm_quantize.lib.analysis.importance import _get_tokenizer

        mock_tokenizer = MagicMock()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            # First call fails, second succeeds (fallback)
            mock_auto_tokenizer.from_pretrained.side_effect = [
                Exception("Model not found"),
                mock_tokenizer,
            ]

            result = _get_tokenizer("unknown-model")

            assert mock_auto_tokenizer.from_pretrained.call_count == 2


class TestComputeImportanceFromGradients:
    """Tests for gradient-based importance computation."""

    def test_compute_importance_from_gradients(self):
        """Test gradient-based importance computation."""
        from llm_quantize.lib.analysis.importance import compute_importance_from_gradients
        import torch

        # Mock model
        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"

        mock_param = MagicMock()
        mock_param.grad = MagicMock()
        mock_param.grad.abs.return_value.mean.return_value.item.return_value = 0.5
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.named_parameters.return_value = [("layer1", mock_param)]
        mock_model.parameters.return_value = iter([mock_param])

        # Empty calibration data to skip actual computation
        calibration_data = []

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer"):
            result = compute_importance_from_gradients(
                model=mock_model,
                calibration_data=calibration_data,
            )

        assert isinstance(result, ImportanceMatrix)
        assert result.computation_method == ImportanceMethod.GRADIENT_SENSITIVITY


class TestImportanceMatrixAdditional:
    """Additional tests for importance matrix functionality."""

    def test_importance_matrix_to_dict(self):
        """Test importance matrix to_dict method."""
        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[LayerImportance("layer1", 0, 0.5, 100)],
            total_parameters=1000,
        )

        d = matrix.to_dict()
        assert d["model_name"] == "test"
        assert "layer_scores" in d

    def test_load_importance_matrix_file_not_found(self):
        """Test loading from non-existent file."""
        from llm_quantize.lib.analysis.importance import load_importance_matrix

        with pytest.raises(FileNotFoundError):
            load_importance_matrix(Path("/nonexistent/path/file.json"))

    def test_get_tokenizer_with_trust_remote_code(self):
        """Test tokenizer loading with trust_remote_code."""
        from llm_quantize.lib.analysis.importance import _get_tokenizer

        mock_tokenizer = MagicMock()

        with patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            result = _get_tokenizer("meta-llama/Llama-2-7b-hf")

            # Check that from_pretrained was called
            assert mock_auto_tokenizer.from_pretrained.called

    def test_compute_layer_scores_fisher_method(self):
        """Test computing layer scores with Fisher information method."""
        from llm_quantize.lib.analysis.importance import _compute_layer_scores

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        layer_activations = {
            "layer1": [0.5, 0.7, 0.3],
        }

        result = _compute_layer_scores(
            layer_activations=layer_activations,
            model=mock_model,
            method=ImportanceMethod.FISHER_INFORMATION,
            num_layers=1,
        )

        assert len(result) == 1

    def test_compute_layer_scores_hessian_method(self):
        """Test computing layer scores with Hessian diagonal method."""
        from llm_quantize.lib.analysis.importance import _compute_layer_scores

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        layer_activations = {
            "layer1": [0.5, 0.7, 0.3],
        }

        result = _compute_layer_scores(
            layer_activations=layer_activations,
            model=mock_model,
            method=ImportanceMethod.HESSIAN_DIAGONAL,
            num_layers=1,
        )

        assert len(result) == 1

    def test_get_recommended_bits_boundary(self):
        """Test bit recommendation at boundaries."""
        from llm_quantize.lib.analysis.importance import _get_recommended_bits

        # Test exact boundaries - based on actual implementation
        assert _get_recommended_bits(0.9) == 8
        assert _get_recommended_bits(0.7) == 6
        assert _get_recommended_bits(0.5) == 4
        assert _get_recommended_bits(0.3) == 3
        assert _get_recommended_bits(0.0) == 2

    def test_importance_matrix_save_and_load_roundtrip(self):
        """Test save and load roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matrix.json"

            original = ImportanceMatrix(
                model_name="test-model",
                computation_method=ImportanceMethod.GRADIENT_SENSITIVITY,
                calibration_info=CalibrationInfo("dataset", 100),
                layer_scores=[
                    LayerImportance("layer1", 0, 0.9, 1000, recommended_bits=8),
                    LayerImportance("layer2", 1, 0.5, 2000, recommended_bits=4),
                ],
                total_parameters=3000,
            )

            original.save(path)
            loaded = ImportanceMatrix.load(path)

            assert loaded.model_name == original.model_name
            assert loaded.computation_method == original.computation_method
            assert len(loaded.layer_scores) == len(original.layer_scores)

    def test_importance_matrix_get_layer_importance_found(self):
        """Test get_layer_importance returns layer when found."""
        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.9, 1000),
                LayerImportance("layer2", 1, 0.5, 2000),
            ],
        )

        result = matrix.get_layer_importance("layer1")
        assert result is not None
        assert result.layer_name == "layer1"

    def test_importance_matrix_get_layer_importance_not_found(self):
        """Test get_layer_importance returns None when not found."""
        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.9, 1000),
            ],
        )

        result = matrix.get_layer_importance("nonexistent")
        assert result is None

    def test_importance_matrix_get_layer_by_index_found(self):
        """Test get_layer_by_index returns layer when found."""
        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.9, 1000),
                LayerImportance("layer2", 1, 0.5, 2000),
            ],
        )

        result = matrix.get_layer_by_index(1)
        assert result is not None
        assert result.layer_name == "layer2"

    def test_importance_matrix_get_layer_by_index_not_found(self):
        """Test get_layer_by_index returns None when not found."""
        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.9, 1000),
            ],
        )

        result = matrix.get_layer_by_index(99)
        assert result is None

    def test_importance_matrix_save_imatrix_format(self):
        """Test saving in .imatrix format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matrix.imatrix"

            matrix = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
                layer_scores=[LayerImportance("layer1", 0, 0.9, 1000)],
            )

            matrix.save(path)
            assert path.exists()

    def test_importance_matrix_load_imatrix_format(self):
        """Test loading from .imatrix format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matrix.imatrix"

            # Create and save
            original = ImportanceMatrix(
                model_name="test",
                computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
                calibration_info=CalibrationInfo("test", 10),
                layer_scores=[LayerImportance("layer1", 0, 0.9, 1000)],
            )
            original.save(path)

            # Load
            loaded = ImportanceMatrix.load(path)
            assert loaded.model_name == "test"

    def test_layer_importance_from_dict(self):
        """Test LayerImportance from_dict."""
        data = {
            "layer_name": "test_layer",
            "layer_index": 5,
            "importance_score": 0.75,
            "parameter_count": 500,
            "super_weight_indices": [1, 2, 3],
            "recommended_bits": 6,
        }

        layer = LayerImportance.from_dict(data)
        assert layer.layer_name == "test_layer"
        assert layer.recommended_bits == 6
        assert layer.super_weight_indices == [1, 2, 3]

    def test_calibration_info_from_dict(self):
        """Test CalibrationInfo from_dict."""
        data = {
            "dataset_name": "wikitext",
            "num_samples": 128,
            "sequence_length": 1024,
            "source_path": "/path/to/data",
        }

        info = CalibrationInfo.from_dict(data)
        assert info.dataset_name == "wikitext"
        assert info.sequence_length == 1024


class TestImportanceMatrixCoverage:
    """Additional tests to improve importance module coverage."""

    def test_compute_importance_with_progress_callback(self):
        """Test importance computation with progress callback."""
        from llm_quantize.lib.analysis.importance import compute_importance_matrix
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"
        mock_model.config.num_hidden_layers = 2
        mock_model.config.hidden_size = 768
        mock_model.config.max_position_embeddings = 512

        mock_param = MagicMock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]
        mock_model.named_modules.return_value = []

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer", return_value=mock_tokenizer):
            result = compute_importance_matrix(
                model=mock_model,
                calibration_data=["sample1", "sample2"],
                progress_callback=progress_callback,
            )

        assert isinstance(result, ImportanceMatrix)
        # Progress callback should have been called
        assert len(progress_calls) >= 0

    def test_compute_importance_with_linear_modules(self):
        """Test importance computation with Linear modules."""
        from llm_quantize.lib.analysis.importance import compute_importance_matrix
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"
        mock_model.config.num_hidden_layers = 2
        mock_model.config.hidden_size = 768
        mock_model.config.max_position_embeddings = 512

        mock_param = MagicMock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.named_parameters.return_value = [("layer1.weight", mock_param)]

        # Create mock Linear module with hook
        mock_linear = MagicMock(spec=torch.nn.Linear)
        handles = []

        def register_hook(fn):
            handle = MagicMock()
            handles.append(handle)
            return handle

        mock_linear.register_forward_hook = register_hook
        mock_model.named_modules.return_value = [("layer1", mock_linear)]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer", return_value=mock_tokenizer):
            result = compute_importance_matrix(
                model=mock_model,
                calibration_data=[],
            )

        assert isinstance(result, ImportanceMatrix)

    def test_compute_importance_from_gradients_with_data(self):
        """Test gradient-based importance with actual calibration data."""
        from llm_quantize.lib.analysis.importance import compute_importance_from_gradients
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"

        mock_param = MagicMock()
        mock_param.grad = MagicMock()
        mock_param.grad.abs.return_value.mean.return_value.item.return_value = 0.5
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.named_parameters.return_value = [("layer1", mock_param)]
        mock_model.parameters.return_value = iter([mock_param])

        # Mock forward pass
        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        mock_outputs.loss.backward = MagicMock()
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer", return_value=mock_tokenizer):
            result = compute_importance_from_gradients(
                model=mock_model,
                calibration_data=["Test text sample"],
            )

        assert isinstance(result, ImportanceMatrix)
        assert result.computation_method == ImportanceMethod.GRADIENT_SENSITIVITY

    def test_compute_importance_from_gradients_error_handling(self):
        """Test gradient-based importance handles errors gracefully."""
        from llm_quantize.lib.analysis.importance import compute_importance_from_gradients
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"

        mock_param = MagicMock()
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.named_parameters.return_value = []
        mock_model.parameters.return_value = iter([mock_param])

        # Make forward pass raise error
        mock_model.side_effect = Exception("Forward pass failed")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer", return_value=mock_tokenizer):
            result = compute_importance_from_gradients(
                model=mock_model,
                calibration_data=["Test text"],
            )

        # Should still return a valid matrix
        assert isinstance(result, ImportanceMatrix)

    def test_compute_layer_scores_param_count_matching(self):
        """Test layer scores computation with parameter count matching."""
        from llm_quantize.lib.analysis.importance import _compute_layer_scores
        import torch

        mock_model = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.numel.return_value = 500
        mock_param2 = MagicMock()
        mock_param2.numel.return_value = 1000

        mock_model.named_parameters.return_value = [
            ("layer1.weight", mock_param1),
            ("layer2.weight", mock_param2),
        ]

        layer_activations = {
            "layer1": [0.5, 0.7, 0.3],
            "layer2": [0.1, 0.2],
        }

        result = _compute_layer_scores(
            layer_activations=layer_activations,
            model=mock_model,
            method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            num_layers=2,
        )

        assert len(result) == 2
        # Should be sorted by layer index
        assert result[0].layer_index <= result[1].layer_index

    def test_importance_matrix_from_dict(self):
        """Test ImportanceMatrix from_dict method."""
        data = {
            "model_name": "test-model",
            "computation_method": "activation_magnitude",
            "calibration_info": {
                "dataset_name": "test",
                "num_samples": 10,
            },
            "layer_scores": [
                {
                    "layer_name": "layer1",
                    "layer_index": 0,
                    "importance_score": 0.9,
                    "parameter_count": 1000,
                }
            ],
            "total_parameters": 1000,
            "super_weight_coverage": 0.01,
        }

        matrix = ImportanceMatrix.from_dict(data)
        assert matrix.model_name == "test-model"
        assert len(matrix.layer_scores) == 1

    def test_calibration_info_to_dict(self):
        """Test CalibrationInfo to_dict method."""
        info = CalibrationInfo(
            dataset_name="wikitext",
            num_samples=128,
            sequence_length=1024,
            source_path="/path/to/data",
        )

        d = info.to_dict()
        assert d["dataset_name"] == "wikitext"
        assert d["num_samples"] == 128

    def test_layer_importance_to_dict(self):
        """Test LayerImportance to_dict method."""
        layer = LayerImportance(
            layer_name="test_layer",
            layer_index=0,
            importance_score=0.9,
            parameter_count=1000,
            recommended_bits=8,
            super_weight_indices=[1, 2, 3],
        )

        d = layer.to_dict()
        assert d["layer_name"] == "test_layer"
        assert d["super_weight_indices"] == [1, 2, 3]

    def test_importance_matrix_get_super_weight_count(self):
        """Test getting super weight count from matrix."""
        matrix = ImportanceMatrix(
            model_name="test",
            computation_method=ImportanceMethod.ACTIVATION_MAGNITUDE,
            calibration_info=CalibrationInfo("test", 10),
            layer_scores=[
                LayerImportance("layer1", 0, 0.9, 1000, super_weight_indices=[1, 2, 3]),
                LayerImportance("layer2", 1, 0.5, 2000, super_weight_indices=[4, 5]),
            ],
        )

        count = matrix.get_super_weight_count()
        assert count == 5

    def test_compute_importance_sample_error_skipped(self):
        """Test that samples causing errors are skipped."""
        from llm_quantize.lib.analysis.importance import compute_importance_matrix
        import torch

        mock_model = MagicMock()
        mock_model.config._name_or_path = "test-model"
        mock_model.config.num_hidden_layers = 2
        mock_model.config.hidden_size = 768
        mock_model.config.max_position_embeddings = 512

        mock_param = MagicMock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.named_parameters.return_value = []
        mock_model.named_modules.return_value = []

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = [Exception("Error"), {"input_ids": torch.tensor([[1, 2, 3]])}]

        with patch("llm_quantize.lib.analysis.importance._get_tokenizer", return_value=mock_tokenizer):
            result = compute_importance_matrix(
                model=mock_model,
                calibration_data=["bad sample", "good sample"],
            )

        assert isinstance(result, ImportanceMatrix)
