"""Tests for quality metrics computation."""

from unittest.mock import MagicMock, patch

import pytest

from llm_quantize.models import (
    CoherenceTestResult,
    LayerError,
    QualityGrade,
    QuantizationQualityReport,
)


class TestComputePerplexity:
    """Tests for perplexity computation."""

    def test_compute_perplexity_basic(self):
        """Test basic perplexity computation."""
        from llm_quantize.lib.analysis.quality import compute_perplexity
        import torch

        # Mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        # Mock outputs
        mock_outputs = MagicMock()
        mock_outputs.loss.item.return_value = 2.0
        mock_model.return_value = mock_outputs

        texts = ["Test text"]
        result = compute_perplexity(mock_model, mock_tokenizer, texts)

        assert isinstance(result, float)

    def test_compute_perplexity_empty_texts(self):
        """Test perplexity with empty text list."""
        from llm_quantize.lib.analysis.quality import compute_perplexity
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()

        result = compute_perplexity(mock_model, mock_tokenizer, [])

        assert result == float("inf")


class TestTestCoherence:
    """Tests for coherence testing."""

    def test_test_coherence_basic(self):
        """Test basic coherence testing."""
        from llm_quantize.lib.analysis.quality import test_coherence
        import torch

        # Mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "The capital of France is Paris."

        # Mock generation
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs

        prompts = ["The capital of France is"]
        results = test_coherence(mock_model, mock_tokenizer, prompts)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], CoherenceTestResult)

    def test_test_coherence_with_default_prompts(self):
        """Test coherence with default prompts."""
        from llm_quantize.lib.analysis.quality import test_coherence
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Output text with reasonable length."

        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs

        results = test_coherence(mock_model, mock_tokenizer)

        assert len(results) == 5  # Default 5 prompts


class TestAnalyzeCoherence:
    """Tests for coherence analysis helper."""

    def test_analyze_coherence_empty(self):
        """Test coherence analysis with empty text."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        is_coherent, rep_score, gram_score = _analyze_coherence("")

        assert not is_coherent
        assert rep_score == 1.0
        assert gram_score == 0.0

    def test_analyze_coherence_short(self):
        """Test coherence analysis with very short text."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        is_coherent, rep_score, gram_score = _analyze_coherence("Hi")

        assert not is_coherent

    def test_analyze_coherence_repetitive(self):
        """Test coherence analysis with repetitive text."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "word word word word word"
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        assert rep_score > 0.5  # High repetition

    def test_analyze_coherence_pattern_repetition(self):
        """Test detection of pattern repetition."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "abc abc abc abc abc abc abc"
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        assert not is_coherent or rep_score >= 0.5

    def test_analyze_coherence_good_text(self):
        """Test coherence analysis with good text."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "This is a well-formed sentence with proper grammar."
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        assert is_coherent
        assert rep_score < 0.5
        assert gram_score > 0.5


class TestComputeLayerErrors:
    """Tests for layer error computation."""

    def test_compute_layer_errors(self):
        """Test layer error computation."""
        from llm_quantize.lib.analysis.quality import compute_layer_errors
        import torch

        # Mock models with no modules to avoid complex mocking
        mock_original = MagicMock()
        mock_quantized = MagicMock()

        mock_original.named_modules.return_value = []
        mock_quantized.named_modules.return_value = []

        sample_input = torch.tensor([[1, 2, 3]])

        results = compute_layer_errors(mock_original, mock_quantized, sample_input)

        assert isinstance(results, list)


class TestGenerateQualityReport:
    """Tests for quality report generation."""

    def test_generate_quality_report_basic(self):
        """Test basic quality report generation."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test-model",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.25,
        )

        assert isinstance(report, QuantizationQualityReport)
        assert report.model_name == "test-model"
        assert report.perplexity_increase == pytest.approx(0.05, 0.001)

    def test_generate_quality_report_with_coherence(self):
        """Test quality report with coherence results."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        coherence_results = [
            CoherenceTestResult("prompt1", "output1", True),
            CoherenceTestResult("prompt2", "output2", False),
        ]

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.5,
            coherence_results=coherence_results,
        )

        assert len(report.coherence_tests) == 2

    def test_generate_quality_report_high_perplexity(self):
        """Test quality report with high perplexity increase."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q2_K",
            perplexity_original=5.0,
            perplexity_quantized=6.0,  # 20% increase
        )

        assert len(report.warnings) > 0
        assert any("perplexity" in w.lower() for w in report.warnings)

    def test_generate_quality_report_low_coherence(self):
        """Test quality report with low coherence rate."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        coherence_results = [
            CoherenceTestResult("p1", "o1", False),
            CoherenceTestResult("p2", "o2", False),
            CoherenceTestResult("p3", "o3", True),
        ]

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q2_K",
            perplexity_original=5.0,
            perplexity_quantized=5.1,
            coherence_results=coherence_results,
        )

        assert any("coherence" in w.lower() for w in report.warnings)

    def test_generate_quality_report_ultra_low_bit(self):
        """Test quality report with ultra-low-bit quantization."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="IQ1_S",
            perplexity_original=5.0,
            perplexity_quantized=6.5,
            average_bits=1.5,
        )

        assert any("ultra-low" in r.lower() for r in report.recommendations)

    def test_generate_quality_report_degraded_adds_recommendations(self):
        """Test that degraded quality adds recommendations."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q2_K",
            perplexity_original=5.0,
            perplexity_quantized=6.0,  # 20% increase
        )

        assert len(report.recommendations) > 0


class TestLayerErrorModel:
    """Tests for LayerError model."""

    def test_layer_error_to_dict(self):
        """Test LayerError to_dict method."""
        error = LayerError(
            layer_name="layer1",
            layer_index=0,
            mse=0.01,
            max_error=0.05,
            relative_error=0.02,
            bit_width=4,
        )

        d = error.to_dict()
        assert d["layer_name"] == "layer1"
        assert d["mse"] == 0.01

    def test_layer_error_from_dict(self):
        """Test LayerError from_dict method."""
        data = {
            "layer_name": "layer2",
            "layer_index": 1,
            "mse": 0.02,
            "max_error": 0.1,
            "relative_error": 0.03,
            "bit_width": 8,
        }

        error = LayerError.from_dict(data)
        assert error.layer_name == "layer2"
        assert error.bit_width == 8


class TestCoherenceTestResultModel:
    """Tests for CoherenceTestResult model."""

    def test_coherence_result_to_dict(self):
        """Test CoherenceTestResult to_dict method."""
        result = CoherenceTestResult(
            prompt="Test prompt",
            output="Test output",
            is_coherent=True,
            repetition_score=0.1,
            grammar_score=0.9,
        )

        d = result.to_dict()
        assert d["prompt"] == "Test prompt"
        assert d["is_coherent"] is True
        assert d["repetition_score"] == 0.1

    def test_coherence_result_from_dict(self):
        """Test CoherenceTestResult from_dict method."""
        data = {
            "prompt": "Another prompt",
            "output": "Another output",
            "is_coherent": False,
            "repetition_score": 0.5,
            "grammar_score": 0.5,
        }

        result = CoherenceTestResult.from_dict(data)
        assert result.prompt == "Another prompt"
        assert result.is_coherent is False


class TestQuantizationQualityReportModel:
    """Tests for QuantizationQualityReport model."""

    def test_report_to_dict(self):
        """Test QuantizationQualityReport to_dict method."""
        report = QuantizationQualityReport(
            model_name="test-model",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.25,
            layer_errors=[
                LayerError("layer1", 0, 0.01, 0.05, 0.02, 4),
            ],
            coherence_tests=[
                CoherenceTestResult("prompt", "output", True),
            ],
        )

        d = report.to_dict()
        assert d["model_name"] == "test-model"
        assert len(d["layer_errors"]) == 1
        assert len(d["coherence_tests"]) == 1

    def test_report_from_dict(self):
        """Test QuantizationQualityReport from_dict method."""
        data = {
            "model_name": "loaded-model",
            "quantization_format": "awq",
            "quantization_level": "4bit",
            "perplexity_original": 4.0,
            "perplexity_quantized": 4.5,
            "layer_errors": [
                {
                    "layer_name": "layer1",
                    "layer_index": 0,
                    "mse": 0.01,
                    "max_error": 0.05,
                    "relative_error": 0.02,
                    "bit_width": 4,
                },
            ],
            "coherence_tests": [
                {
                    "prompt": "test",
                    "output": "output",
                    "is_coherent": True,
                },
            ],
            "warnings": ["warning1"],
            "recommendations": ["rec1"],
        }

        report = QuantizationQualityReport.from_dict(data)
        assert report.model_name == "loaded-model"
        assert len(report.layer_errors) == 1
        assert len(report.coherence_tests) == 1
        assert len(report.warnings) == 1

    def test_report_generate_summary_with_coherence(self):
        """Test generate_summary with coherence tests."""
        report = QuantizationQualityReport(
            model_name="test-model",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.25,
            coherence_tests=[
                CoherenceTestResult("prompt", "output", True),
                CoherenceTestResult("prompt2", "output2", False),
            ],
        )

        summary = report.generate_summary()
        assert "Coherence Rate" in summary

    def test_report_generate_summary_with_warnings_and_recommendations(self):
        """Test generate_summary with warnings and recommendations."""
        report = QuantizationQualityReport(
            model_name="test-model",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.25,
        )
        report.add_warning("Test warning")
        report.add_recommendation("Test recommendation")

        summary = report.generate_summary()
        assert "Warnings" in summary
        assert "Test warning" in summary
        assert "Recommendations" in summary
        assert "Test recommendation" in summary

    def test_report_compute_quality_grade_excellent(self):
        """Test compute_quality_grade returns EXCELLENT for low perplexity increase."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_increase=0.01,
        )

        grade = report.compute_quality_grade()
        assert grade == QualityGrade.EXCELLENT

    def test_report_compute_quality_grade_failed(self):
        """Test compute_quality_grade returns FAILED for low coherence rate."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_increase=0.01,
            coherence_tests=[
                CoherenceTestResult("p1", "o1", False),
                CoherenceTestResult("p2", "o2", False),
                CoherenceTestResult("p3", "o3", False),
            ],
        )

        grade = report.compute_quality_grade()
        assert grade == QualityGrade.FAILED

    def test_report_get_worst_layers(self):
        """Test get_worst_layers returns layers sorted by MSE."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            layer_errors=[
                LayerError("layer1", 0, 0.01, 0.05, 0.02, 4),
                LayerError("layer2", 1, 0.05, 0.10, 0.05, 4),
                LayerError("layer3", 2, 0.03, 0.07, 0.03, 4),
            ],
        )

        worst = report.get_worst_layers(2)
        assert len(worst) == 2
        assert worst[0].layer_name == "layer2"  # Highest MSE
        assert worst[1].layer_name == "layer3"

    def test_report_get_coherence_rate_no_tests(self):
        """Test get_coherence_rate returns 1.0 when no tests."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
        )

        rate = report.get_coherence_rate()
        assert rate == 1.0

    def test_report_get_coherence_rate_with_tests(self):
        """Test get_coherence_rate calculates correctly."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            coherence_tests=[
                CoherenceTestResult("p1", "o1", True),
                CoherenceTestResult("p2", "o2", True),
                CoherenceTestResult("p3", "o3", False),
            ],
        )

        rate = report.get_coherence_rate()
        assert rate == pytest.approx(2/3, 0.01)


class TestQualityAnalysisCoverage:
    """Additional tests to improve quality module coverage."""

    def test_compute_perplexity_with_long_sequence(self):
        """Test perplexity with sequence longer than max_length."""
        from llm_quantize.lib.analysis.quality import compute_perplexity
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        # Return a longer sequence to trigger sliding window
        mock_input_ids = torch.tensor([[i for i in range(1000)]])
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        mock_outputs = MagicMock()
        mock_outputs.loss.item.return_value = 2.0
        mock_model.return_value = mock_outputs

        result = compute_perplexity(mock_model, mock_tokenizer, ["Long text"], max_length=100, stride=50)
        assert isinstance(result, float)
        assert result > 0

    def test_compute_perplexity_tokenizer_error(self):
        """Test perplexity when tokenizer raises error."""
        from llm_quantize.lib.analysis.quality import compute_perplexity
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = Exception("Tokenization failed")

        result = compute_perplexity(mock_model, mock_tokenizer, ["Test"])
        assert result == float("inf")

    def test_test_coherence_generation_error(self):
        """Test coherence testing when generation fails."""
        from llm_quantize.lib.analysis.quality import test_coherence
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.side_effect = [MagicMock(), Exception("Generate failed")]

        mock_model.generate.side_effect = Exception("Generate failed")

        results = test_coherence(mock_model, mock_tokenizer, ["Test prompt"])
        # Should have error result
        assert len(results) == 1
        assert not results[0].is_coherent

    def test_analyze_coherence_two_words(self):
        """Test coherence analysis with exactly two words."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        is_coherent, rep_score, gram_score = _analyze_coherence("hi there")
        assert not is_coherent
        assert rep_score == 1.0

    def test_analyze_coherence_long_pattern_repetition(self):
        """Test detection of longer pattern repetition."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        # Create text with pattern repetition
        text = "hello world " * 20
        is_coherent, rep_score, gram_score = _analyze_coherence(text)
        assert rep_score >= 0.7

    def test_analyze_coherence_no_punctuation(self):
        """Test coherence with no punctuation."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "This is a sentence without punctuation"
        is_coherent, rep_score, gram_score = _analyze_coherence(text)
        # Has lower grammar score but still may be coherent
        assert isinstance(gram_score, float)

    def test_compute_layer_errors_with_modules(self):
        """Test layer error computation with actual modules."""
        from llm_quantize.lib.analysis.quality import compute_layer_errors
        import torch

        mock_original = MagicMock()
        mock_quantized = MagicMock()

        # Create mock Linear modules
        mock_linear1 = MagicMock(spec=torch.nn.Linear)
        mock_linear2 = MagicMock(spec=torch.nn.Linear)

        mock_original.named_modules.return_value = [("layer1", mock_linear1)]
        mock_quantized.named_modules.return_value = [("layer1", mock_linear2)]

        # Setup hooks manually
        captured_hooks = []

        def capture_hook(fn):
            captured_hooks.append(fn)
            return MagicMock()

        mock_linear1.register_forward_hook = capture_hook
        mock_linear2.register_forward_hook = capture_hook

        sample_input = torch.tensor([[1, 2, 3]])

        results = compute_layer_errors(mock_original, mock_quantized, sample_input)
        assert isinstance(results, list)

    def test_generate_quality_report_zero_perplexity(self):
        """Test quality report with zero original perplexity."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=0.0,  # Edge case
            perplexity_quantized=5.0,
        )

        assert report.perplexity_increase == 0.0

    def test_generate_quality_report_poor_grade(self):
        """Test quality report with POOR grade."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q2_K",
            perplexity_original=5.0,
            perplexity_quantized=8.0,  # 60% increase - poor
        )

        # Should have recommendations for poor quality
        assert len(report.recommendations) > 0

    def test_generate_quality_report_with_layer_errors(self):
        """Test quality report with layer errors."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        layer_errors = [
            LayerError("layer1", 0, 0.01, 0.05, 0.02, 4),
            LayerError("layer2", 1, 0.02, 0.08, 0.03, 4),
        ]

        report = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.2,
            layer_errors=layer_errors,
        )

        assert len(report.layer_errors) == 2

    def test_report_quality_grades(self):
        """Test all quality grades."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        # Test GOOD grade
        report_good = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.15,  # 3% increase
        )
        assert report_good.quality_grade == QualityGrade.GOOD

        # Test ACCEPTABLE grade
        report_acceptable = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.35,  # 7% increase
        )
        assert report_acceptable.quality_grade == QualityGrade.ACCEPTABLE

        # Test DEGRADED grade
        report_degraded = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=5.75,  # 15% increase
        )
        assert report_degraded.quality_grade == QualityGrade.DEGRADED

        # Test POOR grade
        report_poor = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_original=5.0,
            perplexity_quantized=6.5,  # 30% increase
        )
        assert report_poor.quality_grade == QualityGrade.POOR


class TestQualityReportEdgeCases:
    """Edge case tests for quality report model."""

    def test_report_add_warning(self):
        """Test adding warnings to report."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
        )

        report.add_warning("Warning 1")
        report.add_warning("Warning 2")

        assert len(report.warnings) == 2

    def test_report_add_recommendation(self):
        """Test adding recommendations to report."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
        )

        report.add_recommendation("Rec 1")
        report.add_recommendation("Rec 2")

        assert len(report.recommendations) == 2

    def test_report_generate_summary_empty(self):
        """Test generating summary for empty report."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
        )

        summary = report.generate_summary()
        assert "test" in summary
        assert "Q4_K_M" in summary

    def test_report_get_worst_layers_empty(self):
        """Test get_worst_layers with no layer errors."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
        )

        worst = report.get_worst_layers(5)
        assert len(worst) == 0

    def test_report_compute_quality_grade_good_coherence(self):
        """Test quality grade with good coherence but high perplexity."""
        report = QuantizationQualityReport(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q4_K_M",
            perplexity_increase=0.15,
            coherence_tests=[
                CoherenceTestResult("p1", "o1", True),
                CoherenceTestResult("p2", "o2", True),
                CoherenceTestResult("p3", "o3", True),
            ],
        )

        grade = report.compute_quality_grade()
        assert grade in [QualityGrade.DEGRADED, QualityGrade.ACCEPTABLE]

    def test_layer_error_with_bit_width(self):
        """Test LayerError with explicit bit_width."""
        error = LayerError(
            layer_name="layer1",
            layer_index=0,
            mse=0.01,
            max_error=0.05,
            relative_error=0.02,
            bit_width=4,
        )
        assert error.bit_width == 4

    def test_coherence_test_result_default_scores(self):
        """Test CoherenceTestResult with default scores."""
        result = CoherenceTestResult(
            prompt="Test",
            output="Output",
            is_coherent=True,
        )
        # Default scores should be set
        assert result.repetition_score is not None
        assert result.grammar_score is not None


class TestQualityFunctionsCoverage:
    """Additional tests for complete quality function coverage."""

    def test_test_coherence_successful_generation(self):
        """Test coherence with successful generation and coherence analysis."""
        from llm_quantize.lib.analysis.quality import test_coherence
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 1

        # Set up tokenizer to return input
        mock_input = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_input_tensor = MagicMock()
        mock_input_tensor.to.return_value = mock_input
        mock_tokenizer.return_value = mock_input_tensor

        # Set up model to return generated output
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        mock_model.generate.return_value = mock_outputs

        # Set up tokenizer decode to return coherent text
        mock_tokenizer.decode.return_value = "The capital of France is Paris, which is known as the city of light."

        results = test_coherence(mock_model, mock_tokenizer, ["The capital of France is"])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result.prompt, str)
        assert isinstance(result.output, str)
        # Coherence should be analyzed
        assert isinstance(result.is_coherent, bool)
        assert 0 <= result.repetition_score <= 1
        assert 0 <= result.grammar_score <= 1

    def test_compute_layer_errors_with_matching_layers(self):
        """Test layer error computation with actually matching layers."""
        from llm_quantize.lib.analysis.quality import compute_layer_errors
        import torch

        # Create real simple models instead of mocks
        class SimpleModel(torch.nn.Module):
            def __init__(self, output_tensor):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.output_tensor = output_tensor

            def forward(self, x):
                return self.linear(x.float())

        # Create two models with different weights
        torch.manual_seed(42)
        model1 = SimpleModel(torch.tensor([[1.0, 2.0, 3.0]]))
        torch.manual_seed(0)
        model2 = SimpleModel(torch.tensor([[1.1, 2.1, 3.1]]))

        sample_input = torch.tensor([[1.0, 2.0, 3.0]])

        with torch.no_grad():
            results = compute_layer_errors(model1, model2, sample_input)

        assert isinstance(results, list)
        # Should have at least one layer error
        assert len(results) >= 1
        assert isinstance(results[0], LayerError)
        assert results[0].mse >= 0
        assert results[0].max_error >= 0

    def test_compute_layer_errors_shape_mismatch(self):
        """Test layer errors when layer outputs have different shapes."""
        from llm_quantize.lib.analysis.quality import compute_layer_errors
        import torch

        # Create models with different output sizes
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x.float())

        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)  # Different output size

            def forward(self, x):
                return self.linear(x.float())

        model1 = Model1()
        model2 = Model2()

        sample_input = torch.tensor([[1.0, 2.0, 3.0]])

        with torch.no_grad():
            results = compute_layer_errors(model1, model2, sample_input)

        # Should skip layers with mismatched shapes
        assert isinstance(results, list)

    def test_compute_layer_errors_multiple_layers(self):
        """Test layer error computation with multiple linear layers."""
        from llm_quantize.lib.analysis.quality import compute_layer_errors
        import torch

        class MultiLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(3, 4)
                self.layer2 = torch.nn.Linear(4, 2)

            def forward(self, x):
                x = self.layer1(x.float())
                return self.layer2(x)

        torch.manual_seed(1)
        model1 = MultiLayerModel()
        torch.manual_seed(2)
        model2 = MultiLayerModel()

        sample_input = torch.tensor([[1.0, 2.0, 3.0]])

        with torch.no_grad():
            results = compute_layer_errors(model1, model2, sample_input)

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_perplexity_with_sliding_window_multiple_windows(self):
        """Test perplexity computation that triggers multiple sliding windows."""
        from llm_quantize.lib.analysis.quality import compute_perplexity
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_tokenizer = MagicMock()
        # Create input longer than max_length to trigger sliding window
        long_input_ids = torch.tensor([[i for i in range(200)]])
        mock_tokenizer.return_value = {"input_ids": long_input_ids}

        mock_outputs = MagicMock()
        mock_outputs.loss.item.return_value = 1.5
        mock_model.return_value = mock_outputs

        result = compute_perplexity(
            mock_model,
            mock_tokenizer,
            ["Long text " * 50],
            max_length=50,
            stride=25
        )

        assert isinstance(result, float)
        assert result > 0

    def test_analyze_coherence_all_caps(self):
        """Test coherence analysis with all caps text."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "THIS IS ALL CAPS TEXT WITHOUT LOWERCASE"
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        assert isinstance(is_coherent, bool)
        # All caps should still have some structure

    def test_analyze_coherence_very_long_text(self):
        """Test coherence analysis with very long text."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "This is a very long sentence. " * 100
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        # Long text might trigger pattern repetition check
        assert rep_score >= 0.7 or is_coherent is False

    def test_analyze_coherence_with_special_chars(self):
        """Test coherence analysis with special characters."""
        from llm_quantize.lib.analysis.quality import _analyze_coherence

        text = "Hello! World? This is a test. It has punctuation!!!"
        is_coherent, rep_score, gram_score = _analyze_coherence(text)

        assert gram_score >= 0.5  # Should have high grammar score with punctuation

    def test_generate_quality_report_all_grades(self):
        """Test generate_quality_report produces all quality grades."""
        from llm_quantize.lib.analysis.quality import generate_quality_report

        # FAILED grade (high perplexity + low coherence)
        report_failed = generate_quality_report(
            model_name="test",
            quantization_format="gguf",
            quantization_level="Q2_K",
            perplexity_original=5.0,
            perplexity_quantized=10.0,  # 100% increase
            coherence_results=[
                CoherenceTestResult("p1", "ERROR", False),
                CoherenceTestResult("p2", "ERROR", False),
            ],
        )
        assert report_failed.quality_grade == QualityGrade.FAILED

    def test_test_coherence_with_move_to_device(self):
        """Test coherence when tokenizer output needs .to(device)."""
        from llm_quantize.lib.analysis.quality import test_coherence
        import torch

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Set up tokenizer that returns object needing .to()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 1

        # Create a tokenizer output that supports .to()
        mock_tokenizer_output = MagicMock()
        mock_tokenizer_output.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        mock_tokenizer.return_value = mock_tokenizer_output

        mock_outputs = torch.tensor([[1, 2, 3, 4]])
        mock_model.generate.return_value = mock_outputs

        mock_tokenizer.decode.return_value = "Prompt output text with good structure."

        results = test_coherence(mock_model, mock_tokenizer, ["Prompt"])

        assert len(results) == 1
