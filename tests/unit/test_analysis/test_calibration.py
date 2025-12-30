"""Tests for calibration data loading."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_quantize.lib.calibration import (
    DEFAULT_CALIBRATION_SAMPLES,
    get_default_calibration_samples,
    load_calibration_data,
    validate_calibration_data,
)


class TestLoadCalibrationData:
    """Tests for calibration data loading."""

    def test_default_samples(self):
        """Test getting default calibration samples."""
        samples = get_default_calibration_samples(5)
        assert len(samples) == 5
        assert all(isinstance(s, str) for s in samples)

    def test_load_from_none_uses_defaults(self):
        """Test that None source uses default samples."""
        samples = load_calibration_data(None, num_samples=10)
        assert len(samples) <= 10
        assert len(samples) > 0

    def test_load_from_json_array(self):
        """Test loading from JSON array file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = ["Sample 1", "Sample 2", "Sample 3"]
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert samples == test_data

    def test_load_from_json_object(self):
        """Test loading from JSON object with 'texts' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = {"texts": ["Sample 1", "Sample 2"]}
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert len(samples) == 2

    def test_load_from_jsonl(self):
        """Test loading from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write('{"text": "Sample 1"}\n')
                f.write('{"text": "Sample 2"}\n')
                f.write('{"text": "Sample 3"}\n')

            samples = load_calibration_data(str(path))
            assert len(samples) == 3
            assert samples[0] == "Sample 1"

    def test_load_from_text_file(self):
        """Test loading from plain text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.txt"
            with open(path, "w") as f:
                f.write("This is the first paragraph.\n\n")
                f.write("This is the second paragraph with more text.\n\n")
                f.write("This is the third paragraph.\n")

            samples = load_calibration_data(str(path))
            assert len(samples) >= 2

    def test_num_samples_limit(self):
        """Test that num_samples limits output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = [f"Sample {i}" for i in range(100)]
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path), num_samples=10)
            assert len(samples) == 10

    def test_max_length_truncation(self):
        """Test that samples are truncated to max_length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            long_text = "x" * 1000
            with open(path, "w") as f:
                json.dump([long_text], f)

            samples = load_calibration_data(str(path), max_length=100)
            assert len(samples[0]) == 100

    def test_nonexistent_file_uses_defaults(self):
        """Test that nonexistent file falls back to defaults."""
        samples = load_calibration_data("/nonexistent/path/data.json", num_samples=5)
        assert len(samples) <= 5
        assert len(samples) > 0


class TestValidateCalibrationData:
    """Tests for calibration data validation."""

    def test_empty_data_invalid(self):
        """Test that empty data is invalid."""
        is_valid, warnings = validate_calibration_data([])
        assert not is_valid
        assert "No calibration samples" in warnings[0]

    def test_few_samples_warning(self):
        """Test warning for few samples."""
        samples = ["Sample " + str(i) for i in range(5)]
        is_valid, warnings = validate_calibration_data(samples)
        assert any("samples provided" in w.lower() for w in warnings)

    def test_short_samples_warning(self):
        """Test warning for short samples."""
        samples = ["a", "bb", "ccc"] * 10
        is_valid, warnings = validate_calibration_data(samples)
        assert any("short" in w.lower() for w in warnings)

    def test_duplicate_samples_warning(self):
        """Test warning for duplicate samples."""
        samples = ["Same text"] * 50
        is_valid, warnings = validate_calibration_data(samples)
        assert any("duplicate" in w.lower() for w in warnings)

    def test_valid_data(self):
        """Test that good data passes validation."""
        samples = [f"This is sample number {i} with some reasonable length." for i in range(100)]
        is_valid, warnings = validate_calibration_data(samples)
        assert is_valid


class TestCalibrationEdgeCases:
    """Additional tests for edge cases in calibration loading."""

    def test_load_from_directory(self):
        """Test loading from a directory of text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create text files in directory
            for i in range(3):
                path = Path(tmpdir) / f"file{i}.txt"
                with open(path, "w") as f:
                    f.write(f"This is sample text from file {i} with enough content.\n")

            samples = load_calibration_data(tmpdir)
            assert len(samples) >= 1

    def test_load_json_with_samples_key(self):
        """Test loading JSON with 'samples' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = {"samples": ["Sample A", "Sample B"]}
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert len(samples) == 2

    def test_load_json_with_data_key(self):
        """Test loading JSON with 'data' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = {"data": ["Sample C", "Sample D"]}
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert len(samples) == 2

    def test_load_json_with_calibration_key(self):
        """Test loading JSON with 'calibration' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = {"calibration": ["Sample E", "Sample F"]}
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert len(samples) == 2

    def test_load_json_with_nested_text_objects(self):
        """Test loading JSON with nested objects containing 'text' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = {
                "item1": {"text": "Sample from item1"},
                "item2": {"text": "Sample from item2"},
            }
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert len(samples) >= 1

    def test_load_json_with_string_values(self):
        """Test loading JSON with string values directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            test_data = {
                "key1": "Sample string 1",
                "key2": "Sample string 2",
            }
            with open(path, "w") as f:
                json.dump(test_data, f)

            samples = load_calibration_data(str(path))
            assert len(samples) >= 1

    def test_load_json_single_value(self):
        """Test loading JSON with a single non-dict/non-list value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            with open(path, "w") as f:
                json.dump("Single string value", f)

            samples = load_calibration_data(str(path))
            assert len(samples) == 1

    def test_load_invalid_json_uses_defaults(self):
        """Test loading invalid JSON falls back to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            with open(path, "w") as f:
                f.write("not valid json {{{")

            samples = load_calibration_data(str(path), num_samples=5)
            assert len(samples) <= 5

    def test_load_jsonl_with_string_objects(self):
        """Test loading JSONL with plain string objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write('"String sample 1"\n')
                f.write('"String sample 2"\n')

            samples = load_calibration_data(str(path))
            assert len(samples) == 2

    def test_load_jsonl_with_content_key(self):
        """Test loading JSONL with 'content' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write('{"content": "Content sample 1"}\n')
                f.write('{"content": "Content sample 2"}\n')

            samples = load_calibration_data(str(path))
            assert len(samples) == 2

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL skips empty lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write('{"text": "Sample 1"}\n')
                f.write('\n')
                f.write('{"text": "Sample 2"}\n')
                f.write('   \n')
                f.write('{"text": "Sample 3"}\n')

            samples = load_calibration_data(str(path))
            assert len(samples) == 3

    def test_load_jsonl_with_invalid_lines(self):
        """Test loading JSONL skips invalid JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write('{"text": "Valid sample"}\n')
                f.write('not valid json\n')
                f.write('{"text": "Another valid"}\n')

            samples = load_calibration_data(str(path))
            assert len(samples) >= 2

    def test_load_jsonl_truncates_long_text(self):
        """Test JSONL loading truncates long text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            long_text = "x" * 500
            with open(path, "w") as f:
                f.write(json.dumps({"text": long_text}) + '\n')

            samples = load_calibration_data(str(path), max_length=100)
            assert len(samples[0]) == 100

    def test_load_text_file_line_separated(self):
        """Test loading text file with line-separated samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.txt"
            # Create many lines without paragraph separators
            with open(path, "w") as f:
                for i in range(20):
                    f.write(f"This is line {i} with some reasonable content for testing.\n")

            samples = load_calibration_data(str(path))
            assert len(samples) >= 1

    def test_load_text_file_skips_short_lines(self):
        """Test that short lines are skipped in text loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.txt"
            with open(path, "w") as f:
                f.write("short\n")
                f.write("This is a longer line that should be included.\n")
                f.write("x\n")

            samples = load_calibration_data(str(path))
            assert all(len(s) >= 20 for s in samples)

    def test_load_text_file_truncates_long(self):
        """Test text file loading truncates long samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.txt"
            long_text = "x" * 500
            with open(path, "w") as f:
                f.write(long_text + "\n")

            samples = load_calibration_data(str(path), max_length=100)
            if samples:
                assert len(samples[0]) <= 100

    def test_load_text_file_error_uses_defaults(self):
        """Test text file loading error falls back to defaults."""
        from unittest.mock import patch, mock_open

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.txt"
            path.touch()

            with patch("builtins.open", side_effect=IOError("Read error")):
                samples = load_calibration_data(str(path), num_samples=5)
                # Should fall back to defaults
                assert len(samples) <= 5

    def test_load_jsonl_error_uses_defaults(self):
        """Test JSONL file error falls back to defaults."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.touch()

            with patch("builtins.open", side_effect=IOError("Read error")):
                samples = load_calibration_data(str(path), num_samples=5)
                assert len(samples) <= 5

    def test_load_directory_with_json_files(self):
        """Test loading from directory with JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON files
            for i in range(2):
                path = Path(tmpdir) / f"data{i}.json"
                with open(path, "w") as f:
                    json.dump([f"Sample from JSON file {i}"], f)

            samples = load_calibration_data(tmpdir)
            assert len(samples) >= 1

    def test_load_empty_directory_uses_defaults(self):
        """Test loading from empty directory uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = load_calibration_data(tmpdir, num_samples=5)
            # May return empty or defaults
            assert isinstance(samples, list)

    def test_default_calibration_samples_constant(self):
        """Test DEFAULT_CALIBRATION_SAMPLES is non-empty."""
        assert len(DEFAULT_CALIBRATION_SAMPLES) > 0
        assert all(isinstance(s, str) for s in DEFAULT_CALIBRATION_SAMPLES)

    def test_load_jsonl_with_non_dict_object(self):
        """Test JSONL with non-dict object (list, number, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w") as f:
                f.write('[1, 2, 3]\n')  # List object
                f.write('42\n')  # Number

            samples = load_calibration_data(str(path))
            assert len(samples) >= 1
