"""Tests for progress reporting functionality."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from llm_quantize.lib.progress import ProgressReporter
from llm_quantize.models import Verbosity


class TestProgressReporterLogging:
    """Tests for logging functionality."""

    def test_log_respects_verbosity_quiet(self) -> None:
        """Test that quiet mode suppresses normal logs."""
        console = Console(file=StringIO(), force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.QUIET, console=console)

        reporter.log_info("This should not appear")

        output = console.file.getvalue()  # type: ignore
        assert output == ""

    def test_log_shows_in_normal_mode(self) -> None:
        """Test that normal mode shows info logs."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        reporter.log_info("Test message")

        assert "Test message" in output.getvalue()

    def test_log_verbose_hidden_in_normal(self) -> None:
        """Test that verbose logs are hidden in normal mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        reporter.log_verbose("Verbose message")

        assert "Verbose message" not in output.getvalue()

    def test_log_verbose_shown_in_verbose(self) -> None:
        """Test that verbose logs are shown in verbose mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE, console=console)

        reporter.log_verbose("Verbose message")

        assert "Verbose message" in output.getvalue()

    def test_log_error_always_shown(self) -> None:
        """Test that errors are always shown."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.QUIET, console=console)

        reporter.log_error("Error message")

        assert "Error" in output.getvalue()

    def test_log_warning_shown_in_normal(self) -> None:
        """Test that warnings are shown in normal mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        reporter.log_warning("Warning message")

        assert "Warning" in output.getvalue()

    def test_log_warning_hidden_in_quiet(self) -> None:
        """Test that warnings are hidden in quiet mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.QUIET, console=console)

        reporter.log_warning("Warning message")

        assert "Warning" not in output.getvalue()

    def test_log_debug_hidden_in_normal(self) -> None:
        """Test that debug logs are hidden in normal mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        reporter.log_debug("Debug message")

        assert "Debug message" not in output.getvalue()

    def test_log_debug_shown_in_debug(self) -> None:
        """Test that debug logs are shown in debug mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.DEBUG, console=console)

        reporter.log_debug("Debug message")

        assert "Debug message" in output.getvalue()


class TestProgressBarUpdate:
    """Tests for progress bar update functionality."""

    def test_update_with_description(self) -> None:
        """Test updating progress with new description."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        with reporter.progress_bar(total=10, description="Initial"):
            reporter.update(advance=1, description="New description")

        # Just verify no exception is raised

    def test_set_total(self) -> None:
        """Test setting new total value."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        with reporter.progress_bar(total=10, description="Testing"):
            reporter.set_total(20)

        # Just verify no exception is raised

    def test_set_total_outside_context(self) -> None:
        """Test set_total outside progress bar context does nothing."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        # Should not raise
        reporter.set_total(20)


class TestFormatSizeEdgeCases:
    """Tests for format_size edge cases."""

    def test_format_size_terabytes(self) -> None:
        """Test formatting terabytes."""
        reporter = ProgressReporter()
        assert reporter.format_size(1610612736000) == "1.5 TB"

    def test_format_size_petabytes(self) -> None:
        """Test formatting petabytes."""
        reporter = ProgressReporter()
        # 1.5 PB
        size = int(1.5 * 1024 * 1024 * 1024 * 1024 * 1024)
        result = reporter.format_size(size)
        assert "PB" in result


class TestProgressBar:
    """Tests for progress bar functionality."""

    def test_progress_bar_context_manager(self) -> None:
        """Test progress bar context manager."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        with reporter.progress_bar(total=10, description="Testing"):
            for i in range(10):
                reporter.update(advance=1)

        # Progress bar should have been shown
        assert reporter._progress is None  # Cleaned up after context

    def test_progress_bar_quiet_mode(self) -> None:
        """Test that progress bar is not shown in quiet mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.QUIET, console=console)

        with reporter.progress_bar(total=10, description="Testing"):
            reporter.update(advance=5)

        # No progress bar output in quiet mode
        assert reporter._progress is None

    def test_update_with_memory_tracking(self) -> None:
        """Test that memory usage is tracked."""
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL)

        reporter.update(memory_usage=1000000)
        reporter.update(memory_usage=2000000)
        reporter.update(memory_usage=1500000)

        assert reporter.peak_memory == 2000000


class TestFormatting:
    """Tests for formatting utilities."""

    def test_format_size_bytes(self) -> None:
        """Test formatting bytes."""
        reporter = ProgressReporter()
        assert reporter.format_size(512) == "512.0 B"

    def test_format_size_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        reporter = ProgressReporter()
        assert reporter.format_size(1536) == "1.5 KB"

    def test_format_size_megabytes(self) -> None:
        """Test formatting megabytes."""
        reporter = ProgressReporter()
        assert reporter.format_size(1572864) == "1.5 MB"

    def test_format_size_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        reporter = ProgressReporter()
        assert reporter.format_size(1610612736) == "1.5 GB"

    def test_format_duration_seconds(self) -> None:
        """Test formatting seconds."""
        reporter = ProgressReporter()
        assert reporter.format_duration(45.5) == "45.5s"

    def test_format_duration_minutes(self) -> None:
        """Test formatting minutes."""
        reporter = ProgressReporter()
        assert reporter.format_duration(125) == "2m 5s"

    def test_format_duration_hours(self) -> None:
        """Test formatting hours."""
        reporter = ProgressReporter()
        assert reporter.format_duration(3725) == "1h 2m"


class TestReportCompletion:
    """Tests for completion reporting."""

    def test_report_completion_normal(self) -> None:
        """Test completion report in normal mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.NORMAL, console=console)

        reporter.report_completion(
            output_path="/path/to/output.gguf",
            file_size=4000000000,  # ~4GB
            duration=1800.0,  # 30 minutes
            compression_ratio=0.28,
        )

        result = output.getvalue()
        assert "Complete" in result
        assert "output.gguf" in result

    def test_report_completion_quiet(self) -> None:
        """Test completion report is suppressed in quiet mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.QUIET, console=console)

        reporter.report_completion(
            output_path="/path/to/output.gguf",
            file_size=4000000000,
            duration=1800.0,
            compression_ratio=0.28,
        )

        assert output.getvalue() == ""

    def test_report_completion_verbose_details(self) -> None:
        """Test completion report shows details in verbose mode."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        reporter = ProgressReporter(verbosity=Verbosity.VERBOSE, console=console)
        reporter._peak_memory = 16000000000  # 16GB

        reporter.report_completion(
            output_path="/path/to/output.gguf",
            file_size=4000000000,
            duration=1800.0,
            compression_ratio=0.28,
        )

        result = output.getvalue()
        assert "Duration" in result
        assert "Compression" in result
        assert "Peak memory" in result
