import os


def pytest_report_teststatus(report, config):
    inline_durations = os.getenv("PYTEST_INLINE_DURATIONS") == "1"
    if report.when == "call" and inline_durations:
        if report.passed:
            return "passed", ".", f"[{report.duration:.3f}s]"
        if report.failed:
            return "failed", "F", f"[{report.duration:.3f}s]"
        if report.skipped:
            return "skipped", "s", f"[{report.duration:.3f}s]"
    return None
