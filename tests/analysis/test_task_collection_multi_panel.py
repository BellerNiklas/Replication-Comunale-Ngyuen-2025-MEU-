import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _normalize_pytask_output(text: str) -> str:
    """Normalize pytask collect output for task-ID matching.

    pytask wraps long task names across terminal lines, inserting line breaks
    and box-drawing characters (``│``, ``├``, ``└``, ``─``, emoji) mid-word.
    Stripping all non-alphanumeric characters except underscores and brackets
    produces a single searchable blob where task IDs like
    ``country_analysis_2021`` appear as contiguous substrings.
    """
    return re.sub(r"[^a-zA-Z0-9_\[\]]", "", text)


@pytest.mark.integration
def test_pytask_collect_reports_all_analysis_panel_ids():
    result = subprocess.run(
        [sys.executable, "-m", "pytask", "collect", "src/meu_replication/analysis"],
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )
    raw_output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, raw_output

    output = _normalize_pytask_output(raw_output)

    assert "analysis_2021" in output
    assert "analysis_2022" in output
    assert "analysis_2025" in output
    assert "country_analysis_2021" in output
    assert "country_analysis_2022" in output
    assert "country_analysis_2025" in output
