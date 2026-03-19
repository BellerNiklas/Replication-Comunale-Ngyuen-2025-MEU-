import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_pytask_collect_reports_all_analysis_panel_ids():
    result = subprocess.run(
        [sys.executable, "-m", "pytask", "collect", "src/meu_replication/analysis"],
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, output
    assert "analysis_2021" in output
    assert "analysis_2022" in output
    assert "analysis_2025" in output
