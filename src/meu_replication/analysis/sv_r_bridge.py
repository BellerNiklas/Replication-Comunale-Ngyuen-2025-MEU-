"""Shared helpers for invoking the R-backed stochastic-volatility script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

SV_R_SCRIPT = Path(__file__).with_name("estimate_sv.R")


def run_sv_r_script(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    options: dict[str, Any],
) -> None:
    """Run the shared R script with a pytask-r compatible JSON config."""
    payload = {
        "depends_on": {key: str(path) for key, path in depends_on.items()},
        "produces": {key: str(path) for key, path in produces.items()},
        **options,
    }
    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="sv_r_") as tmp_dir:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            ["Rscript", "--vanilla", str(SV_R_SCRIPT), str(config_path)],
            capture_output=True,
            check=False,
            cwd=SV_R_SCRIPT.parent,
            text=True,
        )

    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        msg = (
            "R stochastic-volatility script failed.\n"
            f"stdout:\n{stdout or '<empty>'}\n"
            f"stderr:\n{stderr or '<empty>'}"
        )
        raise RuntimeError(msg)
