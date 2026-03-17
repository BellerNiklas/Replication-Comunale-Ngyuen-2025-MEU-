"""Aggregate the stage-4 uncertainty output to the baseline EA MEU."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meu_replication.analysis.aggregate_meu import aggregate_ea_meu
from meu_replication.config import ANALYSIS


def _aggregation_dependencies() -> dict[str, Path]:
    return {
        "uncertainty_variance": ANALYSIS / "uncertainty_variance.parquet",
        "series_order": ANALYSIS / "series_order.parquet",
    }


def _aggregation_outputs() -> dict[str, Path]:
    return {
        "meu_ea": ANALYSIS / "meu_ea.parquet",
    }


def task_aggregate_meu(
    depends_on: dict[str, Path] = _aggregation_dependencies(),
    produces: dict[str, Path] = _aggregation_outputs(),
) -> None:
    """Run the baseline EA-wide aggregation step."""
    run_meu_aggregation_stage(depends_on=depends_on, produces=produces)


def run_meu_aggregation_stage(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
) -> None:
    """Aggregate uncertainty to the public baseline EA MEU artifact."""
    meu_ea = aggregate_ea_meu(
        uncertainty_variance=pd.read_parquet(depends_on["uncertainty_variance"]),
        series_order=pd.read_parquet(depends_on["series_order"]),
    )

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    meu_ea.to_parquet(produces["meu_ea"], index=False)

    print(
        "EA aggregation complete: "
        f"{meu_ea['date'].nunique()} dates, "
        f"{meu_ea['horizon'].nunique()} horizons, "
        f"{len(meu_ea)} rows."
    )
