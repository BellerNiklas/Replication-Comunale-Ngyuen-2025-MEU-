"""Aggregate the stage-4 uncertainty output to the baseline EA MEU."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.analysis.aggregate_meu import aggregate_ea_meu
from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
    DEFAULT_ANALYSIS_PANEL,
)


def _aggregation_dependencies(base_dir: Path) -> dict[str, Path]:
    return {
        "uncertainty_variance": base_dir / "uncertainty_variance.parquet",
        "series_order": base_dir / "series_order.parquet",
    }


def _aggregation_outputs(base_dir: Path) -> dict[str, Path]:
    return {
        "meu_ea": base_dir / "meu_ea.parquet",
    }


def run_meu_aggregation_stage(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    panel_name: str = DEFAULT_ANALYSIS_PANEL.panel_name,
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
        f"[{panel_name}] EA aggregation complete: "
        f"{meu_ea['date'].nunique()} dates, "
        f"{meu_ea['horizon'].nunique()} horizons, "
        f"{len(meu_ea)} rows."
    )


for _spec in ANALYSIS_PANELS:

    @pytask.task(id=_spec.task_id)
    def task_aggregate_meu(
        depends_on: dict[str, Path] = _aggregation_dependencies(_spec.output_dir),
        produces: dict[str, Path] = _aggregation_outputs(_spec.output_dir),
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Run EA aggregation for one cleaned panel."""
        run_meu_aggregation_stage(
            depends_on=depends_on,
            produces=produces,
            panel_name=panel_name,
        )
