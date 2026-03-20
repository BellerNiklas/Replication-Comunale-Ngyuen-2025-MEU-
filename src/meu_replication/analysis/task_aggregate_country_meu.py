"""Aggregate the stage-4 uncertainty output to country-level MEUs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.analysis.aggregate_meu import (
    aggregate_country_meu,
    build_basket_membership,
    build_country_baskets,
)
from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
    DEFAULT_ANALYSIS_PANEL,
)


def _country_aggregation_dependencies(base_dir: Path) -> dict[str, Path]:
    return {
        "uncertainty_variance": base_dir / "uncertainty_variance.parquet",
        "series_order": base_dir / "series_order.parquet",
    }


def _country_aggregation_outputs(base_dir: Path) -> dict[str, Path]:
    return {
        "all_countries_meu": base_dir / "countries" / "all_countries_meu.parquet",
        "basket_membership": base_dir / "countries" / "basket_membership.parquet",
    }


def run_country_meu_aggregation_stage(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    panel_name: str = DEFAULT_ANALYSIS_PANEL.panel_name,
) -> None:
    """Aggregate uncertainty to country-level MEU artifacts."""
    series_order = pd.read_parquet(depends_on["series_order"])
    uncertainty_variance = pd.read_parquet(depends_on["uncertainty_variance"])

    all_countries_meu = aggregate_country_meu(
        uncertainty_variance=uncertainty_variance,
        series_order=series_order,
    )
    baskets = build_country_baskets(series_order)
    membership = build_basket_membership(baskets)

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    all_countries_meu.to_parquet(produces["all_countries_meu"], index=False)
    membership.to_parquet(produces["basket_membership"], index=False)

    n_countries = all_countries_meu["country_iso2"].nunique()
    n_dates = all_countries_meu["date"].nunique()
    n_horizons = all_countries_meu["horizon"].nunique()
    print(
        f"[{panel_name}] Country aggregation complete: "
        f"{n_countries} countries, {n_dates} dates, "
        f"{n_horizons} horizons, {len(all_countries_meu)} rows."
    )


for _spec in ANALYSIS_PANELS:

    @pytask.task(id=f"country_{_spec.task_id}")
    def task_aggregate_country_meu(
        depends_on: dict[str, Path] = _country_aggregation_dependencies(
            _spec.output_dir,
        ),
        produces: dict[str, Path] = _country_aggregation_outputs(_spec.output_dir),
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Run country aggregation for one cleaned panel."""
        run_country_meu_aggregation_stage(
            depends_on=depends_on,
            produces=produces,
            panel_name=panel_name,
        )
