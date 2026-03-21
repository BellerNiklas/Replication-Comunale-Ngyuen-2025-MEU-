"""Validate stochastic-volatility convergence before uncertainty estimation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.output_layout import AnalysisOutputLayout
from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
)
from meu_replication.analysis.sv_validation import (
    assert_sv_validation_passed,
    build_stability_metrics,
    build_sv_validation_summary,
    build_validation_subset_metrics,
)


def _validation_dependencies(layout: AnalysisOutputLayout) -> dict[str, Path]:
    return {
        "forecast_errors_y": layout.forecasts_dir / "forecast_errors_y.parquet",
        "forecast_errors_f": layout.forecasts_dir / "forecast_errors_f.parquet",
        "series_order": layout.factors_dir / "series_order.parquet",
        "sv_params_y": layout.sv_dir / "sv_params_y.parquet",
        "sv_params_f": layout.sv_dir / "sv_params_f.parquet",
        "sv_diagnostics": layout.sv_diagnostics_dir / "sv_diagnostics.parquet",
    }


def _validation_outputs(layout: AnalysisOutputLayout) -> dict[str, Path]:
    return {
        "sv_validation_summary": layout.sv_diagnostics_dir / "sv_validation_summary.parquet",
        "sv_validation_subset_metrics": layout.sv_diagnostics_dir
        / "sv_validation_subset_metrics.parquet",
    }


def run_stochastic_volatility_validation(
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    config: MEUConfig,
) -> None:
    """Execute the SV validation gate and persist the results."""
    series_order = pd.read_parquet(depends_on["series_order"])
    sv_params_y = pd.read_parquet(depends_on["sv_params_y"])
    sv_params_f = pd.read_parquet(depends_on["sv_params_f"])
    sv_diagnostics = pd.read_parquet(depends_on["sv_diagnostics"])

    subset_metrics = build_validation_subset_metrics(
        forecast_errors_y_path=depends_on["forecast_errors_y"],
        forecast_errors_f_path=depends_on["forecast_errors_f"],
        series_order=series_order,
        config=config,
    )
    stability_metrics = build_stability_metrics(
        forecast_errors_y_path=depends_on["forecast_errors_y"],
        series_order=series_order,
        config=config,
    )
    summary = build_sv_validation_summary(
        subset_metrics=subset_metrics,
        diagnostics=sv_diagnostics,
        params_y=sv_params_y,
        params_f=sv_params_f,
        stability_metrics=stability_metrics,
        config=config,
    )

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_metrics.to_parquet(produces["sv_validation_subset_metrics"], index=False)
    summary.to_parquet(produces["sv_validation_summary"], index=False)
    assert_sv_validation_passed(summary)

    print(
        f"[{config.panel_name}] SV validation complete: "
        f"passed={bool(summary.loc[0, 'validation_passed'])}, "
        f"failed_metrics={summary.loc[0, 'failed_metrics'] or 'none'}."
    )


for _spec in ANALYSIS_PANELS:

    @pytask.task(id=_spec.task_id)
    def task_validate_stochastic_volatility(
        depends_on: dict[str, Path] = _validation_dependencies(_spec.layout),
        produces: dict[str, Path] = _validation_outputs(_spec.layout),
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Run SV validation for one cleaned panel."""
        run_stochastic_volatility_validation(
            depends_on=depends_on,
            produces=produces,
            config=MEUConfig(panel_name=panel_name, sv_mode="fast"),
        )
