"""Validate stochastic-volatility convergence before uncertainty estimation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.sv_validation import (
    assert_sv_validation_passed,
    build_stability_metrics,
    build_sv_validation_summary,
    build_validation_subset_metrics,
)
from meu_replication.config import ANALYSIS


def _validation_dependencies() -> dict[str, Path]:
    return {
        "forecast_errors_y": ANALYSIS / "forecast_errors_y.parquet",
        "forecast_errors_f": ANALYSIS / "forecast_errors_f.parquet",
        "series_order": ANALYSIS / "series_order.parquet",
        "sv_params_y": ANALYSIS / "sv_params_y.parquet",
        "sv_params_f": ANALYSIS / "sv_params_f.parquet",
        "sv_diagnostics": ANALYSIS / "sv_diagnostics.parquet",
    }


def _validation_outputs() -> dict[str, Path]:
    return {
        "sv_validation_summary": ANALYSIS / "sv_validation_summary.parquet",
        "sv_validation_subset_metrics": ANALYSIS
        / "sv_validation_subset_metrics.parquet",
    }


def task_validate_stochastic_volatility(
    depends_on: dict[str, Path] = _validation_dependencies(),
    produces: dict[str, Path] = _validation_outputs(),
    config: MEUConfig | None = None,
) -> None:
    """Run stochastic-volatility validation on sentinel y-series and factors."""
    run_stochastic_volatility_validation(
        depends_on=depends_on,
        produces=produces,
        config=MEUConfig(sv_mode="fast") if config is None else config,
    )


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
        "SV validation complete: "
        f"passed={bool(summary.loc[0, 'validation_passed'])}, "
        f"failed_metrics={summary.loc[0, 'failed_metrics'] or 'none'}."
    )
