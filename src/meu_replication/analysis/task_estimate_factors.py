"""Estimate the factor objects used by the MEU pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.analysis.factor_estimation import (
    estimate_predictor_set,
    prepare_analysis_panel,
)
from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
    DEFAULT_ANALYSIS_PANEL,
)


def _factor_outputs(base_dir: Path) -> dict[str, Path]:
    return {
        "panel_wide": base_dir / "panel_wide.parquet",
        "series_order": base_dir / "series_order.parquet",
        "date_index": base_dir / "date_index.parquet",
        "series_standardization": base_dir / "series_standardization.parquet",
        "fhat": base_dir / "fhat.parquet",
        "ghat": base_dir / "ghat.parquet",
        "predictor_set": base_dir / "predictor_set.parquet",
        "factor_metadata": base_dir / "factor_metadata.parquet",
    }


def run_factor_estimation_stage(
    *,
    depends_on: Path,
    produces: dict[str, Path],
    config: MEUConfig,
) -> None:
    """Execute the factor stage for one cleaned panel."""
    panel = pd.read_parquet(depends_on)
    wide, series_order, date_index = prepare_analysis_panel(panel)
    result = estimate_predictor_set(
        wide.to_numpy(dtype=float),
        kmax=config.kmax,
        ic_criterion=config.ic_criterion,
    )

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    wide.reset_index().to_parquet(produces["panel_wide"], index=False)

    date_index = date_index.assign(
        usable_for_forecast_errors=date_index["date_position"]
        >= config.forecast_lag_order
    )
    date_index.to_parquet(produces["date_index"], index=False)
    series_order.to_parquet(produces["series_order"], index=False)

    series_standardization = series_order[["series_position", "series_id"]].assign(
        mean=result.series_means,
        std=result.series_stds,
    )
    series_standardization.to_parquet(
        produces["series_standardization"],
        index=False,
    )

    _matrix_frame(
        dates=wide.index,
        values=result.fhat,
        prefix="F",
    ).to_parquet(produces["fhat"], index=False)
    _matrix_frame(
        dates=wide.index,
        values=result.ghat,
        prefix="G",
    ).to_parquet(produces["ghat"], index=False)
    predictor_columns = [
        *[f"F{i}" for i in range(1, result.factor_count + 1)],
        "F1_sq",
        "G1",
    ]
    pd.DataFrame(
        {
            "date": wide.index.astype(str),
            **{
                column: result.predictor_set[:, idx]
                for idx, column in enumerate(predictor_columns)
            },
        }
    ).to_parquet(produces["predictor_set"], index=False)

    effective_start = wide.index[config.forecast_lag_order]
    metadata = pd.DataFrame(
        [
            {
                "panel_name": config.panel_name,
                "n_obs": wide.shape[0],
                "n_series": wide.shape[1],
                "selected_factor_count": result.factor_count,
                "selected_squared_factor_count": result.squared_factor_count,
                "predictor_count": result.predictor_set.shape[1],
                "kmax": config.kmax,
                "ic_criterion": config.ic_criterion,
                "sample_start": str(wide.index[0]),
                "sample_end": str(wide.index[-1]),
                "effective_forecast_start": str(effective_start),
                "effective_forecast_end": str(wide.index[-1]),
            }
        ]
    )
    metadata.to_parquet(produces["factor_metadata"], index=False)

    print(
        f"[{config.panel_name}] Factor stage complete: "
        f"{wide.shape[0]} dates, {wide.shape[1]} series, "
        f"{result.factor_count} X factors, "
        f"{result.squared_factor_count} X^2 factors, "
        f"{result.predictor_set.shape[1]} predictors."
    )


def _matrix_frame(
    dates: pd.Index,
    values: pd.DataFrame | pd.Series | pd.Index | object,
    prefix: str,
) -> pd.DataFrame:
    """Convert a dense matrix to a date-indexed DataFrame with named columns."""
    matrix = pd.DataFrame(values)
    matrix.columns = [f"{prefix}{idx}" for idx in range(1, matrix.shape[1] + 1)]
    matrix.insert(0, "date", dates.astype(str))
    return matrix


for _spec in ANALYSIS_PANELS:

    @pytask.task(id=_spec.task_id)
    def task_estimate_factors(
        depends_on: Path = _spec.cleaned_panel_path,
        produces: dict[str, Path] = _factor_outputs(_spec.output_dir),
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Run the factor stage for one cleaned panel."""
        run_factor_estimation_stage(
            depends_on=depends_on,
            produces=produces,
            config=MEUConfig(panel_name=panel_name),
        )


class _TaskEstimateFactorsCallable:
    """Provide a direct-call interface without participating in pytask collection."""

    def __call__(
        self,
        depends_on: Path = DEFAULT_ANALYSIS_PANEL.cleaned_panel_path,
        produces: dict[str, Path] = _factor_outputs(DEFAULT_ANALYSIS_PANEL.output_dir),
        config: MEUConfig | None = None,
    ) -> None:
        run_factor_estimation_stage(
            depends_on=depends_on,
            produces=produces,
            config=MEUConfig(panel_name=DEFAULT_ANALYSIS_PANEL.panel_name)
            if config is None
            else config,
        )


task_estimate_factors = _TaskEstimateFactorsCallable()
