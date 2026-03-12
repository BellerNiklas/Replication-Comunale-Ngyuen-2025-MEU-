"""Generate forecast-error artifacts for the MEU pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from meu_replication.analysis.forecast_errors import generate_forecast_errors
from meu_replication.analysis.model_config import MEUConfig
from meu_replication.config import ANALYSIS

FloatArray = NDArray[np.float64]


def _forecast_dependencies() -> dict[str, Path]:
    return {
        "panel_wide": ANALYSIS / "panel_wide.parquet",
        "series_order": ANALYSIS / "series_order.parquet",
        "series_standardization": ANALYSIS / "series_standardization.parquet",
        "predictor_set": ANALYSIS / "predictor_set.parquet",
        "factor_metadata": ANALYSIS / "factor_metadata.parquet",
    }


def _forecast_outputs() -> dict[str, Path]:
    return {
        "forecast_errors_y": ANALYSIS / "forecast_errors_y.parquet",
        "forecast_errors_f": ANALYSIS / "forecast_errors_f.parquet",
        "regression_coefs_y": ANALYSIS / "regression_coefs_y.parquet",
        "regression_coefs_f": ANALYSIS / "regression_coefs_f.parquet",
        "predictor_selection_masks": ANALYSIS / "predictor_selection_masks.parquet",
        "forecast_metadata": ANALYSIS / "forecast_metadata.parquet",
    }


def task_forecast_errors(
    depends_on: dict[str, Path] = _forecast_dependencies(),
    produces: dict[str, Path] = _forecast_outputs(),
) -> None:
    """Estimate the Milestone 2 forecast equations and persist the outputs."""
    config = MEUConfig()
    panel_wide = pd.read_parquet(depends_on["panel_wide"])
    series_order = pd.read_parquet(depends_on["series_order"])
    series_stats = pd.read_parquet(depends_on["series_standardization"])
    predictor_set = pd.read_parquet(depends_on["predictor_set"])
    factor_metadata = pd.read_parquet(depends_on["factor_metadata"])

    series_ids = series_order["series_id"].tolist()
    predictor_names = predictor_set.columns.tolist()[1:]
    dates = panel_wide["date"].astype(str).tolist()
    standardized_y = _standardize_panel(
        panel_wide=panel_wide,
        series_stats=series_stats,
        series_ids=series_ids,
    )
    result = generate_forecast_errors(
        standardized_y=standardized_y,
        predictor_values=predictor_set[predictor_names].to_numpy(dtype=float),
        dates=dates,
        predictor_names=predictor_names,
        py=config.py,
        pz=config.pz,
        pf=config.pf,
        threshold=config.threshold,
    )

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": result.y_dates,
            **{
                series_id: result.y_residuals[:, idx]
                for idx, series_id in enumerate(series_ids)
            },
        }
    ).to_parquet(produces["forecast_errors_y"], index=False)

    pd.DataFrame(
        {
            "date": result.f_dates,
            **{
                predictor_name: result.f_residuals[:, idx]
                for idx, predictor_name in enumerate(predictor_names)
            },
        }
    ).to_parquet(produces["forecast_errors_f"], index=False)

    pd.DataFrame(
        {
            "series_position": series_order["series_position"].to_numpy(),
            "series_id": series_ids,
            **{
                name: result.y_coefficients[:, idx]
                for idx, name in enumerate(result.y_regressor_names)
            },
        }
    ).assign(
        n_selected_predictors=result.selection_mask.sum(axis=1).astype("int64"),
    ).to_parquet(produces["regression_coefs_y"], index=False)

    pd.DataFrame(
        {
            "predictor_position": range(len(predictor_names)),
            "predictor_name": predictor_names,
            **{
                name: result.f_coefficients[:, idx]
                for idx, name in enumerate(result.factor_regressor_names)
            },
        }
    ).to_parquet(produces["regression_coefs_f"], index=False)

    pd.DataFrame(
        {
            "series_position": series_order["series_position"].to_numpy(),
            "series_id": series_ids,
            **{
                name: result.selection_mask[:, idx]
                for idx, name in enumerate(result.selection_regressor_names)
            },
        }
    ).assign(
        n_selected_predictors=result.selection_mask.sum(axis=1).astype("int64"),
    ).to_parquet(produces["predictor_selection_masks"], index=False)

    pd.DataFrame(
        [
            {
                "panel_name": config.panel_name,
                "n_obs_input": len(dates),
                "n_obs_forecast_errors_y": len(result.y_dates),
                "n_obs_forecast_errors_f": len(result.f_dates),
                "n_series": len(series_ids),
                "n_predictors": len(predictor_names),
                "py": config.py,
                "pz": config.pz,
                "pf": config.pf,
                "threshold": config.threshold,
                "bandwidth_y": result.bandwidth_y,
                "bandwidth_f": result.bandwidth_f,
                "y_sample_start": result.y_dates[0],
                "y_sample_end": result.y_dates[-1],
                "f_sample_start": result.f_dates[0],
                "f_sample_end": result.f_dates[-1],
                "selected_factor_count": int(
                    factor_metadata.loc[0, "selected_factor_count"]
                ),
                "predictor_count": int(factor_metadata.loc[0, "predictor_count"]),
            }
        ]
    ).to_parquet(produces["forecast_metadata"], index=False)

    print(
        "Forecast-error stage complete: "
        f"{len(series_ids)} series, "
        f"{len(predictor_names)} predictors, "
        f"{len(result.y_dates)} y residual dates, "
        f"{len(result.f_dates)} factor residual dates."
    )


def _standardize_panel(
    panel_wide: pd.DataFrame,
    series_stats: pd.DataFrame,
    series_ids: list[str],
) -> FloatArray:
    """Recreate the MATLAB-style z-scored panel from persisted stats."""
    if series_stats["series_id"].tolist() != series_ids:
        msg = "Series standardization stats must align with the persisted series order."
        raise ValueError(msg)

    panel_values = panel_wide[series_ids].to_numpy(dtype=float)
    means = series_stats["mean"].to_numpy(dtype=float)
    stds = series_stats["std"].to_numpy(dtype=float)
    return np.asarray((panel_values - means) / stds, dtype=np.float64)
