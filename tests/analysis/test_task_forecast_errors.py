from pathlib import Path

import numpy as np
import pandas as pd

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.output_layout import build_panel_output_layout
from meu_replication.analysis.task_estimate_factors import task_estimate_factors
from meu_replication.analysis.task_forecast_errors import task_forecast_errors

EXPECTED_SERIES = 4
EXPECTED_EFFECTIVE_OBS = 32


def _make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = [
        f"{year}-{month:02d}" for year in range(2020, 2023) for month in range(1, 13)
    ]
    rows = []
    common = np.linspace(-1.5, 1.5, len(dates))
    for series_id, multiplier in {
        "A_SER": 1.0,
        "B_SER": 1.4,
        "C_SER": -0.7,
        "D_SER": 0.5,
    }.items():
        values = multiplier * common + rng.normal(scale=0.1, size=len(dates))
        for date, value in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": date,
                    "value": float(value),
                    "series_id": series_id,
                    "country_iso2": "DE",
                    "variable_name": f"name_{series_id}",
                    "category": 1,
                    "category_name": "test",
                    "source": "test",
                    "transformationcode": 1,
                }
            )
    return pd.DataFrame(rows)


def test_task_forecast_errors_writes_expected_outputs(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    _make_panel().to_parquet(panel_path, index=False)
    layout = build_panel_output_layout(
        "panel_2003_2021_strict_corr",
        panels_dir=tmp_path / "analysis" / "panels",
    )

    factor_outputs = {
        "panel_wide": layout.factors_dir / "panel_wide.parquet",
        "series_order": layout.factors_dir / "series_order.parquet",
        "date_index": layout.factors_dir / "date_index.parquet",
        "series_standardization": layout.factors_dir / "series_standardization.parquet",
        "fhat": layout.factors_dir / "fhat.parquet",
        "ghat": layout.factors_dir / "ghat.parquet",
        "predictor_set": layout.factors_dir / "predictor_set.parquet",
        "factor_metadata": layout.factors_dir / "factor_metadata.parquet",
    }
    config = MEUConfig(panel_name="panel_2003_2021_strict_corr")
    task_estimate_factors(depends_on=panel_path, produces=factor_outputs, config=config)

    forecast_outputs = {
        "forecast_errors_y": layout.forecasts_dir / "forecast_errors_y.parquet",
        "forecast_errors_f": layout.forecasts_dir / "forecast_errors_f.parquet",
        "regression_coefs_y": layout.forecasts_dir / "regression_coefs_y.parquet",
        "regression_coefs_f": layout.forecasts_dir / "regression_coefs_f.parquet",
        "predictor_selection_masks": layout.forecasts_dir
        / "predictor_selection_masks.parquet",
        "forecast_metadata": layout.forecasts_dir / "forecast_metadata.parquet",
    }
    task_forecast_errors(
        depends_on=factor_outputs,
        produces=forecast_outputs,
        config=config,
    )

    y_residuals = pd.read_parquet(forecast_outputs["forecast_errors_y"])
    f_residuals = pd.read_parquet(forecast_outputs["forecast_errors_f"])
    y_coefs = pd.read_parquet(forecast_outputs["regression_coefs_y"])
    masks = pd.read_parquet(forecast_outputs["predictor_selection_masks"])
    metadata = pd.read_parquet(forecast_outputs["forecast_metadata"])

    assert y_residuals.columns.tolist()[0] == "date"
    assert f_residuals.columns.tolist()[0] == "date"
    assert y_residuals.loc[0, "date"] == "2020-05"
    assert f_residuals.loc[0, "date"] == "2020-05"
    assert y_coefs.shape[0] == EXPECTED_SERIES
    assert masks.shape[0] == EXPECTED_SERIES
    assert metadata.loc[0, "panel_name"] == config.panel_name
    assert metadata.loc[0, "n_obs_forecast_errors_y"] == EXPECTED_EFFECTIVE_OBS
    assert metadata.loc[0, "y_sample_start"] == "2020-05"
