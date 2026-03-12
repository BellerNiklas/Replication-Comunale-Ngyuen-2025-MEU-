from pathlib import Path

import numpy as np
import pandas as pd

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

    factor_outputs = {
        "panel_wide": tmp_path / "panel_wide.parquet",
        "series_order": tmp_path / "series_order.parquet",
        "date_index": tmp_path / "date_index.parquet",
        "series_standardization": tmp_path / "series_standardization.parquet",
        "fhat": tmp_path / "fhat.parquet",
        "ghat": tmp_path / "ghat.parquet",
        "predictor_set": tmp_path / "predictor_set.parquet",
        "factor_metadata": tmp_path / "factor_metadata.parquet",
    }
    task_estimate_factors(depends_on=panel_path, produces=factor_outputs)

    forecast_outputs = {
        "forecast_errors_y": tmp_path / "forecast_errors_y.parquet",
        "forecast_errors_f": tmp_path / "forecast_errors_f.parquet",
        "regression_coefs_y": tmp_path / "regression_coefs_y.parquet",
        "regression_coefs_f": tmp_path / "regression_coefs_f.parquet",
        "predictor_selection_masks": tmp_path / "predictor_selection_masks.parquet",
        "forecast_metadata": tmp_path / "forecast_metadata.parquet",
    }
    task_forecast_errors(depends_on=factor_outputs, produces=forecast_outputs)

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
    assert metadata.loc[0, "n_obs_forecast_errors_y"] == EXPECTED_EFFECTIVE_OBS
    assert metadata.loc[0, "y_sample_start"] == "2020-05"
