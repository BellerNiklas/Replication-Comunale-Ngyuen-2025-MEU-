from pathlib import Path

import pandas as pd

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.task_estimate_factors import task_estimate_factors


def _make_panel() -> pd.DataFrame:
    dates = [f"2020-{month:02d}" for month in range(1, 9)]
    rows = []
    for series_id, multiplier in {
        "A_SER": 1.0,
        "B_SER": 2.0,
        "C_SER": -0.5,
        "D_SER": 1.5,
    }.items():
        for position, date in enumerate(dates, start=1):
            rows.append(
                {
                    "date": date,
                    "value": multiplier * position,
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


def test_task_estimate_factors_writes_expected_outputs(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    _make_panel().to_parquet(panel_path, index=False)

    produces = {
        "panel_wide": tmp_path / "panel_wide.parquet",
        "series_order": tmp_path / "series_order.parquet",
        "date_index": tmp_path / "date_index.parquet",
        "series_standardization": tmp_path / "series_standardization.parquet",
        "fhat": tmp_path / "fhat.parquet",
        "ghat": tmp_path / "ghat.parquet",
        "predictor_set": tmp_path / "predictor_set.parquet",
        "factor_metadata": tmp_path / "factor_metadata.parquet",
    }

    config = MEUConfig(panel_name="panel_2003_2025_strict_corr")
    task_estimate_factors(depends_on=panel_path, produces=produces, config=config)

    metadata = pd.read_parquet(produces["factor_metadata"])
    wide = pd.read_parquet(produces["panel_wide"])
    predictors = pd.read_parquet(produces["predictor_set"])
    series_order = pd.read_parquet(produces["series_order"])

    assert metadata.loc[0, "n_obs"] == 8
    assert metadata.loc[0, "n_series"] == 4
    assert metadata.loc[0, "panel_name"] == config.panel_name
    assert list(series_order["series_id"]) == ["A_SER", "B_SER", "C_SER", "D_SER"]
    assert wide.columns.tolist()[0] == "date"
    assert predictors.columns.tolist()[0] == "date"
    assert predictors.shape[0] == 8
