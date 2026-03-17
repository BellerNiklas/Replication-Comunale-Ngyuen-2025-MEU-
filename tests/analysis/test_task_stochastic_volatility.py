from pathlib import Path

import pandas as pd

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.task_estimate_factors import task_estimate_factors
from meu_replication.analysis.task_forecast_errors import task_forecast_errors
from meu_replication.analysis.task_stochastic_volatility import (
    run_stochastic_volatility_stage,
)
from tests.analysis.test_task_forecast_errors import _make_panel

FAST_TASK_CONFIG = MEUConfig(
    sv_mode="fast",
    sv_draws_fast=120,
    sv_burnin_fast=100,
    sv_thin_para_fast=2,
    sv_thin_latent_fast=2,
    num_workers=1,
)
EXPECTED_SERIES = 4


def test_run_stochastic_volatility_stage_writes_expected_outputs(tmp_path: Path):
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

    sv_outputs = {
        "sv_params_y": tmp_path / "sv_params_y.parquet",
        "sv_latent_y": tmp_path / "sv_latent_y.parquet",
        "sv_params_f": tmp_path / "sv_params_f.parquet",
        "sv_latent_f": tmp_path / "sv_latent_f.parquet",
        "sv_diagnostics": tmp_path / "sv_diagnostics.parquet",
    }
    run_stochastic_volatility_stage(
        depends_on={
            "forecast_errors_y": forecast_outputs["forecast_errors_y"],
            "forecast_errors_f": forecast_outputs["forecast_errors_f"],
            "series_order": factor_outputs["series_order"],
            "forecast_metadata": forecast_outputs["forecast_metadata"],
        },
        produces=sv_outputs,
        config=FAST_TASK_CONFIG,
    )

    params_y = pd.read_parquet(sv_outputs["sv_params_y"])
    latent_y = pd.read_parquet(sv_outputs["sv_latent_y"])
    diagnostics = pd.read_parquet(sv_outputs["sv_diagnostics"])
    series_order = pd.read_parquet(factor_outputs["series_order"]).sort_values(
        "series_position"
    )
    expected_latent_columns = ["date", *series_order["series_id"].tolist()]

    assert params_y.shape[0] == EXPECTED_SERIES
    assert params_y["series_position"].tolist() == sorted(params_y["series_position"])
    assert latent_y.columns.tolist() == expected_latent_columns
    assert latent_y.loc[0, "date"] == "2020-05"
    assert set(diagnostics["series_type"]) == {"f", "y"}
    assert (params_y["sigma"] > 0).all()
    assert "acceptance_phi" in diagnostics.columns
    assert diagnostics["acceptance_phi"].isna().all()
    assert "acceptance_sigma" not in diagnostics.columns
    assert (tmp_path / "_sv_r" / "sv_params_y_core.parquet").exists()
