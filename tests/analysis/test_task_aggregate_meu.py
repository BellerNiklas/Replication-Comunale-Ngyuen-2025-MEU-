from pathlib import Path

import numpy as np
import pandas as pd

from meu_replication.analysis.task_aggregate_meu import run_meu_aggregation_stage
from meu_replication.analysis.task_compute_uncertainty import run_uncertainty_stage
from meu_replication.analysis.task_estimate_factors import task_estimate_factors
from meu_replication.analysis.task_forecast_errors import task_forecast_errors
from tests.analysis._synthetic_sv_helpers import write_synthetic_sv_outputs
from tests.analysis.test_task_forecast_errors import _make_panel
from tests.analysis.test_task_stochastic_volatility import FAST_TASK_CONFIG

EXPECTED_FIRST_DATE = "2020-05"
EXPECTED_FORECAST_OBS = 32
EXPECTED_HORIZONS = FAST_TASK_CONFIG.h_max
EXPECTED_AGGREGATE_ROWS = EXPECTED_FORECAST_OBS * EXPECTED_HORIZONS


def test_run_meu_aggregation_stage_writes_expected_outputs(tmp_path: Path):
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

    sv_outputs = write_synthetic_sv_outputs(
        tmp_path=tmp_path,
        series_order_path=factor_outputs["series_order"],
        regression_coefs_f_path=forecast_outputs["regression_coefs_f"],
        forecast_errors_y_path=forecast_outputs["forecast_errors_y"],
        forecast_errors_f_path=forecast_outputs["forecast_errors_f"],
    )

    validation_summary_path = tmp_path / "sv_validation_summary.parquet"
    pd.DataFrame(
        [{"validation_passed": True, "failed_metrics": ""}]
    ).to_parquet(validation_summary_path, index=False)

    uncertainty_output = tmp_path / "uncertainty_variance.parquet"
    run_uncertainty_stage(
        depends_on={
            "series_order": factor_outputs["series_order"],
            "forecast_metadata": forecast_outputs["forecast_metadata"],
            "regression_coefs_y": forecast_outputs["regression_coefs_y"],
            "regression_coefs_f": forecast_outputs["regression_coefs_f"],
            "sv_params_y": sv_outputs["sv_params_y"],
            "sv_latent_y": sv_outputs["sv_latent_y"],
            "sv_params_f": sv_outputs["sv_params_f"],
            "sv_latent_f": sv_outputs["sv_latent_f"],
            "sv_validation_summary": validation_summary_path,
        },
        produces={"uncertainty_variance": uncertainty_output},
        config=FAST_TASK_CONFIG,
    )

    meu_output = tmp_path / "meu_ea.parquet"
    run_meu_aggregation_stage(
        depends_on={
            "uncertainty_variance": uncertainty_output,
            "series_order": factor_outputs["series_order"],
        },
        produces={"meu_ea": meu_output},
    )

    meu_ea = pd.read_parquet(meu_output)
    uncertainty = pd.read_parquet(uncertainty_output)
    direct = (
        uncertainty.assign(
            meu=np.sqrt(uncertainty["variance"].to_numpy(dtype=np.float64))
        )
        .groupby(["date", "horizon"], as_index=False, observed=True)["meu"]
        .mean()
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    direct["date"] = direct["date"].astype(str)
    direct["horizon"] = direct["horizon"].astype("int16")
    direct["meu"] = direct["meu"].astype("float64")

    assert meu_ea.columns.tolist() == ["date", "horizon", "meu"]
    assert len(meu_ea) == EXPECTED_AGGREGATE_ROWS
    assert meu_ea.loc[0, "date"] == EXPECTED_FIRST_DATE
    assert int(meu_ea["horizon"].min()) == 1
    assert int(meu_ea["horizon"].max()) == EXPECTED_HORIZONS
    assert np.isfinite(meu_ea["meu"]).all()
    pd.testing.assert_frame_equal(meu_ea, direct)
