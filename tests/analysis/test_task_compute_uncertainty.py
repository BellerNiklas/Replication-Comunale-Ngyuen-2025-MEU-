from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meu_replication.analysis.task_compute_uncertainty import run_uncertainty_stage
from meu_replication.analysis.task_estimate_factors import task_estimate_factors
from meu_replication.analysis.task_forecast_errors import task_forecast_errors
from tests.analysis.test_task_forecast_errors import _make_panel
from tests.analysis.test_task_stochastic_volatility import FAST_TASK_CONFIG

EXPECTED_FIRST_DATE = "2020-05"
EXPECTED_SERIES = 4
EXPECTED_FORECAST_OBS = 32
EXPECTED_ROWS = EXPECTED_FORECAST_OBS * EXPECTED_SERIES * FAST_TASK_CONFIG.h_max


def test_run_uncertainty_stage_writes_expected_outputs(tmp_path: Path):
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

    sv_outputs = _write_synthetic_sv_outputs(
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

    result = pd.read_parquet(uncertainty_output)
    series_order = pd.read_parquet(factor_outputs["series_order"]).sort_values(
        "series_position"
    )

    assert result.columns.tolist() == ["date", "series_id", "horizon", "variance"]
    assert len(result) == EXPECTED_ROWS
    assert str(result.iloc[0]["date"]) == EXPECTED_FIRST_DATE
    assert int(result["horizon"].min()) == 1
    assert int(result["horizon"].max()) == FAST_TASK_CONFIG.h_max
    assert np.isfinite(result["variance"]).all()
    assert (result["variance"] > 0.0).all()

    first_block = result[
        (result["date"].astype(str) == EXPECTED_FIRST_DATE) & (result["horizon"] == 1)
    ]
    assert first_block["series_id"].astype(str).tolist() == series_order[
        "series_id"
    ].astype(str).tolist()


def test_run_uncertainty_stage_stops_when_sv_validation_failed(tmp_path: Path):
    summary_path = tmp_path / "sv_validation_summary.parquet"
    pd.DataFrame(
        [{"validation_passed": False, "failed_metrics": "subset_y_rhat"}]
    ).to_parquet(summary_path, index=False)

    with pytest.raises(RuntimeError, match="subset_y_rhat"):
        run_uncertainty_stage(
            depends_on={
                "series_order": tmp_path / "missing_series_order.parquet",
                "forecast_metadata": tmp_path / "missing_forecast_metadata.parquet",
                "regression_coefs_y": tmp_path / "missing_regression_coefs_y.parquet",
                "regression_coefs_f": tmp_path / "missing_regression_coefs_f.parquet",
                "sv_params_y": tmp_path / "missing_sv_params_y.parquet",
                "sv_latent_y": tmp_path / "missing_sv_latent_y.parquet",
                "sv_params_f": tmp_path / "missing_sv_params_f.parquet",
                "sv_latent_f": tmp_path / "missing_sv_latent_f.parquet",
                "sv_validation_summary": summary_path,
            },
            produces={"uncertainty_variance": tmp_path / "unused.parquet"},
            config=FAST_TASK_CONFIG,
        )


def _write_synthetic_sv_outputs(
    *,
    tmp_path: Path,
    series_order_path: Path,
    regression_coefs_f_path: Path,
    forecast_errors_y_path: Path,
    forecast_errors_f_path: Path,
) -> dict[str, Path]:
    series_order = pd.read_parquet(series_order_path).sort_values("series_position")
    regression_coefs_f = pd.read_parquet(regression_coefs_f_path).sort_values(
        "predictor_position"
    )
    forecast_errors_y = pd.read_parquet(forecast_errors_y_path)
    forecast_errors_f = pd.read_parquet(forecast_errors_f_path)

    sv_outputs = {
        "sv_params_y": tmp_path / "sv_params_y.parquet",
        "sv_latent_y": tmp_path / "sv_latent_y.parquet",
        "sv_params_f": tmp_path / "sv_params_f.parquet",
        "sv_latent_f": tmp_path / "sv_latent_f.parquet",
    }

    pd.DataFrame(
        {
            "series_position": series_order["series_position"].to_numpy(),
            "series_id": series_order["series_id"].astype(str).to_numpy(),
            "mu": np.full(len(series_order), -0.25),
            "phi": np.full(len(series_order), 0.85),
            "sigma": np.full(len(series_order), 0.4),
            "offset_applied": np.zeros(len(series_order)),
            "adjusted_for_zero": np.zeros(len(series_order), dtype=bool),
        }
    ).to_parquet(sv_outputs["sv_params_y"], index=False)

    latent_y = pd.DataFrame({"date": forecast_errors_y["date"].astype(str)})
    for series_id in series_order["series_id"].astype(str):
        latent_y[series_id] = np.log(np.square(forecast_errors_y[series_id]) + 0.1)
    latent_y.to_parquet(sv_outputs["sv_latent_y"], index=False)

    pd.DataFrame(
        {
            "predictor_position": regression_coefs_f["predictor_position"].to_numpy(),
            "predictor_name": regression_coefs_f["predictor_name"].astype(str).to_numpy(),
            "mu": np.full(len(regression_coefs_f), -0.5),
            "phi": np.full(len(regression_coefs_f), 0.8),
            "sigma": np.full(len(regression_coefs_f), 0.35),
            "offset_applied": np.zeros(len(regression_coefs_f)),
            "adjusted_for_zero": np.zeros(len(regression_coefs_f), dtype=bool),
        }
    ).to_parquet(sv_outputs["sv_params_f"], index=False)

    latent_f = pd.DataFrame({"date": forecast_errors_f["date"].astype(str)})
    predictor_names = regression_coefs_f["predictor_name"].astype(str).tolist()
    for predictor_name in predictor_names:
        latent_f[predictor_name] = np.log(np.square(forecast_errors_f[predictor_name]) + 0.1)
    latent_f.to_parquet(sv_outputs["sv_latent_f"], index=False)

    return sv_outputs
