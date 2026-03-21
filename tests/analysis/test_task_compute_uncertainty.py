from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meu_replication.analysis.output_layout import build_panel_output_layout
from meu_replication.analysis.task_compute_uncertainty import run_uncertainty_stage
from meu_replication.analysis.task_estimate_factors import task_estimate_factors
from meu_replication.analysis.task_forecast_errors import task_forecast_errors
from tests.analysis._synthetic_sv_helpers import write_synthetic_sv_outputs
from tests.analysis.test_task_forecast_errors import _make_panel
from tests.analysis.test_task_stochastic_volatility import FAST_TASK_CONFIG

EXPECTED_FIRST_DATE = "2020-05"
EXPECTED_SERIES = 4
EXPECTED_FORECAST_OBS = 32
EXPECTED_ROWS = EXPECTED_FORECAST_OBS * EXPECTED_SERIES * FAST_TASK_CONFIG.h_max


def test_run_uncertainty_stage_writes_expected_outputs(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    _make_panel().to_parquet(panel_path, index=False)
    layout = build_panel_output_layout(
        "panel_2003_2022_strict_corr",
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
    task_estimate_factors(depends_on=panel_path, produces=factor_outputs)

    forecast_outputs = {
        "forecast_errors_y": layout.forecasts_dir / "forecast_errors_y.parquet",
        "forecast_errors_f": layout.forecasts_dir / "forecast_errors_f.parquet",
        "regression_coefs_y": layout.forecasts_dir / "regression_coefs_y.parquet",
        "regression_coefs_f": layout.forecasts_dir / "regression_coefs_f.parquet",
        "predictor_selection_masks": layout.forecasts_dir
        / "predictor_selection_masks.parquet",
        "forecast_metadata": layout.forecasts_dir / "forecast_metadata.parquet",
    }
    task_forecast_errors(depends_on=factor_outputs, produces=forecast_outputs)

    sv_outputs = write_synthetic_sv_outputs(
        tmp_path=tmp_path,
        series_order_path=factor_outputs["series_order"],
        regression_coefs_f_path=forecast_outputs["regression_coefs_f"],
        forecast_errors_y_path=forecast_outputs["forecast_errors_y"],
        forecast_errors_f_path=forecast_outputs["forecast_errors_f"],
        sv_dir=layout.sv_dir,
    )

    validation_summary_path = layout.sv_diagnostics_dir / "sv_validation_summary.parquet"
    validation_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"validation_passed": True, "failed_metrics": ""}]
    ).to_parquet(validation_summary_path, index=False)

    uncertainty_output = layout.uncertainty_dir / "uncertainty_variance.parquet"
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
