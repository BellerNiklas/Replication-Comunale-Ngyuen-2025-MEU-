from pathlib import Path

import pandas as pd
import pytest

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.output_layout import build_panel_output_layout
from meu_replication.analysis import task_validate_stochastic_volatility as validation_task
from meu_replication.analysis.task_estimate_factors import task_estimate_factors
from meu_replication.analysis.task_forecast_errors import task_forecast_errors
from meu_replication.analysis.task_stochastic_volatility import (
    run_stochastic_volatility_stage,
)
from meu_replication.analysis.task_validate_stochastic_volatility import (
    run_stochastic_volatility_validation,
)
from tests.analysis.test_task_forecast_errors import _make_panel

FAST_VALIDATION_CONFIG = MEUConfig(
    sv_mode="fast",
    sv_draws_fast=240,
    sv_burnin_fast=200,
    sv_thin_para_fast=2,
    sv_thin_latent_fast=2,
    sv_validation_chains=2,
    sv_validation_subset_size=2,
    sv_rhat_threshold=1.2,
    sv_mu_rhat_threshold=1.2,
    sv_geweke_threshold=10.0,
    sv_geweke_p90_threshold=10.0,
    sv_stability_relative_tolerance=0.5,
    num_workers=1,
)


def test_run_stochastic_volatility_validation_writes_expected_outputs(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    _make_panel().to_parquet(panel_path, index=False)
    layout = build_panel_output_layout(
        "panel_2003_2025_strict_corr",
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

    sv_outputs = {
        "sv_params_y": layout.sv_dir / "sv_params_y.parquet",
        "sv_latent_y": layout.sv_dir / "sv_latent_y.parquet",
        "sv_params_f": layout.sv_dir / "sv_params_f.parquet",
        "sv_latent_f": layout.sv_dir / "sv_latent_f.parquet",
        "sv_diagnostics": layout.sv_diagnostics_dir / "sv_diagnostics.parquet",
    }
    run_stochastic_volatility_stage(
        depends_on={
            "forecast_errors_y": forecast_outputs["forecast_errors_y"],
            "forecast_errors_f": forecast_outputs["forecast_errors_f"],
            "series_order": factor_outputs["series_order"],
            "forecast_metadata": forecast_outputs["forecast_metadata"],
        },
        produces=sv_outputs,
        config=FAST_VALIDATION_CONFIG,
    )

    validation_outputs = {
        "sv_validation_summary": layout.sv_diagnostics_dir
        / "sv_validation_summary.parquet",
        "sv_validation_subset_metrics": layout.sv_diagnostics_dir
        / "sv_validation_subset_metrics.parquet",
    }
    run_stochastic_volatility_validation(
        depends_on={
            "forecast_errors_y": forecast_outputs["forecast_errors_y"],
            "forecast_errors_f": forecast_outputs["forecast_errors_f"],
            "series_order": factor_outputs["series_order"],
            "sv_params_y": sv_outputs["sv_params_y"],
            "sv_params_f": sv_outputs["sv_params_f"],
            "sv_diagnostics": sv_outputs["sv_diagnostics"],
        },
        produces=validation_outputs,
        config=FAST_VALIDATION_CONFIG,
    )

    summary = pd.read_parquet(validation_outputs["sv_validation_summary"])
    subset_metrics = pd.read_parquet(validation_outputs["sv_validation_subset_metrics"])

    assert summary.shape[0] == 1
    assert "validation_passed" in summary.columns
    assert "failed_metrics" in summary.columns
    assert {"f", "y"} <= set(subset_metrics["series_type"])
    assert {"rhat_mu", "rhat_phi", "rhat_sigma"} <= set(subset_metrics.columns)


def test_run_stochastic_volatility_validation_raises_after_writing_failed_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    series_order_path = tmp_path / "series_order.parquet"
    sv_params_y_path = tmp_path / "sv_params_y.parquet"
    sv_params_f_path = tmp_path / "sv_params_f.parquet"
    sv_diagnostics_path = tmp_path / "sv_diagnostics.parquet"
    summary_path = tmp_path / "sv_validation_summary.parquet"
    subset_metrics_path = tmp_path / "sv_validation_subset_metrics.parquet"

    pd.DataFrame({"series_position": [0], "series_id": ["a"]}).to_parquet(
        series_order_path, index=False
    )
    pd.DataFrame(
        {
            "series_position": [0],
            "series_id": ["a"],
            "mu": [0.1],
            "phi": [0.9],
            "sigma": [0.2],
            "offset_applied": [0.0],
            "adjusted_for_zero": [False],
        }
    ).to_parquet(sv_params_y_path, index=False)
    pd.DataFrame(
        {
            "predictor_position": [0],
            "predictor_name": ["factor_1"],
            "mu": [0.1],
            "phi": [0.9],
            "sigma": [0.2],
            "offset_applied": [0.0],
            "adjusted_for_zero": [False],
        }
    ).to_parquet(sv_params_f_path, index=False)
    pd.DataFrame(
        {
            "series_type": ["y"],
            "series_position": [0],
            "series_key": ["a"],
            "geweke_mu": [0.0],
            "geweke_phi": [0.0],
            "geweke_sigma": [0.0],
            "acceptance_phi": [float("nan")],
            "offset_applied": [0.0],
            "adjusted_for_zero": [False],
            "n_param_draws": [10],
            "n_latent_draws": [10],
        }
    ).to_parquet(sv_diagnostics_path, index=False)

    monkeypatch.setattr(
        validation_task,
        "build_validation_subset_metrics",
        lambda **_: pd.DataFrame(
            {
                "series_type": ["y"],
                "series_position": [0],
                "series_key": ["a"],
                "n_chains": [2],
                "n_draws_per_chain": [10],
                "rhat_mu": [1.0],
                "rhat_phi": [1.20],
                "rhat_sigma": [1.20],
                "passes_mu_rhat": [True],
                "passes_phi_sigma_rhat": [False],
            }
        ),
    )
    monkeypatch.setattr(
        validation_task,
        "build_stability_metrics",
        lambda **_: pd.DataFrame(
            {
                "series_position": [0],
                "series_id": ["a"],
                "phi_relative_diff": [0.01],
                "sigma_relative_diff": [0.01],
                "passes_stability": [True],
            }
        ),
    )
    monkeypatch.setattr(
        validation_task,
        "build_sv_validation_summary",
        lambda **_: pd.DataFrame(
            {
                "validation_passed": [False],
                "failed_metrics": ["subset_y_rhat"],
                "subset_y_pass_count": [0],
                "subset_y_sampled_count": [1],
                "subset_y_pass_threshold": [1],
                "subset_f_pass_all": [True],
                "subset_mu_pass_all": [True],
                "full_panel_params_are_finite": [True],
                "full_panel_params_are_admissible": [True],
                "full_panel_diagnostics_are_finite": [True],
                "median_abs_geweke_phi_y": [0.0],
                "median_abs_geweke_sigma_y": [0.0],
                "p90_abs_geweke_phi_y": [0.0],
                "p90_abs_geweke_sigma_y": [0.0],
                "factor_geweke_pass_all": [True],
                "stability_pass_all": [True],
                "stability_max_phi_relative_diff": [0.01],
                "stability_max_sigma_relative_diff": [0.01],
                "sv_validation_chains": [2],
                "sv_validation_subset_size": [1],
                "sv_rhat_threshold": [1.05],
                "sv_mu_rhat_threshold": [1.02],
                "sv_geweke_threshold": [2.6],
                "sv_geweke_p90_threshold": [5.0],
                "sv_stability_relative_tolerance": [0.20],
            }
        ),
    )

    with pytest.raises(RuntimeError, match="subset_y_rhat"):
        run_stochastic_volatility_validation(
            depends_on={
                "forecast_errors_y": tmp_path / "unused_y.parquet",
                "forecast_errors_f": tmp_path / "unused_f.parquet",
                "series_order": series_order_path,
                "sv_params_y": sv_params_y_path,
                "sv_params_f": sv_params_f_path,
                "sv_diagnostics": sv_diagnostics_path,
            },
            produces={
                "sv_validation_summary": summary_path,
                "sv_validation_subset_metrics": subset_metrics_path,
            },
            config=FAST_VALIDATION_CONFIG,
        )

    assert summary_path.exists()
    assert subset_metrics_path.exists()
