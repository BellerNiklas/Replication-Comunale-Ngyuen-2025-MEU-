from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.sv_validation import (
    assert_sv_validation_passed,
    build_stability_metrics,
    build_sv_validation_summary,
    build_validation_subset_metrics,
    select_evenly_spaced_positions,
    select_validation_subset,
    split_rhat,
)

EXPECTED_SUBSET_START = 0
EXPECTED_SUBSET_END = 3
EXPECTED_SUBSET_COUNT = 3
EXPECTED_RHAT_THRESHOLD = 1.05
FAST_VALIDATION_CONFIG = MEUConfig(
    sv_mode="fast",
    sv_draws_fast=120,
    sv_burnin_fast=100,
    sv_thin_para_fast=2,
    sv_thin_latent_fast=2,
    sv_validation_chains=2,
    sv_validation_subset_size=2,
)
EXPECTED_PARAM_DRAWS = (
    FAST_VALIDATION_CONFIG.sv_draws // FAST_VALIDATION_CONFIG.sv_thin_para
)
EXPECTED_FACTOR_COUNT = 2
EXPECTED_STABILITY_SERIES = 3


def _simulate_residuals(seed: int, n_obs: int = 36) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.empty(n_obs)
    eps = rng.normal(size=n_obs)
    h[0] = -0.4
    for idx in range(1, n_obs):
        h[idx] = -0.2 + 0.95 * (h[idx - 1] + 0.2) + 0.15 * rng.normal()
    return np.exp(h / 2.0) * eps


def _make_forecast_errors_y() -> pd.DataFrame:
    dates = pd.period_range("2020-01", periods=36, freq="M").astype(str)
    return pd.DataFrame(
        {
            "date": dates,
            "a": _simulate_residuals(1),
            "b": _simulate_residuals(2),
            "c": _simulate_residuals(3),
            "d": _simulate_residuals(4),
        }
    )


def _make_forecast_errors_f() -> pd.DataFrame:
    dates = pd.period_range("2020-01", periods=36, freq="M").astype(str)
    return pd.DataFrame(
        {
            "date": dates,
            "factor_1": _simulate_residuals(11),
            "factor_2": _simulate_residuals(12),
        }
    )


def _make_series_order() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_position": [0, 1, 2, 3],
            "series_id": ["a", "b", "c", "d"],
        }
    )


def test_select_evenly_spaced_positions_includes_endpoints():
    assert select_evenly_spaced_positions(12, 4) == [0, 4, 7, 11]
    assert select_evenly_spaced_positions(4, 12) == [0, 1, 2, 3]


def test_select_validation_subset_uses_sorted_series_positions():
    series_order = pd.DataFrame(
        {
            "series_position": [2, 0, 3, 1],
            "series_id": ["c", "a", "d", "b"],
        }
    )

    subset = select_validation_subset(series_order, subset_size=3)

    assert subset["series_position"].tolist() == sorted(
        subset["series_position"].tolist()
    )
    assert subset["series_position"].tolist()[0] == EXPECTED_SUBSET_START
    assert subset["series_position"].tolist()[-1] == EXPECTED_SUBSET_END
    assert subset.shape[0] == EXPECTED_SUBSET_COUNT


def test_split_rhat_is_close_to_one_for_well_mixed_draws():
    rng = np.random.default_rng(123)
    draws = rng.normal(size=(4, 200))

    rhat = split_rhat(draws)

    assert np.isfinite(rhat)
    assert rhat < EXPECTED_RHAT_THRESHOLD


def test_build_validation_subset_metrics_runs_batched_r_backend(tmp_path: Path):
    forecast_errors_y_path = tmp_path / "forecast_errors_y.parquet"
    forecast_errors_f_path = tmp_path / "forecast_errors_f.parquet"
    _make_forecast_errors_y().to_parquet(forecast_errors_y_path, index=False)
    _make_forecast_errors_f().to_parquet(forecast_errors_f_path, index=False)

    subset_metrics = build_validation_subset_metrics(
        forecast_errors_y_path=forecast_errors_y_path,
        forecast_errors_f_path=forecast_errors_f_path,
        series_order=_make_series_order(),
        config=FAST_VALIDATION_CONFIG,
    )

    assert subset_metrics.shape[0] == (
        FAST_VALIDATION_CONFIG.sv_validation_subset_size + EXPECTED_FACTOR_COUNT
    )
    assert {"f", "y"} <= set(subset_metrics["series_type"])
    assert subset_metrics["n_chains"].eq(FAST_VALIDATION_CONFIG.sv_validation_chains).all()
    assert subset_metrics["n_draws_per_chain"].ge(EXPECTED_PARAM_DRAWS).all()
    assert np.isfinite(subset_metrics[["rhat_mu", "rhat_phi", "rhat_sigma"]]).all().all()


def test_build_validation_subset_metrics_retries_failed_subset_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    forecast_errors_y_path = tmp_path / "forecast_errors_y.parquet"
    forecast_errors_f_path = tmp_path / "forecast_errors_f.parquet"
    _make_forecast_errors_y()[["date", "a"]].to_parquet(forecast_errors_y_path, index=False)
    _make_forecast_errors_f()[["date", "factor_1"]].to_parquet(
        forecast_errors_f_path,
        index=False,
    )

    config = MEUConfig(
        sv_mode="fast",
        sv_draws_fast=120,
        sv_burnin_fast=100,
        sv_thin_para_fast=2,
        sv_thin_latent_fast=2,
        sv_validation_chains=2,
        sv_validation_subset_size=1,
    )
    calls: list[dict[str, int]] = []

    def fake_run_validation_batch(
        *,
        forecast_errors_y_path: Path,
        forecast_errors_f_path: Path,
        jobs: list[dict[str, int | str]],
        config: MEUConfig,
    ) -> pd.DataFrame:
        calls.append({"n_jobs": len(jobs), "sv_draws": config.sv_draws})
        return pd.DataFrame({"attempt": [len(calls)]})

    def fake_summarize_validation_draws(*, draws: pd.DataFrame, config: MEUConfig) -> pd.DataFrame:
        attempt = int(draws.loc[0, "attempt"])
        if attempt == 1:
            return pd.DataFrame(
                {
                    "series_type": ["y", "f"],
                    "series_position": [0, 0],
                    "series_key": ["a", "factor_1"],
                    "n_chains": [2, 2],
                    "n_draws_per_chain": [60, 60],
                    "rhat_mu": [1.0, 1.0],
                    "rhat_phi": [1.01, 1.20],
                    "rhat_sigma": [1.01, 1.20],
                    "passes_mu_rhat": [True, True],
                    "passes_phi_sigma_rhat": [True, False],
                }
            )
        return pd.DataFrame(
            {
                "series_type": ["f"],
                "series_position": [0],
                "series_key": ["factor_1"],
                "n_chains": [2],
                "n_draws_per_chain": [240],
                "rhat_mu": [1.0],
                "rhat_phi": [1.01],
                "rhat_sigma": [1.01],
                "passes_mu_rhat": [True],
                "passes_phi_sigma_rhat": [True],
            }
        )

    monkeypatch.setattr(
        "meu_replication.analysis.sv_validation._run_validation_batch",
        fake_run_validation_batch,
    )
    monkeypatch.setattr(
        "meu_replication.analysis.sv_validation._summarize_validation_draws",
        fake_summarize_validation_draws,
    )

    subset_metrics = build_validation_subset_metrics(
        forecast_errors_y_path=forecast_errors_y_path,
        forecast_errors_f_path=forecast_errors_f_path,
        series_order=pd.DataFrame({"series_position": [0], "series_id": ["a"]}),
        config=config,
    )

    assert len(calls) == 2
    assert calls[0]["n_jobs"] == 4
    assert calls[1]["n_jobs"] == 2
    assert calls[1]["sv_draws"] == calls[0]["sv_draws"] * 4
    assert subset_metrics["passes_phi_sigma_rhat"].all()


def test_build_stability_metrics_runs_batched_r_backend(tmp_path: Path):
    forecast_errors_y_path = tmp_path / "forecast_errors_y.parquet"
    _make_forecast_errors_y().to_parquet(forecast_errors_y_path, index=False)

    stability_metrics = build_stability_metrics(
        forecast_errors_y_path=forecast_errors_y_path,
        series_order=_make_series_order(),
        config=FAST_VALIDATION_CONFIG,
    )

    assert stability_metrics.shape[0] == EXPECTED_STABILITY_SERIES
    assert stability_metrics["series_position"].tolist() == sorted(
        stability_metrics["series_position"].tolist()
    )
    assert {"phi_relative_diff", "sigma_relative_diff", "passes_stability"} <= set(
        stability_metrics.columns
    )
    assert stability_metrics["passes_stability"].isin([True, False]).all()


def test_build_sv_validation_summary_collects_failed_metrics():
    config = MEUConfig(sv_mode="fast")
    subset_metrics = pd.DataFrame(
        {
            "series_type": ["y", "y", "f"],
            "series_position": [0, 1, 0],
            "series_key": ["a", "b", "factor_1"],
            "rhat_mu": [1.0, 1.0, 1.0],
            "rhat_phi": [1.01, 1.20, 1.01],
            "rhat_sigma": [1.01, 1.20, 1.01],
            "passes_mu_rhat": [True, True, True],
            "passes_phi_sigma_rhat": [True, False, True],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "series_type": ["y", "y", "f"],
            "series_position": [0, 1, 0],
            "series_key": ["a", "b", "factor_1"],
            "geweke_mu": [0.1, 0.2, 0.1],
            "geweke_phi": [0.2, 0.3, 3.0],
            "geweke_sigma": [0.2, 0.3, 0.4],
            "acceptance_phi": [float("nan"), float("nan"), float("nan")],
        }
    )
    params_y = pd.DataFrame(
        {
            "series_position": [0, 1],
            "series_id": ["a", "b"],
            "mu": [0.1, 0.2],
            "phi": [0.8, 0.9],
            "sigma": [0.5, 0.6],
        }
    )
    params_f = pd.DataFrame(
        {
            "predictor_position": [0],
            "predictor_name": ["factor_1"],
            "mu": [0.0],
            "phi": [0.7],
            "sigma": [0.4],
        }
    )
    stability_metrics = pd.DataFrame(
        {
            "series_position": [0, 10, 20],
            "series_id": ["a", "m", "z"],
            "phi_relative_diff": [0.01, 0.02, 0.03],
            "sigma_relative_diff": [0.01, 0.02, 0.03],
            "passes_stability": [True, True, True],
        }
    )

    summary = build_sv_validation_summary(
        subset_metrics=subset_metrics,
        diagnostics=diagnostics,
        params_y=params_y,
        params_f=params_f,
        stability_metrics=stability_metrics,
        config=config,
    )

    assert not bool(summary.loc[0, "validation_passed"])
    assert "subset_y_rhat" in summary.loc[0, "failed_metrics"]
    assert "factor_geweke" in summary.loc[0, "failed_metrics"]
    assert int(summary.loc[0, "subset_y_sampled_count"]) == 2
    assert int(summary.loc[0, "subset_y_pass_threshold"]) == 2


def test_assert_sv_validation_passed_raises_on_failed_summary(tmp_path: Path):
    summary_path = tmp_path / "summary.parquet"
    pd.DataFrame(
        [
            {
                "validation_passed": False,
                "failed_metrics": "subset_y_rhat,stability",
            }
        ]
    ).to_parquet(summary_path, index=False)

    with pytest.raises(RuntimeError, match="subset_y_rhat,stability"):
        assert_sv_validation_passed(summary_path)
