"""Validation helpers for the stochastic-volatility stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.sv_r_bridge import run_sv_r_script

FloatArray = NDArray[np.float64]
_TWO_DIMENSIONS = 2
_FULL_PANEL_PERCENTILE = 90
_STABILITY_SERIES_COUNT = 3
_MIN_SPLIT_RHAT_CHAINS = 2
_MIN_SPLIT_RHAT_HALVES = 2
_SUBSET_Y_REQUIRED_PASS_COUNT = 10
_PHI_SUPPORT_MIN = -0.999
_PHI_SUPPORT_MAX = 0.999


class ValidationJob(TypedDict):
    """Typed metadata for one validation-chain run."""

    series_type: str
    series_position: int
    series_key: str
    chain_id: int
    seed: int


class SubsetGateSummary(TypedDict):
    """Summary of the subset split-Rhat gate."""

    failed_metrics: list[str]
    subset_y_pass_count: int
    subset_y_sampled_count: int
    subset_y_pass_threshold: int
    subset_f_pass_all: bool
    subset_mu_pass_all: bool


class FullPanelGateSummary(TypedDict):
    """Summary of the full-panel admissibility and Geweke gate."""

    failed_metrics: list[str]
    params_are_finite: bool
    params_are_admissible: bool
    diagnostics_are_finite: bool
    median_phi: float
    median_sigma: float
    p90_phi: float
    p90_sigma: float
    factor_geweke_pass: bool


def select_evenly_spaced_positions(n_items: int, count: int) -> list[int]:
    """Select evenly spaced integer positions including the endpoints."""
    if n_items <= 0 or count <= 0:
        return []
    actual = min(n_items, count)
    if actual == 1:
        return [0]
    if actual == n_items:
        return list(range(n_items))
    return [round(index * (n_items - 1) / (actual - 1)) for index in range(actual)]


def select_validation_subset(
    series_order: pd.DataFrame,
    subset_size: int,
) -> pd.DataFrame:
    """Select a deterministic evenly spaced subset of y-series."""
    ordered = series_order.sort_values("series_position").reset_index(drop=True)
    positions = select_evenly_spaced_positions(len(ordered), subset_size)
    subset = ordered.iloc[positions][["series_position", "series_id"]].copy()
    subset["subset_rank"] = np.arange(len(subset))
    return subset.reset_index(drop=True)


def split_rhat(chain_draws: FloatArray) -> float:
    """Compute the split-Rhat diagnostic for retained MCMC draws."""
    draws = np.asarray(chain_draws, dtype=np.float64)
    if draws.ndim != _TWO_DIMENSIONS:
        msg = "chain_draws must be a two-dimensional array."
        raise ValueError(msg)

    n_chains, n_draws = draws.shape
    half = n_draws // _TWO_DIMENSIONS
    if n_chains < _MIN_SPLIT_RHAT_CHAINS or half < _MIN_SPLIT_RHAT_HALVES:
        return np.nan

    split = np.concatenate((draws[:, :half], draws[:, -half:]), axis=0)
    chain_means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean()
    between = half * chain_means.var(ddof=1)

    if not np.isfinite(within) or within < 0:
        return np.nan
    if within == 0:
        return 1.0 if between == 0 else np.inf

    variance_hat = ((half - 1) / half) * within + between / half
    return float(np.sqrt(max(variance_hat / within, 0.0)))


def build_validation_subset_metrics(
    forecast_errors_y_path: Path,
    forecast_errors_f_path: Path,
    series_order: pd.DataFrame,
    config: MEUConfig,
) -> pd.DataFrame:
    """Run multi-chain validation on sentinel y-series and all factor predictors."""
    subset = select_validation_subset(
        series_order=series_order,
        subset_size=config.sv_validation_subset_size,
    )
    predictor_names = pd.read_parquet(forecast_errors_f_path).columns.tolist()[1:]
    jobs = _build_validation_jobs(
        subset=subset,
        predictor_names=predictor_names,
        config=config,
    )
    draws = _run_validation_batch(
        forecast_errors_y_path=forecast_errors_y_path,
        forecast_errors_f_path=forecast_errors_f_path,
        jobs=jobs,
        config=config,
    )
    return _summarize_validation_draws(draws=draws, config=config)


def build_sv_validation_summary(
    subset_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    params_y: pd.DataFrame,
    params_f: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    config: MEUConfig,
) -> pd.DataFrame:
    """Combine subset and full-panel diagnostics into a one-row validation summary."""
    subset_summary = _evaluate_subset_gate(subset_metrics=subset_metrics, config=config)
    full_panel_summary = _evaluate_full_panel_gate(
        diagnostics=diagnostics,
        params_y=params_y,
        params_f=params_f,
        config=config,
    )

    failed_metrics = (
        subset_summary["failed_metrics"] + full_panel_summary["failed_metrics"]
    )
    stability_pass = bool(stability_metrics["passes_stability"].all())
    if not stability_pass:
        failed_metrics.append("stability")

    return pd.DataFrame(
        [
            {
                "validation_passed": not failed_metrics,
                "failed_metrics": ",".join(failed_metrics),
                "subset_y_pass_count": subset_summary["subset_y_pass_count"],
                "subset_y_sampled_count": subset_summary["subset_y_sampled_count"],
                "subset_y_pass_threshold": subset_summary["subset_y_pass_threshold"],
                "subset_f_pass_all": subset_summary["subset_f_pass_all"],
                "subset_mu_pass_all": subset_summary["subset_mu_pass_all"],
                "full_panel_params_are_finite": full_panel_summary["params_are_finite"],
                "full_panel_params_are_admissible": full_panel_summary[
                    "params_are_admissible"
                ],
                "full_panel_diagnostics_are_finite": full_panel_summary[
                    "diagnostics_are_finite"
                ],
                "median_abs_geweke_phi_y": full_panel_summary["median_phi"],
                "median_abs_geweke_sigma_y": full_panel_summary["median_sigma"],
                "p90_abs_geweke_phi_y": full_panel_summary["p90_phi"],
                "p90_abs_geweke_sigma_y": full_panel_summary["p90_sigma"],
                "factor_geweke_pass_all": full_panel_summary["factor_geweke_pass"],
                "stability_pass_all": stability_pass,
                "stability_max_phi_relative_diff": float(
                    stability_metrics["phi_relative_diff"].max()
                ),
                "stability_max_sigma_relative_diff": float(
                    stability_metrics["sigma_relative_diff"].max()
                ),
                "sv_validation_chains": config.sv_validation_chains,
                "sv_validation_subset_size": config.sv_validation_subset_size,
                "sv_rhat_threshold": config.sv_rhat_threshold,
                "sv_mu_rhat_threshold": config.sv_mu_rhat_threshold,
                "sv_geweke_threshold": config.sv_geweke_threshold,
                "sv_geweke_p90_threshold": config.sv_geweke_p90_threshold,
                "sv_stability_relative_tolerance": (
                    config.sv_stability_relative_tolerance
                ),
            }
        ]
    )


def build_stability_metrics(
    forecast_errors_y_path: Path,
    series_order: pd.DataFrame,
    config: MEUConfig,
) -> pd.DataFrame:
    """Compare fast and doubled-fast runs on representative y-series."""
    ordered = series_order.sort_values("series_position").reset_index(drop=True)
    positions = select_evenly_spaced_positions(len(ordered), _STABILITY_SERIES_COUNT)
    subset = ordered.iloc[positions][["series_position", "series_id"]].reset_index(
        drop=True
    )

    short_config = replace(config, sv_mode="fast")
    long_config = replace(
        short_config,
        sv_draws_fast=short_config.sv_draws_fast * _TWO_DIMENSIONS,
        sv_burnin_fast=short_config.sv_burnin_fast * _TWO_DIMENSIONS,
    )

    series = [
        {
            "series_position": int(record["series_position"]),
            "series_id": str(record["series_id"]),
            "seed": config.sv_seed + 200_000 + int(record["series_position"]),
        }
        for _, record in subset.iterrows()
    ]
    stability = _run_stability_batch(
        forecast_errors_y_path=forecast_errors_y_path,
        series=series,
        config=config,
        short_config=short_config,
        long_config=long_config,
    )
    return stability.assign(
        passes_stability=lambda frame: (
            frame["phi_relative_diff"] <= config.sv_stability_relative_tolerance
        )
        & (frame["sigma_relative_diff"] <= config.sv_stability_relative_tolerance)
    )


def assert_sv_validation_passed(summary_source: Path | pd.DataFrame) -> None:
    """Raise if the SV validation summary indicates a failed gate."""
    summary = (
        pd.read_parquet(summary_source)
        if isinstance(summary_source, Path)
        else summary_source.copy()
    )
    if summary.empty:
        msg = "SV validation summary is empty."
        raise ValueError(msg)
    if not bool(summary.loc[0, "validation_passed"]):
        failed_metrics = str(summary.loc[0, "failed_metrics"])
        msg = (
            "Stochastic-volatility validation failed. "
            f"Failed metrics: {failed_metrics}."
        )
        raise RuntimeError(msg)


def _build_validation_jobs(
    subset: pd.DataFrame,
    predictor_names: list[str],
    config: MEUConfig,
) -> list[ValidationJob]:
    """Create deterministic multi-chain jobs for validation."""
    jobs: list[ValidationJob] = []
    for _, record in subset.iterrows():
        series_position = int(record["series_position"])
        series_id = str(record["series_id"])
        for chain_id in range(config.sv_validation_chains):
            jobs.append(
                {
                    "series_type": "y",
                    "series_position": series_position,
                    "series_key": series_id,
                    "chain_id": chain_id,
                    "seed": config.sv_seed
                    + 100_000 * (chain_id + 1)
                    + series_position,
                }
            )

    for predictor_position, predictor_name in enumerate(predictor_names):
        for chain_id in range(config.sv_validation_chains):
            jobs.append(
                {
                    "series_type": "f",
                    "series_position": predictor_position,
                    "series_key": predictor_name,
                    "chain_id": chain_id,
                    "seed": config.sv_seed
                    + 1_000_000
                    + 100_000 * (chain_id + 1)
                    + predictor_position,
                }
            )
    return jobs


def _run_validation_batch(
    *,
    forecast_errors_y_path: Path,
    forecast_errors_f_path: Path,
    jobs: list[ValidationJob],
    config: MEUConfig,
) -> pd.DataFrame:
    with TemporaryDirectory(prefix="sv_validation_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        draws_path = tmp_path / "validation_draws.parquet"

        run_sv_r_script(
            depends_on={
                "forecast_errors_y": forecast_errors_y_path,
                "forecast_errors_f": forecast_errors_f_path,
            },
            produces={"draws": draws_path},
            options={
                "task_kind": "validation",
                "jobs": jobs,
                "sv_burnin": config.sv_burnin,
                "sv_draws": config.sv_draws,
                "sv_offset": config.sv_offset,
                "sv_thin_latent": config.sv_thin_latent,
                "sv_thin_para": config.sv_thin_para,
            },
        )
        return pd.read_parquet(draws_path)


def _run_stability_batch(
    *,
    forecast_errors_y_path: Path,
    series: list[dict[str, int | str]],
    config: MEUConfig,
    short_config: MEUConfig,
    long_config: MEUConfig,
) -> pd.DataFrame:
    with TemporaryDirectory(prefix="sv_stability_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        stability_path = tmp_path / "stability.parquet"

        run_sv_r_script(
            depends_on={"forecast_errors_y": forecast_errors_y_path},
            produces={"stability": stability_path},
            options={
                "task_kind": "stability",
                "series": series,
                "sv_offset": config.sv_offset,
                "short_sv_burnin": short_config.sv_burnin,
                "short_sv_draws": short_config.sv_draws,
                "short_sv_thin_latent": short_config.sv_thin_latent,
                "short_sv_thin_para": short_config.sv_thin_para,
                "long_sv_burnin": long_config.sv_burnin,
                "long_sv_draws": long_config.sv_draws,
                "long_sv_thin_latent": long_config.sv_thin_latent,
                "long_sv_thin_para": long_config.sv_thin_para,
            },
        )
        stability = pd.read_parquet(stability_path)

    return stability.astype(
        {
            "series_position": "int64",
            "series_id": "string",
            "phi_relative_diff": "float64",
            "sigma_relative_diff": "float64",
        }
    ).sort_values("series_position", ignore_index=True)


def _summarize_validation_draws(
    *,
    draws: pd.DataFrame,
    config: MEUConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    series_keys = (
        draws[["series_type", "series_position", "series_key"]]
        .drop_duplicates()
        .sort_values(["series_type", "series_position", "series_key"])
    )
    for record in series_keys.itertuples(index=False):
        series_type = str(record.series_type)
        series_position = int(record.series_position)
        series_key = str(record.series_key)
        frame = draws.loc[
            (draws["series_type"] == series_type)
            & (draws["series_position"] == series_position)
            & (draws["series_key"] == series_key)
        ]
        ordered_chains = []
        for chain_id in sorted(frame["chain_id"].unique().tolist()):
            ordered = frame.loc[frame["chain_id"] == chain_id].sort_values("draw_index")
            ordered_chains.append(
                (
                    chain_id,
                    ordered["mu_draw"].to_numpy(dtype=np.float64),
                    ordered["phi_draw"].to_numpy(dtype=np.float64),
                    ordered["sigma_draw"].to_numpy(dtype=np.float64),
                )
            )

        mu_draws = np.vstack([item[1] for item in ordered_chains])
        phi_draws = np.vstack([item[2] for item in ordered_chains])
        sigma_draws = np.vstack([item[3] for item in ordered_chains])
        rhat_mu = split_rhat(mu_draws)
        rhat_phi = split_rhat(phi_draws)
        rhat_sigma = split_rhat(sigma_draws)
        rows.append(
            {
                "series_type": str(series_type),
                "series_position": int(series_position),
                "series_key": str(series_key),
                "n_chains": mu_draws.shape[0],
                "n_draws_per_chain": mu_draws.shape[1],
                "rhat_mu": rhat_mu,
                "rhat_phi": rhat_phi,
                "rhat_sigma": rhat_sigma,
                "passes_mu_rhat": bool(rhat_mu <= config.sv_mu_rhat_threshold),
                "passes_phi_sigma_rhat": bool(
                    rhat_phi <= config.sv_rhat_threshold
                    and rhat_sigma <= config.sv_rhat_threshold
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["series_type", "series_position", "series_key"])
        .reset_index(drop=True)
    )


def _evaluate_subset_gate(
    subset_metrics: pd.DataFrame,
    config: MEUConfig,
) -> SubsetGateSummary:
    """Evaluate the split-Rhat gate on the validation subset."""
    subset_y = subset_metrics.loc[subset_metrics["series_type"] == "y"]
    subset_f = subset_metrics.loc[subset_metrics["series_type"] == "f"]
    subset_y_pass_count = int(subset_y["passes_phi_sigma_rhat"].sum())
    subset_y_required = min(config.sv_validation_subset_size, len(subset_y))
    subset_y_threshold = min(_SUBSET_Y_REQUIRED_PASS_COUNT, subset_y_required)
    subset_y_pass = subset_y_pass_count >= subset_y_threshold
    subset_f_pass = bool(subset_f["passes_phi_sigma_rhat"].all())
    subset_mu_pass = bool(subset_metrics["passes_mu_rhat"].all())

    failed_metrics: list[str] = []
    if not subset_y_pass:
        failed_metrics.append("subset_y_rhat")
    if not subset_f_pass:
        failed_metrics.append("subset_factor_rhat")
    if not subset_mu_pass:
        failed_metrics.append("subset_mu_rhat")

    return {
        "failed_metrics": failed_metrics,
        "subset_y_pass_count": subset_y_pass_count,
        "subset_y_sampled_count": subset_y_required,
        "subset_y_pass_threshold": subset_y_threshold,
        "subset_f_pass_all": subset_f_pass,
        "subset_mu_pass_all": subset_mu_pass,
    }


def _evaluate_full_panel_gate(
    diagnostics: pd.DataFrame,
    params_y: pd.DataFrame,
    params_f: pd.DataFrame,
    config: MEUConfig,
) -> FullPanelGateSummary:
    """Evaluate admissibility and Geweke gates on the full-panel SV outputs."""
    params_all = pd.concat(
        [
            params_y.assign(series_type="y", series_key=params_y["series_id"]),
            params_f.assign(series_type="f", series_key=params_f["predictor_name"]),
        ],
        ignore_index=True,
    )
    params_are_finite = bool(
        np.isfinite(params_all[["mu", "phi", "sigma"]]).all().all()
    )
    params_are_admissible = bool(
        (
            (params_all["phi"] > _PHI_SUPPORT_MIN)
            & (params_all["phi"] < _PHI_SUPPORT_MAX)
        ).all()
        and (params_all["sigma"] > 0.0).all()
    )

    diagnostics_y = diagnostics.loc[diagnostics["series_type"] == "y"].copy()
    diagnostics_f = diagnostics.loc[diagnostics["series_type"] == "f"].copy()
    diagnostics_are_finite = bool(
        np.isfinite(diagnostics[["geweke_mu", "geweke_phi", "geweke_sigma"]])
        .all()
        .all()
    )
    y_abs_phi = diagnostics_y["geweke_phi"].abs()
    y_abs_sigma = diagnostics_y["geweke_sigma"].abs()
    median_phi = float(np.nanmedian(y_abs_phi))
    median_sigma = float(np.nanmedian(y_abs_sigma))
    p90_phi = float(np.nanpercentile(y_abs_phi, _FULL_PANEL_PERCENTILE))
    p90_sigma = float(np.nanpercentile(y_abs_sigma, _FULL_PANEL_PERCENTILE))
    factor_geweke_pass = bool(
        (diagnostics_f["geweke_phi"].abs() <= config.sv_geweke_threshold).all()
        and (diagnostics_f["geweke_sigma"].abs() <= config.sv_geweke_threshold).all()
    )

    failed_metrics: list[str] = []
    if not params_are_finite:
        failed_metrics.append("full_panel_params_nonfinite")
    if not params_are_admissible:
        failed_metrics.append("full_panel_params_support")
    if not diagnostics_are_finite:
        failed_metrics.append("full_panel_geweke_nonfinite")
    if (
        median_phi > config.sv_geweke_threshold
        or median_sigma > config.sv_geweke_threshold
    ):
        failed_metrics.append("full_panel_geweke_median")
    if (
        p90_phi > config.sv_geweke_p90_threshold
        or p90_sigma > config.sv_geweke_p90_threshold
    ):
        failed_metrics.append("full_panel_geweke_p90")
    if not factor_geweke_pass:
        failed_metrics.append("factor_geweke")

    return {
        "failed_metrics": failed_metrics,
        "params_are_finite": params_are_finite,
        "params_are_admissible": params_are_admissible,
        "diagnostics_are_finite": diagnostics_are_finite,
        "median_phi": median_phi,
        "median_sigma": median_sigma,
        "p90_phi": p90_phi,
        "p90_sigma": p90_sigma,
        "factor_geweke_pass": factor_geweke_pass,
    }
