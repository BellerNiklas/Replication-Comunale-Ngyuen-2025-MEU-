from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.sv_r_bridge import run_sv_r_script
from meu_replication.config import ROOT

EXPECTED_PARITY_RTOL = 1e-8
EXPECTED_PARITY_ATOL = 1e-8
FULL_CONFIG = MEUConfig(sv_mode="full")


def _simulate_residuals(seed: int, n_obs: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.empty(n_obs)
    h[0] = -0.4
    eps = rng.normal(size=n_obs)
    for idx in range(1, n_obs):
        h[idx] = -0.2 + 0.95 * (h[idx - 1] + 0.2) + 0.15 * rng.normal()
    return np.exp(h / 2.0) * eps


def _run_reference_script(
    *,
    script_path: Path,
    input_name: str,
    output_name: str,
    matrix: np.ndarray,
    workdir: Path,
) -> np.ndarray:
    compatible_script = workdir / script_path.name
    compatible_text = script_path.read_text(encoding="utf-8")
    compatible_text = compatible_text.replace(
        "library(stochvol)",
        "library(stochvol)\nlibrary(coda)",
    ).replace(
        "all    = cbind(draws$para,draws$latent)",
        (
            "all    = cbind("
            "as.matrix(draws$para)[,c('mu','phi','sigma'),drop=FALSE],"
            "as.matrix(draws$latent)"
            ")"
        ),
    ).replace(
        "g[,i]  = geweke.diag(draws$para)$z",
        (
            "g[,i]  = tryCatch("
            "as.numeric(geweke.diag(draws$para)$z),"
            "error = function(err) tryCatch("
            "as.numeric(geweke.diag(coda::mcmc(as.matrix(draws$para)[,c('mu','phi','sigma'),drop=FALSE]))$z),"
            "error = function(inner_err) rep(NaN, 3)"
            ")"
            ")"
        ),
    )
    compatible_script.write_text(compatible_text, encoding="utf-8")
    np.savetxt(workdir / input_name, matrix, delimiter="\t", fmt="%.17g")
    result = subprocess.run(
        ["Rscript", "--vanilla", str(compatible_script)],
        capture_output=True,
        check=False,
        cwd=workdir,
        text=True,
    )
    if result.returncode != 0:
        msg = (
            "Reference SV script failed.\n"
            f"stdout:\n{result.stdout.strip() or '<empty>'}\n"
            f"stderr:\n{result.stderr.strip() or '<empty>'}"
        )
        raise RuntimeError(msg)
    return np.loadtxt(workdir / output_name)


def _run_repo_panel(
    *,
    forecast_errors_path: Path,
    output_dir: Path,
    series_type: str,
    seed: int,
    series_order_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    depends_on = {"forecast_errors": forecast_errors_path}
    if series_order_path is not None:
        depends_on["series_order"] = series_order_path

    produces = {
        "params": output_dir / f"sv_params_{series_type}_core.parquet",
        "latent": output_dir / f"sv_latent_{series_type}_core.parquet",
        "diagnostics": output_dir / f"sv_diagnostics_{series_type}_core.parquet",
    }
    run_sv_r_script(
        depends_on=depends_on,
        produces=produces,
        options={
            "task_kind": "panel",
            "seed": seed,
            "series_type": series_type,
            "sv_burnin": FULL_CONFIG.sv_burnin,
            "sv_draws": FULL_CONFIG.sv_draws,
            "sv_offset": FULL_CONFIG.sv_offset,
            "sv_thin_latent": FULL_CONFIG.sv_thin_latent,
            "sv_thin_para": FULL_CONFIG.sv_thin_para,
        },
    )
    return (
        pd.read_parquet(produces["params"]),
        pd.read_parquet(produces["latent"]),
        pd.read_parquet(produces["diagnostics"]),
    )


def _repo_output_matrix(
    params: pd.DataFrame,
    latent: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> np.ndarray:
    ordered_params = params.sort_values("series_position").reset_index(drop=True)
    rows: list[np.ndarray] = []
    for record in ordered_params.itertuples(index=False):
        series_key = str(record.series_key)
        diagnostics_row = diagnostics.loc[
            (diagnostics["series_position"] == int(record.series_position))
            & (diagnostics["series_key"] == series_key)
        ].iloc[0]
        rows.append(
            np.concatenate(
                [
                    np.array([record.mu, record.phi, record.sigma], dtype=np.float64),
                    latent[series_key].to_numpy(dtype=np.float64),
                    np.array(
                        [
                            diagnostics_row["geweke_mu"],
                            diagnostics_row["geweke_phi"],
                            diagnostics_row["geweke_sigma"],
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        )
    return np.column_stack(rows)


def test_estimate_sv_panel_matches_reference_y_script(tmp_path: Path):
    dates = pd.period_range("2020-01", periods=12, freq="M").astype(str)
    y_order = ["a", "b"]
    y_matrix = np.column_stack([_simulate_residuals(1), _simulate_residuals(2)])

    forecast_errors_y = pd.DataFrame(
        {
            "date": dates,
            "b": y_matrix[:, 1],
            "a": y_matrix[:, 0],
        }
    )
    forecast_errors_y_path = tmp_path / "forecast_errors_y.parquet"
    forecast_errors_y.to_parquet(forecast_errors_y_path, index=False)

    series_order_path = tmp_path / "series_order.parquet"
    pd.DataFrame(
        {"series_position": [0, 1], "series_id": y_order}
    ).to_parquet(series_order_path, index=False)

    reference_output = _run_reference_script(
        script_path=ROOT / "references" / "jln_macro_code" / "generate_svydraws.r",
        input_name="vyt2014.txt",
        output_name="svymeans2014.txt",
        matrix=y_matrix,
        workdir=tmp_path,
    )
    params, latent, diagnostics = _run_repo_panel(
        forecast_errors_path=forecast_errors_y_path,
        output_dir=tmp_path / "repo_y",
        series_type="y",
        seed=FULL_CONFIG.sv_seed_y,
        series_order_path=series_order_path,
    )

    assert params.sort_values("series_position")["series_key"].tolist() == y_order
    np.testing.assert_allclose(
        _repo_output_matrix(params, latent, diagnostics),
        reference_output,
        rtol=EXPECTED_PARITY_RTOL,
        atol=EXPECTED_PARITY_ATOL,
    )


def test_estimate_sv_panel_matches_reference_f_script(tmp_path: Path):
    dates = pd.period_range("2020-01", periods=12, freq="M").astype(str)
    factor_order = ["factor_2", "factor_1"]
    factor_matrix = np.column_stack([_simulate_residuals(11), _simulate_residuals(12)])

    forecast_errors_f = pd.DataFrame(
        {
            "date": dates,
            "factor_2": factor_matrix[:, 0],
            "factor_1": factor_matrix[:, 1],
        }
    )
    forecast_errors_f_path = tmp_path / "forecast_errors_f.parquet"
    forecast_errors_f.to_parquet(forecast_errors_f_path, index=False)

    reference_output = _run_reference_script(
        script_path=ROOT / "references" / "jln_macro_code" / "generate_svfdraws.r",
        input_name="vft2014.txt",
        output_name="svfmeans2014.txt",
        matrix=factor_matrix,
        workdir=tmp_path,
    )
    params, latent, diagnostics = _run_repo_panel(
        forecast_errors_path=forecast_errors_f_path,
        output_dir=tmp_path / "repo_f",
        series_type="f",
        seed=FULL_CONFIG.sv_seed_f,
    )

    assert params.sort_values("series_position")["series_key"].tolist() == factor_order
    np.testing.assert_allclose(
        _repo_output_matrix(params, latent, diagnostics),
        reference_output,
        rtol=EXPECTED_PARITY_RTOL,
        atol=EXPECTED_PARITY_ATOL,
    )
