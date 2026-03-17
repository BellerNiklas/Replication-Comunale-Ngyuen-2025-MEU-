"""Stage-4 uncertainty computation for the MEU pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from meu_replication.analysis.forecast_errors import build_y_regressor_names

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UncertaintyResult:
    """Dense uncertainty outputs before long-format serialization."""

    dates: tuple[str, ...]
    series_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    variances: FloatArray


def expected_variance(
    alpha: float | FloatArray,
    beta: float | FloatArray,
    tau2: float | FloatArray,
    x_t: float | FloatArray,
    h: int,
) -> FloatArray:
    """Compute E_t[exp{x_(t+h)}] for the JLN AR(1) log-volatility law."""
    if h < 1:
        msg = "h must be at least 1."
        raise ValueError(msg)

    alpha_arr = np.asarray(alpha, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    tau2_arr = np.asarray(tau2, dtype=np.float64)
    x_arr = np.asarray(x_t, dtype=np.float64)

    beta_powers = np.power(beta_arr[..., None], np.arange(h, dtype=np.float64))
    beta_sq_powers = np.power(
        np.square(beta_arr)[..., None],
        np.arange(h, dtype=np.float64),
    )
    beta_h = np.power(beta_arr, h)

    return np.exp(
        alpha_arr * beta_powers.sum(axis=-1)
        + 0.5 * tau2_arr * beta_sq_powers.sum(axis=-1)
        + beta_h * x_arr,
    )


def sv_params_to_jln(
    mu: float | FloatArray,
    phi: float | FloatArray,
    sigma: float | FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert stochvol posterior means to the JLN AR(1) parameterization."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    return (
        np.asarray(mu_arr * (1.0 - phi_arr), dtype=np.float64),
        phi_arr,
        np.asarray(np.square(sigma_arr), dtype=np.float64),
    )


def build_predictor_companion_matrix(
    regression_coefs_f: pd.DataFrame,
    *,
    pf: int,
) -> FloatArray:
    """Build the shared predictor companion matrix used in `compute_uf.m`."""
    if pf < 1:
        msg = "pf must be at least 1."
        raise ValueError(msg)

    predictor_frame = regression_coefs_f.sort_values("predictor_position").reset_index(
        drop=True,
    )
    r = len(predictor_frame)
    lag_columns = [f"lag{lag}" for lag in range(1, pf + 1)]
    missing = [column for column in lag_columns if column not in predictor_frame.columns]
    if missing:
        msg = f"Predictor regression coefficients are missing lag columns: {missing}"
        raise ValueError(msg)

    top = np.hstack(
        [
            np.diag(predictor_frame[column].to_numpy(dtype=np.float64))
            for column in lag_columns
        ],
    )
    if pf == 1:
        return np.asarray(top, dtype=np.float64)

    bottom = np.hstack(
        (
            np.eye(r * (pf - 1), dtype=np.float64),
            np.zeros((r * (pf - 1), r), dtype=np.float64),
        ),
    )
    return np.vstack((top, bottom))


def build_series_transition_blocks(
    y_coefficients: pd.Series,
    *,
    predictor_names: Sequence[str],
    py: int,
    pz: int,
    pf: int,
) -> tuple[FloatArray, FloatArray]:
    """Build the series-specific `lambda` and `phiy` blocks from `compute_uy.m`."""
    if py < 1 or pz < 1 or pf < 1:
        msg = "py, pz, and pf must all be at least 1."
        raise ValueError(msg)
    if pz > pf:
        msg = "pz cannot exceed pf when building the combined companion matrix."
        raise ValueError(msg)

    y_lag_columns = [f"y_lag{lag}" for lag in range(1, py + 1)]
    predictor_columns = list(build_y_regressor_names(predictor_names, py=py, pz=pz))[
        1 + py :
    ]
    missing = [
        column
        for column in [*y_lag_columns, *predictor_columns]
        if column not in y_coefficients.index
    ]
    if missing:
        msg = f"Series coefficients are missing required columns: {missing}"
        raise ValueError(msg)

    r = len(predictor_names)
    lambda_matrix = np.zeros((py, r * pf), dtype=np.float64)
    lambda_matrix[0, : r * pz] = y_coefficients[predictor_columns].to_numpy(
        dtype=np.float64,
    )

    phiy_top = y_coefficients[y_lag_columns].to_numpy(dtype=np.float64).reshape(1, py)
    if py == 1:
        phiy = phiy_top
    else:
        phiy_bottom = np.hstack(
            (
                np.eye(py - 1, dtype=np.float64),
                np.zeros((py - 1, 1), dtype=np.float64),
            ),
        )
        phiy = np.vstack((phiy_top, phiy_bottom))
    return lambda_matrix, phiy


def build_combined_companion_matrix(
    *,
    phif: FloatArray,
    lambda_matrix: FloatArray,
    phiy: FloatArray,
) -> FloatArray:
    """Combine the shared predictor block with series-specific y dynamics."""
    predictor_state_dim = phif.shape[0]
    py = phiy.shape[0]
    if phif.shape[0] != phif.shape[1]:
        msg = "phif must be square."
        raise ValueError(msg)
    if lambda_matrix.shape != (py, predictor_state_dim):
        msg = "lambda_matrix must have shape (py, predictor_state_dim)."
        raise ValueError(msg)
    if phiy.shape[0] != phiy.shape[1]:
        msg = "phiy must be square."
        raise ValueError(msg)

    top_right = np.zeros((predictor_state_dim, py), dtype=np.float64)
    top = np.hstack((phif, top_right))
    bottom = np.hstack((lambda_matrix, phiy))
    return np.vstack((top, bottom))


def compute_irf_squares(
    phi: FloatArray,
    *,
    y_position: int,
    shock_positions: Sequence[int],
    h_max: int,
) -> FloatArray:
    """Compute squared IRFs from the direct shock entries to the y state."""
    if h_max < 1:
        msg = "h_max must be at least 1."
        raise ValueError(msg)

    shock_index = np.asarray(shock_positions, dtype=np.int64)
    if phi.shape[0] != phi.shape[1]:
        msg = "phi must be square."
        raise ValueError(msg)
    if y_position < 0 or y_position >= phi.shape[0]:
        msg = "y_position must index a valid state."
        raise ValueError(msg)
    if np.any(shock_index < 0) or np.any(shock_index >= phi.shape[0]):
        msg = "shock_positions must index valid states."
        raise ValueError(msg)

    row = np.zeros(phi.shape[0], dtype=np.float64)
    row[y_position] = 1.0

    irf_squares = np.empty((h_max, len(shock_index)), dtype=np.float64)
    for horizon_idx in range(h_max):
        irf_squares[horizon_idx, :] = np.square(row[shock_index])
        row = row @ phi
    return irf_squares


def compute_scalar_uncertainty_from_irf(
    *,
    irf_squares: FloatArray,
    shock_variance_paths: Sequence[FloatArray],
) -> FloatArray:
    """Combine squared IRFs with horizon-specific shock variances."""
    h_max = len(shock_variance_paths)
    if h_max == 0:
        msg = "shock_variance_paths cannot be empty."
        raise ValueError(msg)
    if irf_squares.shape[0] < h_max:
        msg = "irf_squares must cover all requested horizons."
        raise ValueError(msg)

    first = np.asarray(shock_variance_paths[0], dtype=np.float64)
    if first.ndim != 2:
        msg = "Each shock-variance path must be a two-dimensional array."
        raise ValueError(msg)

    n_obs, n_shocks = first.shape
    path_arrays = [np.asarray(path, dtype=np.float64) for path in shock_variance_paths]
    output = np.zeros((n_obs, h_max), dtype=np.float64)
    for horizon_idx, path_arr in enumerate(path_arrays):
        if path_arr.shape != (n_obs, n_shocks):
            msg = "All shock-variance paths must share the same shape."
            raise ValueError(msg)
        total = np.zeros(n_obs, dtype=np.float64)
        for step in range(horizon_idx + 1):
            total += path_arrays[horizon_idx - step] @ irf_squares[step, :]
        output[:, horizon_idx] = total
    return output


def _companion_recursion_scalar_uncertainty(
    *,
    phi: FloatArray,
    y_position: int,
    shock_positions: Sequence[int],
    shock_variance_paths: Sequence[FloatArray],
) -> FloatArray:
    """Literal companion recursion used for parity tests against the IRF engine."""
    if phi.shape[0] != phi.shape[1]:
        msg = "phi must be square."
        raise ValueError(msg)

    shock_index = np.asarray(shock_positions, dtype=np.int64)
    first = np.asarray(shock_variance_paths[0], dtype=np.float64)
    n_obs, n_shocks = first.shape
    if len(shock_index) != n_shocks:
        msg = "shock_positions must align with the shock-variance path width."
        raise ValueError(msg)

    path_arrays = [np.asarray(path, dtype=np.float64) for path in shock_variance_paths]
    output = np.zeros((n_obs, len(shock_variance_paths)), dtype=np.float64)
    for obs_idx in range(n_obs):
        u = np.zeros_like(phi)
        for horizon_idx, path_arr in enumerate(path_arrays):
            diagonal = np.zeros(phi.shape[0], dtype=np.float64)
            diagonal[shock_index] = path_arr[obs_idx, :]
            ev = np.diag(diagonal)
            u = ev if horizon_idx == 0 else phi @ u @ phi.T + ev
            output[obs_idx, horizon_idx] = u[y_position, y_position]
    return output


def compute_uncertainty(
    *,
    forecast_metadata: pd.DataFrame,
    series_order: pd.DataFrame,
    regression_coefs_y: pd.DataFrame,
    regression_coefs_f: pd.DataFrame,
    sv_params_y: pd.DataFrame,
    sv_latent_y: pd.DataFrame,
    sv_params_f: pd.DataFrame,
    sv_latent_f: pd.DataFrame,
    h_max: int,
) -> UncertaintyResult:
    """Compute horizon-specific uncertainty from forecast and SV artifacts."""
    metadata_row = _validate_metadata(forecast_metadata)
    py = int(metadata_row["py"])
    pz = int(metadata_row["pz"])
    pf = int(metadata_row["pf"])

    ordered_series = _ordered_frame(
        series_order,
        position_col="series_position",
        id_col="series_id",
        label="series_order",
    )
    ordered_y_coefs = _ordered_frame(
        regression_coefs_y,
        position_col="series_position",
        id_col="series_id",
        label="regression_coefs_y",
    )
    ordered_y_params = _ordered_frame(
        sv_params_y,
        position_col="series_position",
        id_col="series_id",
        label="sv_params_y",
    )
    ordered_f_coefs = _ordered_frame(
        regression_coefs_f,
        position_col="predictor_position",
        id_col="predictor_name",
        label="regression_coefs_f",
    )
    ordered_f_params = _ordered_frame(
        sv_params_f,
        position_col="predictor_position",
        id_col="predictor_name",
        label="sv_params_f",
    )

    series_ids = tuple(ordered_series["series_id"].astype(str).tolist())
    predictor_names = tuple(ordered_f_coefs["predictor_name"].astype(str).tolist())
    _validate_order_match(
        expected_ids=series_ids,
        candidate=ordered_y_coefs,
        id_col="series_id",
        label="regression_coefs_y",
    )
    _validate_order_match(
        expected_ids=series_ids,
        candidate=ordered_y_params,
        id_col="series_id",
        label="sv_params_y",
    )
    _validate_order_match(
        expected_ids=predictor_names,
        candidate=ordered_f_params,
        id_col="predictor_name",
        label="sv_params_f",
    )

    dates = _validate_dates(
        forecast_metadata=forecast_metadata,
        sv_latent_y=sv_latent_y,
        sv_latent_f=sv_latent_f,
        series_ids=series_ids,
        predictor_names=predictor_names,
    )

    phif = build_predictor_companion_matrix(ordered_f_coefs, pf=pf)
    predictor_state_dim = phif.shape[0]
    y_position = predictor_state_dim
    shock_positions = tuple([*range(len(predictor_names)), y_position])

    alpha_f, beta_f, tau2_f = sv_params_to_jln(
        ordered_f_params["mu"].to_numpy(dtype=np.float64),
        ordered_f_params["phi"].to_numpy(dtype=np.float64),
        ordered_f_params["sigma"].to_numpy(dtype=np.float64),
    )
    xf = sv_latent_f.loc[:, predictor_names].to_numpy(dtype=np.float64)
    predictor_paths = tuple(
        expected_variance(alpha_f, beta_f, tau2_f, xf, h=horizon).astype(np.float64)
        for horizon in range(1, h_max + 1)
    )

    variances = np.empty((len(dates), len(series_ids), h_max), dtype=np.float64)
    for series_idx, series_id in enumerate(series_ids):
        lambda_matrix, phiy = build_series_transition_blocks(
            ordered_y_coefs.iloc[series_idx],
            predictor_names=predictor_names,
            py=py,
            pz=pz,
            pf=pf,
        )
        phi = build_combined_companion_matrix(
            phif=phif,
            lambda_matrix=lambda_matrix,
            phiy=phiy,
        )
        irf_squares = compute_irf_squares(
            phi,
            y_position=y_position,
            shock_positions=shock_positions,
            h_max=h_max,
        )

        alpha_y, beta_y, tau2_y = sv_params_to_jln(
            ordered_y_params.loc[series_idx, "mu"],
            ordered_y_params.loc[series_idx, "phi"],
            ordered_y_params.loc[series_idx, "sigma"],
        )
        xy = sv_latent_y.loc[:, series_id].to_numpy(dtype=np.float64)
        y_paths = tuple(
            expected_variance(alpha_y, beta_y, tau2_y, xy, h=horizon).astype(
                np.float64,
            )
            for horizon in range(1, h_max + 1)
        )
        shock_variance_paths = tuple(
            np.column_stack((predictor_paths[horizon_idx], y_paths[horizon_idx]))
            for horizon_idx in range(h_max)
        )
        variances[:, series_idx, :] = compute_scalar_uncertainty_from_irf(
            irf_squares=irf_squares,
            shock_variance_paths=shock_variance_paths,
        )

    if not np.isfinite(variances).all():
        msg = "Computed uncertainty contains non-finite values."
        raise ValueError(msg)
    if np.any(variances <= 0.0):
        msg = "Computed uncertainty contains non-positive variances."
        raise ValueError(msg)

    return UncertaintyResult(
        dates=dates,
        series_ids=series_ids,
        horizons=tuple(range(1, h_max + 1)),
        variances=variances,
    )


def uncertainty_result_to_long_frame(result: UncertaintyResult) -> pd.DataFrame:
    """Serialize dense uncertainty arrays to the deterministic public contract."""
    n_dates = len(result.dates)
    n_series = len(result.series_ids)
    n_horizons = len(result.horizons)
    if result.variances.shape != (n_dates, n_series, n_horizons):
        msg = "Result variances must align with the provided dates, series, and horizons."
        raise ValueError(msg)

    date_codes = np.repeat(
        np.arange(n_dates, dtype=np.int32),
        n_horizons * n_series,
    )
    horizon_values = np.tile(
        np.repeat(np.asarray(result.horizons, dtype=np.int16), n_series),
        n_dates,
    )
    series_codes = np.tile(np.arange(n_series, dtype=np.int32), n_dates * n_horizons)

    frame = pd.DataFrame(
        {
            "date": pd.Categorical.from_codes(
                date_codes,
                categories=list(result.dates),
                ordered=True,
            ),
            "series_id": pd.Categorical.from_codes(
                series_codes,
                categories=list(result.series_ids),
                ordered=True,
            ),
            "horizon": horizon_values,
            "variance": result.variances.transpose(0, 2, 1).reshape(-1),
        },
    )
    return frame


def _ordered_frame(
    frame: pd.DataFrame,
    *,
    position_col: str,
    id_col: str,
    label: str,
) -> pd.DataFrame:
    if frame.empty:
        msg = f"{label} cannot be empty."
        raise ValueError(msg)
    if frame[position_col].duplicated().any():
        msg = f"{label} contains duplicated positions."
        raise ValueError(msg)
    if frame[id_col].astype(str).duplicated().any():
        msg = f"{label} contains duplicated identifiers."
        raise ValueError(msg)

    ordered = frame.sort_values(position_col).reset_index(drop=True).copy()
    expected_positions = np.arange(len(ordered), dtype=np.int64)
    positions = ordered[position_col].to_numpy(dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        msg = f"{label} positions must be consecutive and zero-based."
        raise ValueError(msg)

    ordered[id_col] = ordered[id_col].astype(str)
    return ordered


def _validate_order_match(
    *,
    expected_ids: Sequence[str],
    candidate: pd.DataFrame,
    id_col: str,
    label: str,
) -> None:
    candidate_ids = tuple(candidate[id_col].astype(str).tolist())
    if candidate_ids != tuple(expected_ids):
        msg = f"{label} does not align with the persisted identifier order."
        raise ValueError(msg)


def _validate_metadata(forecast_metadata: pd.DataFrame) -> pd.Series:
    if len(forecast_metadata) != 1:
        msg = "forecast_metadata must contain exactly one row."
        raise ValueError(msg)
    return forecast_metadata.iloc[0]


def _validate_dates(
    *,
    forecast_metadata: pd.DataFrame,
    sv_latent_y: pd.DataFrame,
    sv_latent_f: pd.DataFrame,
    series_ids: Sequence[str],
    predictor_names: Sequence[str],
) -> tuple[str, ...]:
    y_columns = ["date", *series_ids]
    f_columns = ["date", *predictor_names]
    if list(sv_latent_y.columns) != y_columns:
        msg = "sv_latent_y columns must be date followed by the persisted series order."
        raise ValueError(msg)
    if list(sv_latent_f.columns) != f_columns:
        msg = "sv_latent_f columns must be date followed by the predictor order."
        raise ValueError(msg)

    y_dates = tuple(sv_latent_y["date"].astype(str).tolist())
    f_dates = tuple(sv_latent_f["date"].astype(str).tolist())
    if y_dates != f_dates:
        msg = "sv_latent_y and sv_latent_f must share the same date index."
        raise ValueError(msg)

    metadata_row = forecast_metadata.iloc[0]
    if y_dates[0] != str(metadata_row["y_sample_start"]):
        msg = "sv_latent_y start date does not match forecast_metadata."
        raise ValueError(msg)
    if y_dates[-1] != str(metadata_row["y_sample_end"]):
        msg = "sv_latent_y end date does not match forecast_metadata."
        raise ValueError(msg)
    if y_dates[0] != str(metadata_row["f_sample_start"]):
        msg = "sv_latent_f start date does not match forecast_metadata."
        raise ValueError(msg)
    if y_dates[-1] != str(metadata_row["f_sample_end"]):
        msg = "sv_latent_f end date does not match forecast_metadata."
        raise ValueError(msg)
    if len(y_dates) != int(metadata_row["n_obs_forecast_errors_y"]):
        msg = "sv_latent_y length does not match forecast_metadata."
        raise ValueError(msg)
    if len(f_dates) != int(metadata_row["n_obs_forecast_errors_f"]):
        msg = "sv_latent_f length does not match forecast_metadata."
        raise ValueError(msg)

    return y_dates
