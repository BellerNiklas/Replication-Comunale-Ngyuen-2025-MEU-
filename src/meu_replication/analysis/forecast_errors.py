"""Forecast-error generation for the Milestone 2 MEU stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from meu_replication.analysis.newey_west import (
    compute_newey_west_bandwidth,
    newey_west,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_TWO_DIMENSIONS = 2


@dataclass(frozen=True)
class ForecastErrorResult:
    """Outputs produced by the forecast-error stage."""

    y_residuals: FloatArray
    f_residuals: FloatArray
    y_coefficients: FloatArray
    f_coefficients: FloatArray
    selection_mask: BoolArray
    y_regressor_names: tuple[str, ...]
    factor_regressor_names: tuple[str, ...]
    selection_regressor_names: tuple[str, ...]
    y_dates: tuple[str, ...]
    f_dates: tuple[str, ...]
    bandwidth_y: int
    bandwidth_f: int


def mlags(
    x: FloatArray,
    k: int = 1,
    p: int | None = None,
    fill_value: float = 0.0,
) -> FloatArray:
    """Match the MATLAB lag-matrix helper used by JLN."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != _TWO_DIMENSIONS:
        msg = "x must be one- or two-dimensional."
        raise ValueError(msg)
    if k < 1:
        msg = "k must be at least 1."
        raise ValueError(msg)

    keep_lags = k if p is None else p
    if keep_lags < 1 or keep_lags > k:
        msg = "p must be between 1 and k."
        raise ValueError(msg)

    n_obs, n_var = values.shape
    z = np.full((n_obs, n_var * k), fill_value, dtype=np.float64)

    for lag in range(1, k + 1):
        z[lag:n_obs, n_var * (lag - 1) : lag * n_var] = values[: n_obs - lag, :]

    return z[:, -n_var * keep_lags :]


def build_y_regressor_names(
    predictor_names: Sequence[str],
    py: int,
    pz: int,
) -> tuple[str, ...]:
    """Construct coefficient names for the y-equation layout."""
    names = ["const", *[f"y_lag{lag}" for lag in range(1, py + 1)]]
    for lag in range(1, pz + 1):
        names.extend(f"{predictor_name}_lag{lag}" for predictor_name in predictor_names)
    return tuple(names)


def build_factor_regressor_names(pf: int) -> tuple[str, ...]:
    """Construct coefficient names for the factor AR equations."""
    return ("const", *[f"lag{lag}" for lag in range(1, pf + 1)])


def generate_forecast_errors(
    standardized_y: FloatArray,
    predictor_values: FloatArray,
    dates: Sequence[str],
    predictor_names: Sequence[str],
    py: int,
    pz: int,
    pf: int,
    threshold: float,
) -> ForecastErrorResult:
    """Estimate the JLN-style forecast equations and return their residuals."""
    y_values = np.asarray(standardized_y, dtype=np.float64)
    f_values = np.asarray(predictor_values, dtype=np.float64)
    date_index = tuple(str(date) for date in dates)
    predictor_name_tuple = tuple(str(name) for name in predictor_names)

    if y_values.ndim != _TWO_DIMENSIONS or f_values.ndim != _TWO_DIMENSIONS:
        msg = "standardized_y and predictor_values must be two-dimensional."
        raise ValueError(msg)
    if y_values.shape[0] != f_values.shape[0]:
        msg = "standardized_y and predictor_values must have the same number of rows."
        raise ValueError(msg)
    if y_values.shape[0] != len(date_index):
        msg = "dates must match the number of observations."
        raise ValueError(msg)
    if f_values.shape[1] != len(predictor_name_tuple):
        msg = "predictor_names must match the predictor matrix width."
        raise ValueError(msg)

    n_obs, n_series = y_values.shape
    n_predictors = f_values.shape[1]
    y_regressor_names = build_y_regressor_names(predictor_name_tuple, py=py, pz=pz)
    selection_regressor_names = y_regressor_names[1 + py :]
    factor_regressor_names = build_factor_regressor_names(pf=pf)

    max_lag = max(py, pz)
    bandwidth_y = compute_newey_west_bandwidth(n_obs)
    bandwidth_f = compute_newey_west_bandwidth(n_obs)

    y_residuals = np.empty((n_obs - max_lag, n_series), dtype=np.float64)
    y_coefficients = np.zeros(
        (n_series, len(y_regressor_names)),
        dtype=np.float64,
    )
    selection_mask = np.zeros(
        (n_series, len(selection_regressor_names)),
        dtype=bool,
    )

    factor_lags = mlags(f_values, pz)
    for series_idx in range(n_series):
        y_lags = mlags(y_values[:, series_idx], py)
        x = np.column_stack((np.ones(n_obs), y_lags, factor_lags))
        initial = newey_west(
            y_values[max_lag:, series_idx],
            x[max_lag:, :],
            bandwidth_y,
        )
        selected = np.abs(initial.tstat[1 + py :]) > threshold
        keep = np.concatenate(
            (
                np.ones(1 + py, dtype=bool),
                selected,
            )
        )
        refit = newey_west(
            y_values[max_lag:, series_idx],
            x[max_lag:, keep],
            bandwidth_y,
        )
        y_residuals[:, series_idx] = refit.resid
        y_coefficients[series_idx, keep] = refit.beta
        selection_mask[series_idx, :] = selected

    f_residuals = np.empty((n_obs - pf, n_predictors), dtype=np.float64)
    f_coefficients = np.zeros(
        (n_predictors, len(factor_regressor_names)),
        dtype=np.float64,
    )

    for predictor_idx in range(n_predictors):
        x = np.column_stack(
            (
                np.ones(n_obs),
                mlags(f_values[:, predictor_idx], pf),
            )
        )
        reg = newey_west(
            f_values[pf:, predictor_idx],
            x[pf:, :],
            bandwidth_f,
        )
        f_residuals[:, predictor_idx] = reg.resid
        f_coefficients[predictor_idx, :] = reg.beta

    return ForecastErrorResult(
        y_residuals=y_residuals,
        f_residuals=f_residuals,
        y_coefficients=y_coefficients,
        f_coefficients=f_coefficients,
        selection_mask=selection_mask,
        y_regressor_names=y_regressor_names,
        factor_regressor_names=factor_regressor_names,
        selection_regressor_names=selection_regressor_names,
        y_dates=date_index[max_lag:],
        f_dates=date_index[pf:],
        bandwidth_y=bandwidth_y,
        bandwidth_f=bandwidth_f,
    )
