"""Newey-West regressions matching the JLN MATLAB helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_TWO_DIMENSIONS = 2


@dataclass(frozen=True)
class NeweyWestResult:
    """Regression outputs used downstream by the MEU pipeline."""

    beta: FloatArray
    tstat: FloatArray
    yhat: FloatArray
    resid: FloatArray
    sige: float
    rsqr: float
    rbar: float
    dw: float
    covariance: FloatArray


def compute_newey_west_bandwidth(n_obs: int) -> int:
    """Match the JLN/MATLAB Newey-West bandwidth rule."""
    if n_obs < 1:
        msg = "n_obs must be positive."
        raise ValueError(msg)
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def newey_west(y: FloatArray, x: FloatArray, nlag: int) -> NeweyWestResult:
    """Run OLS and attach Newey-West HAC standard errors."""
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    x_values = np.asarray(x, dtype=np.float64)

    if x_values.ndim != _TWO_DIMENSIONS:
        msg = "x must be a two-dimensional design matrix."
        raise ValueError(msg)
    if y_values.shape[0] != x_values.shape[0]:
        msg = "y and x must have the same number of observations."
        raise ValueError(msg)
    if nlag < 0:
        msg = "nlag must be non-negative."
        raise ValueError(msg)

    n_obs, n_var = x_values.shape
    if n_obs <= n_var:
        msg = "Newey-West regression requires more observations than regressors."
        raise ValueError(msg)

    xtx = x_values.T @ x_values
    xtx_inv = _invert_gram(xtx)
    beta = xtx_inv @ (x_values.T @ y_values)
    yhat = x_values @ beta
    resid = y_values - yhat
    sigu = float(resid.T @ resid)
    sige = sigu / (n_obs - n_var)

    hhat = np.tile(resid, (n_var, 1)) * x_values.T
    covariance_core = np.zeros((n_var, n_var), dtype=np.float64)

    for lag in range(nlag + 1):
        weight = (nlag + 1 - lag) / (nlag + 1)
        za = hhat[:, lag:n_obs] @ hhat[:, : n_obs - lag].T
        ga = za if lag == 0 else za + za.T
        covariance_core += weight * ga

    covariance = xtx_inv @ covariance_core @ xtx_inv
    nwerr = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    tstat = np.divide(
        beta,
        nwerr,
        out=np.full_like(beta, np.inf),
        where=nwerr > 0,
    )

    y_centered = y_values - y_values.mean()
    total_ss = float(y_centered.T @ y_centered)
    rsqr = np.nan if total_ss == 0 else 1.0 - sigu / total_ss
    rsqr1 = sigu / (n_obs - n_var)
    rsqr2 = np.nan if n_obs == 1 else total_ss / (n_obs - 1.0)
    rbar = np.nan if np.isnan(rsqr2) or rsqr2 == 0.0 else 1.0 - (rsqr1 / rsqr2)
    ediff = resid[1:n_obs] - resid[: n_obs - 1]
    dw = float((ediff.T @ ediff) / sigu) if sigu > 0 else np.nan

    return NeweyWestResult(
        beta=beta,
        tstat=tstat,
        yhat=yhat,
        resid=resid,
        sige=sige,
        rsqr=rsqr,
        rbar=rbar,
        dw=dw,
        covariance=covariance,
    )


def _invert_gram(gram: FloatArray) -> FloatArray:
    """Invert the Gram matrix with a pseudoinverse fallback for stability."""
    try:
        return np.asarray(np.linalg.inv(gram), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.pinv(gram), dtype=np.float64)
