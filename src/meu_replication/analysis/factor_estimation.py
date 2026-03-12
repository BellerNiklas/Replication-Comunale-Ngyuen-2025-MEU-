"""Panel preparation and factor estimation for MEU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_PENALTY_FORMULAS = tuple(range(1, 9))

_REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    "date",
    "value",
    "series_id",
    "country_iso2",
    "variable_name",
    "category",
    "category_name",
    "source",
    "transformationcode",
)


@dataclass(frozen=True)
class FactorEstimationResult:
    """Outputs needed downstream from the factor stage."""

    standardized_values: FloatArray
    series_means: FloatArray
    series_stds: FloatArray
    factor_count: int
    squared_factor_count: int
    fhat: FloatArray
    ghat: FloatArray
    predictor_set: FloatArray
    loadings: FloatArray
    squared_loadings: FloatArray
    eigenvalues: FloatArray
    squared_eigenvalues: FloatArray


def prepare_analysis_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot the long panel to a deterministic wide matrix and metadata tables."""
    missing = [col for col in _REQUIRED_PANEL_COLUMNS if col not in panel.columns]
    if missing:
        msg = f"Panel is missing required columns: {missing}"
        raise ValueError(msg)

    wide = (
        panel.pivot_table(
            index="date",
            columns="series_id",
            values="value",
            aggfunc="first",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    wide.columns.name = None
    wide.index.name = "date"

    if wide.isna().any().any():
        msg = "Analysis panel must be balanced after the long-to-wide pivot."
        raise ValueError(msg)

    series_metadata = (
        panel[
            [
                "series_id",
                "country_iso2",
                "variable_name",
                "category",
                "category_name",
                "source",
                "transformationcode",
            ]
        ]
        .drop_duplicates(subset=["series_id"])
        .set_index("series_id")
        .reindex(wide.columns)
    )
    series_metadata.index.name = "series_id"
    series_metadata = series_metadata.reset_index()
    series_metadata.insert(
        0,
        "series_position",
        pd.Series(np.arange(len(series_metadata)), dtype="int64"),
    )

    date_index = pd.DataFrame(
        {
            "date_position": np.arange(len(wide.index), dtype=np.int64),
            "date": wide.index.astype(str),
        }
    )

    return wide, series_metadata, date_index


def standardize_columns(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Z-score columns using sample standard deviation to match MATLAB."""
    values = np.asarray(x, dtype=np.float64)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1)

    if np.any(~np.isfinite(stds)) or np.any(stds == 0):
        msg = "Cannot standardize columns with zero or non-finite sample std."
        raise ValueError(msg)

    standardized = (values - means) / stds
    return standardized, means, stds


def select_factor_count(
    x: FloatArray,
    kmax: int,
    ic_criterion: int = 2,
) -> int:
    """Select the number of factors using Bai-Ng information criteria."""
    values = np.asarray(x, dtype=np.float64)
    n_obs, n_series = values.shape
    kmax_eff = min(kmax, n_obs, n_series)
    if kmax_eff < 1:
        msg = "kmax must be positive and no larger than the matrix rank bound."
        raise ValueError(msg)

    penalties = _compute_penalty_terms(
        n_obs=n_obs,
        n_series=n_series,
        kmax=kmax_eff,
        ic_criterion=ic_criterion,
    )
    factor_space, loading_space, _ = _initial_factor_space(values)

    ic_values = np.empty(kmax_eff + 1, dtype=np.float64)
    ic_values[0] = np.log(np.mean(values**2))

    for factor_count in range(1, kmax_eff + 1):
        fitted = factor_space[:, :factor_count] @ loading_space[:, :factor_count].T
        sigma = np.mean((values - fitted) ** 2)
        ic_values[factor_count] = np.log(sigma) + penalties[factor_count - 1]

    return int(np.argmin(ic_values))


def extract_static_factors(
    x: FloatArray,
    n_factors: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract static factors and loadings from a standardized panel."""
    values = np.asarray(x, dtype=np.float64)
    if n_factors < 1:
        msg = "At least one factor is required for extraction."
        raise ValueError(msg)

    factor_space, loading_space, eigenvalues = _pc_factor_space(values)
    if n_factors > factor_space.shape[1]:
        msg = "Requested more factors than are available in the factor space."
        raise ValueError(msg)

    return (
        factor_space[:, :n_factors],
        loading_space[:, :n_factors],
        eigenvalues,
    )


def estimate_predictor_set(
    x: FloatArray,
    kmax: int = 20,
    ic_criterion: int = 2,
) -> FactorEstimationResult:
    """Estimate factors on X and X^2, then build the predictor set."""
    values = np.asarray(x, dtype=np.float64)

    standardized_values, means, stds = standardize_columns(values)
    factor_count = select_factor_count(
        standardized_values,
        kmax=kmax,
        ic_criterion=ic_criterion,
    )
    if factor_count < 1:
        msg = "The selected factor count must be at least 1."
        raise ValueError(msg)

    fhat, loadings, eigenvalues = extract_static_factors(
        standardized_values,
        n_factors=factor_count,
    )

    squared_standardized, _, _ = standardize_columns(values**2)
    squared_factor_count = select_factor_count(
        squared_standardized,
        kmax=kmax,
        ic_criterion=ic_criterion,
    )
    if squared_factor_count < 1:
        msg = "The squared-data factor count must be at least 1."
        raise ValueError(msg)

    ghat, squared_loadings, squared_eigenvalues = extract_static_factors(
        squared_standardized,
        n_factors=squared_factor_count,
    )
    predictor_set = np.column_stack((fhat, fhat[:, 0] ** 2, ghat[:, 0]))

    return FactorEstimationResult(
        standardized_values=standardized_values,
        series_means=means,
        series_stds=stds,
        factor_count=factor_count,
        squared_factor_count=squared_factor_count,
        fhat=fhat,
        ghat=ghat,
        predictor_set=predictor_set,
        loadings=loadings,
        squared_loadings=squared_loadings,
        eigenvalues=eigenvalues,
        squared_eigenvalues=squared_eigenvalues,
    )


def _compute_penalty_terms(
    n_obs: int,
    n_series: int,
    kmax: int,
    ic_criterion: int,
) -> FloatArray:
    """Compute Bai-Ng penalty terms."""
    ii = np.arange(1, kmax + 1, dtype=np.float64)
    nt = float(n_obs * n_series)
    nt1 = float(n_obs + n_series)
    gct = float(min(n_obs, n_series))

    penalties = {
        _PENALTY_FORMULAS[0]: np.log(nt / nt1) * ii * nt1 / nt,
        _PENALTY_FORMULAS[1]: ii * (nt1 / nt) * np.log(gct),
        _PENALTY_FORMULAS[2]: ii * np.log(gct) / gct,
        _PENALTY_FORMULAS[3]: 2 * ii / n_obs,
        _PENALTY_FORMULAS[4]: np.log(n_obs) * ii / n_obs,
        _PENALTY_FORMULAS[5]: 2 * ii * nt1 / nt,
        _PENALTY_FORMULAS[6]: np.log(nt) * ii * nt1 / nt,
        _PENALTY_FORMULAS[7]: 2 * ii * (np.sqrt(n_series) + np.sqrt(n_obs)) ** 2 / nt,
    }
    if ic_criterion in penalties:
        return penalties[ic_criterion]

    msg = f"Unsupported Bai-Ng information criterion: {ic_criterion}"
    raise ValueError(msg)


def _initial_factor_space(
    x: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct the IC-evaluation factor space used by MATLAB's nbplog helper."""
    n_obs, n_series = x.shape

    if n_obs < n_series:
        gram = x @ x.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
        eigenvectors = eigenvectors[:, order]
        factor_space = np.sqrt(n_obs) * eigenvectors
        loading_space = x.T @ factor_space / n_obs
        return factor_space, loading_space, eigenvalues

    gram = x.T @ x
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]
    loading_space = np.sqrt(n_series) * eigenvectors
    factor_space = x @ loading_space / n_series
    return factor_space, loading_space, eigenvalues


def _pc_factor_space(
    x: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Match the factor normalization used by MATLAB's pc() helper."""
    n_series = x.shape[1]
    left_singular, singular_values, right_singular_t = np.linalg.svd(
        x,
        full_matrices=False,
    )
    factor_space = left_singular * (singular_values / np.sqrt(n_series))
    loading_space = right_singular_t.T * np.sqrt(n_series)
    eigenvalues = singular_values**2
    return factor_space, loading_space, eigenvalues
