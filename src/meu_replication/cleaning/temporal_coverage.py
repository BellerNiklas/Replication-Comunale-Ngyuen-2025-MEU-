"""Pure functions for temporal coverage filtering."""

import math
from typing import Any

import pandas as pd

_COVERAGE_THRESHOLD = 0.02  # 98% coverage -> 2% missing allowed


def _build_expected_months(sample_start: str, sample_end: str) -> set[str]:
    """Build canonical set of YYYY-MM strings for the full sample period.

    Args:
        sample_start: First month of sample period ("YYYY-MM").
        sample_end: Last month of sample period ("YYYY-MM").

    Returns:
        Set of YYYY-MM strings for every month in [sample_start, sample_end].
    """
    periods = pd.period_range(sample_start, sample_end, freq="M")
    return {p.strftime("%Y-%m") for p in periods}


def build_variants(
    windows: list[tuple[str, str, str]],
    thresholds: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Build filter variant configs from windows x thresholds.

    Args:
        windows: List of (label_suffix, start, end) tuples.
        thresholds: List of (label_suffix, allowed_missing) tuples.

    Returns:
        List of variant dicts with keys: label, key, start, end, allowed_missing.
    """
    return [
        {
            "label": f"{wlabel}_{tlabel}",
            "key": f"panel_{wlabel}_{tlabel}",
            "start": start,
            "end": end,
            "allowed_missing": missing,
        }
        for wlabel, start, end in windows
        for tlabel, missing in thresholds
    ]


def compute_allowed_missing(sample_start: str, sample_end: str) -> int:
    """Compute allowed missing months for the 98% coverage threshold."""
    n_months = len(_build_expected_months(sample_start, sample_end))
    return math.floor(_COVERAGE_THRESHOLD * n_months)


def filter_by_temporal_coverage(
    df: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    allowed_missing: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter panel to series with sufficient monthly coverage.

    Pure, vectorized function: no Python loops, no mutations.

    Args:
        df: Clean macro panel with columns [date, value, series_id,
            country_iso2, variable_name, category, category_name, source].
        sample_start: First month of sample period ("YYYY-MM").
        sample_end: Last month of sample period ("YYYY-MM").
        allowed_missing: Maximum number of missing months tolerated.
            0 = strict (every month required), >0 = near-complete.

    Returns:
        Tuple of (filtered_panel, drop_info) where:
        - filtered_panel: rows restricted to sample period, only for series
          with at most allowed_missing months absent.
        - drop_info: DataFrame with columns [series_id, country_iso2,
          variable_name, category_name, n_months, n_missing] for every
          series that was dropped due to insufficient coverage.
    """
    drop_cols: tuple[str, ...] = (
        "series_id",
        "country_iso2",
        "variable_name",
        "category_name",
        "n_months",
        "n_missing",
    )
    empty_drop = pd.DataFrame(columns=pd.Index(drop_cols))

    if df.empty:
        return df.copy(), empty_drop

    # Restrict to sample period
    in_period = df[(df["date"] >= sample_start) & (df["date"] <= sample_end)].copy()

    expected_n = len(pd.period_range(sample_start, sample_end, freq="M"))

    # Vectorized coverage check
    n_months = in_period.groupby("series_id")["date"].nunique()
    n_missing = (expected_n - n_months).clip(lower=0)
    keep_ids = n_missing[n_missing <= allowed_missing].index
    drop_ids = n_missing[n_missing > allowed_missing].index

    # Build filtered panel
    filtered_panel = (
        in_period[in_period["series_id"].isin(keep_ids)]
        .sort_values(["country_iso2", "series_id", "date"])
        .reset_index(drop=True)
    )

    # Build drop info
    if len(drop_ids) == 0:
        return filtered_panel, empty_drop.copy()

    meta = in_period.groupby("series_id").agg(
        country_iso2=("country_iso2", "first"),
        variable_name=("variable_name", "first"),
        category_name=("category_name", "first"),
    )
    drop_info = meta.loc[drop_ids].copy()
    drop_info["n_months"] = n_months.loc[drop_ids]
    drop_info["n_missing"] = n_missing.loc[drop_ids]
    drop_info = (
        drop_info.reset_index()
        .sort_values(["country_iso2", "series_id"])
        .reset_index(drop=True)
    )

    return filtered_panel, drop_info


def run_all_filter_variants(
    panel: pd.DataFrame,
    filter_variants: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Run all filter variants and return filtered panels.

    Args:
        panel: Clean macro panel DataFrame.
        filter_variants: List of dicts with keys:
            label, key, start, end, allowed_missing.

    Returns:
        Dict mapping key -> filtered DataFrame.
    """
    return {
        v["key"]: filter_by_temporal_coverage(
            panel, str(v["start"]), str(v["end"]), int(v["allowed_missing"])
        )[0]
        for v in filter_variants
    }
