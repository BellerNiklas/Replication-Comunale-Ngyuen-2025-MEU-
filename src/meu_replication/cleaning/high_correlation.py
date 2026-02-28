"""Pure functions for removing highly correlated variables within each country."""

from __future__ import annotations

import numpy as np
import pandas as pd


def remove_high_correlation(
    panel: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop one variable from each highly correlated pair, per country.

    For each country_iso2 group, pivots to wide form, computes the pairwise
    Pearson correlation matrix, and greedily removes redundant series.
    The alphabetically-later series_id is dropped from each correlated pair.

    Pure function: constructs new DataFrames, no mutation of input.

    Args:
        panel: Long-format panel with columns [date, value, series_id,
            country_iso2, variable_name, category, category_name, source].
        threshold: Absolute correlation above which one series is dropped.

    Returns:
        Tuple of (filtered_panel, drop_info) where:
        - filtered_panel has the same schema as input, minus dropped rows.
        - drop_info has columns [country_iso2, dropped_series, kept_series,
          abs_correlation].
    """
    drop_cols = ["country_iso2", "dropped_series", "kept_series", "abs_correlation"]
    empty_drop = pd.DataFrame(columns=drop_cols)

    if panel.empty:
        return panel.copy(), empty_drop

    kept_frames: list[pd.DataFrame] = []
    drop_records: list[dict] = []

    for country, country_df in panel.groupby("country_iso2"):
        ids_to_drop, records = _find_redundant_series(
            country_df, threshold, str(country)
        )
        drop_records.extend(records)
        kept_frames.append(country_df[~country_df["series_id"].isin(ids_to_drop)])

    filtered = (
        pd.concat(kept_frames, ignore_index=True)
        .sort_values(["country_iso2", "series_id", "date"])
        .reset_index(drop=True)
    )

    drop_info = pd.DataFrame(drop_records, columns=drop_cols)

    return filtered, drop_info


def _find_redundant_series(
    country_df: pd.DataFrame,
    threshold: float,
    country: str,
) -> tuple[set[str], list[dict]]:
    """Identify which series to drop for a single country.

    Uses a greedy maximum-independent-set approach: iterates through
    series in alphabetical order and keeps each series only if none of
    its high-correlation neighbours have already been kept. This
    guarantees that no two kept series are correlated above the
    threshold and maximises the number of retained variables.

    Args:
        country_df: Panel rows for one country.
        threshold: Absolute correlation cutoff.
        country: Country ISO2 code (for metadata).

    Returns:
        Tuple of (set of series_ids to drop, list of record dicts).
    """
    wide = _pivot_to_wide(country_df)
    corr = wide.corr()
    pairs = _correlated_pairs(corr, threshold)

    # Build adjacency: series -> set of correlated neighbours
    adj: dict[str, set[str]] = {s: set() for s in wide.columns}
    pair_corr: dict[tuple[str, str], float] = {}
    for s1, s2, abs_corr in pairs:
        adj[s1].add(s2)
        adj[s2].add(s1)
        pair_corr[(min(s1, s2), max(s1, s2))] = abs_corr

    # Greedy independent set: alphabetical order, keep if no neighbour kept
    kept: set[str] = set()
    for series in sorted(wide.columns):
        if not adj[series] & kept:
            kept.add(series)

    dropped = set(wide.columns) - kept

    # Build drop records: for each dropped series, report its highest-
    # correlation kept neighbour as the reason
    records: list[dict] = []
    for drop in sorted(dropped):
        kept_neighbours = adj[drop] & kept
        if kept_neighbours:
            best_kept = max(
                kept_neighbours,
                key=lambda k: pair_corr.get((min(drop, k), max(drop, k)), 0.0),
            )
            key = (min(drop, best_kept), max(drop, best_kept))
            records.append(
                {
                    "country_iso2": country,
                    "dropped_series": drop,
                    "kept_series": best_kept,
                    "abs_correlation": round(pair_corr[key], 6),
                }
            )

    return dropped, records


def _pivot_to_wide(country_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a single-country long panel to wide: rows=date, cols=series_id.

    Args:
        country_df: Long-format panel for one country.

    Returns:
        Wide DataFrame with dates as index and series_ids as columns.
    """
    return country_df.pivot(index="date", columns="series_id", values="value")


def _correlated_pairs(
    corr: pd.DataFrame,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Extract unique pairs with |correlation| > threshold, sorted descending.

    Args:
        corr: Square correlation matrix.
        threshold: Absolute correlation cutoff.

    Returns:
        List of (series_a, series_b, abs_correlation) sorted by
        abs_correlation descending, then alphabetically for determinism.
    """
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    abs_corr = corr.abs()
    above = abs_corr.where(mask & (abs_corr > threshold))

    pairs: list[tuple[str, str, float]] = []
    for col in above.columns:
        for idx in above.index:
            val = above.at[idx, col]
            if pd.notna(val):
                a, b = sorted([str(idx), str(col)])
                pairs.append((a, b, float(val)))

    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    return pairs
