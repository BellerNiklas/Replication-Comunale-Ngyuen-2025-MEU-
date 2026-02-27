"""Filter clean macro panel to series with full temporal coverage."""

import math
from pathlib import Path

import pandas as pd

from meu_replication.config import (
    BLD,
    SAMPLE_END,
    SAMPLE_END_ALT,
    SAMPLE_START,
)

_COVERAGE_THRESHOLD = 0.02  # 98% coverage


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


def _compute_allowed_missing(sample_start: str, sample_end: str) -> int:
    """Compute allowed missing months for the 98% coverage threshold."""
    n_months = len(_build_expected_months(sample_start, sample_end))
    return math.floor(_COVERAGE_THRESHOLD * n_months)


# Module-level variant configuration (computed once at import time)
_FILTER_VARIANTS = [
    {
        "label": "2022_strict",
        "key": "panel_2022_strict",
        "start": SAMPLE_START,
        "end": SAMPLE_END,
        "allowed_missing": 0,
    },
    {
        "label": "2022_cov98",
        "key": "panel_2022_cov98",
        "start": SAMPLE_START,
        "end": SAMPLE_END,
        "allowed_missing": _compute_allowed_missing(SAMPLE_START, SAMPLE_END),
    },
    {
        "label": "2021_strict",
        "key": "panel_2021_strict",
        "start": SAMPLE_START,
        "end": SAMPLE_END_ALT,
        "allowed_missing": 0,
    },
    {
        "label": "2021_cov98",
        "key": "panel_2021_cov98",
        "start": SAMPLE_START,
        "end": SAMPLE_END_ALT,
        "allowed_missing": _compute_allowed_missing(SAMPLE_START, SAMPLE_END_ALT),
    },
]


def filter_by_temporal_coverage(
    df: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    allowed_missing: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter panel to series with sufficient monthly coverage.

    Pure function: constructs new DataFrames, no mutations.

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
    expected_months = _build_expected_months(sample_start, sample_end)

    empty_drop = pd.DataFrame(
        columns=[
            "series_id",
            "country_iso2",
            "variable_name",
            "category_name",
            "n_months",
            "n_missing",
        ]
    )

    if df.empty:
        return df.copy(), empty_drop

    # Restrict to sample period
    in_period = df[(df["date"] >= sample_start) & (df["date"] <= sample_end)].copy()

    # For each series, check coverage against canonical month set
    series_meta = in_period.groupby("series_id").agg(
        country_iso2=("country_iso2", "first"),
        variable_name=("variable_name", "first"),
        category_name=("category_name", "first"),
    )

    present_by_series = in_period.groupby("series_id")["date"].apply(set)

    # Classify each series
    keep_ids = []
    drop_rows = []

    for series_id, present in present_by_series.items():
        missing = expected_months - present
        if len(missing) <= allowed_missing:
            keep_ids.append(series_id)
        else:
            meta = series_meta.loc[series_id]
            drop_rows.append(
                {
                    "series_id": series_id,
                    "country_iso2": meta["country_iso2"],
                    "variable_name": meta["variable_name"],
                    "category_name": meta["category_name"],
                    "n_months": len(present),
                    "n_missing": len(missing),
                }
            )

    # Build filtered panel (only kept series, only sample period rows)
    filtered_panel = (
        in_period[in_period["series_id"].isin(keep_ids)]
        .sort_values(["country_iso2", "series_id", "date"])
        .reset_index(drop=True)
    )

    # Build drop info
    if drop_rows:
        drop_info = (
            pd.DataFrame(drop_rows)
            .sort_values(["country_iso2", "series_id"])
            .reset_index(drop=True)
        )
    else:
        drop_info = empty_drop.copy()

    return filtered_panel, drop_info


def run_all_filter_variants(
    panel: pd.DataFrame,
    filter_variants: list[dict],
) -> dict[str, pd.DataFrame]:
    """Run all filter variants and return filtered panels.

    Pure function: no I/O, no side effects.

    Args:
        panel: Clean macro panel DataFrame.
        filter_variants: List of dicts with keys:
            label, key, start, end, allowed_missing.

    Returns:
        Dict mapping key -> filtered DataFrame.
    """
    panels = {}

    for variant in filter_variants:
        filtered, _ = filter_by_temporal_coverage(
            panel,
            variant["start"],
            variant["end"],
            variant["allowed_missing"],
        )
        panels[variant["key"]] = filtered

    return panels


def task_filter_temporal_coverage(
    depends_on: Path = BLD / "data" / "clean" / "macro_panel.parquet",
    produces: dict[str, Path] = {
        "panel_2022_strict": BLD / "data" / "clean" / "panel_2003_2022_strict.parquet",
        "panel_2022_cov98": BLD / "data" / "clean" / "panel_2003_2022_cov98.parquet",
        "panel_2021_strict": BLD / "data" / "clean" / "panel_2003_2021_strict.parquet",
        "panel_2021_cov98": BLD / "data" / "clean" / "panel_2003_2021_cov98.parquet",
    },
) -> None:
    """Filter macro panel to series with sufficient temporal coverage.

    Produces 4 filtered panels (2 windows x 2 thresholds).
    Task only handles I/O; real logic in run_all_filter_variants().
    """
    panel = pd.read_parquet(depends_on)
    print(f"Input: {len(panel)} rows, {panel['series_id'].nunique()} series")

    panels = run_all_filter_variants(panel, _FILTER_VARIANTS)

    for key, filtered in panels.items():
        out_path = produces[key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(out_path, index=False)

    print(f"Wrote {len(panels)} filtered panels")
