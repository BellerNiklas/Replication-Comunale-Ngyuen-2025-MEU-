"""Fetch all series and build unified macro panel (Germany only).

This task replaces the 4 separate fetcher tasks with a single unified task.
Follows EPP: short task (I/O only), logic in pure helper functions.
"""

from pathlib import Path

import pandas as pd

from template_project.config import BLD
from template_project.data_fetch.fetch import fetch_many
from template_project.data_management.registry.registry_io import load_registry


def task_fetch_and_build_macro_panel(
    produces: Path = BLD / "data" / "clean" / "macro_panel.parquet",
) -> None:
    """Fetch all series from registry and build unified panel (short and boring).

    Task only handles I/O. Real logic in pure helper functions.

    Produces:
        Unified macro panel in Parquet format with all 148 series.
    """
    print("Loading series registry...")
    registry = load_registry()
    all_series_ids = registry.series_id.tolist()
    print(f"Found {len(all_series_ids)} series in registry")

    print("\nFetching data from all sources...")
    raw_combined = fetch_many(all_series_ids, registry=registry)
    print(f"\nFetched total: {len(raw_combined)} rows")

    print("\nCleaning and standardizing data...")
    cleaned = _clean_macro_panel(raw_combined)
    print(f"After cleaning: {len(cleaned)} rows")

    print(f"\nWriting to {produces}...")
    produces.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(produces, index=False)
    print(f"Successfully wrote {len(cleaned)} rows to {produces.name}")


def _clean_macro_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean combined macro panel following EPP functional rules.

    Pure function: constructs new DataFrame, no mutations.

    Args:
        df: Raw combined DataFrame from all sources

    Returns:
        Cleaned DataFrame with standardized schema.
    """
    return (
        pd.DataFrame(
            {
                "date": _parse_dates(df["date"]),
                "value": pd.to_numeric(df["value"], errors="coerce"),
                "series_id": df["series_id"].astype(str),
                "country_iso2": df["country_iso2"].astype(str),
                "variable_name": df["variable_name"].astype(str),
                "category": pd.to_numeric(df["category"], errors="coerce").astype(
                    "Int64"
                ),
                "category_name": df["category_name"].astype(str),
                "source": df["source"].astype(str),
            }
        )
        .dropna(subset=["value", "date"])
        .drop_duplicates(subset=["series_id", "date"])
        .sort_values(["country_iso2", "series_id", "date"])
        .reset_index(drop=True)
    )


def _parse_dates(dates: pd.Series) -> pd.Series:
    """Parse monthly dates to consistent YYYY-MM string format.

    Pure function: no side effects.

    Args:
        dates: Series of date strings in various formats

    Returns:
        Series of date strings in YYYY-MM format.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    return parsed.dt.strftime("%Y-%m")
