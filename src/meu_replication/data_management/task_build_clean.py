"""Build clean macro panel from per-country raw snapshots."""

from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, MEU_COUNTRIES


def _build_raw_depends() -> dict[str, Path]:
    """Build dependency dict for all 77 per-country raw snapshot files."""
    deps = {}
    for country in MEU_COUNTRIES:
        for source in ("eurostat", "ecb", "oecd", "bis"):
            deps[f"{source}_{country}"] = (
                BLD / "data" / "raw" / source / f"{country}_snapshot.parquet"
            )
    # ECB U2 (EA aggregates)
    deps["ecb_U2"] = BLD / "data" / "raw" / "ecb" / "U2_snapshot.parquet"
    return deps


def task_build_clean(
    depends_on: dict = _build_raw_depends(),
    produces: Path = BLD / "data" / "clean" / "macro_panel.parquet",
) -> None:
    """Build clean macro panel from per-country snapshots (short and boring task).

    Task only handles I/O. Real logic in _clean_macro_panel() helper.
    """
    print("Loading raw snapshots from all per-country files...")

    # Every fetch task always writes a file (empty or with data),
    # so no existence checks needed — a missing file is a real error.
    dfs = []
    for label, path in depends_on.items():
        df = pd.read_parquet(path)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        msg = "No raw data found — cannot build clean panel"
        raise ValueError(msg)

    raw_combined = pd.concat(dfs, ignore_index=True)
    n_series = raw_combined["series_id"].nunique()
    n_countries = raw_combined["country_iso2"].nunique()
    print(f"Combined: {len(raw_combined)} rows, {n_series} series, {n_countries} countries")

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
