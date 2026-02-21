"""Build clean macro panel from all raw sources."""

from pathlib import Path

import pandas as pd

from template_project.config import BLD


def task_build_macro_panel(
    depends_on: dict[str, Path] = {
        # Eurostat categories 1-6
        "eurostat_cat1": BLD
        / "data"
        / "raw"
        / "eurostat"
        / "category_1_Industrial_production.csv",
        "eurostat_cat2": BLD
        / "data"
        / "raw"
        / "eurostat"
        / "category_2_Labor_market_indicators.csv",
        "eurostat_cat3": BLD / "data" / "raw" / "eurostat" / "category_3_Prices.csv",
        "eurostat_cat4": BLD
        / "data"
        / "raw"
        / "eurostat"
        / "category_4_Activity_indicators.csv",
        "eurostat_cat5": BLD / "data" / "raw" / "eurostat" / "category_5_Trade.csv",
        "eurostat_cat6": BLD / "data" / "raw" / "eurostat" / "category_6_Sentiment.csv",
        # ECB categories 4,7,8
        "ecb_cat4": BLD
        / "data"
        / "raw"
        / "ecb"
        / "category_4_Activity_indicators.csv",
        "ecb_cat7": BLD / "data" / "raw" / "ecb" / "category_7_Financial.csv",
        "ecb_cat8": BLD / "data" / "raw" / "ecb" / "category_8_EA_Financial.csv",
        # OECD categories 6,7
        "oecd_cat6": BLD / "data" / "raw" / "oecd" / "category_6_Sentiment.csv",
        "oecd_cat7": BLD / "data" / "raw" / "oecd" / "category_7_Financial.csv",
        # BIS category 7
        "bis_cat7": BLD / "data" / "raw" / "bis" / "category_7_Financial.csv",
    },
    produces: Path = BLD / "data" / "clean" / "macro_panel.csv",
) -> None:
    """Concatenate all raw sources into unified macro panel."""
    # Read all sources
    dfs = [pd.read_csv(path) for path in depends_on.values()]

    # Concatenate
    combined = pd.concat(dfs, ignore_index=True)

    # Clean and standardize
    cleaned = _clean_macro_panel(combined)

    # Save as CSV
    produces.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(produces, index=False)


def _clean_macro_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean combined macro panel following EPP functional rules.

    Returns new DataFrame with standardized schema.
    """
    return pd.DataFrame(
        {
            "date": _parse_dates(df["date"]),
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "series_id": df["series_id"].astype(str),
            "country_iso2": df["country_iso2"].astype(str),
            "variable_name": df["variable_name"].astype(str),
            "category": pd.to_numeric(df["category"], errors="coerce").astype("Int64"),
            "category_name": df["category_name"].astype(str),
        }
    ).dropna(subset=["value", "date"]).drop_duplicates(
        subset=["series_id", "date"]
    ).sort_values(
        ["country_iso2", "series_id", "date"]
    ).reset_index(
        drop=True
    )


def _parse_dates(dates: pd.Series) -> pd.Series:
    """Parse monthly dates to consistent YYYY-MM string format."""
    parsed = pd.to_datetime(dates, errors="coerce")
    return parsed.dt.strftime("%Y-%m")
