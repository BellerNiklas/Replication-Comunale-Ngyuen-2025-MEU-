"""Transform raw API responses to canonical long format.

All functions follow EPP functional rules:
- Start with empty DataFrame (no mutations)
- Touch each variable once
- Pure functions (no side effects)
"""

from typing import Any

import pandas as pd


def standardize_from_eurostat(raw: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Transform Eurostat wide format to canonical long (pure function).

    Follows EPP: constructs new DataFrame, no mutations.

    Args:
        raw: Wide format DataFrame from Eurostat API (time columns + dimension columns)
        spec: Series specification dict with keys: series_id, country_iso2,
              variable_name, category, category_name

    Returns:
        Long format DataFrame with canonical schema.
    """
    # Identify time columns (those that look like dates: YYYY, YYYY-MM, etc.)
    # Eurostat returns columns like: freq, nace_r2, unit, s_adj, geo, 2003-01, 2003-02, etc.
    # Time columns are those that start with a digit
    time_cols = [c for c in raw.columns if str(c)[0].isdigit()]

    if not time_cols:
        msg = f"No time columns found in Eurostat response for {spec['series_id']}"
        raise ValueError(msg)

    # Melt to long format (pure operation - creates new DataFrame)
    long = raw.melt(
        id_vars=[c for c in raw.columns if c not in time_cols],
        value_vars=time_cols,
        var_name="date",
        value_name="value",
    )

    # Construct new standardized DataFrame (EPP rule 1: start with empty)
    # Touch each column once (EPP rule 2)
    return pd.DataFrame(
        {
            "date": long["date"].astype(str),
            "value": pd.to_numeric(long["value"], errors="coerce"),
            "series_id": spec["series_id"],
            "country_iso2": spec["country_iso2"],
            "variable_name": spec["variable_name"],
            "category": spec["category"],
            "category_name": spec["category_name"],
            "source": "eurostat",
        }
    ).dropna(subset=["value"]).reset_index(drop=True)


def standardize_from_ecb_like(raw: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Transform ECB/OECD long format to canonical long (pure function).

    Used for both ECB and OECD (same API response format).
    Follows EPP: constructs new DataFrame.

    Args:
        raw: Long format DataFrame with TIME_PERIOD and OBS_VALUE columns
        spec: Series specification dict with keys: series_id, country_iso2,
              variable_name, category, category_name, source

    Returns:
        Long format DataFrame with canonical schema.
    """
    # Handle case variations in column names
    time_col = _find_column(raw, ["TIME_PERIOD", "time_period", "TIME"])
    value_col = _find_column(raw, ["OBS_VALUE", "obs_value", "VALUE"])

    if time_col not in raw.columns or value_col not in raw.columns:
        msg = (
            f"Expected TIME_PERIOD and OBS_VALUE columns not found "
            f"for {spec['series_id']}. Available: {list(raw.columns)}"
        )
        raise ValueError(msg)

    # Construct new standardized DataFrame (EPP: start empty, touch once)
    return pd.DataFrame(
        {
            "date": raw[time_col].astype(str),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
            "series_id": spec["series_id"],
            "country_iso2": spec["country_iso2"],
            "variable_name": spec["variable_name"],
            "category": spec["category"],
            "category_name": spec["category_name"],
            "source": spec["source"],
        }
    ).dropna(subset=["value"]).reset_index(drop=True)


def standardize_from_bis(raw: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Transform BIS long format to canonical long (pure function).

    Follows EPP: constructs new DataFrame.

    Args:
        raw: Long format DataFrame with TIME_PERIOD and OBS_VALUE columns
        spec: Series specification dict with keys: series_id, country_iso2,
              variable_name, category, category_name, unit_measure_filter (optional)

    Returns:
        Long format DataFrame with canonical schema.
    """
    # Handle case variations
    time_col = _find_column(raw, ["TIME_PERIOD", "time_period", "TIME"])
    value_col = _find_column(raw, ["OBS_VALUE", "obs_value", "VALUE"])

    if time_col not in raw.columns or value_col not in raw.columns:
        msg = (
            f"Expected TIME_PERIOD and OBS_VALUE columns not found "
            f"for {spec['series_id']}. Available: {list(raw.columns)}"
        )
        raise ValueError(msg)

    # Filter by unit_measure if specified (pure operation - creates new DataFrame)
    filtered = raw
    unit_measure_filter = spec.get("unit_measure_filter")
    if (
        unit_measure_filter
        and not pd.isna(unit_measure_filter)
        and "UNIT_MEASURE" in raw.columns
    ):
        # Convert to int for comparison if it's a string
        if isinstance(unit_measure_filter, str) and unit_measure_filter.isdigit():
            unit_measure_filter = int(unit_measure_filter)
        filtered = raw[raw["UNIT_MEASURE"] == unit_measure_filter]

    # Construct new standardized DataFrame (EPP: start empty, touch once)
    return pd.DataFrame(
        {
            "date": filtered[time_col].astype(str),
            "value": pd.to_numeric(filtered[value_col], errors="coerce"),
            "series_id": spec["series_id"],
            "country_iso2": spec["country_iso2"],
            "variable_name": spec["variable_name"],
            "category": spec["category"],
            "category_name": spec["category_name"],
            "source": "bis",
        }
    ).dropna(subset=["value"]).reset_index(drop=True)


def standardize_oecd_bulk(
    raw: pd.DataFrame,
    registry: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Transform bulk OECD response to canonical long format (pure function).

    Reconstructs SDMX keys from dimension columns and matches each row
    to its registry series_id. Follows EPP: constructs new DataFrame.

    Args:
        raw: Concatenated bulk OECD CSV with 'dataset' column added.
        registry: Full series registry DataFrame.
        countries: Countries DataFrame for ISO3-to-ISO2 mapping.

    Returns:
        Canonical long DataFrame with standard schema.
    """
    oecd_reg = registry[registry.source == "oecd"]

    # SDMX dimension columns in key order (verified from OECD API response)
    _KEY_DIMS = [
        "REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE",
        "ACTIVITY", "ADJUSTMENT", "TRANSFORMATION", "TIME_HORIZ", "METHODOLOGY",
    ]

    # Build lookup: (dataset, key) -> registry spec dict
    key_lookup = {
        (row["dataset"], row["key"]): row.to_dict()
        for _, row in oecd_reg.iterrows()
    }

    # Reconstruct SDMX key per row and match to registry
    keys = raw[_KEY_DIMS].apply(
        lambda row: ".".join(str(v) for v in row), axis=1
    )
    datasets = raw["dataset"]

    series_ids = []
    country_iso2s = []
    variable_names = []
    categories = []
    category_names = []

    for key, dataset in zip(keys, datasets):
        spec = key_lookup.get((dataset, key))
        if spec is not None:
            series_ids.append(spec["series_id"])
            country_iso2s.append(spec["country_iso2"])
            variable_names.append(spec["variable_name"])
            categories.append(spec["category"])
            category_names.append(spec["category_name"])
        else:
            series_ids.append(None)
            country_iso2s.append(None)
            variable_names.append(None)
            categories.append(None)
            category_names.append(None)

    time_col = _find_column(raw, ["TIME_PERIOD", "time_period", "TIME"])
    value_col = _find_column(raw, ["OBS_VALUE", "obs_value", "VALUE"])

    # Construct new DataFrame (EPP: start empty, touch each column once)
    result = pd.DataFrame(
        {
            "date": raw[time_col].astype(str),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
            "series_id": series_ids,
            "country_iso2": country_iso2s,
            "variable_name": variable_names,
            "category": categories,
            "category_name": category_names,
            "source": "oecd",
        }
    )

    # Drop rows with no registry match and no value
    return (
        result
        .dropna(subset=["series_id", "value"])
        .reset_index(drop=True)
    )


def validate_long(df: pd.DataFrame, series_id: str) -> None:
    """Validate standardized long DataFrame (pure function - no side effects).

    Args:
        df: Standardized long format DataFrame to validate
        series_id: Series identifier for error messages

    Raises:
        ValueError: If validation fails with detailed message.
    """
    if df.empty:
        msg = f"{series_id}: No data returned"
        raise ValueError(msg)

    if df["value"].isna().all():
        msg = f"{series_id}: All values are NaN"
        raise ValueError(msg)

    required_cols = [
        "date",
        "value",
        "series_id",
        "country_iso2",
        "variable_name",
        "category",
        "category_name",
        "source",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        msg = f"{series_id}: Missing columns {missing}"
        raise ValueError(msg)

    # Check for duplicates
    dups = df.duplicated(subset=["date", "series_id"], keep=False)
    if dups.any():
        dup_count = dups.sum()
        msg = f"{series_id}: Found {dup_count} duplicate (date, series_id) pairs"
        raise ValueError(msg)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find first matching column name from candidates (pure helper function).

    Args:
        df: DataFrame to search
        candidates: List of possible column names to try

    Returns:
        First matching column name.

    Raises:
        ValueError: If no candidate found.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    msg = f"None of {candidates} found in columns: {list(df.columns)}"
    raise ValueError(msg)
