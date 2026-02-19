"""Functions for fetching macroeconomic data from the BIS Statistics API.

Fetches country-specific Category 7 (Financial) variables used in
Comunale & Nguyen (2025):
  - Nominal Effective Exchange Rate (NEER)
  - Long-term interest rate

REST API: https://stats.bis.org/api/v1/data/{dataset}/{key}
Documentation: https://www.bis.org/statistics/api.htm

BIS SDMX key formats (confirmed via live API exploration):
  WS_EER:      FREQ.EER_TYPE.EER_BASKET.REF_AREA
               e.g. M.N.B.DE  (Monthly, Nominal, Broad, Germany)
  WS_LONG_CPI: FREQ.REF_AREA
               e.g. M.DE
               Response contains two UNIT_MEASURE codes:
                 628 = CPI index (filtered out)
                 771 = long-term interest rate (retained)

Note: The dataset was previously named WS_NEER; the current name is WS_EER.

Currently fetches for Germany (DE) as a proof-of-concept; extend
`COUNTRIES` list to add more euro area member states.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import requests

from template_project.config import BLD, SRC  # noqa: F401

BIS_BASE_URL: str = "https://stats.bis.org/api/v1/data"

# Countries to fetch – BIS REF_AREA uses ISO 3166-1 alpha-2 codes.
# Extend to all 19 EA members as needed.
COUNTRIES: list[str] = ["DE"]

# ============================================================================
# DATASET_CONFIGS
#
# Each top-level key is a descriptive label.
# Required fields per config:
#   dataset           – BIS dataset ID
#   category          – integer category number
#   category_name     – string label used in the output filename
#   series_prefix     – prefix for auto-generated series IDs
#   unit_measure_filter – optional int; if set, filter rows by UNIT_MEASURE
#   variables         – list of {id, key_template, desc}
#                       key_template may contain {country} replaced at fetch time.
#
# All keys confirmed via live BIS API queries.
# ============================================================================

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # ========================================================================
    # NOMINAL EFFECTIVE EXCHANGE RATE (NEER) – Category 7, Financial
    # Source: BIS WS_EER dataset (formerly WS_NEER)
    # CONFIRMED: M.N.B.{country} returns 200 for DE.
    # Key format: FREQ.EER_TYPE.EER_BASKET.REF_AREA
    #   EER_TYPE: N=Nominal, R=Real
    #   EER_BASKET: B=Broad (64 economies), N=Narrow (27 economies)
    # Paper uses broad nominal, which is the standard NEER for EA members.
    # ========================================================================
    "NEER": {
        "dataset": "WS_EER",
        "category": 7,
        "category_name": "Financial",
        "series_prefix": "BIS_NEER",
        "unit_measure_filter": None,
        "variables": [
            {
                "id": "001",
                "key_template": "M.N.B.{country}",
                "desc": "Nominal Effective Exchange Rate – broad (index 2020=100)",
            },
        ],
    },
}

# ============================================================================
# TEST CONFIG – single confirmed series for smoke test
# ============================================================================

TEST_CONFIG: dict[str, Any] = {
    "dataset": "WS_EER",
    "key": "M.N.B.DE",
    "desc": "NEER – Germany, nominal broad (index 2020=100)",
    "unit_measure_filter": None,
}


# ============================================================================
# generate_all_variable_configs
# ============================================================================


def generate_all_variable_configs() -> list[dict[str, Any]]:
    """Expand DATASET_CONFIGS into one entry per (dataset, variable, country).

    Returns:
        List of flat config dicts ready to pass to fetch/transform functions.
    """
    configs: list[dict[str, Any]] = []
    for config_name, config in DATASET_CONFIGS.items():
        for var in config["variables"]:
            for country in COUNTRIES:
                country_iso2 = country  # BIS already uses ISO alpha-2
                key = var["key_template"].format(country=country)
                series_id = (
                    f"{config['series_prefix']}_{var['id']}_{country_iso2}"
                )
                configs.append(
                    {
                        "config_name": config_name,
                        "dataset": config["dataset"],
                        "category": config["category"],
                        "category_name": config["category_name"],
                        "series_prefix": config["series_prefix"],
                        "unit_measure_filter": config.get("unit_measure_filter"),
                        "series_id": series_id,
                        "key": key,
                        "country": country,
                        "country_iso2": country_iso2,
                        "variable_name": var["desc"],
                        "var_id": var["id"],
                    }
                )
    return configs


# ============================================================================
# fetch_bis_series
# ============================================================================


def fetch_bis_series(dataset: str, key: str) -> pd.DataFrame:
    """Fetch a single BIS series and return the raw CSV as a DataFrame.

    Args:
        dataset: BIS dataset ID (e.g. "WS_EER").
        key: SDMX key string (e.g. "M.N.B.DE").

    Returns:
        Raw DataFrame from the BIS API response.

    Raises:
        requests.HTTPError: On non-2xx response.
        ValueError: If the response is empty or cannot be parsed.
    """
    url = f"{BIS_BASE_URL}/{dataset}/{key}"
    params: dict[str, str] = {
        "startPeriod": "2003-01",
        "format": "csv",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    if not response.text.strip():
        msg = f"Empty response for BIS {dataset}/{key}"
        raise ValueError(msg)
    return pd.read_csv(StringIO(response.text))


# ============================================================================
# transform_bis_data
# ============================================================================


def transform_bis_data(
    raw: pd.DataFrame,
    series_id: str,
    metadata: dict[str, Any],
    *,
    unit_measure_filter: int | None = None,
) -> pd.DataFrame:
    """Transform raw BIS API response into standardised long format.

    The BIS CSV format contains TIME_PERIOD and OBS_VALUE columns along
    with dimension label columns.

    For WS_LONG_CPI, the response contains two measure types:
      UNIT_MEASURE=628 → CPI index (discarded)
      UNIT_MEASURE=771 → long-term interest rate (retained)
    Pass unit_measure_filter=771 to keep only the rate rows.

    Args:
        raw: Raw DataFrame from fetch_bis_series.
        series_id: Unique series identifier string.
        metadata: Dict with keys: country_iso2, variable_name, category,
            category_name.
        unit_measure_filter: If not None, keep only rows where UNIT_MEASURE
            equals this value.

    Returns:
        Long-format DataFrame with columns:
            date | value | series_id | country_iso2 |
            variable_name | category | category_name
    """
    # BIS CSV format: TIME_PERIOD + OBS_VALUE
    time_col = "TIME_PERIOD"
    value_col = "OBS_VALUE"

    # Handle capitalisation variants
    col_upper = {c.upper(): c for c in raw.columns}
    time_col = col_upper.get("TIME_PERIOD", time_col)
    value_col = col_upper.get("OBS_VALUE", value_col)

    if time_col not in raw.columns or value_col not in raw.columns:
        available = list(raw.columns)
        msg = (
            f"Expected columns 'TIME_PERIOD' and 'OBS_VALUE' not found in "
            f"BIS response for series {series_id}. Available: {available}"
        )
        raise ValueError(msg)

    filtered = raw
    if unit_measure_filter is not None and "UNIT_MEASURE" in raw.columns:
        filtered = raw[raw["UNIT_MEASURE"] == unit_measure_filter]

    result = pd.DataFrame(
        {
            "date": filtered[time_col].astype(str),
            "value": pd.to_numeric(filtered[value_col], errors="coerce"),
            "series_id": series_id,
            "country_iso2": metadata.get("country_iso2", ""),
            "variable_name": metadata.get("variable_name", ""),
            "category": metadata.get("category"),
            "category_name": metadata.get("category_name", ""),
        }
    )
    result = (
        result[result["date"] >= "2003"]
        .dropna(subset=["value"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return result


# ============================================================================
# validate_series_data
# ============================================================================


def validate_series_data(data: pd.DataFrame, series_id: str) -> None:
    """Validate that a fetched series has usable data.

    Args:
        data: Transformed long-format DataFrame.
        series_id: Series identifier for error messages.

    Raises:
        ValueError: If data is empty or entirely NaN.
    """
    if data.empty:
        msg = f"No data returned for series {series_id}"
        raise ValueError(msg)
    if data["value"].isna().all():
        msg = f"All values are NaN for series {series_id}"
        raise ValueError(msg)


# ============================================================================
# fetch_category_variables
# ============================================================================


def fetch_category_variables(
    category_num: int,
    all_configs: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch all BIS variables belonging to a given category.

    Args:
        category_num: Integer category to fetch.
        all_configs: Full list from generate_all_variable_configs().

    Returns:
        Tuple of (series_data, errors):
          series_data – dict mapping series_id to transformed DataFrame.
          errors      – dict mapping series_id to error message string.
    """
    cat_configs = [c for c in all_configs if c["category"] == category_num]
    series_data: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for cfg in cat_configs:
        series_id = cfg["series_id"]
        print(f"  Fetching {series_id}: {cfg['variable_name']}")
        try:
            raw = fetch_bis_series(cfg["dataset"], cfg["key"])
            transformed = transform_bis_data(
                raw,
                series_id,
                {
                    "country_iso2": cfg["country_iso2"],
                    "variable_name": cfg["variable_name"],
                    "category": cfg["category"],
                    "category_name": cfg["category_name"],
                },
                unit_measure_filter=cfg.get("unit_measure_filter"),
            )
            validate_series_data(transformed, series_id)
            series_data[series_id] = transformed
            print(f"    OK: {len(transformed)} rows")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            errors[series_id] = msg
            print(f"    FAILED: {msg}")
        sleep(0.5)  # polite rate-limiting

    return series_data, errors


# ============================================================================
# concatenate_category_data
# ============================================================================


def concatenate_category_data(
    series_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate per-series DataFrames into a single long DataFrame.

    Args:
        series_data: Dict mapping series_id to transformed DataFrame.

    Returns:
        Single long-format DataFrame sorted by series_id then date.
    """
    if not series_data:
        return pd.DataFrame()
    combined = pd.concat(list(series_data.values()), ignore_index=True)
    return combined.sort_values(["series_id", "date"]).reset_index(drop=True)


# ============================================================================
# save_category_csv
# ============================================================================


def save_category_csv(
    data: pd.DataFrame,
    category_num: int,
    category_name: str,
    output_dir: Path,
) -> Path:
    """Save a category-level DataFrame to CSV.

    Args:
        data: Long-format DataFrame for the category.
        category_num: Integer category number (used in filename).
        category_name: String name (used in filename).
        output_dir: Directory to write the file.

    Returns:
        Path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"category_{category_num}_{category_name}.csv"
    path = output_dir / filename
    data.to_csv(path, index=False)
    return path


# ============================================================================
# test_fetch
# ============================================================================


def test_fetch() -> bool:
    """Run a minimal smoke test against the BIS API.

    Fetches NEER (broad nominal) for Germany – a single confirmed series.

    Returns:
        True if the test passes, False otherwise.
    """
    print("Running BIS API smoke test...")
    print(f"  Dataset: {TEST_CONFIG['dataset']}")
    print(f"  Series: {TEST_CONFIG['desc']}")
    print(f"  Key:    {TEST_CONFIG['key']}")
    try:
        raw = fetch_bis_series(TEST_CONFIG["dataset"], TEST_CONFIG["key"])
        print(f"  Raw columns: {list(raw.columns)}")
        transformed = transform_bis_data(
            raw,
            "BIS_NEER_001_DE_TEST",
            {
                "country_iso2": "DE",
                "variable_name": TEST_CONFIG["desc"],
                "category": 7,
                "category_name": "Financial",
            },
            unit_measure_filter=TEST_CONFIG.get("unit_measure_filter"),
        )
        validate_series_data(transformed, "BIS_NEER_001_DE_TEST")
        print(f"  Rows: {len(transformed)}")
        print(f"  Date range: {transformed['date'].min()} – {transformed['date'].max()}")
        print(f"  Value range: {transformed['value'].min():.2f} – {transformed['value'].max():.2f}")
        print("  SMOKE TEST PASSED")
        return True  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        print(f"  SMOKE TEST FAILED: {e}")
        return False


# ============================================================================
# fetch_all_bis_variables
# ============================================================================


def fetch_all_bis_variables(
    output_dir: Path,
    *,
    test_mode: bool = False,
) -> dict[str, Path]:
    """Fetch all BIS variables and save category-level CSV files.

    Args:
        output_dir: Root directory for output files (bld/data/raw/bis/).
        test_mode: If True, run only the smoke test and return.

    Returns:
        Dict mapping category filenames to their output Paths.
    """
    if test_mode:
        success = test_fetch()
        if not success:
            print("Aborting: smoke test failed.")
        return {}

    all_configs = generate_all_variable_configs()
    categories = sorted({c["category"] for c in all_configs})

    output_paths: dict[str, Path] = {}
    all_errors: dict[str, str] = {}

    for cat_num in categories:
        cat_name = next(
            c["category_name"] for c in all_configs if c["category"] == cat_num
        )
        print(f"\nFetching category {cat_num}: {cat_name}")
        series_data, errors = fetch_category_variables(cat_num, all_configs)
        all_errors.update(errors)

        if series_data:
            combined = concatenate_category_data(series_data)
            path = save_category_csv(combined, cat_num, cat_name, output_dir)
            output_paths[f"category_{cat_num}_{cat_name}"] = path
            print(f"  Saved: {path} ({len(combined)} rows)")
        else:
            print(f"  No data saved for category {cat_num}.")

    if all_errors:
        print(f"\nSummary: {len(all_errors)} series failed:")
        for sid, msg in all_errors.items():
            print(f"  {sid}: {msg}")
    else:
        print("\nAll BIS series fetched successfully.")

    return output_paths


# ============================================================================
# main
# ============================================================================


def main() -> None:
    """Entry point for running the BIS fetcher as a script.

    Usage:
        pixi run python -m template_project.data_fetch.bis          # full fetch
        pixi run python -m template_project.data_fetch.bis --test   # smoke test
    """
    output_dir = BLD / "data" / "raw" / "bis"
    test_mode = "--test" in sys.argv
    fetch_all_bis_variables(output_dir, test_mode=test_mode)


if __name__ == "__main__":
    main()
