"""Functions for fetching macroeconomic data from the OECD SDMX REST API.

Fetches two groups of variables used in Comunale & Nguyen (2025):
1. Category 6 (Sentiment): Business tendency surveys (4 sectors),
   consumer confidence, and CLI component (CS confidence indicator).
2. Category 7 (Financial): Long-term interest rate and share prices.

REST API: https://sdmx.oecd.org/public/rest/data/{flow}/{key}
Documentation: https://data-explorer.oecd.org/

Key formats confirmed via live API exploration:
  DF_BTS (Business Tendency Surveys):
    REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.ADJUSTMENT.TRANSFORMATION.TIME_HORIZ.METHODOLOGY
    e.g. DEU.M.BCICP.PB.F.Y._Z._Z.N

  DF_CS (Consumer Opinion Surveys):
    Same dimension order as DF_BTS.
    e.g. DEU.M.CCICP.PB._Z.Y._Z._Z.N

  DF_CLI (Composite Leading Indicators):
    REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.TRANSFORMATION.TIME_HORIZ.METHODOLOGY_DETAIL.METHODOLOGY
    e.g. DEU.M.LOCOCI.IX._Z.NOR.IX._Z.H

  DF_FINMARK (Financial Markets):
    Same dimension order as DF_BTS.
    e.g. DEU.M.IRLT.PA._Z._Z._Z._Z.N

Important:
  - OECD uses ISO 3166-1 alpha-3 country codes (DEU, not DE).
  - Always request format=csvfilewithlabels to get tidy CSV output.
  - Pin every dimension to avoid pulling thousands of unwanted series.
  - DF_CLI is version 4.1 (not 4.0 like the others).
"""

from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import requests

from template_project.config import BLD, SRC  # noqa: F401

OECD_BASE_URL: str = "https://sdmx.oecd.org/public/rest/data"

# Countries to fetch — OECD uses ISO 3166-1 alpha-3 codes.
# Extend to all 19 EA members as needed.
COUNTRIES: list[str] = ["DEU"]

# Mapping from OECD alpha-3 to ISO 3166-1 alpha-2 (used in output CSVs).
OECD_TO_ISO2: dict[str, str] = {
    "DEU": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL",
    "BEL": "BE", "AUT": "AT", "FIN": "FI", "GRC": "GR", "PRT": "PT",
    "IRL": "IE", "SVK": "SK", "SVN": "SI", "LTU": "LT", "LVA": "LV",
    "EST": "EE", "LUX": "LU", "CYP": "CY", "MLT": "MT",
}

# ============================================================================
# DATASET_CONFIGS
#
# Each top-level key is a descriptive label.
# Required fields per config:
#   flow          – OECD SDMX dataflow reference
#   category      – integer category number
#   category_name – string label for the output filename
#   series_prefix – prefix for auto-generated series IDs
#   variables     – list of {id, key_template, desc}
#                   key_template contains {country} replaced at fetch time.
#
# All keys marked CONFIRMED were verified against the live OECD SDMX API.
# ============================================================================

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # ========================================================================
    # BUSINESS TENDENCY SURVEYS (BTS) – Category 6, Sentiment
    # Flow: OECD.SDD.STES,DSD_STES@DF_BTS,4.0
    # Key: REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.ADJUSTMENT.
    #      TRANSFORMATION.TIME_HORIZ.METHODOLOGY
    # CONFIRMED: all four keys return 200 for DEU.
    # ========================================================================
    "BTS": {
        "flow": "OECD.SDD.STES,DSD_STES@DF_BTS,4.0",
        "category": 6,
        "category_name": "Sentiment",
        "series_prefix": "OECD_SENT",
        "variables": [
            {
                "id": "001",
                "key_template": "{country}.M.BCICP.PB.F.Y._Z._Z.N",
                "desc": "BTS Construction confidence indicator",
            },
            {
                "id": "002",
                "key_template": "{country}.M.BCICP.PB.G47.Y._Z._Z.N",
                "desc": "BTS Retail trade confidence indicator",
            },
            {
                "id": "003",
                "key_template": "{country}.M.BCICP.PB.C.Y._Z._Z.N",
                "desc": "BTS Manufacturing confidence indicator",
            },
            {
                "id": "004",
                "key_template": "{country}.M.BCICP.PB.GTU.Y._Z._Z.N",
                "desc": "BTS Services confidence indicator",
            },
        ],
    },
    # ========================================================================
    # CONSUMER OPINION SURVEYS (COS) – Category 6, Sentiment
    # Flow: OECD.SDD.STES,DSD_STES@DF_CS,4.0
    # CONFIRMED: key returns 200 for DEU.
    # ========================================================================
    "COS": {
        "flow": "OECD.SDD.STES,DSD_STES@DF_CS,4.0",
        "category": 6,
        "category_name": "Sentiment",
        "series_prefix": "OECD_SENT",
        "variables": [
            {
                "id": "005",
                "key_template": "{country}.M.CCICP.PB._Z.Y._Z._Z.N",
                "desc": "Consumer confidence indicator",
            },
        ],
    },
    # ========================================================================
    # COMPOSITE LEADING INDICATORS (CLI) – Category 6, Sentiment
    # Flow: OECD.SDD.STES,DSD_STES@DF_CLI,4.1  (note: version 4.1, not 4.0)
    # MEASURE=LOCOCI (Consumer confidence as CLI component)
    # TRANSFORMATION=NOR (Normalised), METHODOLOGY=H (OECD harmonised)
    # CONFIRMED: key returns 200 for DEU.
    # ========================================================================
    "CLI": {
        "flow": "OECD.SDD.STES,DSD_STES@DF_CLI,4.1",
        "category": 6,
        "category_name": "Sentiment",
        "series_prefix": "OECD_SENT",
        "variables": [
            {
                "id": "006",
                "key_template": "{country}.M.LOCOCI.IX._Z.NOR.IX._Z.H",
                "desc": "CLI component: CS confidence indicator (normalised)",
            },
        ],
    },
    # ========================================================================
    # FINANCIAL MARKETS (FINMARK) – Category 7, Financial
    # Flow: OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0
    # IRLT = Long-term interest rates (percent per annum)
    # SHARE = Share prices (index, base 2015=100)
    # CONFIRMED: both keys return 200 for DEU.
    # ========================================================================
    "FINMARK": {
        "flow": "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0",
        "category": 7,
        "category_name": "Financial",
        "series_prefix": "OECD_FIN",
        "variables": [
            {
                "id": "001",
                "key_template": "{country}.M.IRLT.PA._Z._Z._Z._Z.N",
                "desc": "Long-term interest rate",
            },
            {
                "id": "002",
                "key_template": "{country}.M.SHARE.IX._Z._Z._Z._Z.N",
                "desc": "Share prices (all shares/broad, index 2015=100)",
            },
        ],
    },
}


def generate_all_variable_configs() -> list[dict[str, Any]]:
    """Expand DATASET_CONFIGS x COUNTRIES into a flat list of per-variable dicts.

    Returns:
        List of dicts, each containing:
        - series_id:     Full identifier (e.g. 'DE_OECD_SENT_001')
        - flow:          OECD SDMX flow reference
        - key:           SDMX key with country substituted
        - description:   Human-readable label
        - category:      Category integer
        - category_name: Category string
        - country_iso2:  ISO 3166-1 alpha-2 code
        - country_iso3:  ISO 3166-1 alpha-3 code (OECD)

    """
    all_configs: list[dict[str, Any]] = []

    for country_iso3 in COUNTRIES:
        country_iso2 = OECD_TO_ISO2[country_iso3]

        for _label, ds_cfg in DATASET_CONFIGS.items():
            for var in ds_cfg["variables"]:
                series_id = f"{country_iso2}_{ds_cfg['series_prefix']}_{var['id']}"
                key = var["key_template"].format(country=country_iso3)

                all_configs.append(
                    {
                        "series_id": series_id,
                        "flow": ds_cfg["flow"],
                        "key": key,
                        "description": var["desc"],
                        "category": ds_cfg["category"],
                        "category_name": ds_cfg["category_name"],
                        "country_iso2": country_iso2,
                        "country_iso3": country_iso3,
                    }
                )

    return all_configs


def fetch_oecd_series(flow: str, key: str) -> pd.DataFrame:
    """Fetch a single time series from the OECD SDMX REST API.

    Args:
        flow: OECD SDMX dataflow reference
              (e.g. 'OECD.SDD.STES,DSD_STES@DF_BTS,4.0').
        key:  SDMX key string (e.g. 'DEU.M.BCICP.PB.F.Y._Z._Z.N').

    Returns:
        DataFrame as returned by the OECD API (CSV with labels).

    Raises:
        ValueError: If flow or key is empty.
        RuntimeError: If the API request fails after all retries.

    """
    if not flow:
        msg = "flow cannot be empty"
        raise ValueError(msg)
    if not key:
        msg = "key cannot be empty"
        raise ValueError(msg)

    url = f"{OECD_BASE_URL}/{flow}/{key}"
    params: dict[str, str] = {
        "startPeriod": "2003-01",
        "dimensionAtObservation": "AllDimensions",
        "format": "csvfilewithlabels",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))
            sleep(1)  # Rate limiting
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                msg = f"Failed to fetch {flow}/{key} after {max_retries} attempts: {e}"
                raise RuntimeError(msg) from e
            sleep(2**attempt)

    msg = "Unreachable code"
    raise RuntimeError(msg)


def transform_oecd_data(
    data: pd.DataFrame,
    series_id: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Transform OECD CSV response to standardised long format.

    The OECD 'csvfilewithlabels' format returns one row per observation
    with TIME_PERIOD and OBS_VALUE as key columns.

    Args:
        data:      Raw DataFrame from fetch_oecd_series().
        series_id: Series identifier (e.g. 'DE_OECD_SENT_001').
        metadata:  Dict containing country_iso2, variable_name, category,
                   category_name.

    Returns:
        Long-format DataFrame with columns:
        date | value | series_id | country_iso2 | variable_name |
        category | category_name

    Raises:
        ValueError: If TIME_PERIOD or OBS_VALUE columns are missing.

    """
    if "TIME_PERIOD" not in data.columns:
        msg = f"No TIME_PERIOD column in OECD response for {series_id}"
        raise ValueError(msg)
    if "OBS_VALUE" not in data.columns:
        msg = f"No OBS_VALUE column in OECD response for {series_id}"
        raise ValueError(msg)

    result = pd.DataFrame(
        {
            "date": data["TIME_PERIOD"].astype(str),
            "value": pd.to_numeric(data["OBS_VALUE"], errors="coerce"),
            "series_id": series_id,
            "country_iso2": metadata.get("country_iso2", "DE"),
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


def validate_series_data(data: pd.DataFrame, series_id: str) -> None:
    """Validate that fetched series meets minimum quality requirements.

    Args:
        data:      Transformed DataFrame to validate.
        series_id: Series identifier used in error messages.

    Raises:
        ValueError: If data is empty or all values are NaN.

    """
    if data.empty:
        msg = f"No data fetched for {series_id}"
        raise ValueError(msg)
    if "value" in data.columns and data["value"].isna().all():
        msg = f"All values are NaN for {series_id}"
        raise ValueError(msg)


def fetch_category_variables(
    category_num: int,
    all_configs: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch all variables belonging to a given category number.

    Args:
        category_num: Target category (e.g. 6 for sentiment, 7 for financial).
        all_configs:  Full flat config list from generate_all_variable_configs().

    Returns:
        Tuple (successful_data, failed_series):
        - successful_data: dict mapping series_id -> transformed DataFrame
        - failed_series:   dict mapping series_id -> error message

    """
    category_configs = [c for c in all_configs if c["category"] == category_num]

    successful_data: dict[str, pd.DataFrame] = {}
    failed_series: dict[str, str] = {}

    for i, cfg in enumerate(category_configs, 1):
        series_id = cfg["series_id"]
        print(f"  [{i}/{len(category_configs)}] {series_id}: {cfg['description']}")

        try:
            raw = fetch_oecd_series(cfg["flow"], cfg["key"])
            metadata = {
                "country_iso2": cfg["country_iso2"],
                "variable_name": cfg["description"],
                "category": cfg["category"],
                "category_name": cfg["category_name"],
            }
            transformed = transform_oecd_data(raw, series_id, metadata)
            validate_series_data(transformed, series_id)
            successful_data[series_id] = transformed
            print(f"    SUCCESS: {len(transformed)} rows")

        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            print(f"    FAILED: {error_msg[:100]}")
            failed_series[series_id] = error_msg

    return successful_data, failed_series


def concatenate_category_data(
    category_dfs: dict[str, pd.DataFrame],
    category_num: int,
    category_name: str,
) -> pd.DataFrame:
    """Concatenate all DataFrames for a category into a single DataFrame.

    Args:
        category_dfs:  Dict mapping series_id -> DataFrame.
        category_num:  Category number (used in error message only).
        category_name: Category name (used in error message only).

    Returns:
        Concatenated DataFrame sorted by date then series_id.

    Raises:
        ValueError: If category_dfs is empty.

    """
    if not category_dfs:
        msg = f"No successful data for category {category_num} ({category_name})"
        raise ValueError(msg)

    concatenated = pd.concat(category_dfs.values(), ignore_index=True)
    return concatenated.sort_values(["date", "series_id"]).reset_index(drop=True)


def save_category_csv(
    data: pd.DataFrame,
    category_num: int,
    category_name: str,
    output_dir: Path,
) -> Path:
    """Save category data to a CSV file following project naming convention.

    Args:
        data:          DataFrame to save.
        category_num:  Category number used in the filename.
        category_name: Category name used in the filename.
        output_dir:    Directory where the CSV will be written.

    Returns:
        Path to the saved CSV file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"category_{category_num}_{category_name}.csv"
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    return output_path


def fetch_all_oecd_variables(
    output_dir: Path,
) -> dict[str, Path]:
    """Fetch all OECD variables and save category CSV files.

    Args:
        output_dir: Directory where category CSV files will be saved.

    Returns:
        Dict mapping 'category_{N}' keys to saved CSV file paths.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_configs = generate_all_variable_configs()
    results: dict[str, Path] = {}
    summary_stats: dict[int, dict[str, Any]] = {}

    category_nums = sorted({c["category"] for c in all_configs})
    category_names = {c["category"]: c["category_name"] for c in all_configs}

    for category_num in category_nums:
        cat_name = category_names[category_num]
        cat_configs = [c for c in all_configs if c["category"] == category_num]
        print(
            f"\nFetching Category {category_num}: {cat_name} "
            f"({len(cat_configs)} variables)"
        )

        successful_data, failed_series = fetch_category_variables(
            category_num, all_configs
        )

        total = len(cat_configs)
        success_count = len(successful_data)
        fail_count = len(failed_series)
        summary_stats[category_num] = {
            "total": total,
            "success": success_count,
            "failed": fail_count,
            "failed_series": failed_series,
            "category_name": cat_name,
        }

        if successful_data:
            cat_df = concatenate_category_data(successful_data, category_num, cat_name)
            output_path = save_category_csv(cat_df, category_num, cat_name, output_dir)
            results[f"category_{category_num}"] = output_path
            print(
                f"\n  Category {category_num}: {success_count}/{total} succeeded "
                f"({fail_count} failed)"
            )
            print(f"    Saved: {output_path.name} ({len(cat_df):,} rows)")
        else:
            print(f"\n  Category {category_num}: all {total} variables failed!")

    # Final summary
    print("\n" + "=" * 70)
    print("OECD FETCH SUMMARY")
    print("=" * 70)
    total_vars = sum(s["total"] for s in summary_stats.values())
    total_success = sum(s["success"] for s in summary_stats.values())
    total_failed = sum(s["failed"] for s in summary_stats.values())
    print(f"\nOverall: {total_success}/{total_vars} variables fetched successfully")
    if total_failed > 0:
        print(f"\nFailed ({total_failed}):")
        for cat_num, stats in summary_stats.items():
            for sid, err in stats["failed_series"].items():
                print(f"  [{cat_num}] {sid}: {err[:80]}")

    return results


def main() -> None:
    """Fetch OECD data and save to bld/data/raw/oecd/."""
    output_dir = BLD / "data" / "raw" / "oecd"

    print("Fetching OECD data...")
    print(f"Countries: {COUNTRIES}")
    print(f"Output directory: {output_dir}\n")

    fetch_all_oecd_variables(output_dir)


if __name__ == "__main__":
    main()
