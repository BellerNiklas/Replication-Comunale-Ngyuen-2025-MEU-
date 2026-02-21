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


def fetch_one(spec: dict[str, Any]) -> pd.DataFrame:
    """Fetch single OECD series from spec dict.

    Args:
        spec: Must contain 'flow', 'key', 'series_id',
              'description', 'category', 'category_name', 'country_iso2'

    Returns:
        Standardized long DataFrame with columns:
        date | value | series_id | country_iso2 | variable_name |
        category | category_name

    Raises:
        ValueError: If spec is missing required keys or data validation fails.
        RuntimeError: If API request fails.

    """
    raw = fetch_oecd_series(spec["flow"], spec["key"])
    metadata = {
        "variable_name": spec["description"],
        "country_iso2": spec.get("country_iso2", "DE"),
        "category": spec["category"],
        "category_name": spec["category_name"],
    }
    transformed = transform_oecd_data(raw, spec["series_id"], metadata)
    validate_series_data(transformed, spec["series_id"])
    return transformed


