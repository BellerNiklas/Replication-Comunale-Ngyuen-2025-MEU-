"""Functions for fetching macroeconomic data from Eurostat API.

This module provides functionality to download time series data from Eurostat
for the MEU (MacroEconomic Uncertainty) database replication (Comunale & Nguyen 2025).
"""

from pathlib import Path
from time import sleep
from typing import Any

import eurostat
import pandas as pd

from template_project.config import BLD, SRC


# Dataset configurations for all Eurostat variables across 6 categories.
# Every non-time dimension must be pinned in filters; omitting one causes the
# API to return all combinations silently (dimension explosion).
#
# Variable specs follow Table 1 of Comunale & Nguyen (2025) appendix.
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # ========================================================================
    # Category 1: Industrial Production (12 variables)
    # ========================================================================
    "STS_INPR_M": {
        "category": 1,
        "category_name": "Industrial_production",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_IP",
        "variables": [
            {"id": "001", "nace_r2": "B-D", "desc": "IP total industry excl construction (B-D)"},
            {"id": "003", "nace_r2": "B", "desc": "IP mining and quarrying (B)"},
            {"id": "004", "nace_r2": "C", "desc": "IP manufacturing (C)"},
            {"id": "005", "nace_r2": "D", "desc": "IP electricity/gas/steam/aircon (D)"},
            {"id": "007", "nace_r2": "MIG_ING", "desc": "IP MIG intermediate goods"},
            {"id": "008", "nace_r2": "MIG_CAG", "desc": "IP MIG capital goods"},
            {"id": "009", "nace_r2": "MIG_DCOG", "desc": "IP MIG durable consumer goods"},
            {"id": "010", "nace_r2": "MIG_NDCOG", "desc": "IP MIG non-durable consumer goods"},
            {"id": "011", "nace_r2": "MIG_COG", "desc": "IP consumer goods industry"},
            {"id": "012", "nace_r2": "MIG_NRG_X_E", "desc": "IP MIG energy (excl section E)"},
        ],
    },

    # Construction production index (missing from previous config)
    "STS_COPR_M": {
        "category": 1,
        "category_name": "Industrial_production",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_IP",
        "variables": [
            {"id": "002", "nace_r2": "F", "desc": "Construction production index (F)"},
            {"id": "006", "nace_r2": "F41", "desc": "Construction: all buildings (F41)"},
        ],
    },

    # ========================================================================
    # Category 2: Labor Market (25 variables)
    # ========================================================================
    # STS_INLB_M carries three indicators via indic_bt: WAGE, EMP, HW.
    # Each indicator is crossed with NACE sections and MIG aggregates.
    "STS_INLB_M": {
        "category": 2,
        "category_name": "Labor_market_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_LAB",
        "variables": [
            # Wages & salaries (5 MIG breakdowns)
            {"id": "WAGE_001", "indic_bt": "WAGE", "nace_r2": "MIG_ING", "desc": "Wages MIG intermediate goods"},
            {"id": "WAGE_002", "indic_bt": "WAGE", "nace_r2": "MIG_CAG", "desc": "Wages MIG capital goods"},
            {"id": "WAGE_003", "indic_bt": "WAGE", "nace_r2": "MIG_DCOG", "desc": "Wages MIG durable consumer goods"},
            {"id": "WAGE_004", "indic_bt": "WAGE", "nace_r2": "MIG_NDCOG", "desc": "Wages MIG non-durable consumer goods"},
            {"id": "WAGE_005", "indic_bt": "WAGE", "nace_r2": "MIG_NRG", "desc": "Wages MIG energy"},
            # Employment index (10: NACE sections + MIG aggregates)
            {"id": "EMP_001", "indic_bt": "EMP", "nace_r2": "B", "desc": "Employment mining and quarrying"},
            {"id": "EMP_002", "indic_bt": "EMP", "nace_r2": "C", "desc": "Employment manufacturing"},
            {"id": "EMP_003", "indic_bt": "EMP", "nace_r2": "D", "desc": "Employment electricity/gas"},
            {"id": "EMP_004", "indic_bt": "EMP", "nace_r2": "B-E36", "desc": "Employment total industry excl construction"},
            {"id": "EMP_005", "indic_bt": "EMP", "nace_r2": "MIG_ING", "desc": "Employment MIG intermediate goods"},
            {"id": "EMP_006", "indic_bt": "EMP", "nace_r2": "MIG_CAG", "desc": "Employment MIG capital goods"},
            {"id": "EMP_007", "indic_bt": "EMP", "nace_r2": "MIG_DCOG", "desc": "Employment MIG durable consumer goods"},
            {"id": "EMP_008", "indic_bt": "EMP", "nace_r2": "MIG_NDCOG", "desc": "Employment MIG non-durable consumer goods"},
            {"id": "EMP_010", "indic_bt": "EMP", "nace_r2": "MIG_NRG", "desc": "Employment MIG energy"},
            # Hours worked index (10: same breakdowns as employment)
            {"id": "HW_001", "indic_bt": "HW", "nace_r2": "B", "desc": "Hours worked mining and quarrying"},
            {"id": "HW_002", "indic_bt": "HW", "nace_r2": "C", "desc": "Hours worked manufacturing"},
            {"id": "HW_003", "indic_bt": "HW", "nace_r2": "D", "desc": "Hours worked electricity/gas"},
            {"id": "HW_004", "indic_bt": "HW", "nace_r2": "B-E36", "desc": "Hours worked total industry excl construction"},
            {"id": "HW_005", "indic_bt": "HW", "nace_r2": "MIG_ING", "desc": "Hours worked MIG intermediate goods"},
            {"id": "HW_006", "indic_bt": "HW", "nace_r2": "MIG_CAG", "desc": "Hours worked MIG capital goods"},
            {"id": "HW_007", "indic_bt": "HW", "nace_r2": "MIG_DCOG", "desc": "Hours worked MIG durable consumer goods"},
            {"id": "HW_008", "indic_bt": "HW", "nace_r2": "MIG_NDCOG", "desc": "Hours worked MIG non-durable consumer goods"},
            {"id": "HW_010", "indic_bt": "HW", "nace_r2": "MIG_NRG", "desc": "Hours worked MIG energy"},
        ],
    },

    # Services labour: employment in accommodation & food (NACE I).
    # May not be available in STS_INLB_M (industry only), so use services dataset.
    "STS_SELB_M": {
        "category": 2,
        "category_name": "Labor_market_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_LAB",
        "variables": [
            {"id": "EMP_011", "indic_bt": "EMP", "nace_r2": "I", "desc": "Employment accommodation & food services (I)"},
        ],
    },

    "UNE_RT_M": {
        "category": 2,
        "category_name": "Labor_market_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SA",
            "unit": "PC_ACT",
            "age": "TOTAL",
            "sex": "T",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_UNEMP",
        "variables": [
            {"id": "001", "desc": "Unemployment rate total"},
        ],
    },

    # ========================================================================
    # Category 3: Prices (25 variables)
    # ========================================================================
    "STS_INPP_M": {
        "category": 3,
        "category_name": "Prices",
        "base_filters": {
            "geo": "DE",
            "s_adj": "NSA",
            "unit": "I21",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_PPI",
        "variables": [
            {"id": "001", "nace_r2": "C", "desc": "PPI manufacturing"},
            {"id": "002", "nace_r2": "B-E36", "desc": "PPI total industry excl construction"},
            {"id": "003", "nace_r2": "MIG_ING", "desc": "PPI MIG intermediate goods"},
            {"id": "004", "nace_r2": "MIG_CAG", "desc": "PPI MIG capital goods"},
            {"id": "005", "nace_r2": "MIG_DCOG", "desc": "PPI MIG durable consumer goods"},
            {"id": "006", "nace_r2": "MIG_NDCOG", "desc": "PPI MIG non-durable consumer goods"},
            {"id": "007", "nace_r2": "MIG_COG", "desc": "PPI consumer goods industry"},
            {"id": "008", "nace_r2": "MIG_NRG", "desc": "PPI MIG energy"},
        ],
    },

    # Import prices: CPA-based classification, pin indic_bt="PRC_IMP".
    "STS_INPI_M": {
        "category": 3,
        "category_name": "Prices",
        "base_filters": {
            "geo": "DE",
            "s_adj": "NSA",
            "unit": "I21",
            "indic_bt": "PRC_IMP",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_IMPPR",
        "variables": [
            {"id": "001", "cpa2_1": "CPA_C", "desc": "Import prices manufactured products"},
            {"id": "002", "cpa2_1": "CPA_MIG_ING", "desc": "Import prices MIG intermediate goods"},
            {"id": "003", "cpa2_1": "CPA_MIG_CAG", "desc": "Import prices MIG capital goods"},
            {"id": "004", "cpa2_1": "CPA_MIG_DCOG", "desc": "Import prices MIG durable consumer goods"},
            {"id": "005", "cpa2_1": "CPA_MIG_NDCOG", "desc": "Import prices MIG non-durable consumer goods"},
            {"id": "006", "cpa2_1": "CPA_MIG_COG", "desc": "Import prices consumer goods industry"},
            {"id": "007", "cpa2_1": "CPA_MIG_NRG_X_E", "desc": "Import prices MIG energy (excl section E)"},
        ],
    },

    "PRC_HICP_MIDX": {
        "category": 3,
        "category_name": "Prices",
        "base_filters": {
            "geo": "DE",
            "unit": "I15",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_HICP",
        "variables": [
            {"id": "001", "coicop": "CP00", "desc": "HICP overall index"},
            {"id": "002", "coicop": "NRG", "desc": "HICP energy"},
            {"id": "003", "coicop": "IGD", "desc": "HICP industrial goods"},
            {"id": "004", "coicop": "GD", "desc": "HICP goods"},
            {"id": "005", "coicop": "FOOD", "desc": "HICP food incl alcohol & tobacco"},
            {"id": "006", "coicop": "SERV", "desc": "HICP services"},
            {"id": "007", "coicop": "CP073", "desc": "HICP transport services"},
            {"id": "008", "coicop": "TOT_X_HOUS", "desc": "HICP excl housing/water/electricity/gas/other fuels"},
            {"id": "009", "coicop": "TOT_X_NRG_FOOD", "desc": "HICP excl energy & food"},
            {"id": "010", "coicop": "TOT_X_EDUC_HLTH_SPR", "desc": "HICP excl education/health/social protection"},
        ],
    },

    # ========================================================================
    # Category 4: Activity Indicators (17 variables)
    # ========================================================================
    # Building permits: appendix specifies SA (seasonally adjusted, not WD-adjusted),
    # but SA is rejected by the SDMX API for STS_COBP_M/DE. Only NSA and SCA are
    # available. Using SCA as closest match; documented deviation from appendix.
    "STS_COBP_M": {
        "category": 4,
        "category_name": "Activity_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "indic_bt": "BPRM_SQM",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_BPERM",
        "variables": [
            {"id": "001", "cpa2_1": "CPA_F41001_41002", "desc": "Building permits all buildings (floor area)"},
            {"id": "002", "cpa2_1": "CPA_F41001", "desc": "Building permits residential (floor area)"},
            {"id": "003", "cpa2_1": "CPA_F41002", "desc": "Building permits non-residential (floor area)"},
        ],
    },

    # Retail turnover: deflated (VOL_SLS), WD+SA.
    "STS_TRTU_M": {
        "category": 4,
        "category_name": "Activity_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "indic_bt": "VOL_SLS",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_RET",
        "variables": [
            {"id": "001", "nace_r2": "G47", "desc": "Retail trade incl fuel"},
            {"id": "002", "nace_r2": "G47_FOOD", "desc": "Retail food"},
            {"id": "003", "nace_r2": "G47_NFOOD_X_G473", "desc": "Retail non-food excl fuel"},
            {"id": "004", "nace_r2": "G473", "desc": "Retail automotive fuel"},
        ],
    },

    # Industry turnover index (missing from previous config).
    "STS_INTV_M": {
        "category": 4,
        "category_name": "Activity_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "indic_bt": "NETTUR",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_TURN",
        "variables": [
            {"id": "001", "nace_r2": "B", "desc": "Turnover mining and quarrying"},
            {"id": "002", "nace_r2": "C", "desc": "Turnover manufacturing"},
            # NOTE: Appendix wants B-E36 ("total industry excl construction") but
            # B-D, B-E, B-E36 are all rejected by the SDMX API for STS_INTV_M/DE.
            # B_C (mining + manufacturing) is the broadest working aggregate;
            # it excludes Energy (D), which is a documented deviation.
            {"id": "004", "nace_r2": "B_C", "desc": "Turnover total industry proxy (B_C, excl Energy D)"},
            {"id": "005", "nace_r2": "MIG_ING", "desc": "Turnover MIG intermediate goods"},
            {"id": "006", "nace_r2": "MIG_CAG", "desc": "Turnover MIG capital goods"},
            {"id": "007", "nace_r2": "MIG_DCOG", "desc": "Turnover MIG durable consumer goods"},
            {"id": "008", "nace_r2": "MIG_NDCOG", "desc": "Turnover MIG non-durable consumer goods"},
            {"id": "009", "nace_r2": "MIG_COG", "desc": "Turnover consumer goods industry"},
            {"id": "010", "nace_r2": "MIG_NRG_X_D_E", "desc": "Turnover MIG energy excl sections D & E"},
        ],
    },

    # Services turnover: accommodation & food (NACE I). Pin indic_bt to avoid explosion.
    "STS_SETU_M": {
        "category": 4,
        "category_name": "Activity_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
            "indic_bt": "NETTUR",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_TURN",
        "variables": [
            {"id": "003", "nace_r2": "I", "desc": "Turnover accommodation & food services (I)"},
        ],
    },

    # ========================================================================
    # Category 5: Trade (2 variables)
    # Source: ei_eteu27_2020_m (EU and euro area international trade)
    # Uses pre-aggregated monthly totals — no Comext API needed.
    # ========================================================================
    "ei_eteu27_2020_m": {
        "category": 5,
        "category_name": "Trade",
        "base_filters": {
            "geo": "DE",
            "indic": "ET-T",
            "partner": "WORLD",
            "unit": "MIO-EUR-SA",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_TRADE",
        "variables": [
            {"id": "001", "stk_flow": "IMP", "desc": "Total imports (world, million EUR, SA)"},
            {"id": "002", "stk_flow": "EXP", "desc": "Total exports (world, million EUR, SA)"},
        ],
    },

    # ========================================================================
    # Category 6: Sentiment (6 variables)
    # ========================================================================
    "EI_BSSI_M_R2": {
        "category": 6,
        "category_name": "Sentiment",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SA",
            "startPeriod": 2003,
        },
        "series_prefix": "DE_SENT",
        "variables": [
            {"id": "001", "indic": "BS-ESI-I", "desc": "Economic Sentiment Indicator"},
            {"id": "002", "indic": "BS-CSMCI-BAL", "desc": "Consumer Confidence Indicator"},
            {"id": "003", "indic": "BS-RCI-BAL", "desc": "Retail Confidence Indicator"},
            {"id": "004", "indic": "BS-CCI-BAL", "desc": "Construction Confidence Indicator"},
            {"id": "005", "indic": "BS-ICI-BAL", "desc": "Industrial Confidence Indicator"},
            {"id": "006", "indic": "BS-SCI-BAL", "desc": "Services Confidence Indicator"},
        ],
    },
}


def generate_all_variable_configs() -> list[dict[str, Any]]:
    """Generate full variable configuration list from DATASET_CONFIGS.

    Expands the nested DATASET_CONFIGS structure into individual variable
    configurations compatible with existing fetch functions.

    Returns:
        List of ~90 dictionaries, each containing:
        - series_id: Full series identifier (e.g., 'DE_IND_PROD_001')
        - dataset_id: Eurostat dataset code (e.g., 'STS_INPR_M')
        - filters: Complete filter dictionary (base + variable-specific)
        - description: Human-readable description
        - category: Category number (1-6)
        - category_name: Category name (e.g., 'Industrial_production')

    """
    all_configs = []

    for dataset_id, dataset_config in DATASET_CONFIGS.items():
        category = dataset_config["category"]
        category_name = dataset_config["category_name"]
        base_filters = dataset_config["base_filters"]
        series_prefix = dataset_config["series_prefix"]
        variables = dataset_config["variables"]

        for var in variables:
            # Build series_id
            series_id = f"{series_prefix}_{var['id']}"

            # Merge base filters with variable-specific filters
            filters = base_filters.copy()
            for key, value in var.items():
                if key not in ("id", "desc"):
                    filters[key] = value

            # Create config dict
            config = {
                "series_id": series_id,
                "dataset_id": dataset_id,
                "filters": filters,
                "description": var["desc"],
                "category": category,
                "category_name": category_name,
            }

            all_configs.append(config)

    return all_configs


_SPECIAL_FILTER_KEYS = {"startPeriod", "endPeriod"}


def validate_filters(dataset_id: str, filters: dict[str, str | list[str]]) -> None:
    """Check that all filter keys are valid dimensions for the dataset.

    Raises ValueError if any filter key is not a real dimension,
    which would cause Eurostat to silently return unfiltered data
    (dimension explosion).
    """
    raw_dims = eurostat.get_pars(dataset_id)
    if raw_dims is None:
        msg = f"{dataset_id}: get_pars() returned None — dataset may not exist"
        raise ValueError(msg)
    dims = set(raw_dims)
    unknown = [k for k in filters if k not in dims and k not in _SPECIAL_FILTER_KEYS]
    if unknown:
        msg = (
            f"{dataset_id}: unknown filter keys {unknown}. "
            f"Valid dims: {sorted(dims)}"
        )
        raise ValueError(msg)


def fetch_eurostat_series(
    dataset_id: str,
    filters: dict[str, str | list[str]],
    *,
    flags: bool = False,
) -> pd.DataFrame:
    """Fetch a single time series from Eurostat API.

    Args:
        dataset_id: Eurostat dataset code (e.g., 'STS_INPR_M').
        filters: Dictionary of filter parameters. Keys are dimension names
            (e.g., 'geo', 'nace_r2'), values are strings or lists.
        flags: If True, include data quality flags. Defaults to False.

    Returns:
        DataFrame in wide format with dimensions as columns and time periods
        as additional columns.

    Raises:
        ValueError: If dataset_id is empty or filters contain invalid keys.
        RuntimeError: If API request fails after all retries.

    """
    if not dataset_id:
        msg = "dataset_id cannot be empty"
        raise ValueError(msg)

    validate_filters(dataset_id, filters)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = eurostat.get_data_df(dataset_id, flags, filter_pars=filters)
            sleep(1)  # Rate limiting
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                msg = f"Failed to fetch {dataset_id} after {max_retries} attempts: {e}"
                raise RuntimeError(msg) from e
            sleep(2**attempt)  # Exponential backoff

    msg = "Unreachable code"
    raise RuntimeError(msg)


def transform_eurostat_data(
    data: pd.DataFrame,
    series_id: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Transform Eurostat data from wide to long format with metadata.

    Args:
        data: Wide format DataFrame from Eurostat API.
        series_id: Series identifier for metadata.
        metadata: Dictionary containing country_iso2, variable_name,
            category, category_name, etc.

    Returns:
        Long format DataFrame with columns: date, value, series_id,
        country_iso2, variable_name, category, category_name.

    Raises:
        ValueError: If data cannot be transformed.

    """
    # Identify non-time columns (metadata dimensions)
    non_time_cols = ["freq", "geo", "nace_r2", "s_adj", "unit", "coicop",
                     "age", "sex", "indic", "stk_flow", "partner",
                     "cpa2_1", "indic_bt", "indic_sb"]  # cpa2_1/indic_bt for prices, indic_sb for car registrations

    # Time columns are those not in the metadata list
    # Also exclude any columns ending with '_flag' as those are quality flags
    time_cols = [
        col for col in data.columns
        if col not in non_time_cols and not col.endswith("_flag")
    ]

    if not time_cols:
        msg = f"No time columns found in data for {series_id}"
        raise ValueError(msg)

    # Melt to long format
    id_vars = [col for col in data.columns if col in non_time_cols]
    melted = pd.melt(
        data,
        id_vars=id_vars,
        value_vars=time_cols,
        var_name="date",
        value_name="value",
    )

    # Keep only necessary columns
    result = pd.DataFrame({
        "date": melted["date"],
        "value": melted["value"],
        "series_id": series_id,
        "country_iso2": metadata.get("country_iso2", "DE"),
        "variable_name": metadata.get("variable_name", ""),
        "category": metadata.get("category"),
        "category_name": metadata.get("category_name", ""),
    })

    # Sort by date
    result = result.sort_values("date").reset_index(drop=True)

    return result


def validate_series_data(data: pd.DataFrame, series_id: str) -> None:
    """Validate that fetched data meets quality requirements.

    Args:
        data: DataFrame to validate.
        series_id: Series identifier for error messages.

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
    """Fetch single Eurostat series from spec dict.

    Args:
        spec: Must contain 'dataset_id', 'filters', 'series_id',
              'description', 'category', 'category_name'

    Returns:
        Standardized long DataFrame with columns:
        date | value | series_id | country_iso2 | variable_name |
        category | category_name

    Raises:
        ValueError: If spec is missing required keys or data validation fails.
        RuntimeError: If API request fails.

    """
    raw = fetch_eurostat_series(spec["dataset_id"], spec["filters"])
    metadata = {
        "variable_name": spec["description"],
        "country_iso2": spec["filters"].get("geo", "DE"),
        "category": spec["category"],
        "category_name": spec["category_name"],
    }
    transformed = transform_eurostat_data(raw, spec["series_id"], metadata)
    validate_series_data(transformed, spec["series_id"])
    return transformed


