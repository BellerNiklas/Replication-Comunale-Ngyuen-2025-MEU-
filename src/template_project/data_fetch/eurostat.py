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


# Dataset configurations for all ~70 Eurostat variables across 6 categories
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # ========================================================================
    # Category 1: Industrial Production (~13 variables)
    # ========================================================================
    "STS_INPR_M": {
        "category": 1,
        "category_name": "Industrial_production",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
        },
        "series_prefix": "DE_IND_PROD",
        "variables": [
            {"id": "001", "nace_r2": "B-D", "desc": "Total Industry (NACE B-D)"},
            {"id": "002", "nace_r2": "C", "desc": "Manufacturing (NACE C)"},
            {"id": "003", "nace_r2": "B", "desc": "Mining and Quarrying (NACE B)"},
            {"id": "004", "nace_r2": "D", "desc": "Energy (NACE D)"},
            {"id": "005", "nace_r2": "C10_C12", "desc": "Food Products (NACE C10-C12)"},
            {"id": "006", "nace_r2": "C13_C15", "desc": "Textiles (NACE C13-C15)"},
            {"id": "007", "nace_r2": "C16_C18", "desc": "Wood and Paper (NACE C16-C18)"},
            {"id": "008", "nace_r2": "C20", "desc": "Chemicals (NACE C20)"},
            {"id": "009", "nace_r2": "C24", "desc": "Basic Metals (NACE C24)"},
            {"id": "010", "nace_r2": "C26", "desc": "Electronics (NACE C26)"},
            {"id": "011", "nace_r2": "C28", "desc": "Machinery (NACE C28)"},
            {"id": "012", "nace_r2": "C29", "desc": "Transport Equipment (NACE C29)"},
            {"id": "013", "nace_r2": "MIG_CAG", "desc": "Capital Goods (MIG)"},
        ],
    },

    # ========================================================================
    # Category 2: Labor Market (~21 variables)
    # ========================================================================
    "UNE_RT_M": {
        "category": 2,
        "category_name": "Labor_market_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SA",
            "unit": "PC_ACT",
        },
        "series_prefix": "DE_LAB_UNEMP",
        "variables": [
            {"id": "001", "age": "TOTAL", "sex": "T", "desc": "Unemployment Rate - Total"},
            {"id": "002", "age": "Y15-24", "sex": "T", "desc": "Unemployment Rate - Youth (15-24)"},
            {"id": "003", "age": "Y25-74", "sex": "T", "desc": "Unemployment Rate - Prime Age (25-74)"},
            {"id": "004", "age": "TOTAL", "sex": "M", "desc": "Unemployment Rate - Male"},
            {"id": "005", "age": "TOTAL", "sex": "F", "desc": "Unemployment Rate - Female"},
        ],
    },

    # Note: STS_INEM_M and STS_INHW_M have known API issues - included for completeness
    # but may fail during fetch (will be skipped gracefully)
    "STS_INLB_M": {
        "category": 2,
        "category_name": "Labor_market_indicators",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SCA",
            "unit": "I21",
        },
        "series_prefix": "DE_LAB_COST",
        "variables": [
            {"id": "001", "nace_r2": "B-E", "desc": "Labor Cost - Industry (NACE B-E)"},
            {"id": "002", "nace_r2": "C", "desc": "Labor Cost - Manufacturing (NACE C)"},
            {"id": "003", "nace_r2": "G-N", "desc": "Labor Cost - Services (NACE G-N)"},
        ],
    },

    # ========================================================================
    # Category 3: Prices (~22 variables)
    # ========================================================================
    "STS_INPP_M": {
        "category": 3,
        "category_name": "Prices",
        "base_filters": {
            "geo": "DE",
            "unit": "I21",
            "s_adj": "NSA",
        },
        "series_prefix": "DE_PRICE_PPI",
        "variables": [
            {"id": "001", "nace_r2": "B-D", "desc": "PPI - Total Industry (NACE B-D)"},
            {"id": "002", "nace_r2": "C", "desc": "PPI - Manufacturing (NACE C)"},
            {"id": "003", "nace_r2": "B", "desc": "PPI - Mining (NACE B)"},
            {"id": "004", "nace_r2": "MIG_CAG", "desc": "PPI - Capital Goods (MIG)"},
            {"id": "005", "nace_r2": "MIG_ING", "desc": "PPI - Intermediate Goods (MIG)"},
            {"id": "006", "nace_r2": "MIG_COG", "desc": "PPI - Consumer Goods (MIG)"},
            {"id": "007", "nace_r2": "MIG_NRG", "desc": "PPI - Energy (MIG)"},
        ],
    },

    "STS_INPI_M": {
        "category": 3,
        "category_name": "Prices",
        "base_filters": {
            "geo": "DE",
            "unit": "I21",
            "s_adj": "NSA",
        },
        "series_prefix": "DE_PRICE_IMP",
        "variables": [
            {"id": "001", "nace_r2": "MIG_TOT", "desc": "Import Price - Total (MIG)"},
            {"id": "002", "nace_r2": "MIG_CAG", "desc": "Import Price - Capital Goods (MIG)"},
            {"id": "003", "nace_r2": "MIG_ING", "desc": "Import Price - Intermediate Goods (MIG)"},
            {"id": "004", "nace_r2": "MIG_COG", "desc": "Import Price - Consumer Goods (MIG)"},
        ],
    },

    "PRC_HICP_MIDX": {
        "category": 3,
        "category_name": "Prices",
        "base_filters": {
            "geo": "DE",
            "unit": "I15",
        },
        "series_prefix": "DE_PRICE_HICP",
        "variables": [
            {"id": "001", "coicop": "CP00", "desc": "HICP - Overall Index (CP00)"},
            {"id": "002", "coicop": "NRG", "desc": "HICP - Energy (NRG)"},
            {"id": "003", "coicop": "FOOD", "desc": "HICP - Food (FOOD)"},
            {"id": "004", "coicop": "CP01", "desc": "HICP - Food and Beverages (CP01)"},
            {"id": "005", "coicop": "CP02", "desc": "HICP - Alcohol and Tobacco (CP02)"},
            {"id": "006", "coicop": "CP03", "desc": "HICP - Clothing (CP03)"},
            {"id": "007", "coicop": "CP04", "desc": "HICP - Housing (CP04)"},
            {"id": "008", "coicop": "CP07", "desc": "HICP - Transport (CP07)"},
            {"id": "009", "coicop": "SERV", "desc": "HICP - Services (SERV)"},
            {"id": "010", "coicop": "TOT_X_NRG_FOOD", "desc": "HICP - Core excl. Energy/Food"},
            {"id": "011", "coicop": "IGD", "desc": "HICP - Industrial Goods (IGD)"},
        ],
    },

    # ========================================================================
    # Category 4: Activity Indicators (~6 variables)
    # ========================================================================
    "STS_TRTU_M": {
        "category": 4,
        "category_name": "Activity_indicators",
        "base_filters": {
            "geo": "DE",
            "unit": "I21",
            "s_adj": "SCA",
        },
        "series_prefix": "DE_ACT_TURN",
        "variables": [
            {"id": "001", "nace_r2": "G47", "desc": "Turnover - Retail Trade (NACE G47)"},
            {"id": "002", "nace_r2": "G47_FOOD", "desc": "Turnover - Food Retail"},
            {"id": "003", "nace_r2": "G47_NFOOD", "desc": "Turnover - Non-Food Retail"},
        ],
    },

    "STS_COBP_M": {
        "category": 4,
        "category_name": "Activity_indicators",
        "base_filters": {
            "geo": "DE",
            "unit": "I21",
            # Note: s_adj not supported for this dataset
        },
        "series_prefix": "DE_ACT_BUILD",
        "variables": [
            {"id": "001", "nace_r2": "F", "desc": "Building Permits - Total Construction (NACE F)"},
            {"id": "002", "nace_r2": "F41", "desc": "Building Permits - Buildings (NACE F41)"},
        ],
    },

    # ========================================================================
    # Category 5: Trade (~2 variables)
    # ========================================================================
    "EXT_ST_EA19": {
        "category": 5,
        "category_name": "Trade",
        "base_filters": {
            "geo": "DE",
            "unit": "MIO_EUR",
            "s_adj": "NSA",
        },
        "series_prefix": "DE_TRADE",
        "variables": [
            {"id": "EXP_001", "stk_flow": "EXP", "partner": "EXT_EU28", "desc": "Exports - Extra EU (Goods)"},
            {"id": "IMP_001", "stk_flow": "IMP", "partner": "EXT_EU28", "desc": "Imports - Extra EU (Goods)"},
        ],
    },

    # ========================================================================
    # Category 6: Sentiment (~6 variables)
    # ========================================================================
    "EI_BSSI_M_R2": {
        "category": 6,
        "category_name": "Sentiment",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SA",
        },
        "series_prefix": "DE_SENT",
        "variables": [
            {"id": "ESI_001", "indic": "BS-ESI-I", "desc": "Economic Sentiment Indicator"},
        ],
    },

    "EI_BSCO_M": {
        "category": 6,
        "category_name": "Sentiment",
        "base_filters": {
            "geo": "DE",
            "s_adj": "SA",
        },
        "series_prefix": "DE_SENT",
        "variables": [
            {"id": "CONS_001", "indic": "BS-CSMCI-BAL", "desc": "Consumer Confidence Indicator"},
            {"id": "RET_001", "indic": "BS-RCI-BAL", "desc": "Retail Confidence Indicator"},
            {"id": "CONST_001", "indic": "BS-CONG-BAL", "desc": "Construction Confidence Indicator"},
            {"id": "IND_001", "indic": "BS-ICI-BAL", "desc": "Industrial Confidence Indicator"},
            {"id": "SERV_001", "indic": "BS-SERV-BAL", "desc": "Services Confidence Indicator"},
        ],
    },
}


def generate_all_variable_configs() -> list[dict[str, Any]]:
    """Generate full variable configuration list from DATASET_CONFIGS.

    Expands the nested DATASET_CONFIGS structure into individual variable
    configurations compatible with existing fetch functions.

    Returns:
        List of ~70 dictionaries, each containing:
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


def get_variable_config() -> list[dict[str, Any]]:
    """Return configuration for Germany variables to fetch.

    Returns:
        List of dictionaries containing:
        - series_id: Identifier in manifest
        - dataset_id: Eurostat dataset code
        - filters: Dictionary of filter parameters
        - description: Human-readable description

    """
    return [
        {
            "series_id": "DE_IND_PROD_001",
            "dataset_id": "STS_INPR_M",
            "filters": {
                "geo": "DE",
                "nace_r2": "B-D",
                "s_adj": "SCA",
                "unit": "I21",
            },
            "description": "Industrial Production - Total Industry",
        },
        {
            "series_id": "DE_IND_PROD_002",
            "dataset_id": "STS_INPR_M",
            "filters": {
                "geo": "DE",
                "nace_r2": "C",
                "s_adj": "SCA",
                "unit": "I21",
            },
            "description": "Industrial Production - Manufacturing",
        },
        {
            "series_id": "DE_LAB_UNEMP_001",
            "dataset_id": "UNE_RT_M",
            "filters": {
                "geo": "DE",
                "age": "TOTAL",
                "sex": "T",
                "unit": "PC_ACT",
                "s_adj": "SA",
            },
            "description": "Unemployment Rate",
        },
        # Note: STS_INEM_M and STS_INHW_M have API structure issues
        # Commented out for now - can be added with correct dataset codes later
        # {
        #     "series_id": "DE_LAB_EMP_001",
        #     "dataset_id": "STS_INEM_M",
        #     "filters": {
        #         "geo": "DE",
        #         "nace_r2": "B-D",
        #         "s_adj": "SCA",
        #         "unit": "I21",
        #     },
        #     "description": "Employment Index - Total",
        # },
        # {
        #     "series_id": "DE_LAB_HOURS_001",
        #     "dataset_id": "STS_INHW_M",
        #     "filters": {
        #         "geo": "DE",
        #         "nace_r2": "B-D",
        #         "s_adj": "SCA",
        #         "unit": "I21",
        #     },
        #     "description": "Hours Worked Index",
        # },
        {
            "series_id": "DE_PRICE_PPI_001",
            "dataset_id": "STS_INPP_M",
            "filters": {
                "geo": "DE",
                "nace_r2": "B-D",
                "unit": "I21",
                "s_adj": "NSA",
            },
            "description": "Producer Price Index - Total",
        },
        {
            "series_id": "DE_PRICE_HICP_001",
            "dataset_id": "PRC_HICP_MIDX",
            "filters": {
                "geo": "DE",
                "coicop": "CP00",
                "unit": "I15",
            },
            "description": "HICP - Overall Index",
        },
        {
            "series_id": "DE_PRICE_HICP_002",
            "dataset_id": "PRC_HICP_MIDX",
            "filters": {
                "geo": "DE",
                "coicop": "NRG",
                "unit": "I15",
            },
            "description": "HICP - Energy",
        },
        {
            "series_id": "DE_SENT_ESI_001",
            "dataset_id": "EI_BSSI_M_R2",
            "filters": {
                "geo": "DE",
                "s_adj": "SA",
                "indic": "BS-ESI-I",
            },
            "description": "Economic Sentiment Indicator",
        },
        {
            "series_id": "DE_ACT_BUILD_001",
            "dataset_id": "STS_COBP_M",
            "filters": {
                "geo": "DE",
                "nace_r2": "F",
                "unit": "I21",
                # Note: s_adj not supported for this dataset
            },
            "description": "Building Permits",
        },
    ]


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
        ValueError: If dataset_id is empty.
        RuntimeError: If API request fails after all retries.

    """
    if not dataset_id:
        msg = "dataset_id cannot be empty"
        raise ValueError(msg)

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
                     "age", "sex", "indic", "stk_flow", "partner"]

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

    # Filter to data from 2003 onwards (as per paper)
    result = result[result["date"] >= "2003"]

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


def save_series_to_csv(data: pd.DataFrame, output_path: Path) -> None:
    """Save series data to CSV file.

    Args:
        data: DataFrame to save.
        output_path: Path where CSV will be saved.

    """
    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    data.to_csv(output_path, index=False)


def fetch_category_variables(
    category_num: int,
    all_configs: list[dict[str, Any]],
    manifest: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch all variables for a specific category.

    Args:
        category_num: Category number to fetch (1-6).
        all_configs: Full list of variable configurations.
        manifest: Manifest DataFrame for metadata lookup.

    Returns:
        Tuple of (successful_data, failed_series):
        - successful_data: Dict mapping series_id to transformed DataFrame
        - failed_series: Dict mapping series_id to error message

    """
    # Filter configs for this category
    category_configs = [c for c in all_configs if c["category"] == category_num]

    successful_data: dict[str, pd.DataFrame] = {}
    failed_series: dict[str, str] = {}

    # Fetch each variable in category
    for i, var_config in enumerate(category_configs, 1):
        series_id = var_config["series_id"]
        dataset_id = var_config["dataset_id"]
        filters = var_config["filters"]
        description = var_config["description"]
        category_name = var_config["category_name"]

        print(f"  [{i}/{len(category_configs)}] {series_id}: {description}")

        try:
            # Fetch from Eurostat
            raw_data = fetch_eurostat_series(dataset_id, filters)

            # Get metadata from manifest or use description
            manifest_row = manifest[manifest["series_id"] == series_id]
            if not manifest_row.empty:
                variable_name = manifest_row.iloc[0]["variable_name"]
                country_iso2 = manifest_row.iloc[0]["country_iso2"]
            else:
                variable_name = description
                country_iso2 = "DE"

            metadata = {
                "variable_name": variable_name,
                "country_iso2": country_iso2,
                "category": category_num,
                "category_name": category_name,
            }

            # Transform to long format
            transformed_data = transform_eurostat_data(raw_data, series_id, metadata)

            # Validate
            validate_series_data(transformed_data, series_id)

            successful_data[series_id] = transformed_data
            print(f"    SUCCESS: {len(transformed_data)} rows")

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
    """Concatenate all DataFrames for a category into single DataFrame.

    Args:
        category_dfs: Dict mapping series_id to DataFrame.
        category_num: Category number for validation.
        category_name: Category name for validation.

    Returns:
        Concatenated DataFrame with columns:
        - date, value, series_id, country_iso2, variable_name,
          category, category_name

        Sorted by date (ascending), then series_id (ascending).

    Raises:
        ValueError: If category_dfs is empty.

    """
    if not category_dfs:
        msg = f"No successful data to concatenate for category {category_num}"
        raise ValueError(msg)

    # Concatenate all DataFrames
    concatenated = pd.concat(category_dfs.values(), ignore_index=True)

    # Sort by date, then series_id
    concatenated = concatenated.sort_values(["date", "series_id"]).reset_index(drop=True)

    return concatenated


def save_category_csv(
    data: pd.DataFrame,
    category_num: int,
    category_name: str,
    output_dir: Path,
) -> Path:
    """Save category data to CSV file.

    Args:
        data: DataFrame with all category variables.
        category_num: Category number for filename.
        category_name: Category name for filename.
        output_dir: Directory where CSV will be saved.

    Returns:
        Path to saved CSV file.

    """
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    filename = f"category_{category_num}_{category_name}.csv"
    output_path = output_dir / filename

    # Save to CSV
    data.to_csv(output_path, index=False)

    return output_path


def fetch_germany_variables(
    output_dir: Path,
    manifest_path: Path,
    *,
    use_category_files: bool = True,
) -> dict[str, Path]:
    """Fetch all Germany variables defined in configuration.

    Args:
        output_dir: Directory where CSV files will be saved.
        manifest_path: Path to series_manifest.csv for metadata lookup.
        use_category_files: If True, save category-based CSVs (default).
                           If False, save individual variable CSVs (legacy).

    Returns:
        Dictionary mapping category/series_id to file path.
        - If use_category_files=True: keys are 'category_1', 'category_2', etc.
        - If use_category_files=False: keys are series_ids (legacy behavior)

    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest for metadata
    manifest = pd.read_csv(manifest_path)

    if use_category_files:
        # === CATEGORY-BASED OUTPUT (NEW) ===
        # Generate all variable configs
        all_configs = generate_all_variable_configs()

        results: dict[str, Path] = {}
        summary_stats: dict[int, dict[str, Any]] = {}

        # Process each category sequentially
        for category_num in range(1, 7):  # Categories 1-6
            # Get category name from first config in category
            category_configs = [c for c in all_configs if c["category"] == category_num]
            if not category_configs:
                continue

            category_name = category_configs[0]["category_name"]

            print(f"\nFetching Category {category_num}: {category_name} ({len(category_configs)} variables)")

            # Fetch all variables in category
            successful_data, failed_series = fetch_category_variables(
                category_num, all_configs, manifest
            )

            # Track statistics
            total = len(category_configs)
            success_count = len(successful_data)
            fail_count = len(failed_series)

            summary_stats[category_num] = {
                "total": total,
                "success": success_count,
                "failed": fail_count,
                "failed_series": failed_series,
                "category_name": category_name,
            }

            if successful_data:
                # Concatenate and save
                category_df = concatenate_category_data(
                    successful_data, category_num, category_name
                )
                output_path = save_category_csv(
                    category_df, category_num, category_name, output_dir
                )
                results[f"category_{category_num}"] = output_path

                print(f"\n  Category {category_num}: {success_count}/{total} "
                      f"variables succeeded ({fail_count} failed)")
                print(f"    Saved: {output_path.name} ({len(category_df):,} rows)")
            else:
                print(f"\n  Category {category_num}: All {total} variables failed!")

        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)

        total_vars = sum(s["total"] for s in summary_stats.values())
        total_success = sum(s["success"] for s in summary_stats.values())
        total_failed = sum(s["failed"] for s in summary_stats.values())

        print(f"\nOverall: {total_success}/{total_vars} variables fetched successfully")
        print(f"Failed: {total_failed} variables")

        if total_failed > 0:
            print("\nFailed variables by category:")
            for cat_num, stats in summary_stats.items():
                if stats["failed"] > 0:
                    print(f"\n  Category {cat_num} ({stats['category_name']}):")
                    for series_id, error in stats["failed_series"].items():
                        print(f"    - {series_id}: {error[:80]}...")

    else:
        # === LEGACY INDIVIDUAL FILE OUTPUT ===
        # Use the old get_variable_config() function
        variables = get_variable_config()

        results: dict[str, Path] = {}
        failed: dict[str, str] = {}

        # Fetch each variable individually
        for var_config in variables:
            series_id = var_config["series_id"]
            dataset_id = var_config["dataset_id"]
            filters = var_config["filters"]
            description = var_config["description"]

            print(f"Fetching {series_id}: {description}")

            try:
                # Fetch from Eurostat
                raw_data = fetch_eurostat_series(dataset_id, filters)

                # Get metadata from manifest
                manifest_row = manifest[manifest["series_id"] == series_id]
                if not manifest_row.empty:
                    variable_name = manifest_row.iloc[0]["variable_name"]
                    country_iso2 = manifest_row.iloc[0]["country_iso2"]
                else:
                    variable_name = description
                    country_iso2 = "DE"

                metadata = {
                    "variable_name": variable_name,
                    "country_iso2": country_iso2,
                }

                # Transform to long format
                transformed_data = transform_eurostat_data(raw_data, series_id, metadata)

                # Validate
                validate_series_data(transformed_data, series_id)

                # Save to CSV
                output_path = output_dir / f"{series_id}.csv"
                save_series_to_csv(transformed_data, output_path)

                results[series_id] = output_path
                print(f"  SUCCESS: {len(transformed_data)} rows saved")

            except (ValueError, RuntimeError) as e:
                error_msg = str(e)
                print(f"  SKIPPED: {error_msg}")
                failed[series_id] = error_msg

        # Print summary
        print()
        print(f"Successfully fetched {len(results)}/{len(variables)} variables")
        if failed:
            print(f"Failed to fetch {len(failed)} variables:")
            for series_id, error in failed.items():
                print(f"  {series_id}: {error}")

    return results


def main() -> None:
    """Fetch Eurostat data for Germany."""
    output_dir = BLD / "data" / "raw" / "eurostat"
    manifest_path = SRC / "data" / "series_manifest.csv"

    print("Fetching Eurostat data for Germany...")
    print(f"Output directory: {output_dir}")
    print()

    results = fetch_germany_variables(output_dir, manifest_path)

    if results:
        print()
        print("Output files:")
        for series_id, path in results.items():
            print(f"  {series_id}: {path.name}")


if __name__ == "__main__":
    main()
