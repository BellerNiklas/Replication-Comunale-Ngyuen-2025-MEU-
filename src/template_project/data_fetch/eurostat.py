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
        metadata: Dictionary containing country_iso2, variable_name, etc.

    Returns:
        Long format DataFrame with columns: date, value, series_id,
        country_iso2, variable_name.

    Raises:
        ValueError: If data cannot be transformed.

    """
    # Identify non-time columns (metadata dimensions)
    non_time_cols = ["freq", "geo", "nace_r2", "s_adj", "unit", "coicop",
                     "age", "sex", "indic"]

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


def fetch_germany_variables(
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Path]:
    """Fetch all Germany variables defined in configuration.

    Args:
        output_dir: Directory where CSV files will be saved.
        manifest_path: Path to series_manifest.csv for metadata lookup.

    Returns:
        Dictionary mapping series_id to file path for each fetched variable.

    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest for metadata
    manifest = pd.read_csv(manifest_path)

    # Get variable configurations
    variables = get_variable_config()

    # Results dictionaries
    results: dict[str, Path] = {}
    failed: dict[str, str] = {}

    # Fetch each variable
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
