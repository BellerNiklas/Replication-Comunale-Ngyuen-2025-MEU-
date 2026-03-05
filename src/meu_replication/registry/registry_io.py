"""Load and validate series registry."""

from pathlib import Path

import pandas as pd

from meu_replication.config import SRC

REGISTRY_PATH = SRC / "registry" / "series_registry.csv"

ALLOWED_SOURCES = {"eurostat", "ecb", "oecd", "bis"}
SUPPORTED_TRANSFORMATION_CODES = {1, 2, 5}


def load_registry() -> pd.DataFrame:
    """Load and validate series registry.

    Returns:
        Validated DataFrame with series definitions.

    Raises:
        FileNotFoundError: If registry file does not exist.
        ValueError: If registry validation fails.
    """
    if not REGISTRY_PATH.exists():
        msg = f"Registry file not found: {REGISTRY_PATH}"
        raise FileNotFoundError(msg)

    registry = pd.read_csv(REGISTRY_PATH)
    validate_registry(registry)
    return registry


def validate_registry(df: pd.DataFrame) -> None:
    """Validate registry integrity.

    Args:
        df: Registry DataFrame to validate.

    Raises:
        ValueError: If validation fails with detailed error message.
    """
    # Check required columns exist
    required_cols = {
        "series_id",
        "source",
        "category",
        "category_name",
        "country_iso2",
        "variable_name",
        "dataset",
        "key",
        "filters_json",
        "unit_measure_filter",
        "frequency",
        "start_period",
        "transformationcode",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        msg = f"Registry missing required columns: {missing_cols}"
        raise ValueError(msg)

    # Check series_id uniqueness
    duplicates = df[df.duplicated(subset=["series_id"], keep=False)]
    if not duplicates.empty:
        dup_ids = duplicates["series_id"].tolist()
        msg = f"Duplicate series_id found: {dup_ids}"
        raise ValueError(msg)

    # Check allowed source values
    invalid_sources = df[~df["source"].isin(ALLOWED_SOURCES)]
    if not invalid_sources.empty:
        invalid_vals = invalid_sources["source"].unique().tolist()
        msg = f"Invalid source values: {invalid_vals}. Allowed: {ALLOWED_SOURCES}"
        raise ValueError(msg)

    # Check required fields per source
    for _, row in df.iterrows():
        source = row["source"]
        series_id = row["series_id"]

        # All sources require dataset
        if pd.isna(row["dataset"]) or row["dataset"] == "":
            msg = f"{series_id}: Missing required field 'dataset' for source '{source}'"
            raise ValueError(msg)

        # Eurostat requires filters_json
        if source == "eurostat":
            if pd.isna(row["filters_json"]) or row["filters_json"] == "":
                msg = f"{series_id}: Eurostat series requires 'filters_json'"
                raise ValueError(msg)
            # Key should be empty for Eurostat
            if not pd.isna(row["key"]) and row["key"] != "":
                msg = f"{series_id}: Eurostat series should have empty 'key'"
                raise ValueError(msg)

        # ECB/OECD/BIS require key
        if source in {"ecb", "oecd", "bis"}:
            if pd.isna(row["key"]) or row["key"] == "":
                msg = f"{series_id}: {source.upper()} series requires 'key'"
                raise ValueError(msg)
            # filters_json should be empty for non-Eurostat
            if not pd.isna(row["filters_json"]) and row["filters_json"] != "":
                msg = f"{series_id}: {source.upper()} series should have empty 'filters_json'"
                raise ValueError(msg)

    # Validate country codes (basic check)
    # Allow ISO 3166-1 alpha-2 (2 letters) or special codes like U2 (Euro Area)
    invalid_countries = df[
        ~df["country_iso2"].str.match(r"^[A-Z0-9]{2}$", na=False)
    ]
    if not invalid_countries.empty:
        invalid_vals = invalid_countries["country_iso2"].unique().tolist()
        msg = (
            "Invalid country_iso2 codes (should be 2 uppercase letters/digits): "
            f"{invalid_vals}"
        )
        raise ValueError(msg)

    # Validate transformation code completeness and domain.
    trans_raw = pd.to_numeric(df["transformationcode"], errors="coerce")
    if trans_raw.isna().any():
        invalid = df.loc[trans_raw.isna(), "series_id"].head(5).tolist()
        msg = (
            "Missing or non-numeric transformationcode for series_ids: "
            f"{invalid}"
        )
        raise ValueError(msg)

    int_mask = trans_raw == trans_raw.astype(int)
    if not int_mask.all():
        invalid = df.loc[~int_mask, "series_id"].head(5).tolist()
        msg = f"Non-integer transformationcode for series_ids: {invalid}"
        raise ValueError(msg)

    trans = trans_raw.astype(int)
    bad_codes = sorted(set(trans.unique()) - SUPPORTED_TRANSFORMATION_CODES)
    if bad_codes:
        msg = (
            "Unsupported transformationcode values: "
            f"{bad_codes}. Allowed: {sorted(SUPPORTED_TRANSFORMATION_CODES)}"
        )
        raise ValueError(msg)
