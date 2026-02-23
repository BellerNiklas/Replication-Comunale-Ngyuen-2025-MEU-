"""Generic fetch dispatcher that works for any series."""

import json

import pandas as pd

from template_project.data_fetch import adapters, standardize
from template_project.data_management.registry.registry_io import load_registry


def fetch_one(series_id: str, *, registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fetch any series by ID using appropriate adapter and standardizer.

    Args:
        series_id: Series identifier (e.g., "DE_IP_001")
        registry: Optional pre-loaded registry (for performance)

    Returns:
        Standardized long DataFrame with canonical schema

    Raises:
        ValueError: If series_id not found or fetch fails
        RuntimeError: If API request fails
    """
    # Load registry if not provided
    if registry is None:
        registry = load_registry()

    # Get spec for this series
    spec = registry[registry.series_id == series_id]
    if spec.empty:
        msg = f"Series {series_id} not found in registry"
        raise ValueError(msg)
    spec = spec.iloc[0].to_dict()

    # Dispatch to appropriate adapter based on source
    source = spec["source"]

    try:
        if source == "eurostat":
            filters = json.loads(spec["filters_json"])
            raw = adapters.fetch_eurostat_raw(spec["dataset"], filters)
            df = standardize.standardize_from_eurostat(raw, spec)

        elif source == "ecb":
            raw = adapters.fetch_ecb_raw(
                spec["dataset"],
                spec["key"],
                start_period=spec.get("start_period", "2003-01"),
            )
            df = standardize.standardize_from_ecb_like(raw, spec)

        elif source == "oecd":
            raw = adapters.fetch_oecd_raw(
                spec["dataset"],
                spec["key"],
                start_period=spec.get("start_period", "2003-01"),
            )
            df = standardize.standardize_from_ecb_like(raw, spec)

        elif source == "bis":
            raw = adapters.fetch_bis_raw(
                spec["dataset"],
                spec["key"],
                start_period=spec.get("start_period", "2003-01"),
            )
            df = standardize.standardize_from_bis(raw, spec)

        else:
            msg = f"Unknown source: {source}"
            raise ValueError(msg)

        # Validate result
        standardize.validate_long(df, series_id)

        return df

    except Exception as e:
        msg = f"Failed to fetch {series_id}: {e}"
        raise RuntimeError(msg) from e


def fetch_many(series_ids: list[str], *, registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fetch multiple series and concatenate.

    Continues on error - prints warning for failed series but doesn't stop.

    Args:
        series_ids: List of series identifiers
        registry: Optional pre-loaded registry

    Returns:
        Concatenated long DataFrame with all successfully fetched series

    Raises:
        ValueError: If no series could be fetched successfully
    """
    if registry is None:
        registry = load_registry()

    dfs = []
    failed = []

    for sid in series_ids:
        try:
            df = fetch_one(sid, registry=registry)
            dfs.append(df)
            print(f"OK Fetched {sid}: {len(df)} rows")
        except Exception as e:
            failed.append((sid, str(e)))
            print(f"SKIPPED {sid}: {e}")

    if not dfs:
        msg = f"No data fetched for any series. All {len(series_ids)} series failed."
        raise ValueError(msg)

    if failed:
        print(f"\nSummary: {len(dfs)}/{len(series_ids)} series fetched successfully")
        print(f"Failed: {len(failed)} series")

    return pd.concat(dfs, ignore_index=True)
