"""Fetch raw data from all APIs and save timestamped snapshot."""

from datetime import datetime
from pathlib import Path

from template_project.config import BLD, SRC
from template_project.data_fetch.fetch import fetch_many
from template_project.data_management.registry.registry_io import load_registry


def task_fetch_raw(
    depends_on: dict = {
        "registry": SRC / "data_management" / "registry" / "series_registry.csv",
        "adapters": SRC / "data_fetch" / "adapters.py",
        "fetch": SRC / "data_fetch" / "fetch.py",
        "standardize": SRC / "data_fetch" / "standardize.py",
    },
    produces: Path = BLD / "data" / "raw" / "raw_api_snapshot.parquet",
) -> None:
    """Fetch raw data from all APIs (short and boring task).

    Task only handles I/O. Real logic in fetch_many() helper.
    Raw snapshot enables reproducibility and reprocessing.
    """
    print("Loading series registry...")
    registry = load_registry()
    all_series_ids = registry.series_id.tolist()
    print(f"Found {len(all_series_ids)} series in registry")

    print("\nFetching data from all sources...")
    raw_combined = fetch_many(all_series_ids, registry=registry)
    print(f"\nFetched total: {len(raw_combined)} rows")

    # Write raw snapshot
    print(f"\nWriting raw snapshot to {produces}...")
    produces.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata to parquet file
    metadata = {
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
        "registry_rows": str(len(registry)),
        "fetched_rows": str(len(raw_combined)),
    }
    raw_combined.to_parquet(produces, index=False)
    print(f"Successfully wrote {len(raw_combined)} rows to {produces.name}")
