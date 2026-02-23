"""Fetch raw data from all APIs and save per-source snapshots."""

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
    produces: dict = {
        "eurostat": BLD / "data" / "raw" / "eurostat_snapshot.parquet",
        "ecb": BLD / "data" / "raw" / "ecb_snapshot.parquet",
        "oecd": BLD / "data" / "raw" / "oecd_snapshot.parquet",
        "bis": BLD / "data" / "raw" / "bis_snapshot.parquet",
    },
) -> None:
    """Fetch raw data from all APIs, write per-source snapshots (short and boring task).

    Task only handles I/O. Real logic in fetch_many() helper.
    Per-source snapshots enable better debugging, resumability, and caching.
    """
    print("Loading series registry...")
    registry = load_registry()
    print(f"Found {len(registry)} series in registry")

    # Group series by source
    by_source = registry.groupby("source")["series_id"].apply(list).to_dict()
    print(f"Sources: {list(by_source.keys())}")

    # Fetch and write each source separately
    total_rows = 0
    for source, series_ids in by_source.items():
        print(f"\nFetching {source} ({len(series_ids)} series)...")
        df = fetch_many(series_ids, registry=registry)
        print(f"Fetched {len(df)} rows from {source}")

        output_path = produces[source]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Wrote {len(df)} rows to {output_path.name}")
        total_rows += len(df)

    print(f"\nTotal: {total_rows} rows across {len(by_source)} sources")
