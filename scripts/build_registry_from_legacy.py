"""Extract series registry from legacy DATASET_CONFIGS.

This script extracts all series definitions from the four fetcher modules
(eurostat, ecb, oecd, bis) and creates a unified CSV registry.

Phase A: Germany (DE) + Euro Area (U2) only (148 series)
Phase B (future): Extend to all 19 EA countries
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path so we can import from meu_replication
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meu_replication.data_fetch.bis import generate_all_variable_configs as bis_configs
from meu_replication.data_fetch.ecb import generate_all_variable_configs as ecb_configs
from meu_replication.data_fetch.eurostat import (
    generate_all_variable_configs as eurostat_configs,
)
from meu_replication.data_fetch.oecd import generate_all_variable_configs as oecd_configs


def main():
    """Build series registry CSV from all four sources."""
    registry_rows = []

    # Eurostat
    print("Extracting Eurostat configs...")
    for cfg in eurostat_configs():
        # Extract country_iso2 from filters (geo field)
        country_iso2 = cfg["filters"].get("geo", "DE")  # Default to DE if not found

        registry_rows.append({
            "series_id": cfg["series_id"],
            "source": "eurostat",
            "category": cfg["category"],
            "category_name": cfg["category_name"],
            "country_iso2": country_iso2,
            "variable_name": cfg["description"],
            "dataset": cfg["dataset_id"],
            "key": "",  # Eurostat doesn't use SDMX keys
            "filters_json": json.dumps(cfg["filters"]),
            "unit_measure_filter": "",
            "frequency": "",
            "start_period": "2003-01",
        })

    # ECB
    print("Extracting ECB configs...")
    for cfg in ecb_configs():
        registry_rows.append({
            "series_id": cfg["series_id"],
            "source": "ecb",
            "category": cfg["category"],
            "category_name": cfg["category_name"],
            "country_iso2": cfg["country_iso2"],
            "variable_name": cfg["description"],  # ECB uses "description"
            "dataset": cfg["flow"],
            "key": cfg["key"],
            "filters_json": "",  # ECB uses SDMX keys, not filters
            "unit_measure_filter": "",
            "frequency": "M",
            "start_period": "2003-01",
        })

    # OECD
    print("Extracting OECD configs...")
    for cfg in oecd_configs():
        registry_rows.append({
            "series_id": cfg["series_id"],
            "source": "oecd",
            "category": cfg["category"],
            "category_name": cfg["category_name"],
            "country_iso2": cfg["country_iso2"],
            "variable_name": cfg["description"],  # OECD uses "description"
            "dataset": cfg["flow"],
            "key": cfg["key"],
            "filters_json": "",  # OECD uses SDMX keys, not filters
            "unit_measure_filter": "",
            "frequency": "M",
            "start_period": "2003-01",
        })

    # BIS
    print("Extracting BIS configs...")
    for cfg in bis_configs():
        # BIS already has "variable_name" field
        registry_rows.append({
            "series_id": cfg["series_id"],
            "source": "bis",
            "category": cfg["category"],
            "category_name": cfg["category_name"],
            "country_iso2": cfg["country_iso2"],
            "variable_name": cfg["variable_name"],
            "dataset": cfg["dataset"],
            "key": cfg["key"],
            "filters_json": "",  # BIS uses SDMX keys, not filters
            "unit_measure_filter": str(cfg.get("unit_measure_filter", "")),
            "frequency": "M",
            "start_period": "2003-01",
        })

    # Create DataFrame
    registry = pd.DataFrame(registry_rows)

    # Validate
    print(f"\nTotal series extracted: {len(registry)}")
    print(f"Unique countries: {sorted(registry.country_iso2.unique())}")
    print(f"Series by source:")
    print(registry.groupby("source").size())

    # Write to CSV
    output_path = Path(__file__).parent.parent / "src" / "meu_replication" / "data_management" / "registry" / "series_registry.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_path, index=False)

    print(f"\nRegistry saved to: {output_path}")
    print(f"Total rows: {len(registry)}")

    # Verify expected count
    if len(registry) != 148:
        print(f"\nWARNING: Expected 148 series, got {len(registry)}")
        sys.exit(1)

    print("\nRegistry extraction complete!")


if __name__ == "__main__":
    main()
