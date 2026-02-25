"""Fetch raw data via pytask DAG (77 tasks + 1 combine per source).

Each task fetches all series for one (country, source) pair, using the
availability manifest to skip series known to be missing. Results are
written as per-country parquet files.

Directory structure:
    bld/data/raw/
    ├── eurostat/{AT,BE,...,SK}_snapshot.parquet   (19 files)
    ├── ecb/{AT,BE,...,SK,U2}_snapshot.parquet     (20 files)
    ├── oecd/{AT,BE,...,SK}_snapshot.parquet       (19 files)
    └── bis/{AT,BE,...,SK}_snapshot.parquet        (19 files)

Task structure:
    - 19 Eurostat tasks (one per EA country)
    - 20 ECB tasks (19 EA countries + U2 for EA-aggregate series)
    - 19 OECD tasks (one per EA country)
    - 19 BIS tasks (one per EA country)

Total: 77 fetch tasks. Each is independent → parallelizable.
"""

from pathlib import Path

import pandas as pd
import pytask

from template_project.config import BLD, MEU_COUNTRIES, SRC

# Standardized column schema for empty DataFrames
_EMPTY_COLUMNS = [
    "date",
    "value",
    "series_id",
    "country_iso2",
    "variable_name",
    "category",
    "category_name",
    "source",
]

# -- Shared dependencies for all fetch tasks --

_FETCH_DEPENDS = {
    "registry": SRC / "data_management" / "registry" / "series_registry.csv",
    "availability": BLD / "meta" / "series_availability.parquet",
    "adapters": SRC / "data_fetch" / "adapters.py",
    "fetch": SRC / "data_fetch" / "fetch.py",
    "standardize": SRC / "data_fetch" / "standardize.py",
}


def _fetch_source_country(
    source: str,
    country: str,
    output_path: Path,
) -> None:
    """Fetch all available series for one (source, country) pair.

    Short and boring: load registry, filter to available series, fetch, write.
    Real logic lives in fetch.fetch_many().
    """
    from template_project.data_fetch.fetch import fetch_many
    from template_project.data_management.registry.registry_io import load_registry

    registry = load_registry()
    availability = pd.read_parquet(BLD / "meta" / "series_availability.parquet")

    # Filter registry to this country + source
    country_series = registry[
        (registry.country_iso2 == country) & (registry.source == source)
    ].series_id.tolist()

    # Only fetch series confirmed as available by probing
    available_series = availability[
        (availability.series_id.isin(country_series))
        & (availability.status.isin(["ok", "ok_short"]))
    ].series_id.tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not available_series:
        print(f"[Fetch/{source}/{country}] No available series — writing empty file")
        pd.DataFrame(columns=_EMPTY_COLUMNS).to_parquet(output_path, index=False)
        return

    print(
        f"[Fetch/{source}/{country}] Fetching {len(available_series)}"
        f"/{len(country_series)} available series..."
    )
    df = fetch_many(available_series, registry=registry)

    df.to_parquet(output_path, index=False)
    print(f"[Fetch/{source}/{country}] Wrote {len(df)} rows")


# -- Eurostat: 19 tasks --

for _country in MEU_COUNTRIES:

    @pytask.task(id=f"eurostat_{_country}")
    def task_fetch_eurostat(
        country: str = _country,
        depends_on: dict = _FETCH_DEPENDS,
        produces: Path = BLD
        / "data"
        / "raw"
        / "eurostat"
        / f"{_country}_snapshot.parquet",
    ) -> None:
        """Fetch Eurostat data for one country."""
        _fetch_source_country("eurostat", country, produces)


# -- ECB: 20 tasks (19 countries + U2 for EA-aggregate series) --

_ECB_COUNTRIES = [*MEU_COUNTRIES, "U2"]

for _country in _ECB_COUNTRIES:

    @pytask.task(id=f"ecb_{_country}")
    def task_fetch_ecb(
        country: str = _country,
        depends_on: dict = _FETCH_DEPENDS,
        produces: Path = BLD
        / "data"
        / "raw"
        / "ecb"
        / f"{_country}_snapshot.parquet",
    ) -> None:
        """Fetch ECB data for one country (or U2 for EA aggregates)."""
        _fetch_source_country("ecb", country, produces)


# -- OECD: 19 tasks --

for _country in MEU_COUNTRIES:

    @pytask.task(id=f"oecd_{_country}")
    def task_fetch_oecd(
        country: str = _country,
        depends_on: dict = _FETCH_DEPENDS,
        produces: Path = BLD
        / "data"
        / "raw"
        / "oecd"
        / f"{_country}_snapshot.parquet",
    ) -> None:
        """Fetch OECD data for one country."""
        _fetch_source_country("oecd", country, produces)


# -- BIS: 19 tasks --

for _country in MEU_COUNTRIES:

    @pytask.task(id=f"bis_{_country}")
    def task_fetch_bis(
        country: str = _country,
        depends_on: dict = _FETCH_DEPENDS,
        produces: Path = BLD
        / "data"
        / "raw"
        / "bis"
        / f"{_country}_snapshot.parquet",
    ) -> None:
        """Fetch BIS data for one country."""
        _fetch_source_country("bis", country, produces)
