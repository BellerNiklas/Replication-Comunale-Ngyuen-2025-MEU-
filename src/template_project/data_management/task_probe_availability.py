"""Probe series availability via pytask DAG.

Each task probes all series for one (country, source) pair using lightweight
fetches (from 2020-01). Results are written as parquet files to
bld/meta/availability/ and combined into a single availability manifest.

OECD availability is derived from the bulk fetch (no per-series probing
needed), avoiding rate-limit issues entirely.

Task structure:
    - 19 Eurostat tasks (one per EA country)
    - 20 ECB tasks (19 EA countries + U2 for EA-aggregate series)
    - 1 OECD availability task (derived from bulk fetch, no API calls)
    - 19 BIS tasks (one per EA country)
    - 1 combine task (merges all probe/availability results)

Total: 60 tasks.
"""

from pathlib import Path

import pandas as pd
import pytask

from template_project.config import BLD, MEU_COUNTRIES, SRC
from template_project.data_management.registry.registry_io import load_registry

# -- Shared dependencies for all probe tasks --

_PROBE_DEPENDS = {
    "registry": SRC / "data_management" / "registry" / "series_registry.csv",
    "probe": SRC / "data_fetch" / "probe.py",
    "adapters": SRC / "data_fetch" / "adapters.py",
    "standardize": SRC / "data_fetch" / "standardize.py",
}


def _probe_source_country(
    source: str,
    country: str,
    output_path: Path,
) -> None:
    """Probe all series for one (source, country) pair (shared helper).

    Short and boring: load registry, filter, probe, write.
    Real logic lives in probe.probe_many().
    """
    from template_project.data_fetch.probe import probe_many

    registry = load_registry()

    # Filter registry to this country + source
    mask = (registry.country_iso2 == country) & (registry.source == source)
    series_ids = registry.loc[mask, "series_id"].tolist()

    if not series_ids:
        print(f"[Probe/{source}/{country}] No series to probe — writing empty file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "series_id",
                "template_id",
                "country_iso2",
                "status",
                "rows_fetched",
                "error_kind",
                "error_message",
            ]
        ).to_parquet(output_path, index=False)
        return

    # OECD has stricter rate limits (429 errors with default 1s delay)
    delay = 5.0 if source == "oecd" else 1.0
    print(f"[Probe/{source}/{country}] Probing {len(series_ids)} series (delay={delay}s)...")
    results = probe_many(series_ids, registry=registry, delay=delay)

    # Summarize
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"[Probe/{source}/{country}] Results: {statuses}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(output_path, index=False)


# -- Eurostat: 19 tasks --

for _country in MEU_COUNTRIES:

    @pytask.task(id=f"eurostat_{_country}")
    def task_probe_eurostat(
        country: str = _country,
        depends_on: dict = _PROBE_DEPENDS,
        produces: Path = BLD / "meta" / "availability" / f"eurostat_{_country}.parquet",
    ) -> None:
        """Probe Eurostat availability for one country."""
        _probe_source_country("eurostat", country, produces)


# -- ECB: 20 tasks (19 countries + U2 for EA-aggregate series) --

_ECB_COUNTRIES = [*MEU_COUNTRIES, "U2"]

for _country in _ECB_COUNTRIES:

    @pytask.task(id=f"ecb_{_country}")
    def task_probe_ecb(
        country: str = _country,
        depends_on: dict = _PROBE_DEPENDS,
        produces: Path = BLD / "meta" / "availability" / f"ecb_{_country}.parquet",
    ) -> None:
        """Probe ECB availability for one country (or U2 for EA aggregates)."""
        _probe_source_country("ecb", country, produces)


# -- OECD: 1 task (derived from bulk fetch, no API calls) --


def _build_oecd_avail_produces() -> dict[str, Path]:
    """Build produces dict for all 19 OECD availability files."""
    return {
        country: BLD / "meta" / "availability" / f"oecd_{country}.parquet"
        for country in MEU_COUNTRIES
    }


def task_derive_oecd_availability(
    depends_on: Path = BLD / "data" / "raw" / "oecd" / "bulk_snapshot.parquet",
    produces: dict = _build_oecd_avail_produces(),
) -> None:
    """Derive OECD availability from bulk fetch results (no API calls).

    Short and boring: read bulk data, check which series have data,
    write availability records in same format as probe results.
    """
    from template_project.data_management.registry.registry_io import load_registry

    registry = load_registry()
    oecd_series = registry[registry.source == "oecd"]

    bulk = pd.read_parquet(depends_on)
    fetched_ids = set(bulk["series_id"].unique()) if not bulk.empty else set()

    for country, path in produces.items():
        country_series = oecd_series[oecd_series.country_iso2 == country]
        results = []
        for _, row in country_series.iterrows():
            sid = row["series_id"]
            if sid in fetched_ids:
                n_rows = len(bulk[bulk.series_id == sid])
                status = "ok" if n_rows >= 10 else "ok_short"
            else:
                n_rows = 0
                status = "missing"
            results.append({
                "series_id": sid,
                "template_id": sid.replace(f"{country}_", "", 1),
                "country_iso2": country,
                "status": status,
                "rows_fetched": n_rows,
                "error_kind": "",
                "error_message": "",
            })

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_parquet(path, index=False)
        statuses = {}
        for r in results:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        print(f"[Avail/oecd/{country}] {statuses}")


# -- BIS: 19 tasks --

for _country in MEU_COUNTRIES:

    @pytask.task(id=f"bis_{_country}")
    def task_probe_bis(
        country: str = _country,
        depends_on: dict = _PROBE_DEPENDS,
        produces: Path = BLD / "meta" / "availability" / f"bis_{_country}.parquet",
    ) -> None:
        """Probe BIS availability for one country."""
        _probe_source_country("bis", country, produces)


# -- Combine: 1 task --


def _build_probe_depends() -> dict[str, Path]:
    """Build dependency dict for all 77 probe output files."""
    deps = {}
    for country in MEU_COUNTRIES:
        for source in ("eurostat", "ecb", "oecd", "bis"):
            deps[f"{source}_{country}"] = (
                BLD / "meta" / "availability" / f"{source}_{country}.parquet"
            )
    # ECB U2 (EA aggregates)
    deps["ecb_U2"] = BLD / "meta" / "availability" / "ecb_U2.parquet"
    return deps


def task_combine_availability(
    depends_on: dict = _build_probe_depends(),
    produces: Path = BLD / "meta" / "series_availability.parquet",
) -> None:
    """Combine all per-country-per-source probe results into one manifest.

    Short and boring: read all parquet files, concatenate, write.
    """
    dfs = []
    for label, path in depends_on.items():
        if path.exists():
            df = pd.read_parquet(path)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        print("WARNING: No probe results found — writing empty availability manifest")
        produces.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "series_id",
                "template_id",
                "country_iso2",
                "status",
                "rows_fetched",
                "error_kind",
                "error_message",
            ]
        ).to_parquet(produces, index=False)
        return

    availability = pd.concat(dfs, ignore_index=True)

    # Summary statistics
    total = len(availability)
    print(f"\nAvailability manifest: {total} series probed")
    for status, count in availability["status"].value_counts().items():
        pct = 100 * count / total
        print(f"  {status}: {count} ({pct:.1f}%)")

    produces.parent.mkdir(parents=True, exist_ok=True)
    availability.to_parquet(produces, index=False)
    print(f"\nWrote {produces}")
