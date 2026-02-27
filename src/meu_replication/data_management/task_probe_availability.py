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

from collections import Counter
from pathlib import Path

import pandas as pd
import pytask

from meu_replication.config import BLD, MEU_COUNTRIES, SRC
from meu_replication.data_management.registry.registry_io import load_registry

_OK_THRESHOLD = 10  # Minimum rows to classify as "ok" (vs "ok_short")

_PROBE_COLUMNS = [
    "series_id",
    "template_id",
    "country_iso2",
    "status",
    "rows_fetched",
    "error_kind",
    "error_message",
]

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
    from meu_replication.data_fetch.probe import probe_many

    registry = load_registry()

    # Filter registry to this country + source
    mask = (registry.country_iso2 == country) & (registry.source == source)
    series_ids = registry.loc[mask, "series_id"].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not series_ids:
        print(f"[Probe/{source}/{country}] No series to probe — writing empty file")
        pd.DataFrame(columns=_PROBE_COLUMNS).to_parquet(output_path, index=False)
        return

    # OECD has stricter rate limits (429 errors with default 1s delay)
    delay = 5.0 if source == "oecd" else 1.0
    n = len(series_ids)
    print(f"[Probe/{source}/{country}] Probing {n} series (delay={delay}s)...")
    results = probe_many(series_ids, registry=registry, delay=delay)

    pd.DataFrame(results).to_parquet(output_path, index=False)
    print(f"[Probe/{source}/{country}] {dict(Counter(r['status'] for r in results))}")


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


def _build_oecd_availability(
    bulk: pd.DataFrame,
    oecd_series: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """Classify OECD series as ok/ok_short/missing from bulk fetch.

    Pure function: no I/O.
    """
    fetched_ids = set(bulk["series_id"].unique()) if not bulk.empty else set()
    country_series = oecd_series[oecd_series.country_iso2 == country]

    rows = []
    for _, row in country_series.iterrows():
        sid = row["series_id"]
        if sid in fetched_ids:
            n_rows = len(bulk[bulk.series_id == sid])
            status = "ok" if n_rows >= _OK_THRESHOLD else "ok_short"
        else:
            n_rows = 0
            status = "missing"
        rows.append(
            {
                "series_id": sid,
                "template_id": sid.replace(f"{country}_", "", 1),
                "country_iso2": country,
                "status": status,
                "rows_fetched": n_rows,
                "error_kind": "",
                "error_message": "",
            }
        )
    return pd.DataFrame(rows, columns=_PROBE_COLUMNS)


def task_derive_oecd_availability(
    depends_on: Path = BLD / "data" / "raw" / "oecd" / "bulk_snapshot.parquet",
    produces: dict = _build_oecd_avail_produces(),
) -> None:
    """Derive OECD availability from bulk fetch results (no API calls)."""
    registry = load_registry()
    oecd_series = registry[registry.source == "oecd"]
    bulk = pd.read_parquet(depends_on)

    for country, path in produces.items():
        result = _build_oecd_availability(bulk, oecd_series, country)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
        print(f"[Avail/oecd/{country}] {dict(Counter(result['status']))}")


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
    for path in depends_on.values():
        if path.exists():
            df = pd.read_parquet(path)
            if not df.empty:
                dfs.append(df)

    produces.parent.mkdir(parents=True, exist_ok=True)

    if not dfs:
        print("WARNING: No probe results found — writing empty manifest")
        pd.DataFrame(columns=_PROBE_COLUMNS).to_parquet(produces, index=False)
        return

    availability = pd.concat(dfs, ignore_index=True)
    availability.to_parquet(produces, index=False)
    print(f"Availability manifest: {len(availability)} series, wrote {produces}")
