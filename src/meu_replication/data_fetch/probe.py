"""Lightweight availability probing for multi-country expansion.

Probes each series with a minimal fetch (last 12 months) to determine
whether data exists for a given country+source combination. Returns
structured status dicts that get aggregated into an availability manifest.

Design:
- Reuses existing adapters (no code duplication)
- Overrides start_period to fetch only recent data (~3x faster than full history)
- Classifies results as ok/ok_short/missing/error
- Pure classification logic; I/O only in probe_one (network call)

Status values:
    ok       : ≥10 observations returned (sufficient data)
    ok_short : 1-9 observations (exists but short — late start, quarterly, etc.)
    missing  : 0 observations or "no data" error (series unavailable for this country)
    error    : API failure unrelated to data availability (transient, rate limit, etc.)
"""

import json
from datetime import datetime, timedelta
from time import sleep
from typing import Any

import pandas as pd

from meu_replication.data_fetch import adapters, standardize

# Minimum rows to classify as "ok" (vs "ok_short")
_OK_THRESHOLD = 10


def probe_one(series_id: str, *, registry: pd.DataFrame) -> dict[str, Any]:
    """Probe a single series for data availability.

    Fetches minimal data (last 12 months) to check whether the series exists
    and returns data for a given country. Does NOT fetch full history.

    Args:
        series_id: Series identifier (e.g., "FR_IP_001").
        registry: Full expanded registry DataFrame.

    Returns:
        Dict with keys: series_id, template_id, country_iso2, status,
        rows_fetched, error_kind, error_message.
    """
    spec = registry[registry.series_id == series_id].iloc[0].to_dict()
    template_id = _extract_template_id(series_id, spec["country_iso2"])
    probe_start = _probe_start_period()

    base = {
        "series_id": series_id,
        "template_id": template_id,
        "country_iso2": spec["country_iso2"],
    }

    try:
        df = _fetch_probe(spec, probe_start)

        rows = len(df)
        if rows == 0:
            return {
                **base,
                "status": "missing",
                "rows_fetched": 0,
                "error_kind": "empty_result",
                "error_message": "",
            }

        status = "ok" if rows >= _OK_THRESHOLD else "ok_short"
        return {
            **base,
            "status": status,
            "rows_fetched": rows,
            "error_kind": "",
            "error_message": "",
        }

    except Exception as e:
        status = "missing" if _is_data_absence_error(e) else "error"
        return {
            **base,
            "status": status,
            "rows_fetched": 0,
            "error_kind": type(e).__name__,
            "error_message": str(e)[:200],
        }


def probe_many(
    series_ids: list[str],
    *,
    registry: pd.DataFrame,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Probe multiple series with rate-limiting delay between calls.

    Args:
        series_ids: List of series identifiers to probe.
        registry: Full expanded registry DataFrame.
        delay: Seconds to wait between API calls (rate limiting).

    Returns:
        List of probe result dicts.
    """
    results = []
    for i, sid in enumerate(series_ids):
        if i > 0 and delay > 0:
            sleep(delay)
        result = probe_one(sid, registry=registry)
        results.append(result)
    return results


def _fetch_probe(spec: dict[str, Any], probe_start: str) -> pd.DataFrame:
    """Fetch minimal data for probing (last 12 months only).

    Dispatches to the appropriate adapter with overridden start_period.
    Standardizes the result to get an accurate count of valid rows.

    Args:
        spec: Series specification dict from registry.
        probe_start: Start period for probe fetch (e.g., "2025-02").

    Returns:
        Standardized DataFrame (may be empty if no data).

    Raises:
        RuntimeError: If adapter call fails.
        ValueError: If standardization finds no valid data.
    """
    source = spec["source"]

    if source == "eurostat":
        return _fetch_probe_eurostat(spec, probe_start)
    if source == "ecb":
        return _fetch_probe_ecb(spec, probe_start)
    if source == "oecd":
        msg = "OECD probing is handled via bulk fetch — use task_derive_oecd_availability"
        raise ValueError(msg)
    if source == "bis":
        return _fetch_probe_bis(spec, probe_start)

    msg = f"Unknown source: {source}"
    raise ValueError(msg)


def _fetch_probe_eurostat(spec: dict[str, Any], probe_start: str) -> pd.DataFrame:
    """Probe Eurostat series with restricted date range."""
    filters = json.loads(spec["filters_json"])

    # Override startPeriod for lightweight probing
    # Eurostat API accepts integer year or YYYY-MM string
    probe_year = int(probe_start[:4])
    filters["startPeriod"] = probe_year

    raw = adapters.fetch_eurostat_raw(spec["dataset"], filters)
    return standardize.standardize_from_eurostat(raw, spec)


def _fetch_probe_ecb(spec: dict[str, Any], probe_start: str) -> pd.DataFrame:
    """Probe ECB series with restricted date range."""
    raw = adapters.fetch_ecb_raw(
        spec["dataset"],
        spec["key"],
        start_period=probe_start,
    )
    return standardize.standardize_from_ecb_like(raw, spec)


def _fetch_probe_bis(spec: dict[str, Any], probe_start: str) -> pd.DataFrame:
    """Probe BIS series with restricted date range."""
    raw = adapters.fetch_bis_raw(
        spec["dataset"],
        spec["key"],
        start_period=probe_start,
    )
    return standardize.standardize_from_bis(raw, spec)


def _probe_start_period() -> str:
    """Return fixed probe start period to catch discontinued series.

    Uses 2020-01 instead of a rolling window because some series
    (e.g., car registrations) were discontinued in 2021/2022.
    A rolling "last 12 months" window would miss these entirely.

    Returns:
        String in YYYY-MM format.
    """
    return "2020-01"


def _extract_template_id(series_id: str, country_iso2: str) -> str:
    """Extract template_id from series_id by stripping country prefix.

    Examples:
        ("DE_IP_001", "DE") → "IP_001"
        ("U2_FX_001", "U2") → "FX_001"

    Args:
        series_id: Full series identifier.
        country_iso2: Country code prefix to strip.

    Returns:
        Template identifier.
    """
    prefix = f"{country_iso2}_"
    if series_id.startswith(prefix):
        return series_id[len(prefix):]
    return series_id


def _is_data_absence_error(error: Exception) -> bool:
    """Classify whether an error indicates data absence vs API failure.

    Data absence errors (→ missing): the series simply doesn't exist for this
    country. These are expected and should not trigger retries.

    API failure errors (→ error): transient issues like rate limiting, timeouts,
    server errors. These may succeed on retry.

    Args:
        error: Exception from adapter or standardize call.

    Returns:
        True if the error indicates data absence (classify as "missing").
    """
    msg = str(error).lower()
    absence_patterns = [
        "no data returned",
        "empty response",
        "empty dataframe",
        "no time columns found",
        "no data",
        "404",
        "not found",
        "400 client error",
        "bad request",
    ]
    return any(pattern in msg for pattern in absence_patterns)
