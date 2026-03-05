"""Deterministic metadata model for raw data snapshot logging."""

from typing import TypedDict

import pandas as pd

from meu_replication.config import MEU_COUNTRIES


class SourceFileInfo(TypedDict):
    """In-memory metadata for one fetched parquet source file."""

    df: pd.DataFrame
    file_size_bytes: int
    checksum: str


class CountryStats(TypedDict):
    """Per-country snapshot metadata written to raw_snapshot_meta.json."""

    series_count: int
    rows: int
    file_size_bytes: int
    checksum: str


class SourceStats(TypedDict):
    """Per-source aggregation metadata written to raw_snapshot_meta.json."""

    total_rows: int
    total_series: int
    countries: dict[str, CountryStats]


class FetchMetadata(TypedDict):
    """Top-level raw snapshot metadata schema (v2, deterministic)."""

    metadata_schema_version: int
    git_commit: str
    git_branch: str
    registry_checksum: str
    registry_rows: int
    availability_summary: dict[str, int]
    sources: dict[str, SourceStats]


def build_fetch_metadata(
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    registry_checksum: str,
    source_files: dict[str, SourceFileInfo | None],
    git_commit: str,
    git_branch: str,
) -> FetchMetadata:
    """Build deterministic fetch metadata dict from pre-loaded data.

    Pure function: receives data and identifiers, returns dict.

    Args:
        availability: Probe availability manifest DataFrame.
        registry: Series registry DataFrame.
        registry_checksum: SHA-256 hex digest of the registry file.
        source_files: Dict mapping label -> source file info or None.
        git_commit: Current git commit hash (or fallback value).
        git_branch: Current git branch name (or fallback value).

    Returns:
        Metadata dict ready for JSON serialization.
    """
    avail_counts = availability["status"].value_counts().to_dict()
    avail_summary = {str(k): int(v) for k, v in avail_counts.items()}

    metadata: FetchMetadata = {
        "metadata_schema_version": 2,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "registry_checksum": registry_checksum,
        "registry_rows": len(registry),
        "availability_summary": avail_summary,
        "sources": {},
    }

    for source in ("eurostat", "ecb", "oecd", "bis"):
        countries = [*MEU_COUNTRIES]
        if source == "ecb":
            countries.append("U2")

        source_rows = 0
        source_series: set[str] = set()
        country_stats: dict[str, CountryStats] = {}

        for country in countries:
            key = f"{source}_{country}"
            info = source_files.get(key)
            if info is not None:
                df = info["df"]
                rows = len(df)
                series_count = df["series_id"].nunique() if rows > 0 else 0
                source_rows += rows
                if rows > 0:
                    source_series.update(
                        str(sid) for sid in df["series_id"].dropna().unique()
                    )
                country_stats[country] = {
                    "series_count": int(series_count),
                    "rows": rows,
                    "file_size_bytes": info["file_size_bytes"],
                    "checksum": info["checksum"],
                }
            else:
                country_stats[country] = {
                    "series_count": 0,
                    "rows": 0,
                    "file_size_bytes": 0,
                    "checksum": "",
                }

        metadata["sources"][source] = {
            "total_rows": source_rows,
            "total_series": len(source_series),
            "countries": country_stats,
        }

    return metadata
