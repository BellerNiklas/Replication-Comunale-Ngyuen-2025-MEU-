import hashlib

import pandas as pd

from meu_replication.data_fetch.fetch_metadata_model import (
    SourceFileInfo,
    build_fetch_metadata,
)


def _source_df(series_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": series_ids,
            "value": [1.0] * len(series_ids),
        }
    )


def test_build_fetch_metadata_schema_v2_is_deterministic():
    availability = pd.DataFrame({"status": ["ok", "ok", "missing"]})
    registry = pd.DataFrame({"series_id": ["A", "B", "C"]})
    checksum = hashlib.sha256(b"registry").hexdigest()

    source_files: dict[str, SourceFileInfo | None] = {
        "eurostat_DE": {
            "df": _source_df(["DE_A", "DE_B"]),
            "file_size_bytes": 100,
            "checksum": "sum1",
        },
        "ecb_U2": {
            "df": _source_df(["U2_A"]),
            "file_size_bytes": 50,
            "checksum": "sum2",
        },
    }

    first = build_fetch_metadata(
        availability=availability,
        registry=registry,
        registry_checksum=checksum,
        source_files=source_files,
        git_commit="abc123",
        git_branch="main",
    )
    second = build_fetch_metadata(
        availability=availability,
        registry=registry,
        registry_checksum=checksum,
        source_files=source_files,
        git_commit="abc123",
        git_branch="main",
    )

    assert first == second
    assert first["metadata_schema_version"] == 2
    assert "fetch_timestamp" not in first
    assert first["registry_checksum"] == checksum
    assert first["availability_summary"]["ok"] == 2


def test_build_fetch_metadata_includes_default_country_stats_for_missing_files():
    availability = pd.DataFrame({"status": ["ok"]})
    registry = pd.DataFrame({"series_id": ["A"]})

    source_files: dict[str, SourceFileInfo | None] = {
        "eurostat_DE": {
            "df": _source_df(["DE_A"]),
            "file_size_bytes": 111,
            "checksum": "sum-de",
        }
    }

    metadata = build_fetch_metadata(
        availability=availability,
        registry=registry,
        registry_checksum="checksum",
        source_files=source_files,
        git_commit="abc123",
        git_branch="main",
    )

    # A missing country/source file should still appear with zeroed stats.
    be_stats = metadata["sources"]["eurostat"]["countries"]["BE"]
    assert be_stats["rows"] == 0
    assert be_stats["checksum"] == ""


