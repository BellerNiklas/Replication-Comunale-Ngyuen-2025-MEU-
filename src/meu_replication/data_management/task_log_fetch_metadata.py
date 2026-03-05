"""Log metadata about raw data fetch for reproducibility audit."""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, MEU_COUNTRIES, SRC
from meu_replication.data_fetch.fetch_metadata_model import (
    SourceFileInfo,
    build_fetch_metadata,
)


def _build_meta_depends() -> dict[str, Path]:
    """Build dependency dict for registry + all 77 per-country snapshot files."""
    deps = {
        "registry": SRC / "registry" / "series_registry.csv",
        "availability": BLD / "meta" / "series_availability.parquet",
    }
    for country in MEU_COUNTRIES:
        for source in ("eurostat", "ecb", "oecd", "bis"):
            deps[f"{source}_{country}"] = (
                BLD / "data" / "raw" / source / f"{country}_snapshot.parquet"
            )
    deps["ecb_U2"] = BLD / "data" / "raw" / "ecb" / "U2_snapshot.parquet"
    return deps


def task_log_fetch_metadata(
    depends_on: dict = _build_meta_depends(),
    produces: Path = BLD / "meta" / "raw_snapshot_meta.json",
) -> None:
    """Log deterministic metadata about raw fetch for replication package.

    Short and boring task: reads all dependency files, delegates aggregation
    to build_fetch_metadata(), writes JSON.
    """
    availability = pd.read_parquet(depends_on["availability"])
    registry = pd.read_csv(depends_on["registry"])
    registry_checksum = _file_sha256(depends_on["registry"])

    source_files: dict[str, SourceFileInfo | None] = {}
    for key, path in depends_on.items():
        if key in ("registry", "availability"):
            continue
        if path.exists():
            df = pd.read_parquet(path)
            source_files[key] = {
                "df": df,
                "file_size_bytes": path.stat().st_size,
                "checksum": _file_sha256(path),
            }
        else:
            source_files[key] = None

    metadata = build_fetch_metadata(
        availability=availability,
        registry=registry,
        registry_checksum=registry_checksum,
        source_files=source_files,
        git_commit=_get_git_hash(),
        git_branch=_get_git_branch(),
    )

    produces.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Logged fetch metadata to {produces}")


def _get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=SRC.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_branch() -> str:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=SRC.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


