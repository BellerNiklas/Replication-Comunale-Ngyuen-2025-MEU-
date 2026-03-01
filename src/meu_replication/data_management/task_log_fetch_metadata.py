"""Log metadata about raw data fetch for reproducibility audit."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, MEU_COUNTRIES, SRC


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


def _build_fetch_metadata(
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    registry_checksum: str,
    source_files: dict[str, dict | None],
) -> dict:
    """Build fetch metadata dict from pre-loaded data.

    Pure function: receives data, returns dict.

    Args:
        availability: Probe availability manifest DataFrame.
        registry: Series registry DataFrame.
        registry_checksum: SHA-256 hex digest of the registry file.
        source_files: Dict mapping label -> {df, file_size_bytes, checksum}
            or None for missing files.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    avail_summary = availability["status"].value_counts().to_dict()

    metadata = {
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _get_git_hash(),
        "git_branch": _get_git_branch(),
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
        source_series = set()
        country_stats = {}

        for country in countries:
            key = f"{source}_{country}"
            info = source_files.get(key)
            if info is not None:
                df = info["df"]
                rows = len(df)
                series_count = df["series_id"].nunique() if rows > 0 else 0
                source_rows += rows
                source_series.update(df["series_id"].unique() if rows > 0 else [])
                country_stats[country] = {
                    "series_count": series_count,
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


def task_log_fetch_metadata(
    depends_on: dict = _build_meta_depends(),
    produces: Path = BLD / "meta" / "raw_snapshot_meta.json",
) -> None:
    """Log metadata about raw fetch for replication package.

    Short and boring task: reads all dependency files, delegates aggregation
    to _build_fetch_metadata(), writes JSON.
    """
    availability = pd.read_parquet(depends_on["availability"])
    registry = pd.read_csv(depends_on["registry"])
    registry_checksum = _file_sha256(depends_on["registry"])

    source_files = {}
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

    metadata = _build_fetch_metadata(
        availability, registry, registry_checksum, source_files
    )

    produces.write_text(json.dumps(metadata, indent=2))
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
