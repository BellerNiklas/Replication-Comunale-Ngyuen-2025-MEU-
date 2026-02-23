"""Log metadata about raw data fetch for reproducibility audit."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from template_project.config import BLD, SRC


def task_log_fetch_metadata(
    depends_on: dict = {
        "registry": SRC / "data_management" / "registry" / "series_registry.csv",
        "eurostat": BLD / "data" / "raw" / "eurostat_snapshot.parquet",
        "ecb": BLD / "data" / "raw" / "ecb_snapshot.parquet",
        "oecd": BLD / "data" / "raw" / "oecd_snapshot.parquet",
        "bis": BLD / "data" / "raw" / "bis_snapshot.parquet",
    },
    produces: Path = BLD / "meta" / "raw_snapshot_meta.json",
) -> None:
    """Log metadata about raw fetch for replication package (short and boring task).

    Task only handles I/O and metadata computation.
    Metadata enables academic reproducibility and audit trails.
    """
    print("Computing fetch metadata...")

    # Compute metadata
    metadata = {
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _get_git_hash(),
        "git_branch": _get_git_branch(),
        "registry_checksum": _file_sha256(depends_on["registry"]),
        "registry_rows": len(pd.read_csv(depends_on["registry"])),
        "sources": {},
    }

    # Per-source statistics
    for source in ["eurostat", "ecb", "oecd", "bis"]:
        print(f"  Processing {source}...")
        df = pd.read_parquet(depends_on[source])
        metadata["sources"][source] = {
            "rows": len(df),
            "series_count": df["series_id"].nunique(),
            "file_size_bytes": depends_on[source].stat().st_size,
            "checksum": _file_sha256(depends_on[source]),
        }

    # Write JSON
    produces.parent.mkdir(parents=True, exist_ok=True)
    produces.write_text(json.dumps(metadata, indent=2))
    print(f"\nLogged fetch metadata to {produces}")
    print(f"  Timestamp: {metadata['fetch_timestamp']}")
    print(f"  Git commit: {metadata['git_commit'][:12]}...")
    print(f"  Sources: {list(metadata['sources'].keys())}")


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
