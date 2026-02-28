"""Remove highly correlated series from the strict 2003-2022 panel."""

from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, HIGH_CORR_THRESHOLD
from meu_replication.data_management.high_correlation import remove_high_correlation


def task_remove_high_correlation(
    depends_on: Path = BLD / "data" / "clean" / "panel_2003_2022_strict.parquet",
    produces: dict[str, Path] = {
        "panel": BLD / "data" / "clean" / "panel_2003_2022_strict_corr.parquet",
        "drop_info": BLD / "data" / "clean" / "high_corr_drop_info.csv",
    },
) -> None:
    """Remove series with pairwise |correlation| > threshold, per country.

    Reads the strict-coverage 2003-2022 panel, removes one variable from
    each highly correlated pair within each country, and writes the
    filtered panel and drop metadata.
    """
    panel = pd.read_parquet(depends_on)
    n_before = panel["series_id"].nunique()

    filtered, drop_info = remove_high_correlation(panel, threshold=HIGH_CORR_THRESHOLD)
    n_after = filtered["series_id"].nunique()

    produces["panel"].parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(produces["panel"], index=False)
    drop_info.to_csv(produces["drop_info"], index=False)

    print(
        f"High-correlation filter: {n_before} -> {n_after} series "
        f"({n_before - n_after} dropped, threshold={HIGH_CORR_THRESHOLD})"
    )
