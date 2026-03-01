"""Remove highly correlated series from all temporal-coverage panels."""

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.cleaning.high_correlation import remove_high_correlation
from meu_replication.config import BLD, HIGH_CORR_THRESHOLD, PANEL_VARIANTS

for _key in PANEL_VARIANTS:

    @pytask.task(id=_key)
    def task_remove_high_correlation(
        key: str = _key,
        depends_on: Path = BLD / "data" / "clean" / f"{_key}.parquet",
        produces: dict[str, Path] = {
            "panel": BLD / "data" / "clean" / f"{_key}_corr.parquet",
            "drop_info": BLD / "data" / "clean" / f"{_key}_corr_drop_info.csv",
        },
    ) -> None:
        """Remove series with pairwise |correlation| > threshold, per country.

        Reads a temporal-coverage panel, removes one variable from each highly
        correlated pair within each country, and writes the filtered panel and
        drop metadata.
        """
        panel = pd.read_parquet(depends_on)
        n_before = panel["series_id"].nunique()

        filtered, drop_info = remove_high_correlation(
            panel, threshold=HIGH_CORR_THRESHOLD
        )
        n_after = filtered["series_id"].nunique()

        filtered.to_parquet(produces["panel"], index=False)
        drop_info.to_csv(produces["drop_info"], index=False)

        print(
            f"[{key}] High-correlation filter: {n_before} -> {n_after} series "
            f"({n_before - n_after} dropped, threshold={HIGH_CORR_THRESHOLD})"
        )
