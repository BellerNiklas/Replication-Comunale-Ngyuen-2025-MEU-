"""Filter clean macro panel to series with full temporal coverage."""

from pathlib import Path

import pandas as pd

from meu_replication.cleaning.temporal_coverage import (
    build_variants,
    compute_allowed_missing,
    run_all_filter_variants,
)
from meu_replication.config import (
    BLD,
    SAMPLE_END,
    SAMPLE_END_ALT,
    SAMPLE_START_TRANSFORMED,
)

_FILTER_VARIANTS = build_variants(
    windows=[
        ("2022", SAMPLE_START_TRANSFORMED, SAMPLE_END),
        ("2021", SAMPLE_START_TRANSFORMED, SAMPLE_END_ALT),
    ],
    thresholds=[
        ("strict", 0),
        ("cov98", compute_allowed_missing(SAMPLE_START_TRANSFORMED, SAMPLE_END)),
    ],
)


def task_filter_temporal_coverage(
    depends_on: Path = BLD / "data" / "clean" / "transformed_panel.parquet",
    produces: dict[str, Path] = {
        v["key"]: BLD
        / "data"
        / "clean"
        / f"{v['key'].replace('panel_', 'panel_2003_')}.parquet"
        for v in _FILTER_VARIANTS
    },
) -> None:
    """Filter macro panel to series with sufficient temporal coverage.

    Produces 4 filtered panels (2 windows x 2 thresholds).
    """
    panel = pd.read_parquet(depends_on)
    print(f"Input: {len(panel)} rows, {panel['series_id'].nunique()} series")

    panels = run_all_filter_variants(panel, _FILTER_VARIANTS)

    for key, filtered in panels.items():
        out_path = produces[key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(out_path, index=False)

    print(f"Wrote {len(panels)} filtered panels")
