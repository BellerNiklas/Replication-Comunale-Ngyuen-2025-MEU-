"""Filter the transformed macro panel to strict endpoint-specific panels."""

from pathlib import Path

import pandas as pd

from meu_replication.cleaning.temporal_coverage import (
    build_variants,
    run_all_filter_variants_with_drop_info,
)
from meu_replication.config import (
    BLD,
    SAMPLE_END_2021,
    SAMPLE_END_2022,
    SAMPLE_END_2025,
    SAMPLE_START_TRANSFORMED,
)

_FILTER_VARIANTS = build_variants(
    windows=[
        ("2021", SAMPLE_START_TRANSFORMED, SAMPLE_END_2021),
        ("2022", SAMPLE_START_TRANSFORMED, SAMPLE_END_2022),
        ("2025", SAMPLE_START_TRANSFORMED, SAMPLE_END_2025),
    ],
    thresholds=[("strict", 0)],
)


def build_filter_variants(prefix: str = "panel") -> list[dict[str, object]]:
    """Build coverage variants for either the current or replication baseline."""
    if prefix == "panel":
        return [dict(variant) for variant in _FILTER_VARIANTS]

    return [
        {
            **variant,
            "key": variant["key"].replace("panel_", f"{prefix}_", 1),
        }
        for variant in _FILTER_VARIANTS
    ]


def _panel_filename_from_key(key: str) -> str:
    if key.startswith("replication_panel_"):
        return key.replace("replication_panel_", "replication_panel_2003_", 1)
    return key.replace("panel_", "panel_2003_", 1)


def _build_filter_outputs(variants: list[dict[str, object]]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for variant in variants:
        key = str(variant["key"])
        base_name = _panel_filename_from_key(key)
        outputs[key] = BLD / "data" / "clean" / f"{base_name}.parquet"
        outputs[f"{key}_drop_info"] = (
            BLD / "data" / "clean" / f"{base_name}_coverage_drop_info.csv"
        )
    return outputs


def task_filter_temporal_coverage(
    depends_on: Path = BLD / "data" / "clean" / "transformed_panel.parquet",
    produces: dict[str, Path] = _build_filter_outputs(_FILTER_VARIANTS),
) -> None:
    """Filter the transformed panel to the strict 2021, 2022, and 2025 windows."""
    panel = pd.read_parquet(depends_on)
    print(f"Input: {len(panel)} rows, {panel['series_id'].nunique()} series")

    panels = run_all_filter_variants_with_drop_info(panel, _FILTER_VARIANTS)

    for variant in _FILTER_VARIANTS:
        key = str(variant["key"])
        filtered, drop_info = panels[key]
        out_path = produces[key]
        filtered.to_parquet(out_path, index=False)
        enriched_drop_info = drop_info.assign(
            sample_start=str(variant["start"]),
            sample_end=str(variant["end"]),
            rule=str(variant["label"]),
        )
        enriched_drop_info.to_csv(produces[f"{key}_drop_info"], index=False)

    print(f"Wrote {len(panels)} filtered panels")
