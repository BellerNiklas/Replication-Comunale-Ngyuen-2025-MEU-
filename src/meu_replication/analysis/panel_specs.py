"""Panel specifications for the multi-panel MEU analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meu_replication.analysis.output_layout import (
    AnalysisOutputLayout,
    build_panel_output_layout,
)
from meu_replication.config import BLD


@dataclass(frozen=True)
class AnalysisPanelSpec:
    """Describe one supported cleaned panel and its analysis output directory."""

    task_id: str
    panel_name: str
    cleaned_panel_path: Path
    output_dir: Path
    layout: AnalysisOutputLayout


_SUPPORTED_PANELS: tuple[tuple[str, str], ...] = (
    ("analysis_2021", "panel_2003_2021_strict_corr"),
    ("analysis_2022", "panel_2003_2022_strict_corr"),
    ("analysis_2025", "panel_2003_2025_strict_corr"),
)

ANALYSIS_PANELS: tuple[AnalysisPanelSpec, ...] = tuple(
    AnalysisPanelSpec(
        task_id=task_id,
        panel_name=panel_name,
        cleaned_panel_path=BLD / "data" / "clean" / f"{panel_name}.parquet",
        output_dir=layout.output_dir,
        layout=layout,
    )
    for task_id, panel_name in _SUPPORTED_PANELS
    for layout in [build_panel_output_layout(panel_name)]
)

ANALYSIS_PANELS_BY_NAME: dict[str, AnalysisPanelSpec] = {
    spec.panel_name: spec for spec in ANALYSIS_PANELS
}

DEFAULT_ANALYSIS_PANEL: AnalysisPanelSpec = ANALYSIS_PANELS_BY_NAME[
    "panel_2003_2022_strict_corr"
]

NON_DEFAULT_ANALYSIS_PANELS: tuple[AnalysisPanelSpec, ...] = tuple(
    spec for spec in ANALYSIS_PANELS if spec != DEFAULT_ANALYSIS_PANEL
)


def get_analysis_panel_spec(panel_name: str) -> AnalysisPanelSpec:
    """Return the registered panel specification for a supported analysis panel."""
    try:
        return ANALYSIS_PANELS_BY_NAME[panel_name]
    except KeyError as exc:
        msg = f"Unsupported analysis panel: {panel_name}"
        raise ValueError(msg) from exc
