"""Render lightweight README guides for the analysis output tree."""

from __future__ import annotations

from pathlib import Path

import pytask

from meu_replication.analysis.output_layout import (
    ANALYSIS_README_PATH,
    build_analysis_root_readme,
    build_panel_readme,
)
from meu_replication.analysis.panel_specs import ANALYSIS_PANELS


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytask.task(id="analysis_index")
def task_render_analysis_root_readme(
    produces: Path = ANALYSIS_README_PATH,
) -> None:
    """Write the top-level analysis output guide."""
    _write_text(
        produces,
        build_analysis_root_readme([spec.panel_name for spec in ANALYSIS_PANELS]),
    )


for _spec in ANALYSIS_PANELS:

    @pytask.task(id=f"readme_{_spec.task_id}")
    def task_render_panel_readme(
        produces: Path = _spec.layout.readme_path,
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Write the panel-level analysis output guide."""
        spec = next(spec for spec in ANALYSIS_PANELS if spec.panel_name == panel_name)
        _write_text(produces, build_panel_readme(spec.layout))
