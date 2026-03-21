"""Helpers for the generated analysis output taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from meu_replication.config import ANALYSIS

ANALYSIS_README_PATH = ANALYSIS / "README.md"
ANALYSIS_PANELS_DIR = ANALYSIS / "panels"
ANALYSIS_AUDITS_DIR = ANALYSIS / "audits"
CORRELATION_AUDIT_DIR = ANALYSIS_AUDITS_DIR / "correlation"


@dataclass(frozen=True)
class AnalysisOutputLayout:
    """Describe the canonical output layout for one analysis panel."""

    panel_name: str
    output_dir: Path
    readme_path: Path
    results_dir: Path
    euro_area_results_dir: Path
    country_results_dir: Path
    diagnostics_dir: Path
    sv_diagnostics_dir: Path
    artifacts_dir: Path
    factors_dir: Path
    forecasts_dir: Path
    sv_dir: Path
    uncertainty_dir: Path
    internal_dir: Path
    sv_r_dir: Path


def build_panel_output_layout(
    panel_name: str,
    *,
    panels_dir: Path = ANALYSIS_PANELS_DIR,
) -> AnalysisOutputLayout:
    """Build the canonical output layout for one panel."""
    output_dir = panels_dir / panel_name
    results_dir = output_dir / "results"
    diagnostics_dir = output_dir / "diagnostics"
    artifacts_dir = output_dir / "artifacts"
    internal_dir = output_dir / "internal"

    return AnalysisOutputLayout(
        panel_name=panel_name,
        output_dir=output_dir,
        readme_path=output_dir / "README.md",
        results_dir=results_dir,
        euro_area_results_dir=results_dir / "euro_area",
        country_results_dir=results_dir / "countries",
        diagnostics_dir=diagnostics_dir,
        sv_diagnostics_dir=diagnostics_dir / "sv",
        artifacts_dir=artifacts_dir,
        factors_dir=artifacts_dir / "factors",
        forecasts_dir=artifacts_dir / "forecasts",
        sv_dir=artifacts_dir / "sv",
        uncertainty_dir=artifacts_dir / "uncertainty",
        internal_dir=internal_dir,
        sv_r_dir=internal_dir / "sv_r",
    )


def build_analysis_root_readme(panel_names: Sequence[str]) -> str:
    """Render a short guide for the analysis output root."""
    rendered_panels = "\n".join(
        f"- `panels/{panel_name}/`" for panel_name in sorted(panel_names)
    )
    return (
        "# Analysis Outputs\n\n"
        "This directory is organized for browsing generated results.\n\n"
        "## Start Here\n\n"
        "- `panels/`: one directory per supported strict analysis panel.\n"
        "- `audits/`: cross-panel audit and review outputs.\n\n"
        "## Panel Structure\n\n"
        "Each panel directory is organized as:\n\n"
        "- `results/`: public deliverables such as EA MEU and country MEUs.\n"
        "- `diagnostics/`: validation and convergence summaries.\n"
        "- `artifacts/`: stage outputs used by downstream tasks.\n"
        "- `internal/`: implementation-specific caches such as raw R SV files.\n\n"
        "## Available Panels\n\n"
        f"{rendered_panels}\n\n"
        "## Audits\n\n"
        "- `audits/correlation/`: strict-panel correlation cleaning review outputs.\n"
    )


def build_panel_readme(layout: AnalysisOutputLayout) -> str:
    """Render a short guide for one panel directory."""
    panel_years = re.match(r"panel_(\d{4})_(\d{4})_", layout.panel_name)
    sample_window = (
        f"{panel_years.group(1)}-{panel_years.group(2)}"
        if panel_years
        else layout.panel_name
    )
    return (
        f"# {layout.panel_name}\n\n"
        f"Strict panel sample window: `{sample_window}`.\n\n"
        "## Results\n\n"
        "- `results/euro_area/meu_ea.parquet`: EA MEU series.\n"
        "- `results/countries/all_countries_meu.parquet`: consolidated country MEUs.\n"
        "- `results/countries/basket_membership.parquet`: country basket audit table.\n\n"
        "## Diagnostics\n\n"
        "- `diagnostics/sv/sv_diagnostics.parquet`\n"
        "- `diagnostics/sv/sv_validation_summary.parquet`\n"
        "- `diagnostics/sv/sv_validation_subset_metrics.parquet`\n\n"
        "## Pipeline Artifacts\n\n"
        "- `artifacts/factors/`: factor-stage objects.\n"
        "- `artifacts/forecasts/`: forecast-error stage objects.\n"
        "- `artifacts/sv/`: normalized stochastic-volatility parameters and latent states.\n"
        "- `artifacts/uncertainty/`: horizon-specific uncertainty panel.\n\n"
        "## Internal Cache\n\n"
        "- `internal/sv_r/`: raw R-backed SV intermediate files.\n"
    )
