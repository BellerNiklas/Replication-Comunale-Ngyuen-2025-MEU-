from pathlib import Path

from meu_replication.analysis.output_layout import (
    ANALYSIS_AUDITS_DIR,
    ANALYSIS_PANELS_DIR,
    ANALYSIS_README_PATH,
    CORRELATION_AUDIT_DIR,
    build_analysis_root_readme,
    build_panel_output_layout,
    build_panel_readme,
)


def test_build_panel_output_layout_uses_results_first_taxonomy(tmp_path: Path):
    layout = build_panel_output_layout(
        "panel_2003_2025_strict_corr",
        panels_dir=tmp_path / "analysis" / "panels",
    )

    assert layout.output_dir == tmp_path / "analysis" / "panels" / "panel_2003_2025_strict_corr"
    assert layout.readme_path == layout.output_dir / "README.md"
    assert layout.euro_area_results_dir == layout.output_dir / "results" / "euro_area"
    assert layout.country_results_dir == layout.output_dir / "results" / "countries"
    assert layout.sv_diagnostics_dir == layout.output_dir / "diagnostics" / "sv"
    assert layout.factors_dir == layout.output_dir / "artifacts" / "factors"
    assert layout.forecasts_dir == layout.output_dir / "artifacts" / "forecasts"
    assert layout.sv_dir == layout.output_dir / "artifacts" / "sv"
    assert layout.uncertainty_dir == layout.output_dir / "artifacts" / "uncertainty"
    assert layout.sv_r_dir == layout.output_dir / "internal" / "sv_r"


def test_analysis_root_constants_point_to_reorganized_tree():
    assert ANALYSIS_README_PATH == ANALYSIS_PANELS_DIR.parent / "README.md"
    assert ANALYSIS_AUDITS_DIR == ANALYSIS_PANELS_DIR.parent / "audits"
    assert CORRELATION_AUDIT_DIR == ANALYSIS_AUDITS_DIR / "correlation"


def test_readme_renderers_reference_new_locations():
    layout = build_panel_output_layout("panel_2003_2022_strict_corr")

    root_readme = build_analysis_root_readme([layout.panel_name])
    panel_readme = build_panel_readme(layout)

    assert "panels/" in root_readme
    assert "audits/correlation/" in root_readme
    assert "results/euro_area/meu_ea.parquet" in panel_readme
    assert "results/countries/all_countries_meu.parquet" in panel_readme
    assert "artifacts/uncertainty/" in panel_readme
    assert "internal/sv_r/" in panel_readme
