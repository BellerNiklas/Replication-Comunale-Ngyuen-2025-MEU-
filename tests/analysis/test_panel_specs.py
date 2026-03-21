import pytest

from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
    DEFAULT_ANALYSIS_PANEL,
    get_analysis_panel_spec,
)
from meu_replication.config import ANALYSIS, BLD


def test_analysis_panel_specs_resolve_supported_inputs_and_output_dirs():
    panel_names = [spec.panel_name for spec in ANALYSIS_PANELS]
    task_ids = [spec.task_id for spec in ANALYSIS_PANELS]

    assert panel_names == [
        "panel_2003_2021_strict_corr",
        "panel_2003_2022_strict_corr",
        "panel_2003_2025_strict_corr",
    ]
    assert task_ids == ["analysis_2021", "analysis_2022", "analysis_2025"]
    assert DEFAULT_ANALYSIS_PANEL.panel_name == "panel_2003_2022_strict_corr"

    output_dirs = {spec.output_dir for spec in ANALYSIS_PANELS}
    assert len(output_dirs) == 3

    for spec in ANALYSIS_PANELS:
        assert spec.cleaned_panel_path == (
            BLD / "data" / "clean" / f"{spec.panel_name}.parquet"
        )
        assert spec.output_dir == ANALYSIS / "panels" / spec.panel_name
        assert spec.layout.output_dir == spec.output_dir
        assert spec.layout.euro_area_results_dir == (
            spec.output_dir / "results" / "euro_area"
        )
        assert spec.layout.country_results_dir == spec.output_dir / "results" / "countries"
        assert spec.layout.sv_diagnostics_dir == spec.output_dir / "diagnostics" / "sv"
        assert spec.layout.factors_dir == spec.output_dir / "artifacts" / "factors"
        assert spec.layout.forecasts_dir == spec.output_dir / "artifacts" / "forecasts"
        assert spec.layout.sv_dir == spec.output_dir / "artifacts" / "sv"
        assert spec.layout.uncertainty_dir == (
            spec.output_dir / "artifacts" / "uncertainty"
        )
        assert spec.layout.sv_r_dir == spec.output_dir / "internal" / "sv_r"


def test_get_analysis_panel_spec_raises_for_unknown_panel():
    with pytest.raises(ValueError, match="Unsupported analysis panel"):
        get_analysis_panel_spec("panel_2003_2099_strict_corr")
