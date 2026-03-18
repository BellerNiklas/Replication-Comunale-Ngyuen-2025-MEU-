from pathlib import Path

import pandas as pd

from meu_replication.cleaning.high_correlation import remove_high_correlation
from meu_replication.data_management.task_audit_high_correlation import (
    task_audit_high_correlation,
)
from tests.cleaning.test_correlation_audit import _build_inputs


def _write_window_inputs(
    *,
    panel: pd.DataFrame,
    base_dir: Path,
    window: str,
    drop_filter: str | None = None,
) -> tuple[Path, Path]:
    panel_path = base_dir / f"{window}.parquet"
    drop_path = base_dir / f"{window}_corr_drop_info.csv"

    window_panel = panel.copy()
    if drop_filter is not None:
        window_panel = window_panel[window_panel["series_id"] != drop_filter].reset_index(
            drop=True
        )

    _, drop_info = remove_high_correlation(window_panel, threshold=0.95)
    window_panel.to_parquet(panel_path, index=False)
    drop_info.to_csv(drop_path, index=False)
    return panel_path, drop_path


def test_task_audit_high_correlation_writes_review_outputs(tmp_path: Path):
    panel, raw_panel, registry, _ = _build_inputs()

    registry_path = tmp_path / "series_registry.csv"
    raw_panel_path = tmp_path / "macro_panel.parquet"
    registry.to_csv(registry_path, index=False)
    raw_panel.to_parquet(raw_panel_path, index=False)

    depends_on = {
        "registry": registry_path,
        "raw_panel": raw_panel_path,
    }
    for window, drop_filter in (
        ("2021_strict", None),
        ("2022_strict", None),
        ("2025_strict", "FR_HICP_002"),
    ):
        panel_path, drop_path = _write_window_inputs(
            panel=panel,
            base_dir=tmp_path,
            window=window,
            drop_filter=drop_filter,
        )
        depends_on[f"{window}_panel"] = panel_path
        depends_on[f"{window}_drop_info"] = drop_path

    analysis_dir = tmp_path / "analysis"
    report_path = tmp_path / "documents" / "correlation_cleaning_review.md"
    produces = {
        "2021_strict_pairs": analysis_dir / "2021_strict_pairs.csv",
        "2021_strict_decisions": analysis_dir / "2021_strict_decisions.csv",
        "2021_strict_country_summary": analysis_dir / "2021_strict_country_summary.csv",
        "2021_strict_source_summary": analysis_dir / "2021_strict_source_summary.csv",
        "2021_strict_category_summary": analysis_dir / "2021_strict_category_summary.csv",
        "2021_strict_family_summary": analysis_dir / "2021_strict_family_summary.csv",
        "2021_strict_components": analysis_dir / "2021_strict_components.csv",
        "2022_strict_pairs": analysis_dir / "2022_strict_pairs.csv",
        "2022_strict_decisions": analysis_dir / "2022_strict_decisions.csv",
        "2022_strict_country_summary": analysis_dir / "2022_strict_country_summary.csv",
        "2022_strict_source_summary": analysis_dir / "2022_strict_source_summary.csv",
        "2022_strict_category_summary": analysis_dir / "2022_strict_category_summary.csv",
        "2022_strict_family_summary": analysis_dir / "2022_strict_family_summary.csv",
        "2022_strict_components": analysis_dir / "2022_strict_components.csv",
        "2025_strict_pairs": analysis_dir / "2025_strict_pairs.csv",
        "2025_strict_decisions": analysis_dir / "2025_strict_decisions.csv",
        "2025_strict_country_summary": analysis_dir / "2025_strict_country_summary.csv",
        "2025_strict_source_summary": analysis_dir / "2025_strict_source_summary.csv",
        "2025_strict_category_summary": analysis_dir / "2025_strict_category_summary.csv",
        "2025_strict_family_summary": analysis_dir / "2025_strict_family_summary.csv",
        "2025_strict_components": analysis_dir / "2025_strict_components.csv",
        "window_overview": analysis_dir / "window_overview.csv",
        "pair_stability": analysis_dir / "pair_stability.csv",
        "fix_readiness": analysis_dir / "fix_readiness.csv",
        "report": report_path,
    }

    task_audit_high_correlation(depends_on=depends_on, produces=produces)

    pairs_2021 = pd.read_csv(produces["2021_strict_pairs"])
    pairs_2025 = pd.read_csv(produces["2025_strict_pairs"])
    stability = pd.read_csv(produces["pair_stability"])
    fix_readiness = pd.read_csv(produces["fix_readiness"])
    report = produces["report"].read_text(encoding="utf-8")

    assert len(pairs_2021) == 6
    assert len(pairs_2025) == 5
    assert "## Fetch-Suspicion Shortlist" in report
    assert "## Next-Step Fix Order" in report

    exact_pair = stability.loc[
        stability["pair_key"] == "DE::DE_OECD_SENT_001::DE_SENT_001"
    ].iloc[0]
    assert exact_pair["window_count"] == 3

    hicp_pair = stability.loc[
        stability["pair_key"] == "FR::FR_HICP_001::FR_HICP_002"
    ].iloc[0]
    assert hicp_pair["window_count"] == 2

    assert "drop_upstream_duplicate" in set(fix_readiness["recommended_disposition"])
    assert "investigate_fetch_or_mapping" in set(
        fix_readiness["recommended_disposition"]
    )
