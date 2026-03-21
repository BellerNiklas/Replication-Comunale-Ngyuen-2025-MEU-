"""Build reproducible audit artifacts for strict-panel high-correlation cleaning."""

from pathlib import Path

import pandas as pd

from meu_replication.analysis.output_layout import CORRELATION_AUDIT_DIR
from meu_replication.cleaning.correlation_audit import (
    build_cross_window_correlation_audit,
    build_window_correlation_audit,
    render_correlation_review_markdown,
)
from meu_replication.config import (
    BLD,
    HIGH_CORR_THRESHOLD,
    SAMPLE_END_2021,
    SAMPLE_END_2022,
    SAMPLE_END_2025,
    SAMPLE_START_TRANSFORMED,
    SRC,
)

_AUDIT_VARIANTS = (
    ("2021_strict", "panel_2003_2021_strict", SAMPLE_END_2021),
    ("2022_strict", "panel_2003_2022_strict", SAMPLE_END_2022),
    ("2025_strict", "panel_2003_2025_strict", SAMPLE_END_2025),
)


def _build_audit_depends() -> dict[str, Path]:
    deps = {
        "registry": SRC / "registry" / "series_registry.csv",
        "raw_panel": BLD / "data" / "clean" / "macro_panel.parquet",
    }
    for window, panel_key, _ in _AUDIT_VARIANTS:
        deps[f"{window}_panel"] = BLD / "data" / "clean" / f"{panel_key}.parquet"
        deps[f"{window}_drop_info"] = (
            BLD / "data" / "clean" / f"{panel_key}_corr_drop_info.csv"
        )
    return deps


def _build_audit_outputs() -> dict[str, Path]:
    base = CORRELATION_AUDIT_DIR
    outputs: dict[str, Path] = {}
    for window, _, _ in _AUDIT_VARIANTS:
        outputs[f"{window}_pairs"] = base / f"{window}_pairs.csv"
        outputs[f"{window}_decisions"] = base / f"{window}_decisions.csv"
        outputs[f"{window}_country_summary"] = base / f"{window}_country_summary.csv"
        outputs[f"{window}_source_summary"] = base / f"{window}_source_summary.csv"
        outputs[f"{window}_category_summary"] = base / f"{window}_category_summary.csv"
        outputs[f"{window}_family_summary"] = base / f"{window}_family_summary.csv"
        outputs[f"{window}_components"] = base / f"{window}_components.csv"
    outputs["window_overview"] = base / "window_overview.csv"
    outputs["pair_stability"] = base / "pair_stability.csv"
    outputs["fix_readiness"] = base / "fix_readiness.csv"
    outputs["report"] = BLD / "documents" / "correlation_cleaning_review.md"
    return outputs


def task_audit_high_correlation(
    depends_on: dict[str, Path] = _build_audit_depends(),
    produces: dict[str, Path] = _build_audit_outputs(),
) -> None:
    """Audit all strict high-correlation cleaning outputs and write diagnostics."""
    registry = pd.read_csv(depends_on["registry"])
    raw_panel = pd.read_parquet(depends_on["raw_panel"])

    window_audits: dict[str, dict[str, pd.DataFrame]] = {}
    for window, _, sample_end in _AUDIT_VARIANTS:
        panel = pd.read_parquet(depends_on[f"{window}_panel"])
        drop_info = pd.read_csv(depends_on[f"{window}_drop_info"])
        window_audits[window] = build_window_correlation_audit(
            panel=panel,
            raw_panel=raw_panel,
            registry=registry,
            drop_info=drop_info,
            window=window,
            sample_start=SAMPLE_START_TRANSFORMED,
            sample_end=sample_end,
            threshold=HIGH_CORR_THRESHOLD,
        )

    combined = build_cross_window_correlation_audit(window_audits)
    report = render_correlation_review_markdown(window_audits, combined)

    for path in produces.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for window, _, _ in _AUDIT_VARIANTS:
        audit = window_audits[window]
        audit["pairs"].to_csv(produces[f"{window}_pairs"], index=False)
        audit["decisions"].to_csv(produces[f"{window}_decisions"], index=False)
        audit["summary_country"].to_csv(
            produces[f"{window}_country_summary"], index=False
        )
        audit["summary_source_pairs"].to_csv(
            produces[f"{window}_source_summary"], index=False
        )
        audit["summary_category_pairs"].to_csv(
            produces[f"{window}_category_summary"], index=False
        )
        audit["summary_family_pairs"].to_csv(
            produces[f"{window}_family_summary"], index=False
        )
        audit["components"].to_csv(produces[f"{window}_components"], index=False)

    combined["window_overview"].to_csv(produces["window_overview"], index=False)
    combined["pair_stability"].to_csv(produces["pair_stability"], index=False)
    combined["fix_readiness"].to_csv(produces["fix_readiness"], index=False)
    produces["report"].write_text(report, encoding="utf-8")
