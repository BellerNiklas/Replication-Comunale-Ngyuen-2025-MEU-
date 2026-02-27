"""Filter clean macro panel to series with full temporal coverage."""

import math
from pathlib import Path

import pandas as pd

from meu_replication.config import (
    BLD,
    SAMPLE_END,
    SAMPLE_END_ALT,
    SAMPLE_START,
)

# Variant definitions: 2 windows x 2 thresholds
_COVERAGE_THRESHOLD = 0.02  # 98% coverage


def _build_expected_months(sample_start: str, sample_end: str) -> set[str]:
    """Build canonical set of YYYY-MM strings for the full sample period.

    Uses pd.period_range to generate every month from start to end inclusive,
    then converts to the same YYYY-MM string format used in the clean panel.

    Args:
        sample_start: First month of sample period ("YYYY-MM").
        sample_end: Last month of sample period ("YYYY-MM").

    Returns:
        Set of YYYY-MM strings for every month in [sample_start, sample_end].
    """
    periods = pd.period_range(sample_start, sample_end, freq="M")
    return {p.strftime("%Y-%m") for p in periods}


def filter_by_temporal_coverage(
    df: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    allowed_missing: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter panel to series with sufficient monthly coverage.

    Pure function: constructs new DataFrames, no mutations.

    Args:
        df: Clean macro panel with columns [date, value, series_id,
            country_iso2, variable_name, category, category_name, source].
        sample_start: First month of sample period ("YYYY-MM").
        sample_end: Last month of sample period ("YYYY-MM").
        allowed_missing: Maximum number of missing months tolerated.
            0 = strict (every month required), >0 = near-complete.

    Returns:
        Tuple of (filtered_panel, drop_info) where:
        - filtered_panel: rows restricted to sample period, only for series
          with at most allowed_missing months absent.
        - drop_info: DataFrame with columns [series_id, country_iso2,
          variable_name, category_name, n_months, n_missing] for every
          series that was dropped due to insufficient coverage.
    """
    expected_months = _build_expected_months(sample_start, sample_end)

    empty_drop = pd.DataFrame(
        columns=[
            "series_id",
            "country_iso2",
            "variable_name",
            "category_name",
            "n_months",
            "n_missing",
        ]
    )

    if df.empty:
        return df.copy(), empty_drop

    # Restrict to sample period
    in_period = df[(df["date"] >= sample_start) & (df["date"] <= sample_end)].copy()

    # For each series, check coverage against canonical month set
    series_meta = in_period.groupby("series_id").agg(
        country_iso2=("country_iso2", "first"),
        variable_name=("variable_name", "first"),
        category_name=("category_name", "first"),
    )

    present_by_series = in_period.groupby("series_id")["date"].apply(set)

    # Classify each series
    keep_ids = []
    drop_rows = []

    for series_id, present in present_by_series.items():
        missing = expected_months - present
        if len(missing) <= allowed_missing:
            keep_ids.append(series_id)
        else:
            meta = series_meta.loc[series_id]
            drop_rows.append(
                {
                    "series_id": series_id,
                    "country_iso2": meta["country_iso2"],
                    "variable_name": meta["variable_name"],
                    "category_name": meta["category_name"],
                    "n_months": len(present),
                    "n_missing": len(missing),
                }
            )

    # Build filtered panel (only kept series, only sample period rows)
    filtered_panel = (
        in_period[in_period["series_id"].isin(keep_ids)]
        .sort_values(["country_iso2", "series_id", "date"])
        .reset_index(drop=True)
    )

    # Build drop info
    if drop_rows:
        drop_info = (
            pd.DataFrame(drop_rows)
            .sort_values(["country_iso2", "series_id"])
            .reset_index(drop=True)
        )
    else:
        drop_info = empty_drop.copy()

    return filtered_panel, drop_info


def generate_filter_report(
    n_total: int,
    n_kept: int,
    n_dropped: int,
    country_totals: dict[str, int],
    drop_info: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    expected_months: int,
    allowed_missing: int = 0,
    variant_label: str = "",
) -> str:
    """Generate markdown report summarizing the temporal coverage filter.

    Pure function: no side effects.

    Args:
        n_total: Total number of series before filtering.
        n_kept: Number of series kept (full coverage).
        n_dropped: Number of series dropped (incomplete coverage).
        country_totals: Dict mapping country_iso2 -> total series count
            (before filtering).
        drop_info: DataFrame with dropped series details.
        sample_start: First month of sample period.
        sample_end: Last month of sample period.
        expected_months: Number of months in the sample period.
        allowed_missing: Maximum missing months tolerated.
        variant_label: Human-readable label for this variant.

    Returns:
        Markdown string with summary, per-country table, and dropped detail.
    """
    lines = [
        "# Temporal Coverage Filter Report",
        "",
        "## Summary",
        "",
    ]

    if variant_label:
        lines.append(f"- **Variant**: {variant_label}")

    lines.extend(
        [
            f"- **Sample period**: {sample_start} to {sample_end}"
            f" ({expected_months} months)",
            f"- **Allowed missing**: {allowed_missing} months",
            f"- **Total series**: {n_total}",
            f"- **Kept**: {n_kept}",
            f"- **Dropped** (incomplete): {n_dropped}",
            "",
        ]
    )

    # Per-country survival table
    lines.extend(
        [
            "## Per-Country Survival",
            "",
            "| Country | Total | Kept | Dropped | Survival % |",
            "|---------|-------|------|---------|------------|",
        ]
    )

    dropped_per_country = (
        drop_info.groupby("country_iso2")["series_id"].count().to_dict()
        if not drop_info.empty
        else {}
    )

    for country in sorted(country_totals):
        total = country_totals[country]
        dropped = dropped_per_country.get(country, 0)
        kept = total - dropped
        pct = 100 * kept / total if total > 0 else 0
        lines.append(f"| {country} | {total} | {kept} | {dropped} | {pct:.1f} |")

    lines.append("")

    # Dropped series detail
    lines.extend(
        [
            "## Dropped Series Detail",
            "",
            "| Country | Series ID | Variable | Category | Months Present | Missing |",
            "|---------|-----------|----------|----------|---------------|---------|",
        ]
    )

    if not drop_info.empty:
        for _, row in drop_info.iterrows():
            lines.append(
                f"| {row['country_iso2']} "
                f"| {row['series_id']} "
                f"| {row['variable_name']} "
                f"| {row['category_name']} "
                f"| {row['n_months']} "
                f"| {row['n_missing']} |"
            )
    else:
        lines.append("| (none) | — | — | — | — | — |")

    lines.append("")

    return "\n".join(lines)


def generate_comparative_report(
    variants: list[dict],
) -> str:
    """Generate comparative markdown report across all filter variants.

    Pure function: no side effects.

    Args:
        variants: List of dicts, each with keys: label, start, end,
            allowed_missing, n_total, n_kept, n_dropped, country_totals,
            drop_info.

    Returns:
        Markdown string with overview table and per-country comparison.
    """
    lines = [
        "# Temporal Coverage Filter Report",
        "",
        "## Overview",
        "",
        "| Variant | Window | Allowed Missing | Total | Kept | Dropped |",
        "|---------|--------|-----------------|-------|------|---------|",
    ]

    lines.extend(
        f"| {v['label']} "
        f"| {v['start']}-{v['end']} "
        f"| {v['allowed_missing']} "
        f"| {v['n_total']} "
        f"| {v['n_kept']} "
        f"| {v['n_dropped']} |"
        for v in variants
    )

    lines.append("")

    # Per-country comparison table
    all_countries = sorted({c for v in variants for c in v["country_totals"]})
    variant_labels = [v["label"] for v in variants]

    lines.extend(
        [
            "## Per-Country Comparison (Kept Series)",
            "",
            "| Country | " + " | ".join(variant_labels) + " |",
            "|---------|"
            + "|".join("-" * (len(lbl) + 2) for lbl in variant_labels)
            + "|",
        ]
    )

    for country in all_countries:
        cells = []
        for v in variants:
            total = v["country_totals"].get(country, 0)
            dropped_map = (
                v["drop_info"].groupby("country_iso2")["series_id"].count().to_dict()
                if not v["drop_info"].empty
                else {}
            )
            dropped = dropped_map.get(country, 0)
            kept = total - dropped
            cells.append(str(kept))
        lines.append(f"| {country} | " + " | ".join(cells) + " |")

    lines.append("")

    # Note about car registrations
    lines.extend(
        [
            "## Notes",
            "",
            "- Commercial vehicle registrations (CARS_002, CARS_003, CARS_004) "
            "end 2021-12 across all countries.",
            "  They appear only in the 2021-window panels.",
            "- Passenger car registrations (CARS_001) have full coverage "
            "through 2022-12.",
            "",
        ]
    )

    return "\n".join(lines)


def _compute_allowed_missing(sample_start: str, sample_end: str) -> int:
    """Compute allowed missing months for the 98% coverage threshold."""
    n_months = len(_build_expected_months(sample_start, sample_end))
    return math.floor(_COVERAGE_THRESHOLD * n_months)


def task_filter_temporal_coverage(
    depends_on: Path = BLD / "data" / "clean" / "macro_panel.parquet",
    produces: dict[str, Path] = {
        "panel_2022_strict": BLD / "data" / "clean" / "panel_2003_2022_strict.parquet",
        "panel_2022_cov98": BLD / "data" / "clean" / "panel_2003_2022_cov98.parquet",
        "panel_2021_strict": BLD / "data" / "clean" / "panel_2003_2021_strict.parquet",
        "panel_2021_cov98": BLD / "data" / "clean" / "panel_2003_2021_cov98.parquet",
        "report": BLD / "documents" / "temporal_filter_report.md",
    },
) -> None:
    """Filter macro panel to series with sufficient temporal coverage.

    Produces 4 filtered panels (2 windows x 2 thresholds) and one
    comparative report. Task only handles I/O; real logic in
    filter_by_temporal_coverage() and generate_comparative_report().
    """
    panel = pd.read_parquet(depends_on)
    n_total = panel["series_id"].nunique()
    country_totals = panel.groupby("country_iso2")["series_id"].nunique().to_dict()

    print(f"Input: {len(panel)} rows, {n_total} series")

    filter_variants = [
        {
            "label": "2022_strict",
            "key": "panel_2022_strict",
            "start": SAMPLE_START,
            "end": SAMPLE_END,
            "allowed_missing": 0,
        },
        {
            "label": "2022_cov98",
            "key": "panel_2022_cov98",
            "start": SAMPLE_START,
            "end": SAMPLE_END,
            "allowed_missing": _compute_allowed_missing(SAMPLE_START, SAMPLE_END),
        },
        {
            "label": "2021_strict",
            "key": "panel_2021_strict",
            "start": SAMPLE_START,
            "end": SAMPLE_END_ALT,
            "allowed_missing": 0,
        },
        {
            "label": "2021_cov98",
            "key": "panel_2021_cov98",
            "start": SAMPLE_START,
            "end": SAMPLE_END_ALT,
            "allowed_missing": _compute_allowed_missing(SAMPLE_START, SAMPLE_END_ALT),
        },
    ]

    variant_results = []

    for variant in filter_variants:
        filtered, drop_info = filter_by_temporal_coverage(
            panel,
            variant["start"],
            variant["end"],
            variant["allowed_missing"],
        )

        n_kept = filtered["series_id"].nunique() if not filtered.empty else 0
        n_dropped = drop_info["series_id"].nunique() if not drop_info.empty else 0

        print(
            f"  {variant['label']}: kept {n_kept}, "
            f"dropped {n_dropped} ({len(filtered)} rows)"
        )

        # Write panel
        out_path = produces[variant["key"]]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(out_path, index=False)

        variant_results.append(
            {
                "label": variant["label"],
                "start": variant["start"],
                "end": variant["end"],
                "allowed_missing": variant["allowed_missing"],
                "n_total": n_total,
                "n_kept": n_kept,
                "n_dropped": n_dropped,
                "country_totals": country_totals,
                "drop_info": drop_info,
            }
        )

    report = generate_comparative_report(variant_results)

    produces["report"].parent.mkdir(parents=True, exist_ok=True)
    produces["report"].write_text(report, encoding="utf-8")
    print(f"Wrote comparative report to {produces['report']}")
