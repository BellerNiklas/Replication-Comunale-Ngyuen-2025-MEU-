"""Filter clean macro panel to series with full temporal coverage."""

from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, SAMPLE_END, SAMPLE_START


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter panel to series with full monthly coverage in the sample period.

    Pure function: constructs new DataFrames, no mutations.

    Args:
        df: Clean macro panel with columns [date, value, series_id,
            country_iso2, variable_name, category, category_name, source].
        sample_start: First month of sample period ("YYYY-MM").
        sample_end: Last month of sample period ("YYYY-MM").

    Returns:
        Tuple of (filtered_panel, drop_info) where:
        - filtered_panel: rows restricted to sample period, only for series
          that have every expected month present.
        - drop_info: DataFrame with columns [series_id, country_iso2,
          variable_name, category_name, n_months, n_missing] for every
          series that was dropped due to incomplete coverage.
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
    series_meta = (
        in_period.groupby("series_id")
        .agg(
            country_iso2=("country_iso2", "first"),
            variable_name=("variable_name", "first"),
            category_name=("category_name", "first"),
        )
    )

    present_by_series = in_period.groupby("series_id")["date"].apply(set)

    # Classify each series
    keep_ids = []
    drop_rows = []

    for series_id, present in present_by_series.items():
        missing = expected_months - present
        if not missing:
            keep_ids.append(series_id)
        else:
            meta = series_meta.loc[series_id]
            drop_rows.append({
                "series_id": series_id,
                "country_iso2": meta["country_iso2"],
                "variable_name": meta["variable_name"],
                "category_name": meta["category_name"],
                "n_months": len(present),
                "n_missing": len(missing),
            })

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
        drop_info = pd.DataFrame(
            columns=[
                "series_id",
                "country_iso2",
                "variable_name",
                "category_name",
                "n_months",
                "n_missing",
            ]
        )

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

    Returns:
        Markdown string with summary, per-country table, and dropped detail.
    """
    lines = [
        "# Temporal Coverage Filter Report",
        "",
        "## Summary",
        "",
        f"- **Sample period**: {sample_start} to {sample_end}"
        f" ({expected_months} months)",
        f"- **Total series**: {n_total}",
        f"- **Kept** (full coverage): {n_kept}",
        f"- **Dropped** (incomplete): {n_dropped}",
        "",
    ]

    # Per-country survival table
    lines.extend([
        "## Per-Country Survival",
        "",
        "| Country | Total | Kept | Dropped | Survival % |",
        "|---------|-------|------|---------|------------|",
    ])

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
    lines.extend([
        "## Dropped Series Detail",
        "",
        "| Country | Series ID | Variable | Category | Months Present | Missing |",
        "|---------|-----------|----------|----------|---------------|---------|",
    ])

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


def task_filter_temporal_coverage(
    depends_on: Path = BLD / "data" / "clean" / "macro_panel.parquet",
    produces: dict[str, Path] = {
        "panel": BLD / "data" / "clean" / "macro_panel_filtered.parquet",
        "report": BLD / "documents" / "temporal_filter_report.md",
    },
) -> None:
    """Filter macro panel to series with full temporal coverage.

    Task only handles I/O. Real logic in filter_by_temporal_coverage()
    and generate_filter_report().
    """
    panel = pd.read_parquet(depends_on)
    n_total = panel["series_id"].nunique()
    country_totals = (
        panel.groupby("country_iso2")["series_id"].nunique().to_dict()
    )

    print(f"Input: {len(panel)} rows, {n_total} series")

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

    n_kept = filtered["series_id"].nunique()
    n_dropped = drop_info["series_id"].nunique() if not drop_info.empty else 0
    expected_months = len(_build_expected_months(SAMPLE_START, SAMPLE_END))

    print(f"Kept: {n_kept} series ({len(filtered)} rows)")
    print(f"Dropped: {n_dropped} series (incomplete coverage)")

    report = generate_filter_report(
        n_total=n_total,
        n_kept=n_kept,
        n_dropped=n_dropped,
        country_totals=country_totals,
        drop_info=drop_info,
        sample_start=SAMPLE_START,
        sample_end=SAMPLE_END,
        expected_months=expected_months,
    )

    produces["panel"].parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(produces["panel"], index=False)
    print(f"Wrote filtered panel to {produces['panel']}")

    produces["report"].parent.mkdir(parents=True, exist_ok=True)
    produces["report"].write_text(report, encoding="utf-8")
    print(f"Wrote filter report to {produces['report']}")
