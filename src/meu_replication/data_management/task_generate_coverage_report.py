"""Generate data coverage report for paper appendix."""

from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, MEU_COUNTRIES


def task_generate_coverage_report(
    depends_on: dict = {
        "panel": BLD / "data" / "clean" / "panel_2003_2022_strict.parquet",
        "availability": BLD / "meta" / "series_availability.parquet",
    },
    produces: Path = BLD / "documents" / "coverage_report.md",
) -> None:
    """Generate coverage report for paper appendix (short and boring task).

    Task only handles I/O. Real logic in _generate_coverage_markdown().
    """
    panel = pd.read_parquet(depends_on["panel"])
    availability = pd.read_parquet(depends_on["availability"])

    report = _generate_coverage_markdown(panel, availability)

    produces.write_text(report, encoding="utf-8")
    print(f"Wrote coverage report to {produces}")


def _generate_coverage_markdown(
    panel: pd.DataFrame,
    availability: pd.DataFrame,
) -> str:
    """Generate markdown coverage report (pure function).

    Args:
        panel: Clean macro panel DataFrame.
        availability: Probe availability manifest DataFrame.

    Returns:
        Markdown string with coverage tables.
    """
    lines = [
        "# Data Coverage Report",
        "",
        "## Summary",
        "",
        f"- **Total series fetched**: {panel.series_id.nunique()}",
        f"- **Total observations**: {len(panel):,}",
        f"- **Countries**: {panel.country_iso2.nunique()}",
        f"- **Date range**: {panel.date.min()} to {panel.date.max()}",
        "",
    ]

    # Availability summary
    lines.extend(
        [
            "## Probe Availability Summary",
            "",
        ]
    )
    for status, count in availability["status"].value_counts().items():
        pct = 100 * count / len(availability)
        lines.append(f"- **{status}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Per-country table
    lines.extend(
        [
            "## Coverage by Country",
            "",
            "| Country | Available | Fetched | Observations | Date Range |",
            "|---------|-----------|---------|--------------|------------|",
        ]
    )

    countries_order = sorted(MEU_COUNTRIES) + ["U2"]
    for country in countries_order:
        # Availability count
        country_avail = availability[
            (availability.country_iso2 == country)
            & (availability.status.isin(["ok", "ok_short"]))
        ]
        n_available = len(country_avail)

        # Panel stats
        country_panel = panel[panel.country_iso2 == country]
        n_fetched = country_panel.series_id.nunique()
        n_obs = len(country_panel)

        if n_obs > 0:
            date_range = f"{country_panel.date.min()} to {country_panel.date.max()}"
        else:
            date_range = "N/A"

        lines.append(
            f"| {country} | {n_available} | {n_fetched} | {n_obs:,} | {date_range} |"
        )
    lines.append("")

    # Per-category table
    lines.extend(
        [
            "## Coverage by Category",
            "",
            "| Category | Series | Observations |",
            "|----------|--------|--------------|",
        ]
    )

    for cat_name in sorted(panel.category_name.unique()):
        cat_data = panel[panel.category_name == cat_name]
        lines.append(
            f"| {cat_name} | {cat_data.series_id.nunique()} | {len(cat_data):,} |"
        )
    lines.append("")

    # Per-country variable count (Figure A1 equivalent)
    lines.extend(
        [
            "## Variables per Country (cf. Comunale & Nguyen 2025, Figure A1)",
            "",
            "| Country | Variables |",
            "|---------|-----------|",
        ]
    )

    country_series_counts = (
        panel[panel.country_iso2 != "U2"]
        .groupby("country_iso2")["series_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    for country, count in country_series_counts.items():
        lines.append(f"| {country} | {count} |")
    lines.append("")

    return "\n".join(lines)
