"""Generate a stage-aware cleaning audit report for the current pipeline."""

from pathlib import Path

import pandas as pd

from meu_replication.config import BLD

_PAPER_TOTAL_OVERALL = 1470
_PAPER_TOTAL_COUNTRY_SPECIFIC = 1445
_PAPER_TOTAL_EA = 30
_PAPER_TOTAL_POST_CORR = 1330
_PAPER_COUNTRY_REFERENCES = {
    "DE": "not transcribed",
    "CY": "29/122",
    "GR": "30/122",
    "MT": "36/122",
    "IE": "not transcribed",
}
_VARIANT_LABELS = ("2022_strict", "2022_cov98", "2021_strict", "2021_cov98")


def _build_depends() -> dict[str, Path]:
    deps = {
        "registry": BLD.parent / "src" / "meu_replication" / "registry" / "series_registry.csv",
        "availability": BLD / "meta" / "series_availability.parquet",
        "raw": BLD / "data" / "clean" / "macro_panel.parquet",
        "transformed": BLD / "data" / "clean" / "transformed_panel.parquet",
        "sentiment_audit": BLD / "meta" / "sentiment_overlap_audit.csv",
    }
    for label in _VARIANT_LABELS:
        deps[label] = BLD / "data" / "clean" / f"panel_2003_{label}.parquet"
        deps[f"{label}_corr"] = BLD / "data" / "clean" / f"panel_2003_{label}_corr.parquet"
    return deps


def task_generate_coverage_report(
    depends_on: dict[str, Path] = _build_depends(),
    produces: Path = BLD / "documents" / "cleaning_audit.md",
) -> None:
    """Generate the cleaning audit report for the current-source pipeline."""
    tables = {key: _read_table(path) for key, path in depends_on.items()}
    report = _generate_cleaning_audit_markdown(tables)
    produces.write_text(report, encoding="utf-8")
    print(f"Wrote cleaning audit report to {produces}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    msg = f"Unsupported tabular input type: {path}"
    raise ValueError(msg)


def _count_registry(df: pd.DataFrame) -> int:
    return int(df["series_id"].nunique())


def _count_available(df: pd.DataFrame) -> int:
    return int(df[df["status"].isin(["ok", "ok_short"])]["series_id"].nunique())


def _count_panel(df: pd.DataFrame) -> int:
    return int(df["series_id"].nunique())


def _stage_summary_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    registry = tables["registry"]
    availability = tables["availability"]
    raw = tables["raw"]
    transformed = tables["transformed"]

    for label in _VARIANT_LABELS:
        rows.append(
            {
                "variant": label,
                "registry": _count_registry(registry),
                "available": _count_available(availability),
                "raw_fetched": _count_panel(raw),
                "transformed": _count_panel(transformed),
                "post_coverage": _count_panel(tables[label]),
                "post_correlation": _count_panel(tables[f"{label}_corr"]),
            }
        )
    return pd.DataFrame(rows)


def _country_compare_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    coverage_2022 = tables["2022_cov98"]
    correlation_2022 = tables["2022_cov98_corr"]
    coverage_2021 = tables["2021_cov98"]
    correlation_2021 = tables["2021_cov98_corr"]

    rows: list[dict[str, object]] = []
    for country in ("DE", "CY", "GR", "MT", "IE"):
        rows.append(
            {
                "country": country,
                "paper_reference": _PAPER_COUNTRY_REFERENCES[country],
                "cov98_2022": int(
                    coverage_2022[coverage_2022["country_iso2"] == country]["series_id"].nunique()
                ),
                "cov98_2022_corr": int(
                    correlation_2022[
                        correlation_2022["country_iso2"] == country
                    ]["series_id"].nunique()
                ),
                "cov98_2021": int(
                    coverage_2021[coverage_2021["country_iso2"] == country]["series_id"].nunique()
                ),
                "cov98_2021_corr": int(
                    correlation_2021[
                        correlation_2021["country_iso2"] == country
                    ]["series_id"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def _paper_totals_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    coverage_2022 = tables["2022_cov98"]
    correlation_2022 = tables["2022_cov98_corr"]
    coverage_2021 = tables["2021_cov98"]
    correlation_2021 = tables["2021_cov98_corr"]

    country_specific_2022 = int(
        coverage_2022[coverage_2022["country_iso2"] != "U2"]["series_id"].nunique()
    )
    country_specific_2021 = int(
        coverage_2021[coverage_2021["country_iso2"] != "U2"]["series_id"].nunique()
    )

    return pd.DataFrame(
        [
            {
                "metric": "total_overall_pre_corr_2022",
                "paper": _PAPER_TOTAL_OVERALL,
                "current_source": _count_panel(coverage_2022),
            },
            {
                "metric": "country_specific_pre_corr_2022",
                "paper": _PAPER_TOTAL_COUNTRY_SPECIFIC,
                "current_source": country_specific_2022,
            },
            {
                "metric": "ea_pre_corr_2022",
                "paper": _PAPER_TOTAL_EA,
                "current_source": _count_panel(
                    coverage_2022[coverage_2022["country_iso2"] == "U2"]
                ),
            },
            {
                "metric": "total_post_corr_2022",
                "paper": _PAPER_TOTAL_POST_CORR,
                "current_source": _count_panel(correlation_2022),
            },
            {
                "metric": "total_overall_pre_corr_2021",
                "paper": _PAPER_TOTAL_OVERALL,
                "current_source": _count_panel(coverage_2021),
            },
            {
                "metric": "country_specific_pre_corr_2021",
                "paper": _PAPER_TOTAL_COUNTRY_SPECIFIC,
                "current_source": country_specific_2021,
            },
            {
                "metric": "ea_pre_corr_2021",
                "paper": _PAPER_TOTAL_EA,
                "current_source": _count_panel(
                    coverage_2021[coverage_2021["country_iso2"] == "U2"]
                ),
            },
            {
                "metric": "total_post_corr_2021",
                "paper": _PAPER_TOTAL_POST_CORR,
                "current_source": _count_panel(correlation_2021),
            },
        ]
    )


def _summarize_sentiment_audit(audit: pd.DataFrame) -> pd.DataFrame:
    exact = audit[audit["all_values_equal"]]
    eurostat_dropped = audit[
        audit["recommended_replication_action"] == "drop_oecd_duplicate"
    ]
    return pd.DataFrame(
        [
            {"metric": "audited_pairs", "value": int(len(audit))},
            {"metric": "exact_duplicate_pairs", "value": int(len(exact))},
            {
                "metric": "countries_with_exact_duplicates",
                "value": int(exact["country_iso2"].nunique()),
            },
            {
                "metric": "pairs_flagged_as_duplicate_source_overlap",
                "value": int(len(eurostat_dropped)),
            },
        ]
    )


def _generate_cleaning_audit_markdown(tables: dict[str, pd.DataFrame]) -> str:
    stage_summary = _stage_summary_rows(tables)
    paper_totals = _paper_totals_rows(tables)
    country_compare = _country_compare_rows(tables)
    sentiment_summary = _summarize_sentiment_audit(tables["sentiment_audit"])

    lines = [
        "# Cleaning Audit Report",
        "",
        "## Stage Summary",
        "",
        stage_summary.to_markdown(index=False),
        "",
        "## Paper Comparison",
        "",
        paper_totals.to_markdown(index=False),
        "",
        "## Selected Countries",
        "",
        country_compare.to_markdown(index=False),
        "",
        "## Sentiment Overlap Summary",
        "",
        sentiment_summary.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `raw_fetched` counts are taken from the clean raw panel, not from coverage-filtered panels.",
        "- The current high-correlation filter is deterministic but source-agnostic, so exact Eurostat/OECD sentiment duplicates are currently resolved by alphabetical `series_id` ordering.",
        "- `2021_cov98` remains the explicit car-registration sensitivity panel because the 2022 car-registration series do not survive the 2022 window.",
    ]
    return "\n".join(lines)
