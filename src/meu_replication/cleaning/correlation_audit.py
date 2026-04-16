"""Helpers for auditing high-correlation cleaning decisions."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd

from meu_replication.cleaning.high_correlation import _correlated_pairs, _pivot_to_wide

_EXACT_CORR_THRESHOLD = 0.999999

_PAIR_COLUMNS: tuple[str, ...] = (
    "window",
    "sample_start",
    "sample_end",
    "threshold",
    "country_iso2",
    "pair_key",
    "series_a",
    "series_b",
    "abs_correlation",
    "raw_abs_correlation",
    "raw_overlap_months",
    "is_exact_corr",
    "raw_is_exact_corr",
    "same_source",
    "same_category",
    "same_family_prefix",
    "same_dataset",
    "same_transformationcode",
    "cross_source_pair",
    "source_pair",
    "category_pair",
    "family_pair",
    "triage_bucket",
    "recommended_disposition",
    "series_a_variable_name",
    "series_a_source",
    "series_a_category",
    "series_a_category_name",
    "series_a_transformationcode",
    "series_a_dataset",
    "series_a_key",
    "series_a_filters_json",
    "series_a_template_id",
    "series_a_family_prefix",
    "series_b_variable_name",
    "series_b_source",
    "series_b_category",
    "series_b_category_name",
    "series_b_transformationcode",
    "series_b_dataset",
    "series_b_key",
    "series_b_filters_json",
    "series_b_template_id",
    "series_b_family_prefix",
)

_DECISION_COLUMNS: tuple[str, ...] = (
    "window",
    "sample_start",
    "sample_end",
    "threshold",
    "country_iso2",
    "series_id",
    "pair_key_count",
    "current_outcome",
    "over_threshold_neighbour_count",
    "kept_neighbour_count",
    "dropped_neighbour_count",
    "exact_neighbour_count",
    "has_exact_neighbour",
    "strongest_neighbour",
    "strongest_abs_correlation",
    "strongest_neighbour_outcome",
    "current_reason_kept_series",
    "current_reason_abs_correlation",
    "current_reason_matches_strongest",
    "component_id",
    "component_size",
    "component_edge_count",
    "component_current_keep_count",
    "component_counterfactual_keep_count",
    "component_greedy_keep_gain",
    "series_variable_name",
    "series_source",
    "series_category",
    "series_category_name",
    "series_transformationcode",
    "series_dataset",
    "series_key",
    "series_filters_json",
    "series_template_id",
    "series_family_prefix",
)

_COMPONENT_COLUMNS: tuple[str, ...] = (
    "window",
    "sample_start",
    "sample_end",
    "threshold",
    "country_iso2",
    "component_id",
    "component_size",
    "component_edge_count",
    "component_edge_density",
    "current_keep_count",
    "current_drop_count",
    "counterfactual_keep_count",
    "counterfactual_drop_count",
    "greedy_keep_gain",
    "kept_pair_violation_count",
    "series_ids",
)

_WINDOW_OVERVIEW_COLUMNS: tuple[str, ...] = (
    "window",
    "sample_start",
    "sample_end",
    "threshold",
    "pair_count",
    "graph_series_count",
    "dropped_series_count",
    "kept_series_count",
    "exact_pair_count",
    "cross_source_pair_count",
    "same_source_pair_count",
    "component_count",
    "counterfactual_keep_count",
    "counterfactual_drop_count",
    "greedy_keep_gain",
    "kept_pair_violation_count",
)

_PAIR_STABILITY_COLUMNS: tuple[str, ...] = (
    "pair_key",
    "country_iso2",
    "series_a",
    "series_b",
    "source_pair",
    "category_pair",
    "family_pair",
    "series_a_variable_name",
    "series_b_variable_name",
    "triage_bucket",
    "recommended_disposition",
    "window_count",
    "windows",
    "min_abs_correlation",
    "max_abs_correlation",
    "exact_in_any_window",
    "exact_in_all_windows",
)

_FIX_READINESS_COLUMNS: tuple[str, ...] = (
    "priority_rank",
    "target_kind",
    "review_target",
    "triage_bucket",
    "recommended_disposition",
    "pair_count",
    "pair_window_count",
    "country_count",
    "window_count",
    "windows",
    "example_country",
    "example_series_a",
    "example_series_b",
    "example_abs_correlation",
    "rationale",
)


def build_window_correlation_audit(
    panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    registry: pd.DataFrame,
    drop_info: pd.DataFrame,
    *,
    window: str,
    sample_start: str,
    sample_end: str,
    threshold: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Build pair, decision, and summary audit tables for one window."""
    series_meta = _build_series_metadata(panel, registry)
    pairs = _build_pair_audit(
        panel=panel,
        raw_panel=raw_panel,
        series_meta=series_meta,
        window=window,
        sample_start=sample_start,
        sample_end=sample_end,
        threshold=threshold,
    )
    decisions, components = _build_decision_audit(
        pairs=pairs,
        drop_info=drop_info,
        series_meta=series_meta,
        window=window,
        sample_start=sample_start,
        sample_end=sample_end,
        threshold=threshold,
    )

    audit_tables = {
        "pairs": pairs,
        "decisions": decisions,
        "summary_country": _summarize_pairs_by_country(pairs),
        "summary_source_pairs": _summarize_pairs_by_dimension(
            pairs, group_cols=("window", "source_pair")
        ),
        "summary_category_pairs": _summarize_pairs_by_dimension(
            pairs, group_cols=("window", "category_pair")
        ),
        "summary_family_pairs": _summarize_pairs_by_dimension(
            pairs, group_cols=("window", "family_pair")
        ),
        "components": components,
        "window_overview": _build_window_overview(
            pairs=pairs, decisions=decisions, components=components
        ),
    }
    return audit_tables


def build_cross_window_correlation_audit(
    window_audits: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Build cross-window stability and fix-readiness tables."""
    all_pairs = pd.concat(
        [audit["pairs"] for audit in window_audits.values()], ignore_index=True
    )
    if all_pairs.empty:
        empty_tables = {
            "window_overview": pd.concat(
                [audit["window_overview"] for audit in window_audits.values()],
                ignore_index=True,
            ),
            "pair_stability": pd.DataFrame(columns=pd.Index(_PAIR_STABILITY_COLUMNS)),
            "fix_readiness": pd.DataFrame(columns=pd.Index(_FIX_READINESS_COLUMNS)),
        }
        return empty_tables

    window_overview = pd.concat(
        [audit["window_overview"] for audit in window_audits.values()],
        ignore_index=True,
    )
    pair_stability = _build_pair_stability(all_pairs)
    fix_readiness = _build_fix_readiness(pair_stability)
    combined_audit = {
        "window_overview": window_overview,
        "pair_stability": pair_stability,
        "fix_readiness": fix_readiness,
    }
    return combined_audit


def render_correlation_review_markdown(
    window_audits: dict[str, dict[str, pd.DataFrame]],
    combined_audit: dict[str, pd.DataFrame],
) -> str:
    """Render a markdown report for the correlation-cleaning review."""
    window_overview = combined_audit["window_overview"]
    pair_stability = combined_audit["pair_stability"]
    fix_readiness = combined_audit["fix_readiness"]
    benchmark = window_audits["2022_strict"]

    exact_targets = fix_readiness[
        fix_readiness["recommended_disposition"] == "drop_upstream_duplicate"
    ].head(15)
    overlap_targets = fix_readiness[
        fix_readiness["recommended_disposition"] == "keep_both_legitimate_overlap"
    ].head(15)
    suspicious_targets = fix_readiness[
        fix_readiness["recommended_disposition"] == "investigate_fetch_or_mapping"
    ].head(15)
    deferred_targets = fix_readiness[
        fix_readiness["recommended_disposition"] == "defer_needs_domain_decision"
    ].head(15)

    component_summary = benchmark["components"].agg(
        {
            "component_id": "count",
            "greedy_keep_gain": "sum",
            "kept_pair_violation_count": "sum",
        }
    )
    stable_pairs = pair_stability[pair_stability["window_count"] == 3].head(20)

    lines = [
        "# Correlation Cleaning Review",
        "",
        "## Window Overview",
        "",
        _to_markdown(window_overview),
        "",
        "## 2022 Strict Benchmark",
        "",
        _to_markdown(benchmark["summary_country"].head(10)),
        "",
        "### Top Source Pairs",
        "",
        _to_markdown(benchmark["summary_source_pairs"].head(10)),
        "",
        "### Top Category Pairs",
        "",
        _to_markdown(benchmark["summary_category_pairs"].head(10)),
        "",
        "### Top Family Pairs",
        "",
        _to_markdown(benchmark["summary_family_pairs"].head(10)),
        "",
        "## Exact Duplicate / Provider-Overlap Families",
        "",
        _to_markdown(exact_targets),
        "",
        "## Same-Source Overlap Families",
        "",
        _to_markdown(overlap_targets),
        "",
        "## Fetch-Suspicion Shortlist",
        "",
        _to_markdown(suspicious_targets),
        "",
        "## Deferred Manual Review",
        "",
        _to_markdown(deferred_targets),
        "",
        "## Stable Pairs Across All Three Windows",
        "",
        _to_markdown(stable_pairs),
        "",
        "## Algorithm Comparison",
        "",
        f"- 2022 connected components: {int(component_summary['component_id'])}",
        f"- 2022 greedy keep gain over one-per-component baseline: {int(component_summary['greedy_keep_gain'])}",
        f"- 2022 kept-pair violations above threshold: {int(component_summary['kept_pair_violation_count'])}",
        "",
        "## Next-Step Fix Order",
        "",
        "1. Remove exact cross-provider duplicate families upstream, starting with Eurostat/OECD sentiment pairs.",
        "2. Keep reviewing same-source overlap families to decide whether they should remain legitimate overlaps or be replaced by less redundant variables.",
        "3. Investigate the fetch or mapping shortlist using raw values, registry keys, and source filters before changing the registry.",
    ]
    return "\n".join(lines)


def _build_series_metadata(panel: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    panel_meta = panel[
        [
            "series_id",
            "country_iso2",
            "variable_name",
            "category",
            "category_name",
            "source",
            "transformationcode",
        ]
    ].drop_duplicates()
    registry_meta = registry[
        ["series_id", "dataset", "key", "filters_json"]
    ].drop_duplicates()
    merged = panel_meta.merge(registry_meta, on="series_id", how="left")
    template_id = [
        _extract_template_id(series_id, country)
        for series_id, country in zip(merged["series_id"], merged["country_iso2"])
    ]
    family_prefix = [_extract_family_prefix(template) for template in template_id]
    series_metadata = merged.assign(
        template_id=template_id, family_prefix=family_prefix
    )
    series_metadata = series_metadata.sort_values(
        ["country_iso2", "series_id"]
    ).reset_index(drop=True)
    return series_metadata


def _build_pair_audit(
    *,
    panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    series_meta: pd.DataFrame,
    window: str,
    sample_start: str,
    sample_end: str,
    threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    raw_window = raw_panel[
        (raw_panel["date"] >= sample_start) & (raw_panel["date"] <= sample_end)
    ].copy()

    for country, country_df in panel.groupby("country_iso2"):
        wide = _pivot_to_wide(country_df)
        pairs = _correlated_pairs(wide.corr(), threshold)
        if not pairs:
            continue

        raw_country = raw_window[
            (raw_window["country_iso2"] == country)
            & (raw_window["series_id"].isin(country_df["series_id"].unique()))
        ]
        raw_corr, raw_overlap = _raw_pair_stats(raw_country)

        for series_a, series_b, abs_correlation in pairs:
            raw_abs_correlation = pd.NA
            raw_overlap_months = 0
            if raw_corr is not None and raw_overlap is not None:
                if series_a in raw_corr.columns and series_b in raw_corr.columns:
                    raw_val = raw_corr.at[series_a, series_b]
                    if pd.notna(raw_val):
                        raw_abs_correlation = float(abs(raw_val))
                    raw_overlap_months = int(raw_overlap.at[series_a, series_b])

            records.append(
                {
                    "window": window,
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "threshold": threshold,
                    "country_iso2": str(country),
                    "pair_key": _pair_key(str(country), series_a, series_b),
                    "series_a": series_a,
                    "series_b": series_b,
                    "abs_correlation": round(abs_correlation, 6),
                    "raw_abs_correlation": raw_abs_correlation,
                    "raw_overlap_months": raw_overlap_months,
                }
            )

    if not records:
        return pd.DataFrame(columns=pd.Index(_PAIR_COLUMNS))

    pairs = pd.DataFrame(records)
    pairs = _join_pair_metadata(pairs, series_meta)
    pairs = _classify_pairs(pairs, threshold=threshold)
    sorted_pairs = pairs.sort_values(
        ["window", "country_iso2", "abs_correlation", "series_a", "series_b"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    return sorted_pairs


def _raw_pair_stats(
    raw_country: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if raw_country.empty:
        return None, None
    wide = raw_country.pivot(index="date", columns="series_id", values="value")
    corr = wide.corr()
    overlap = wide.notna().astype("int64").T.dot(wide.notna().astype("int64"))
    return corr, overlap


def _join_pair_metadata(pairs: pd.DataFrame, series_meta: pd.DataFrame) -> pd.DataFrame:
    meta_a = _prefixed_series_meta(series_meta, prefix="series_a")
    meta_b = _prefixed_series_meta(series_meta, prefix="series_b")
    pairs = pairs.merge(
        meta_a,
        on=["country_iso2", "series_a"],
        how="left",
        validate="many_to_one",
    )
    pairs = pairs.merge(
        meta_b,
        on=["country_iso2", "series_b"],
        how="left",
        validate="many_to_one",
    )
    return pairs


def _prefixed_series_meta(series_meta: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    renamed = series_meta.rename(
        columns={
            "series_id": prefix,
            "variable_name": f"{prefix}_variable_name",
            "source": f"{prefix}_source",
            "category": f"{prefix}_category",
            "category_name": f"{prefix}_category_name",
            "transformationcode": f"{prefix}_transformationcode",
            "dataset": f"{prefix}_dataset",
            "key": f"{prefix}_key",
            "filters_json": f"{prefix}_filters_json",
            "template_id": f"{prefix}_template_id",
            "family_prefix": f"{prefix}_family_prefix",
        }
    )
    return renamed


def _classify_pairs(pairs: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    result = pairs.assign(
        is_exact_corr=pairs["abs_correlation"] >= _EXACT_CORR_THRESHOLD,
        raw_is_exact_corr=pairs["raw_abs_correlation"].fillna(-1)
        >= _EXACT_CORR_THRESHOLD,
        same_source=pairs["series_a_source"] == pairs["series_b_source"],
        same_category=pairs["series_a_category"] == pairs["series_b_category"],
        same_family_prefix=(
            pairs["series_a_family_prefix"] == pairs["series_b_family_prefix"]
        ),
        same_dataset=pairs["series_a_dataset"] == pairs["series_b_dataset"],
        same_transformationcode=(
            pairs["series_a_transformationcode"]
            == pairs["series_b_transformationcode"]
        ),
        cross_source_pair=pairs["series_a_source"] != pairs["series_b_source"],
        source_pair=[
            _normalized_pair_label(a, b)
            for a, b in zip(pairs["series_a_source"], pairs["series_b_source"])
        ],
        category_pair=[
            _normalized_pair_label(str(a), str(b))
            for a, b in zip(
                pairs["series_a_category_name"], pairs["series_b_category_name"]
            )
        ],
        family_pair=[
            _normalized_pair_label(str(a), str(b))
            for a, b in zip(
                pairs["series_a_family_prefix"], pairs["series_b_family_prefix"]
            )
        ],
    )

    triage_bucket = [
        _triage_bucket(row, threshold=threshold) for row in result.to_dict("records")
    ]
    disposition = [_recommended_disposition(bucket) for bucket in triage_bucket]
    classified = result.assign(
        triage_bucket=triage_bucket,
        recommended_disposition=disposition,
    )
    return classified.loc[:, list(_PAIR_COLUMNS)]


def _triage_bucket(row: dict[str, Any], *, threshold: float) -> str:
    raw_abs_correlation = row.get("raw_abs_correlation")
    raw_corr_is_valid = raw_abs_correlation is not pd.NA and pd.notna(raw_abs_correlation)

    if row["cross_source_pair"] and row["is_exact_corr"]:
        return "exact_cross_provider_duplicate"
    if (
        raw_corr_is_valid
        and float(raw_abs_correlation) <= threshold
        and float(row["abs_correlation"]) > threshold
    ):
        return "transformation_induced_near_duplicate"
    if (
        row["same_source"]
        and row["is_exact_corr"]
        and not row["same_family_prefix"]
        and row["series_a_variable_name"] != row["series_b_variable_name"]
    ):
        return "likely_fetch_or_mapping_issue"
    if row["same_source"] and (row["same_category"] or row["same_family_prefix"]):
        return "same_source_overlap"
    return "needs_manual_review"


def _recommended_disposition(triage_bucket: str) -> str:
    mapping = {
        "exact_cross_provider_duplicate": "drop_upstream_duplicate",
        "same_source_overlap": "keep_both_legitimate_overlap",
        "likely_fetch_or_mapping_issue": "investigate_fetch_or_mapping",
        "transformation_induced_near_duplicate": "defer_needs_domain_decision",
        "needs_manual_review": "defer_needs_domain_decision",
    }
    return mapping[triage_bucket]


def _build_decision_audit(
    *,
    pairs: pd.DataFrame,
    drop_info: pd.DataFrame,
    series_meta: pd.DataFrame,
    window: str,
    sample_start: str,
    sample_end: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairs.empty:
        return (
            pd.DataFrame(columns=pd.Index(_DECISION_COLUMNS)),
            pd.DataFrame(columns=pd.Index(_COMPONENT_COLUMNS)),
        )

    drop_info = drop_info.copy()
    drop_map = drop_info.set_index(["country_iso2", "dropped_series"]).to_dict("index")
    dropped_set = set(
        zip(drop_info["country_iso2"].astype(str), drop_info["dropped_series"].astype(str))
    )

    records: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []

    for country, country_pairs in pairs.groupby("country_iso2"):
        adjacency = _build_adjacency(country_pairs)
        pair_corr = {
            (row.series_a, row.series_b): float(row.abs_correlation)
            for row in country_pairs.itertuples(index=False)
        }
        components = _connected_components(adjacency)
        component_lookup: dict[str, dict[str, object]] = {}
        dropped_series = {series for c, series in dropped_set if c == country}
        series_in_graph = set(adjacency)
        kept_series = series_in_graph - dropped_series

        for idx, nodes in enumerate(components, start=1):
            node_set = set(nodes)
            comp_pairs = country_pairs[
                country_pairs["series_a"].isin(node_set)
                & country_pairs["series_b"].isin(node_set)
            ]
            current_keep_count = len(node_set & kept_series)
            component_id = f"{country}_{idx:03d}"
            kept_pair_violation_count = int(
                (
                    ~comp_pairs["series_a"].isin(dropped_series)
                    & ~comp_pairs["series_b"].isin(dropped_series)
                ).sum()
            )
            component_records.append(
                {
                    "window": window,
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "threshold": threshold,
                    "country_iso2": country,
                    "component_id": component_id,
                    "component_size": len(node_set),
                    "component_edge_count": len(comp_pairs),
                    "component_edge_density": round(
                        _edge_density(len(node_set), len(comp_pairs)), 6
                    ),
                    "current_keep_count": current_keep_count,
                    "current_drop_count": len(node_set) - current_keep_count,
                    "counterfactual_keep_count": 1,
                    "counterfactual_drop_count": max(len(node_set) - 1, 0),
                    "greedy_keep_gain": max(current_keep_count - 1, 0),
                    "kept_pair_violation_count": kept_pair_violation_count,
                    "series_ids": " | ".join(sorted(node_set)),
                }
            )
            for series_id in node_set:
                component_lookup[series_id] = {
                    "component_id": component_id,
                    "component_size": len(node_set),
                    "component_edge_count": len(comp_pairs),
                    "component_current_keep_count": current_keep_count,
                    "component_counterfactual_keep_count": 1,
                    "component_greedy_keep_gain": max(current_keep_count - 1, 0),
                }

        for series_id, neighbours in sorted(adjacency.items()):
            strongest = None
            strongest_corr = 0.0
            if neighbours:
                strongest = max(
                    neighbours,
                    key=lambda other: pair_corr.get(
                        _sorted_pair_key(series_id, other), 0.0
                    ),
                )
                strongest_corr = pair_corr[_sorted_pair_key(series_id, strongest)]

            kept_neighbours = {n for n in neighbours if n in kept_series}
            series_drop_info = drop_map.get((country, series_id), {})
            current_reason_kept_series = series_drop_info.get("kept_series", "")
            current_reason_abs_correlation = series_drop_info.get("abs_correlation", pd.NA)

            records.append(
                {
                    "window": window,
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "threshold": threshold,
                    "country_iso2": country,
                    "series_id": series_id,
                    "pair_key_count": sum(
                        country_pairs["series_a"].eq(series_id)
                        | country_pairs["series_b"].eq(series_id)
                    ),
                    "current_outcome": (
                        "dropped" if series_id in dropped_series else "kept"
                    ),
                    "over_threshold_neighbour_count": len(neighbours),
                    "kept_neighbour_count": len(kept_neighbours),
                    "dropped_neighbour_count": len(neighbours) - len(kept_neighbours),
                    "exact_neighbour_count": sum(
                        pair_corr[_sorted_pair_key(series_id, other)]
                        >= _EXACT_CORR_THRESHOLD
                        for other in neighbours
                    ),
                    "has_exact_neighbour": any(
                        pair_corr[_sorted_pair_key(series_id, other)]
                        >= _EXACT_CORR_THRESHOLD
                        for other in neighbours
                    ),
                    "strongest_neighbour": strongest or "",
                    "strongest_abs_correlation": round(strongest_corr, 6)
                    if strongest
                    else pd.NA,
                    "strongest_neighbour_outcome": (
                        "dropped"
                        if strongest in dropped_series
                        else "kept"
                        if strongest
                        else ""
                    ),
                    "current_reason_kept_series": current_reason_kept_series,
                    "current_reason_abs_correlation": current_reason_abs_correlation,
                    "current_reason_matches_strongest": bool(
                        strongest
                        and current_reason_kept_series == strongest
                        and pd.notna(current_reason_abs_correlation)
                        and abs(float(current_reason_abs_correlation) - strongest_corr)
                        < 1e-6
                    ),
                    **component_lookup.get(
                        series_id,
                        {
                            "component_id": "",
                            "component_size": 0,
                            "component_edge_count": 0,
                            "component_current_keep_count": 0,
                            "component_counterfactual_keep_count": 0,
                            "component_greedy_keep_gain": 0,
                        },
                    ),
                }
            )

    decisions = pd.DataFrame(records)
    decisions = decisions.merge(
        series_meta.rename(
            columns={
                "variable_name": "series_variable_name",
                "source": "series_source",
                "category": "series_category",
                "category_name": "series_category_name",
                "transformationcode": "series_transformationcode",
                "dataset": "series_dataset",
                "key": "series_key",
                "filters_json": "series_filters_json",
                "template_id": "series_template_id",
                "family_prefix": "series_family_prefix",
            }
        ),
        on=["country_iso2", "series_id"],
        how="left",
        validate="many_to_one",
    )
    decisions = decisions.loc[:, list(_DECISION_COLUMNS)].sort_values(
        ["window", "country_iso2", "current_outcome", "series_id"]
    )
    components = pd.DataFrame(component_records, columns=pd.Index(_COMPONENT_COLUMNS))
    components = components.sort_values(
        ["window", "country_iso2", "component_size", "component_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return decisions.reset_index(drop=True), components


def _build_adjacency(country_pairs: pd.DataFrame) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in country_pairs.itertuples(index=False):
        adjacency[str(row.series_a)].add(str(row.series_b))
        adjacency[str(row.series_b)].add(str(row.series_a))
    return {key: set(value) for key, value in adjacency.items()}


def _connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for series_id in sorted(adjacency):
        if series_id in seen:
            continue
        queue = deque([series_id])
        component: list[str] = []
        seen.add(series_id)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    return components


def _build_window_overview(
    *,
    pairs: pd.DataFrame,
    decisions: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=pd.Index(_WINDOW_OVERVIEW_COLUMNS))

    kept_series = _kept_series(decisions)
    kept_pair_violation_count = int(
        (
            pairs["series_a"].isin(kept_series) & pairs["series_b"].isin(kept_series)
        ).sum()
    )
    row = {
        "window": str(pairs["window"].iloc[0]),
        "sample_start": str(pairs["sample_start"].iloc[0]),
        "sample_end": str(pairs["sample_end"].iloc[0]),
        "threshold": float(pairs["threshold"].iloc[0]),
        "pair_count": len(pairs),
        "graph_series_count": decisions["series_id"].nunique(),
        "dropped_series_count": int((decisions["current_outcome"] == "dropped").sum()),
        "kept_series_count": int((decisions["current_outcome"] == "kept").sum()),
        "exact_pair_count": int(pairs["is_exact_corr"].sum()),
        "cross_source_pair_count": int(pairs["cross_source_pair"].sum()),
        "same_source_pair_count": int(pairs["same_source"].sum()),
        "component_count": len(components),
        "counterfactual_keep_count": int(components["counterfactual_keep_count"].sum()),
        "counterfactual_drop_count": int(components["counterfactual_drop_count"].sum()),
        "greedy_keep_gain": int(components["greedy_keep_gain"].sum()),
        "kept_pair_violation_count": kept_pair_violation_count,
    }
    overview = pd.DataFrame([row], columns=pd.Index(_WINDOW_OVERVIEW_COLUMNS))
    return overview


def _kept_series(decisions: pd.DataFrame) -> set[str]:
    return set(
        decisions.loc[decisions["current_outcome"] == "kept", "series_id"].astype(str)
    )


def _summarize_pairs_by_country(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "window",
                    "country_iso2",
                    "pair_count",
                    "exact_pair_count",
                    "cross_source_pair_count",
                    "same_source_pair_count",
                ]
            )
        )
    summary = (
        pairs.groupby(["window", "country_iso2"], as_index=False)
        .agg(
            pair_count=("pair_key", "count"),
            exact_pair_count=("is_exact_corr", "sum"),
            cross_source_pair_count=("cross_source_pair", "sum"),
            same_source_pair_count=("same_source", "sum"),
        )
        .sort_values(
            ["window", "pair_count", "country_iso2"], ascending=[True, False, True]
        )
        .reset_index(drop=True)
    )
    return summary


def _summarize_pairs_by_dimension(
    pairs: pd.DataFrame,
    *,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    if pairs.empty:
        cols = [
            *group_cols,
            "pair_count",
            "exact_pair_count",
            "country_count",
            "window_count",
        ]
        return pd.DataFrame(columns=pd.Index(cols))
    summary = (
        pairs.groupby(list(group_cols), as_index=False)
        .agg(
            pair_count=("pair_key", "count"),
            exact_pair_count=("is_exact_corr", "sum"),
            country_count=("country_iso2", "nunique"),
            window_count=("window", "nunique"),
        )
        .sort_values(
            [*group_cols[:-1], "pair_count", group_cols[-1]],
            ascending=[True] * max(len(group_cols) - 1, 0) + [False, True],
        )
        .reset_index(drop=True)
    )
    return summary


def _build_pair_stability(all_pairs: pd.DataFrame) -> pd.DataFrame:
    grouped = all_pairs.groupby("pair_key", as_index=False)
    records: list[dict[str, object]] = []
    for _, group in grouped:
        first = group.iloc[0]
        triage_values = sorted(group["triage_bucket"].dropna().unique())
        disposition_values = sorted(group["recommended_disposition"].dropna().unique())
        records.append(
            {
                "pair_key": first["pair_key"],
                "country_iso2": first["country_iso2"],
                "series_a": first["series_a"],
                "series_b": first["series_b"],
                "source_pair": first["source_pair"],
                "category_pair": first["category_pair"],
                "family_pair": first["family_pair"],
                "series_a_variable_name": first["series_a_variable_name"],
                "series_b_variable_name": first["series_b_variable_name"],
                "triage_bucket": " | ".join(triage_values),
                "recommended_disposition": " | ".join(disposition_values),
                "window_count": group["window"].nunique(),
                "windows": " | ".join(sorted(group["window"].astype(str).unique())),
                "min_abs_correlation": round(group["abs_correlation"].min(), 6),
                "max_abs_correlation": round(group["abs_correlation"].max(), 6),
                "exact_in_any_window": bool(group["is_exact_corr"].any()),
                "exact_in_all_windows": bool(group["is_exact_corr"].all()),
            }
        )
    result = pd.DataFrame(records, columns=pd.Index(_PAIR_STABILITY_COLUMNS))
    sorted_result = result.sort_values(
        ["window_count", "max_abs_correlation", "pair_key"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return sorted_result


def _build_fix_readiness(pair_stability: pd.DataFrame) -> pd.DataFrame:
    if pair_stability.empty:
        return pd.DataFrame(columns=pd.Index(_FIX_READINESS_COLUMNS))

    family_targets = pair_stability[
        pair_stability["triage_bucket"].isin(
            {
                "exact_cross_provider_duplicate",
                "same_source_overlap",
                "transformation_induced_near_duplicate",
            }
        )
    ]
    pair_targets = pair_stability[
        pair_stability["triage_bucket"].isin(
            {"likely_fetch_or_mapping_issue", "needs_manual_review"}
        )
    ]

    records: list[dict[str, object]] = []

    if not family_targets.empty:
        grouped = family_targets.groupby(
            [
                "triage_bucket",
                "recommended_disposition",
                "source_pair",
                "category_pair",
                "family_pair",
            ],
            as_index=False,
        )
        for _, group in grouped:
            first = group.iloc[0]
            records.append(
                {
                    "priority_rank": _priority_rank(first["triage_bucket"]),
                    "target_kind": "family",
                    "review_target": (
                        f"{first['source_pair']} | {first['family_pair']} | {first['category_pair']}"
                    ),
                    "triage_bucket": first["triage_bucket"],
                    "recommended_disposition": first["recommended_disposition"],
                    "pair_count": len(group),
                    "pair_window_count": int(group["window_count"].sum()),
                    "country_count": int(group["country_iso2"].nunique()),
                    "window_count": int(group["window_count"].max()),
                    "windows": " | ".join(
                        sorted(
                            {
                                value
                                for windows in group["windows"]
                                for value in str(windows).split(" | ")
                            }
                        )
                    ),
                    "example_country": first["country_iso2"],
                    "example_series_a": first["series_a"],
                    "example_series_b": first["series_b"],
                    "example_abs_correlation": first["max_abs_correlation"],
                    "rationale": _rationale(first["triage_bucket"]),
                }
            )

    if not pair_targets.empty:
        for row in pair_targets.itertuples(index=False):
            records.append(
                {
                    "priority_rank": _priority_rank(row.triage_bucket),
                    "target_kind": "pair",
                    "review_target": f"{row.country_iso2} | {row.series_a} vs {row.series_b}",
                    "triage_bucket": row.triage_bucket,
                    "recommended_disposition": row.recommended_disposition,
                    "pair_count": 1,
                    "pair_window_count": int(row.window_count),
                    "country_count": 1,
                    "window_count": int(row.window_count),
                    "windows": row.windows,
                    "example_country": row.country_iso2,
                    "example_series_a": row.series_a,
                    "example_series_b": row.series_b,
                    "example_abs_correlation": row.max_abs_correlation,
                    "rationale": _rationale(row.triage_bucket),
                }
            )

    result = pd.DataFrame(records, columns=pd.Index(_FIX_READINESS_COLUMNS))
    sorted_result = result.sort_values(
        ["priority_rank", "pair_window_count", "pair_count", "review_target"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return sorted_result


def _priority_rank(triage_bucket: str) -> int:
    priority = {
        "exact_cross_provider_duplicate": 1,
        "same_source_overlap": 2,
        "likely_fetch_or_mapping_issue": 3,
        "transformation_induced_near_duplicate": 4,
        "needs_manual_review": 5,
    }
    return priority[triage_bucket]


def _rationale(triage_bucket: str) -> str:
    rationale = {
        "exact_cross_provider_duplicate": "Exact cross-provider duplicates should be removed upstream before the correlation screen.",
        "same_source_overlap": "These look like legitimate overlapping concepts and should stay in review rather than be treated as fetch bugs.",
        "likely_fetch_or_mapping_issue": "These exact same-source pairs need raw-data and registry checks before any registry change.",
        "transformation_induced_near_duplicate": "The transformed series cross the threshold while the raw series do not, so this needs a domain decision rather than an immediate fetch fix.",
        "needs_manual_review": "The pair does not fit a strong heuristic bucket and needs manual review.",
    }
    return rationale[triage_bucket]


def _extract_template_id(series_id: str, country_iso2: str) -> str:
    prefix = f"{country_iso2}_"
    if series_id.startswith(prefix):
        return series_id[len(prefix) :]
    return series_id


def _extract_family_prefix(template_id: str) -> str:
    parts = template_id.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def _pair_key(country_iso2: str, series_a: str, series_b: str) -> str:
    left, right = sorted((series_a, series_b))
    return f"{country_iso2}::{left}::{right}"


def _sorted_pair_key(series_a: str, series_b: str) -> tuple[str, str]:
    return tuple(sorted((series_a, series_b)))


def _normalized_pair_label(left: str, right: str) -> str:
    a, b = sorted((str(left), str(right)))
    return f"{a} | {b}"


def _edge_density(node_count: int, edge_count: int) -> float:
    if node_count <= 1:
        return 0.0
    max_edges = node_count * (node_count - 1) / 2
    return edge_count / max_edges


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    return df.to_markdown(index=False)
