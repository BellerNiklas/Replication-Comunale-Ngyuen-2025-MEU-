import pandas as pd

from meu_replication.cleaning.correlation_audit import build_window_correlation_audit
from meu_replication.cleaning.high_correlation import remove_high_correlation

_DATES = [f"2020-0{idx}" for idx in range(1, 7)]


def _series_specs() -> list[dict[str, object]]:
    return [
        {
            "series_id": "DE_SENT_001",
            "country_iso2": "DE",
            "source": "eurostat",
            "category": 6,
            "category_name": "Sentiment",
            "variable_name": "Consumer confidence indicator",
            "dataset": "DE_SENT",
            "key": None,
            "filters_json": '{"geo": "DE"}',
            "transformationcode": 1,
            "values": [1, 2, 3, 4, 5, 6],
            "raw_values": [1, 2, 3, 4, 5, 6],
        },
        {
            "series_id": "DE_OECD_SENT_001",
            "country_iso2": "DE",
            "source": "oecd",
            "category": 6,
            "category_name": "Sentiment",
            "variable_name": "Consumer confidence indicator",
            "dataset": "DE_OECD_SENT",
            "key": "DEU.SENT",
            "filters_json": None,
            "transformationcode": 1,
            "values": [1, 2, 3, 4, 5, 6],
            "raw_values": [1, 2, 3, 4, 5, 6],
        },
        {
            "series_id": "FR_HICP_001",
            "country_iso2": "FR",
            "source": "eurostat",
            "category": 3,
            "category_name": "Prices",
            "variable_name": "HICP overall",
            "dataset": "FR_HICP",
            "key": None,
            "filters_json": '{"geo": "FR", "coicop": "CP00"}',
            "transformationcode": 5,
            "values": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "raw_values": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        },
        {
            "series_id": "FR_HICP_002",
            "country_iso2": "FR",
            "source": "eurostat",
            "category": 3,
            "category_name": "Prices",
            "variable_name": "HICP excluding category",
            "dataset": "FR_HICP",
            "key": None,
            "filters_json": '{"geo": "FR", "coicop": "CP01"}',
            "transformationcode": 5,
            "values": [9.8, 10.9, 12.1, 12.9, 14.0, 15.2],
            "raw_values": [9.8, 10.9, 12.1, 12.9, 14.0, 15.2],
        },
        {
            "series_id": "IT_CHAIN_A",
            "country_iso2": "IT",
            "source": "eurostat",
            "category": 4,
            "category_name": "Activity_indicators",
            "variable_name": "Chain A",
            "dataset": "IT_CHAIN",
            "key": None,
            "filters_json": '{"geo": "IT", "nace_r2": "A"}',
            "transformationcode": 5,
            "values": [-1.2884, 0.3951, 0.4299, 0.6960, -1.1841, -0.6617],
            "raw_values": [-1.2884, 0.3951, 0.4299, 0.6960, -1.1841, -0.6617],
        },
        {
            "series_id": "IT_CHAIN_B",
            "country_iso2": "IT",
            "source": "eurostat",
            "category": 4,
            "category_name": "Activity_indicators",
            "variable_name": "Chain B",
            "dataset": "IT_CHAIN",
            "key": None,
            "filters_json": '{"geo": "IT", "nace_r2": "B"}',
            "transformationcode": 5,
            "values": [-1.3756, 0.1612, 0.7777, 0.5969, -1.1183, -0.7134],
            "raw_values": [-1.3756, 0.1612, 0.7777, 0.5969, -1.1183, -0.7134],
        },
        {
            "series_id": "IT_CHAIN_C",
            "country_iso2": "IT",
            "source": "eurostat",
            "category": 4,
            "category_name": "Activity_indicators",
            "variable_name": "Chain C",
            "dataset": "IT_CHAIN",
            "key": None,
            "filters_json": '{"geo": "IT", "nace_r2": "C"}',
            "transformationcode": 5,
            "values": [-1.0590, 0.4252, 0.9044, 0.1562, -1.1079, -0.5767],
            "raw_values": [-1.0590, 0.4252, 0.9044, 0.1562, -1.1079, -0.5767],
        },
        {
            "series_id": "ES_TRANS_001",
            "country_iso2": "ES",
            "source": "eurostat",
            "category": 1,
            "category_name": "Industrial_production",
            "variable_name": "Transform A",
            "dataset": "ES_TRANS",
            "key": None,
            "filters_json": '{"geo": "ES", "nace_r2": "B"}',
            "transformationcode": 5,
            "values": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "raw_values": [1, 2, 1, 2, 1, 2],
        },
        {
            "series_id": "ES_TRANS_002",
            "country_iso2": "ES",
            "source": "eurostat",
            "category": 1,
            "category_name": "Industrial_production",
            "variable_name": "Transform B",
            "dataset": "ES_TRANS",
            "key": None,
            "filters_json": '{"geo": "ES", "nace_r2": "C"}',
            "transformationcode": 5,
            "values": [0.11, 0.19, 0.29, 0.41, 0.51, 0.59],
            "raw_values": [1, 1, 2, 2, 1, 2],
        },
        {
            "series_id": "NL_FETCH_A_001",
            "country_iso2": "NL",
            "source": "eurostat",
            "category": 4,
            "category_name": "Activity_indicators",
            "variable_name": "Suspicious A",
            "dataset": "NL_FETCH_A",
            "key": None,
            "filters_json": '{"geo": "NL", "indic": "A"}',
            "transformationcode": 1,
            "values": [3, 1, 4, 1, 5, 9],
            "raw_values": [3, 1, 4, 1, 5, 9],
        },
        {
            "series_id": "NL_FETCH_B_001",
            "country_iso2": "NL",
            "source": "eurostat",
            "category": 4,
            "category_name": "Activity_indicators",
            "variable_name": "Suspicious B",
            "dataset": "NL_FETCH_B",
            "key": None,
            "filters_json": '{"geo": "NL", "indic": "B"}',
            "transformationcode": 1,
            "values": [3, 1, 4, 1, 5, 9],
            "raw_values": [3, 1, 4, 1, 5, 9],
        },
        {
            "series_id": "BE_FREE_001",
            "country_iso2": "BE",
            "source": "eurostat",
            "category": 2,
            "category_name": "Labor_market_indicators",
            "variable_name": "Standalone series",
            "dataset": "BE_FREE",
            "key": None,
            "filters_json": '{"geo": "BE"}',
            "transformationcode": 1,
            "values": [0, 1, 0, 2, 1, 3],
            "raw_values": [0, 1, 0, 2, 1, 3],
        },
    ]


def _make_panel(series_specs: list[dict[str, object]], *, use_raw: bool) -> pd.DataFrame:
    rows = []
    for spec in series_specs:
        values = spec["raw_values"] if use_raw else spec["values"]
        for date, value in zip(_DATES, values):
            row = {
                "date": date,
                "value": float(value),
                "series_id": spec["series_id"],
                "country_iso2": spec["country_iso2"],
                "variable_name": spec["variable_name"],
                "category": spec["category"],
                "category_name": spec["category_name"],
                "source": spec["source"],
            }
            if not use_raw:
                row["transformationcode"] = spec["transformationcode"]
            rows.append(row)
    return pd.DataFrame(rows)


def _make_registry(series_specs: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": spec["series_id"],
                "source": spec["source"],
                "category": spec["category"],
                "category_name": spec["category_name"],
                "country_iso2": spec["country_iso2"],
                "variable_name": spec["variable_name"],
                "dataset": spec["dataset"],
                "key": spec["key"],
                "filters_json": spec["filters_json"],
                "unit_measure_filter": None,
                "frequency": "M",
                "start_period": "2020-01",
                "transformationcode": spec["transformationcode"],
            }
            for spec in series_specs
        ]
    )


def _build_inputs():
    specs = _series_specs()
    panel = _make_panel(specs, use_raw=False)
    raw_panel = _make_panel(specs, use_raw=True)
    registry = _make_registry(specs)
    _, drop_info = remove_high_correlation(panel, threshold=0.95)
    return panel, raw_panel, registry, drop_info


def test_build_window_correlation_audit_classifies_pairs():
    panel, raw_panel, registry, drop_info = _build_inputs()

    audit = build_window_correlation_audit(
        panel=panel,
        raw_panel=raw_panel,
        registry=registry,
        drop_info=drop_info,
        window="2022_strict",
        sample_start="2020-01",
        sample_end="2020-06",
        threshold=0.95,
    )

    pairs = audit["pairs"]
    assert len(pairs) == 6

    exact_pair = pairs.loc[
        pairs["pair_key"] == "DE::DE_OECD_SENT_001::DE_SENT_001"
    ].iloc[0]
    assert exact_pair["triage_bucket"] == "exact_cross_provider_duplicate"
    assert exact_pair["recommended_disposition"] == "drop_upstream_duplicate"

    overlap_pair = pairs.loc[
        pairs["pair_key"] == "FR::FR_HICP_001::FR_HICP_002"
    ].iloc[0]
    assert overlap_pair["triage_bucket"] == "same_source_overlap"
    assert overlap_pair["recommended_disposition"] == "keep_both_legitimate_overlap"

    transform_pair = pairs.loc[
        pairs["pair_key"] == "ES::ES_TRANS_001::ES_TRANS_002"
    ].iloc[0]
    assert transform_pair["triage_bucket"] == "transformation_induced_near_duplicate"
    assert transform_pair["raw_abs_correlation"] < 0.95

    fetch_pair = pairs.loc[
        pairs["pair_key"] == "NL::NL_FETCH_A_001::NL_FETCH_B_001"
    ].iloc[0]
    assert fetch_pair["triage_bucket"] == "likely_fetch_or_mapping_issue"
    assert fetch_pair["recommended_disposition"] == "investigate_fetch_or_mapping"


def test_build_window_correlation_audit_maps_drops_and_components():
    panel, raw_panel, registry, drop_info = _build_inputs()

    audit = build_window_correlation_audit(
        panel=panel,
        raw_panel=raw_panel,
        registry=registry,
        drop_info=drop_info,
        window="2022_strict",
        sample_start="2020-01",
        sample_end="2020-06",
        threshold=0.95,
    )

    decisions = audit["decisions"]
    components = audit["components"]
    overview = audit["window_overview"]

    chain_b = decisions.loc[decisions["series_id"] == "IT_CHAIN_B"].iloc[0]
    assert chain_b["current_outcome"] == "dropped"
    assert chain_b["over_threshold_neighbour_count"] == 2
    assert chain_b["current_reason_matches_strongest"]

    chain_component = components.loc[
        components["country_iso2"] == "IT"
    ].iloc[0]
    assert chain_component["component_size"] == 3
    assert chain_component["current_keep_count"] == 2
    assert chain_component["greedy_keep_gain"] == 1

    assert overview.iloc[0]["kept_pair_violation_count"] == 0
