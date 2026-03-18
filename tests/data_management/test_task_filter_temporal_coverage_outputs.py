from pathlib import Path

import pandas as pd

from meu_replication.data_management.task_filter_temporal_coverage import (
    _build_filter_outputs,
    build_filter_variants,
    task_filter_temporal_coverage,
)


def _make_transformed_panel() -> pd.DataFrame:
    months = pd.period_range("2003-02", "2025-12", freq="M").strftime("%Y-%m")

    series_months = {
        "A_SER": set(months),
        "B_SER": set(months) - {"2025-12"},
        "C_SER": set(months) - {"2022-12"},
    }

    rows = []
    for series_id, valid_months in series_months.items():
        for date in months:
            if date not in valid_months:
                continue
            rows.append(
                {
                    "date": date,
                    "value": 100.0,
                    "series_id": series_id,
                    "country_iso2": "DE",
                    "variable_name": f"var_{series_id}",
                    "category": 1,
                    "category_name": "test_category",
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def test_build_filter_variants_is_strict_only():
    variants = build_filter_variants()

    assert [variant["label"] for variant in variants] == [
        "2021_strict",
        "2022_strict",
        "2025_strict",
    ]
    assert [variant["key"] for variant in variants] == [
        "panel_2021_strict",
        "panel_2022_strict",
        "panel_2025_strict",
    ]
    assert [variant["end"] for variant in variants] == [
        "2021-12",
        "2022-12",
        "2025-12",
    ]
    assert all(variant["allowed_missing"] == 0 for variant in variants)


def test_build_filter_outputs_uses_only_strict_artifacts():
    outputs = _build_filter_outputs(build_filter_variants())

    assert set(outputs) == {
        "panel_2021_strict",
        "panel_2021_strict_drop_info",
        "panel_2022_strict",
        "panel_2022_strict_drop_info",
        "panel_2025_strict",
        "panel_2025_strict_drop_info",
    }
    assert all("cov98" not in str(path) for path in outputs.values())


def test_task_filter_temporal_coverage_writes_three_strict_outputs(tmp_path: Path):
    panel_path = tmp_path / "transformed_panel.parquet"
    _make_transformed_panel().to_parquet(panel_path, index=False)

    produces = {}
    for variant in build_filter_variants():
        key = str(variant["key"])
        produces[key] = tmp_path / f"{key}.parquet"
        produces[f"{key}_drop_info"] = tmp_path / f"{key}_coverage_drop_info.csv"

    task_filter_temporal_coverage(depends_on=panel_path, produces=produces)

    panel_2021 = pd.read_parquet(produces["panel_2021_strict"])
    panel_2022 = pd.read_parquet(produces["panel_2022_strict"])
    panel_2025 = pd.read_parquet(produces["panel_2025_strict"])
    drop_2021 = pd.read_csv(produces["panel_2021_strict_drop_info"])
    drop_2022 = pd.read_csv(produces["panel_2022_strict_drop_info"])
    drop_2025 = pd.read_csv(produces["panel_2025_strict_drop_info"])

    assert panel_2021["series_id"].nunique() == 3
    assert panel_2022["series_id"].nunique() == 2
    assert panel_2025["series_id"].nunique() == 1

    assert drop_2021.empty
    assert set(drop_2022["series_id"]) == {"C_SER"}
    assert set(drop_2025["series_id"]) == {"B_SER", "C_SER"}
    assert set(drop_2022["rule"]) == {"2022_strict"}
    assert set(drop_2025["rule"]) == {"2025_strict"}
