from pathlib import Path

import pandas as pd

from meu_replication.data_management.task_probe_availability import (
    _PROBE_COLUMNS,
    _build_oecd_availability,
    _build_oecd_avail_produces,
    _build_probe_depends,
    task_combine_availability,
)


def test_build_oecd_avail_produces_has_all_countries():
    produces = _build_oecd_avail_produces()
    assert len(produces) == 19
    assert all(path.suffix == ".parquet" for path in produces.values())
    assert all(path.name.startswith("oecd_") for path in produces.values())


def test_build_probe_depends_includes_expected_keys():
    depends = _build_probe_depends()
    assert "ecb_U2" in depends
    assert len(depends) == 77


def test_build_oecd_availability_classifies_ok_short_and_missing():
    bulk = pd.DataFrame(
        {
            "series_id": ["DE_A", "DE_A", "DE_B"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    oecd_series = pd.DataFrame(
        {
            "series_id": ["DE_A", "DE_B", "DE_C"],
            "country_iso2": ["DE", "DE", "DE"],
        }
    )

    result = _build_oecd_availability(bulk, oecd_series, "DE")

    status_map = dict(zip(result["series_id"], result["status"], strict=True))
    assert status_map["DE_A"] == "ok_short"
    assert status_map["DE_B"] == "ok_short"
    assert status_map["DE_C"] == "missing"


def test_task_combine_availability_writes_combined_manifest(tmp_path: Path):
    cols = pd.Index(_PROBE_COLUMNS)
    df1 = pd.DataFrame(
        [
            {
                "series_id": "DE_X",
                "template_id": "X",
                "country_iso2": "DE",
                "status": "ok",
                "rows_fetched": 12,
                "error_kind": "",
                "error_message": "",
            }
        ],
        columns=cols,
    )
    df2 = pd.DataFrame(
        [
            {
                "series_id": "FR_X",
                "template_id": "X",
                "country_iso2": "FR",
                "status": "ok_short",
                "rows_fetched": 4,
                "error_kind": "",
                "error_message": "",
            }
        ],
        columns=cols,
    )
    empty = pd.DataFrame(columns=cols)

    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    p3 = tmp_path / "empty.parquet"
    df1.to_parquet(p1, index=False)
    df2.to_parquet(p2, index=False)
    empty.to_parquet(p3, index=False)

    out = tmp_path / "combined.parquet"
    depends = {
        "a": p1,
        "b": p2,
        "empty": p3,
        "missing": tmp_path / "missing.parquet",
    }

    task_combine_availability(depends_on=depends, produces=out)

    result = pd.read_parquet(out)
    assert len(result) == 2
    assert set(result["series_id"]) == {"DE_X", "FR_X"}


def test_task_combine_availability_writes_empty_when_no_inputs(tmp_path: Path):
    out = tmp_path / "combined.parquet"
    depends = {
        "missing_a": tmp_path / "a.parquet",
        "missing_b": tmp_path / "b.parquet",
    }

    task_combine_availability(depends_on=depends, produces=out)

    result = pd.read_parquet(out)
    assert result.empty
    assert list(result.columns) == list(_PROBE_COLUMNS)
