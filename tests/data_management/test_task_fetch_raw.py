from pathlib import Path
from unittest.mock import patch

import pandas as pd

from meu_replication.data_management.task_fetch_raw import (
    _EMPTY_COLUMNS,
    _build_oecd_split_produces,
    _fetch_source_country,
    task_split_oecd_by_country,
)


def test_build_oecd_split_produces_has_all_countries():
    produces = _build_oecd_split_produces()
    assert len(produces) == 19
    assert all(path.name.endswith("_snapshot.parquet") for path in produces.values())


@patch("meu_replication.data_fetch.fetch.fetch_many")
def test_fetch_source_country_returns_empty_when_no_available_series(mock_fetch_many):
    registry = pd.DataFrame(
        {
            "country_iso2": ["DE"],
            "source": ["oecd"],
            "series_id": ["DE_X"],
        }
    )
    availability = pd.DataFrame(
        {
            "series_id": ["DE_X"],
            "status": ["missing"],
        }
    )

    result = _fetch_source_country("oecd", "DE", registry, availability)

    assert result.empty
    assert list(result.columns) == list(_EMPTY_COLUMNS)
    mock_fetch_many.assert_not_called()


@patch("meu_replication.data_fetch.fetch.fetch_many")
def test_fetch_source_country_fetches_only_ok_and_ok_short(mock_fetch_many):
    registry = pd.DataFrame(
        {
            "country_iso2": ["DE", "DE", "DE"],
            "source": ["oecd", "oecd", "oecd"],
            "series_id": ["DE_A", "DE_B", "DE_C"],
        }
    )
    availability = pd.DataFrame(
        {
            "series_id": ["DE_A", "DE_B", "DE_C"],
            "status": ["ok", "ok_short", "error"],
        }
    )
    mock_fetch_many.return_value = pd.DataFrame(
        {
            "date": ["2024-01"],
            "value": [1.0],
            "series_id": ["DE_A"],
            "country_iso2": ["DE"],
            "variable_name": ["X"],
            "category": [1],
            "category_name": ["C"],
            "source": ["oecd"],
        }
    )

    _ = _fetch_source_country("oecd", "DE", registry, availability)

    called_ids = mock_fetch_many.call_args.args[0]
    assert called_ids == ["DE_A", "DE_B"]


def test_task_split_oecd_by_country_writes_nonempty_and_empty_files(tmp_path: Path):
    bulk = pd.DataFrame(
        {
            "date": ["2024-01", "2024-01"],
            "value": [1.0, 2.0],
            "series_id": ["AT_X", "DE_X"],
            "country_iso2": ["AT", "DE"],
            "variable_name": ["var", "var"],
            "category": [1, 1],
            "category_name": ["cat", "cat"],
            "source": ["oecd", "oecd"],
        }
    )
    bulk_path = tmp_path / "bulk.parquet"
    bulk.to_parquet(bulk_path, index=False)

    produces = {
        "AT": tmp_path / "AT_snapshot.parquet",
        "BE": tmp_path / "BE_snapshot.parquet",
    }

    task_split_oecd_by_country(depends_on=bulk_path, produces=produces)

    at_df = pd.read_parquet(produces["AT"])
    be_df = pd.read_parquet(produces["BE"])

    assert len(at_df) == 1
    assert at_df["country_iso2"].iloc[0] == "AT"
    assert be_df.empty
    assert list(be_df.columns) == list(_EMPTY_COLUMNS)
