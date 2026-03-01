import pandas as pd
import pytest

from meu_replication.data_fetch.standardize import (
    standardize_from_bis,
    standardize_from_ecb_like,
    standardize_from_eurostat,
    validate_long,
)

_EXPECTED_COLUMNS = [
    "date",
    "value",
    "series_id",
    "country_iso2",
    "variable_name",
    "category",
    "category_name",
    "source",
]


def _eurostat_raw():
    return pd.DataFrame(
        {
            "freq": ["M", "M"],
            "geo": ["DE", "DE"],
            "nace_r2": ["B-D", "B-D"],
            "2024-01": [100.5, 100.5],
            "2024-02": [101.2, 101.2],
            "2024-03": [99.8, 99.8],
        }
    )


def _eurostat_spec():
    return {
        "series_id": "DE_IP_001",
        "country_iso2": "DE",
        "variable_name": "Test variable",
        "category": 1,
        "category_name": "Industrial_production",
    }


def _ecb_raw():
    return pd.DataFrame(
        {
            "TIME_PERIOD": ["2024-01", "2024-02", "2024-03"],
            "OBS_VALUE": ["1.15", "1.16", "1.14"],
            "FREQ": ["M", "M", "M"],
        }
    )


def _ecb_spec():
    return {
        "series_id": "EA_FX_001",
        "country_iso2": "U2",
        "variable_name": "USD/EUR exchange rate",
        "category": 8,
        "category_name": "EA_Financial",
        "source": "ecb",
    }


# --- Eurostat ---


def test_standardize_from_eurostat_output_schema():
    result = standardize_from_eurostat(_eurostat_raw(), _eurostat_spec())
    assert len(result) > 0
    assert list(result.columns) == _EXPECTED_COLUMNS


def test_standardize_from_eurostat_metadata_values():
    result = standardize_from_eurostat(_eurostat_raw(), _eurostat_spec())
    assert result["series_id"].iloc[0] == "DE_IP_001"
    assert result["country_iso2"].iloc[0] == "DE"
    assert result["source"].iloc[0] == "eurostat"
    assert result["date"].iloc[0] in ["2024-01", "2024-02", "2024-03"]


def test_standardize_from_eurostat_no_time_columns_raises():
    raw = pd.DataFrame({"freq": ["M"], "geo": ["DE"]})
    spec = _eurostat_spec()
    with pytest.raises(ValueError, match="No time columns found"):
        standardize_from_eurostat(raw, spec)


# --- ECB ---


def test_standardize_from_ecb_like_output_schema():
    result = standardize_from_ecb_like(_ecb_raw(), _ecb_spec())
    assert len(result) == 3
    assert list(result.columns) == _EXPECTED_COLUMNS


def test_standardize_from_ecb_like_metadata_and_values():
    result = standardize_from_ecb_like(_ecb_raw(), _ecb_spec())
    assert result["series_id"].iloc[0] == "EA_FX_001"
    assert result["country_iso2"].iloc[0] == "U2"
    assert result["source"].iloc[0] == "ecb"
    assert result["date"].iloc[0] == "2024-01"
    assert abs(result["value"].iloc[0] - 1.15) < 0.01


def test_standardize_from_ecb_like_missing_columns_raises():
    raw = pd.DataFrame({"FREQ": ["M"], "SOME_OTHER_COL": ["X"]})
    spec = _ecb_spec()
    with pytest.raises(ValueError, match="None of .* found in columns"):
        standardize_from_ecb_like(raw, spec)


# --- BIS ---


def test_standardize_from_bis_typical():
    raw = pd.DataFrame(
        {
            "TIME_PERIOD": ["2024-01", "2024-02", "2024-03"],
            "OBS_VALUE": ["105.2", "106.1", "104.8"],
            "FREQ": ["M", "M", "M"],
        }
    )
    spec = {
        "series_id": "BIS_NEER_001_DE",
        "country_iso2": "DE",
        "variable_name": "NEER broad index",
        "category": 7,
        "category_name": "Financial",
        "unit_measure_filter": None,
    }
    result = standardize_from_bis(raw, spec)
    assert len(result) == 3
    assert result["series_id"].iloc[0] == "BIS_NEER_001_DE"
    assert result["source"].iloc[0] == "bis"
    assert result["date"].iloc[0] == "2024-01"


def test_standardize_from_bis_with_unit_measure_filter():
    raw = pd.DataFrame(
        {
            "TIME_PERIOD": ["2024-01", "2024-02", "2024-03", "2024-04"],
            "OBS_VALUE": ["105.2", "106.1", "2.5", "2.6"],
            "UNIT_MEASURE": [771, 771, 628, 628],
        }
    )
    spec = {
        "series_id": "BIS_INT_001_DE",
        "country_iso2": "DE",
        "variable_name": "Long-term rate",
        "category": 7,
        "category_name": "Financial",
        "unit_measure_filter": 771,
    }
    result = standardize_from_bis(raw, spec)
    assert len(result) == 2
    assert result["date"].iloc[0] == "2024-01"
    assert result["date"].iloc[1] == "2024-02"


# --- validate_long ---


def test_validate_long_empty_raises():
    with pytest.raises(ValueError, match="No data returned"):
        validate_long(pd.DataFrame(), "TEST_001")


def test_validate_long_all_nan_raises():
    df = pd.DataFrame(
        {
            "date": ["2024-01"],
            "value": [None],
            "series_id": ["TEST_001"],
            "country_iso2": ["DE"],
            "variable_name": ["Test"],
            "category": [1],
            "category_name": ["Test"],
            "source": ["test"],
        }
    )
    with pytest.raises(ValueError, match="All values are NaN"):
        validate_long(df, "TEST_001")


def test_validate_long_missing_columns_raises():
    df = pd.DataFrame(
        {
            "date": ["2024-01"],
            "value": [100.0],
            "series_id": ["TEST_001"],
        }
    )
    with pytest.raises(ValueError, match="Missing columns"):
        validate_long(df, "TEST_001")


def test_validate_long_duplicates_raises():
    df = pd.DataFrame(
        {
            "date": ["2024-01", "2024-01"],
            "value": [100.0, 101.0],
            "series_id": ["TEST_001", "TEST_001"],
            "country_iso2": ["DE", "DE"],
            "variable_name": ["Test", "Test"],
            "category": [1, 1],
            "category_name": ["Test", "Test"],
            "source": ["test", "test"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_long(df, "TEST_001")


def test_validate_long_valid_passes():
    df = pd.DataFrame(
        {
            "date": ["2024-01", "2024-02"],
            "value": [100.0, 101.0],
            "series_id": ["TEST_001", "TEST_001"],
            "country_iso2": ["DE", "DE"],
            "variable_name": ["Test", "Test"],
            "category": [1, 1],
            "category_name": ["Test", "Test"],
            "source": ["test", "test"],
        }
    )
    validate_long(df, "TEST_001")
