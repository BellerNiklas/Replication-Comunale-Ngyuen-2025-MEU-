import pandas as pd
import pytest

from meu_replication.data_management.task_build_clean import (
    _clean_macro_panel,
    _parse_dates,
)


def _make_raw():
    return pd.DataFrame({
        "date": ["2024-01", "2024-02", "2024-01"],
        "value": [100.5, 101.2, 100.5],
        "series_id": ["DE_IP_001", "DE_IP_001", "DE_IP_002"],
        "country_iso2": ["DE", "DE", "DE"],
        "variable_name": ["IP manufacturing", "IP manufacturing", "IP mining"],
        "category": [1, 1, 1],
        "category_name": [
            "Industrial_production",
            "Industrial_production",
            "Industrial_production",
        ],
        "source": ["eurostat", "eurostat", "eurostat"],
    })


def test_clean_macro_panel_output_columns():
    result = _clean_macro_panel(_make_raw())
    assert list(result.columns) == [
        "date",
        "value",
        "series_id",
        "country_iso2",
        "variable_name",
        "category",
        "category_name",
        "source",
    ]


def test_clean_macro_panel_dtypes():
    result = _clean_macro_panel(_make_raw())
    assert result["date"].dtype == object
    assert result["value"].dtype == float
    assert result["series_id"].dtype == object
    assert result["country_iso2"].dtype == object
    assert result["variable_name"].dtype == object
    assert result["category"].dtype == "Int64"
    assert result["category_name"].dtype == object
    assert result["source"].dtype == object


def test_clean_macro_panel_sorted_by_series_id():
    result = _clean_macro_panel(_make_raw())
    assert list(result["series_id"]) == ["DE_IP_001", "DE_IP_001", "DE_IP_002"]


def test_clean_macro_panel_removes_nan_values():
    raw = pd.DataFrame({
        "date": ["2024-01", "2024-02", "2024-03"],
        "value": [100.5, None, 102.3],
        "series_id": ["DE_IP_001", "DE_IP_001", "DE_IP_001"],
        "country_iso2": ["DE", "DE", "DE"],
        "variable_name": ["IP manufacturing", "IP manufacturing", "IP manufacturing"],
        "category": [1, 1, 1],
        "category_name": [
            "Industrial_production",
            "Industrial_production",
            "Industrial_production",
        ],
        "source": ["eurostat", "eurostat", "eurostat"],
    })
    result = _clean_macro_panel(raw)
    assert len(result) == 2
    assert not result["value"].isna().any()


def test_clean_macro_panel_removes_duplicates():
    raw = pd.DataFrame({
        "date": ["2024-01", "2024-01", "2024-02"],
        "value": [100.5, 100.5, 101.2],
        "series_id": ["DE_IP_001", "DE_IP_001", "DE_IP_001"],
        "country_iso2": ["DE", "DE", "DE"],
        "variable_name": ["IP manufacturing", "IP manufacturing", "IP manufacturing"],
        "category": [1, 1, 1],
        "category_name": [
            "Industrial_production",
            "Industrial_production",
            "Industrial_production",
        ],
        "source": ["eurostat", "eurostat", "eurostat"],
    })
    result = _clean_macro_panel(raw)
    assert len(result) == 2


def test_parse_dates_typical():
    dates = pd.Series(["2024-01", "2024-02", "2024-03"])
    result = _parse_dates(dates)
    assert list(result) == ["2024-01", "2024-02", "2024-03"]
    assert result.dtype == object


def test_parse_dates_handles_invalid():
    dates = pd.Series(["2024-01", "invalid", "2024-03"])
    result = _parse_dates(dates)
    assert pd.isna(result.iloc[1])
