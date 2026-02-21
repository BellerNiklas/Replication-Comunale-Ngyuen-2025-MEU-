"""Unit tests for data cleaning functions."""

import pandas as pd
import pytest

from template_project.data_management.task_build_clean import (
    _clean_macro_panel,
    _parse_dates,
)


def test_clean_macro_panel_typical():
    """Test cleaning with typical combined data."""
    # Create fake combined raw data
    raw = pd.DataFrame({
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
    })

    result = _clean_macro_panel(raw)

    # Check output schema
    assert list(result.columns) == [
        "date",
        "value",
        "series_id",
        "country_iso2",
        "variable_name",
        "category",
        "category_name",
    ]

    # Check data types
    assert result["date"].dtype == object  # String
    assert result["value"].dtype == float
    assert result["series_id"].dtype == object
    assert result["country_iso2"].dtype == object
    assert result["variable_name"].dtype == object
    assert result["category"].dtype == "Int64"  # Nullable integer
    assert result["category_name"].dtype == object

    # Check sorting
    assert list(result["series_id"]) == ["DE_IP_001", "DE_IP_001", "DE_IP_002"]


def test_clean_macro_panel_removes_nan_values():
    """Test that NaN values are removed."""
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
    })

    result = _clean_macro_panel(raw)

    # Should only have 2 rows (NaN row dropped)
    assert len(result) == 2
    assert not result["value"].isna().any()


def test_clean_macro_panel_removes_duplicates():
    """Test that duplicate (series_id, date) pairs are removed."""
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
    })

    result = _clean_macro_panel(raw)

    # Should only have 2 rows (duplicate removed)
    assert len(result) == 2


def test_parse_dates_typical():
    """Test date parsing with typical YYYY-MM format."""
    dates = pd.Series(["2024-01", "2024-02", "2024-03"])
    result = _parse_dates(dates)

    assert list(result) == ["2024-01", "2024-02", "2024-03"]
    assert result.dtype == object  # String


def test_parse_dates_handles_invalid():
    """Test that invalid dates become NaT."""
    dates = pd.Series(["2024-01", "invalid", "2024-03"])
    result = _parse_dates(dates)

    # Second element should be NaN (NaT converted to None in strftime)
    assert pd.isna(result.iloc[1])
