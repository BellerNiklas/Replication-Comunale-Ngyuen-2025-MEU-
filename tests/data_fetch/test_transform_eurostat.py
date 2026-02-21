"""Unit tests for Eurostat transformation logic (no network)."""

import pandas as pd
import pytest

from template_project.data_fetch.eurostat import (
    transform_eurostat_data,
    validate_series_data,
)


def test_transform_eurostat_data_typical():
    """Test transformation with typical wide-format input."""
    # Create fake Eurostat API response (wide format)
    raw = pd.DataFrame({
        "freq": ["M", "M"],
        "geo": ["DE", "DE"],
        "nace_r2": ["C", "C"],
        "2024-01": [100.5, 100.5],
        "2024-02": [101.2, 101.2],
    })

    series_id = "DE_IP_001"
    metadata = {
        "variable_name": "IP manufacturing",
        "country_iso2": "DE",
        "category": 1,
        "category_name": "Industrial_production",
    }

    result = transform_eurostat_data(raw, series_id, metadata)

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

    # Check data
    assert len(result) == 4  # 2 rows × 2 time cols
    assert result["series_id"].unique()[0] == "DE_IP_001"
    assert result["country_iso2"].unique()[0] == "DE"


def test_validate_series_data_empty_raises():
    """Test that empty data raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="No data fetched"):
        validate_series_data(empty_df, "TEST_001")


def test_validate_series_data_all_nan_raises():
    """Test that all-NaN values raise ValueError."""
    df = pd.DataFrame({"value": [None, None, None]})
    with pytest.raises(ValueError, match="All values are NaN"):
        validate_series_data(df, "TEST_001")
