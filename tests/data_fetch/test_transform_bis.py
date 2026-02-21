"""Unit tests for BIS transformation logic (no network)."""

import pandas as pd
import pytest

from template_project.data_fetch.bis import transform_bis_data, validate_series_data


def test_transform_bis_data_typical():
    """Test transformation with typical BIS API response."""
    # Create fake BIS API response (long format with TIME_PERIOD and OBS_VALUE)
    raw = pd.DataFrame({
        "TIME_PERIOD": ["2024-01", "2024-02", "2024-03"],
        "OBS_VALUE": [100.0, 101.5, 99.8],
        "EER_TYPE": ["N", "N", "N"],
        "EER_BASKET": ["B", "B", "B"],
    })

    series_id = "BIS_NEER_001_DE"
    metadata = {
        "variable_name": "Nominal Effective Exchange Rate – broad",
        "country_iso2": "DE",
        "category": 7,
        "category_name": "Financial",
    }

    result = transform_bis_data(raw, series_id, metadata)

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
    assert len(result) == 3
    assert result["series_id"].unique()[0] == "BIS_NEER_001_DE"
    assert result["country_iso2"].unique()[0] == "DE"


def test_validate_series_data_empty_raises():
    """Test that empty data raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="No data returned"):
        validate_series_data(empty_df, "TEST_001")


def test_validate_series_data_all_nan_raises():
    """Test that all-NaN values raise ValueError."""
    df = pd.DataFrame({"value": [None, None, None]})
    with pytest.raises(ValueError, match="All values are NaN"):
        validate_series_data(df, "TEST_001")
