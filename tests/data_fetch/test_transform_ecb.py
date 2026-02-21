"""Unit tests for ECB transformation logic (no network)."""

import pandas as pd
import pytest

from template_project.data_fetch.ecb import transform_ecb_data, validate_series_data


def test_transform_ecb_data_typical():
    """Test transformation with typical ECB API response."""
    # Create fake ECB API response (long format with TIME_PERIOD and OBS_VALUE)
    raw = pd.DataFrame({
        "TIME_PERIOD": ["2024-01", "2024-02", "2024-03"],
        "OBS_VALUE": [1.05, 1.06, 1.07],
        "FREQ": ["M", "M", "M"],
        "CURRENCY": ["USD", "USD", "USD"],
    })

    series_id = "EA_FX_001"
    metadata = {
        "variable_name": "USD/EUR exchange rate",
        "country_iso2": "U2",
        "category": 8,
        "category_name": "EA_Financial",
    }

    result = transform_ecb_data(raw, series_id, metadata)

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
    assert result["series_id"].unique()[0] == "EA_FX_001"
    assert result["country_iso2"].unique()[0] == "U2"


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
