"""Unit tests for OECD transformation logic (no network)."""

import pandas as pd
import pytest

from template_project.data_fetch.oecd import transform_oecd_data, validate_series_data


def test_transform_oecd_data_typical():
    """Test transformation with typical OECD API response."""
    # Create fake OECD API response (long format with TIME_PERIOD and OBS_VALUE)
    raw = pd.DataFrame({
        "TIME_PERIOD": ["2024-01", "2024-02", "2024-03"],
        "OBS_VALUE": [95.5, 96.2, 97.1],
        "MEASURE": ["BCICP", "BCICP", "BCICP"],
        "FREQUENCY": ["M", "M", "M"],
    })

    series_id = "DE_OECD_SENT_001"
    metadata = {
        "variable_name": "BTS Construction confidence indicator",
        "country_iso2": "DE",
        "category": 6,
        "category_name": "Sentiment",
    }

    result = transform_oecd_data(raw, series_id, metadata)

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
    assert result["series_id"].unique()[0] == "DE_OECD_SENT_001"
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
