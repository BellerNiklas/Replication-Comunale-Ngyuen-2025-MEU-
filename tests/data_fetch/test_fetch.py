"""Tests for fetch module (mocked to avoid network calls)."""

from unittest.mock import patch

import pandas as pd
import pytest

from meu_replication.data_fetch.fetch import fetch_many, fetch_one


@pytest.fixture()
def mock_registry():
    """Create mock registry with test series."""
    return pd.DataFrame(
        [
            {
                "series_id": "DE_IP_001",
                "source": "eurostat",
                "category": 1,
                "category_name": "Industrial_production",
                "country_iso2": "DE",
                "variable_name": "IP total",
                "dataset": "STS_INPR_M",
                "key": "",
                "filters_json": '{"geo": "DE", "nace_r2": "B-D"}',
                "unit_measure_filter": "",
                "frequency": "M",
                "start_period": "2003-01",
            },
            {
                "series_id": "EA_FX_001",
                "source": "ecb",
                "category": 8,
                "category_name": "EA_Financial",
                "country_iso2": "U2",
                "variable_name": "USD/EUR",
                "dataset": "EXR",
                "key": "M.USD.EUR.SP00.A",
                "filters_json": "",
                "unit_measure_filter": "",
                "frequency": "M",
                "start_period": "2003-01",
            },
            {
                "series_id": "BIS_NEER_001_DE",
                "source": "bis",
                "category": 7,
                "category_name": "Financial",
                "country_iso2": "DE",
                "variable_name": "NEER broad",
                "dataset": "WS_EER",
                "key": "M.N.B.DE",
                "filters_json": "",
                "unit_measure_filter": "",
                "frequency": "M",
                "start_period": "2003-01",
            },
        ]
    )


@pytest.fixture()
def mock_standardized_df():
    """Create mock standardized DataFrame."""
    return pd.DataFrame(
        {
            "date": ["2024-01", "2024-02"],
            "value": [100.0, 101.0],
            "series_id": ["TEST", "TEST"],
            "country_iso2": ["DE", "DE"],
            "variable_name": ["Test", "Test"],
            "category": [1, 1],
            "category_name": ["Test", "Test"],
            "source": ["test", "test"],
        }
    )


@pytest.mark.parametrize(
    ("series_id", "adapter_method", "standardize_method"),
    [
        ("DE_IP_001", "fetch_eurostat_raw", "standardize_from_eurostat"),
        ("EA_FX_001", "fetch_ecb_raw", "standardize_from_ecb_like"),
        ("BIS_NEER_001_DE", "fetch_bis_raw", "standardize_from_bis"),
    ],
)
@patch("meu_replication.data_fetch.fetch.adapters")
@patch("meu_replication.data_fetch.fetch.standardize")
def test_fetch_one_dispatches_to_correct_adapter(
    mock_std,
    mock_adapters,
    mock_registry,
    mock_standardized_df,
    series_id,
    adapter_method,
    standardize_method,
):
    """Test fetch_one dispatches to the correct adapter per source."""
    getattr(mock_adapters, adapter_method).return_value = pd.DataFrame()
    getattr(mock_std, standardize_method).return_value = mock_standardized_df
    mock_std.validate_long.return_value = None

    result = fetch_one(series_id, registry=mock_registry)

    getattr(mock_adapters, adapter_method).assert_called_once()
    getattr(mock_std, standardize_method).assert_called_once()
    assert isinstance(result, pd.DataFrame)


def test_fetch_one_not_found_raises(mock_registry):
    """Test that unknown series_id raises ValueError."""
    with pytest.raises(ValueError, match="not found in registry"):
        fetch_one("NONEXISTENT", registry=mock_registry)


@patch("meu_replication.data_fetch.fetch.fetch_one")
def test_fetch_many_concatenates(mock_fetch_one, mock_registry):
    """Test fetch_many concatenates results from multiple series."""
    # Setup mock to return different DataFrames for each series
    df1 = pd.DataFrame(
        {
            "date": ["2024-01"],
            "value": [100.0],
            "series_id": ["DE_IP_001"],
            "country_iso2": ["DE"],
            "variable_name": ["IP"],
            "category": [1],
            "category_name": ["Ind"],
            "source": ["eurostat"],
        }
    )
    df2 = pd.DataFrame(
        {
            "date": ["2024-01"],
            "value": [1.15],
            "series_id": ["EA_FX_001"],
            "country_iso2": ["U2"],
            "variable_name": ["FX"],
            "category": [8],
            "category_name": ["Fin"],
            "source": ["ecb"],
        }
    )

    mock_fetch_one.side_effect = [df1, df2]

    # Call
    result = fetch_many(["DE_IP_001", "EA_FX_001"], registry=mock_registry)

    # Verify
    assert len(result) == 2  # Should have concatenated both
    assert "DE_IP_001" in result["series_id"].values
    assert "EA_FX_001" in result["series_id"].values


@patch("meu_replication.data_fetch.fetch.fetch_one")
def test_fetch_many_continues_on_error(mock_fetch_one, mock_registry, capsys):
    """Test fetch_many continues when some series fail."""
    # First series succeeds, second fails
    df1 = pd.DataFrame(
        {
            "date": ["2024-01"],
            "value": [100.0],
            "series_id": ["DE_IP_001"],
            "country_iso2": ["DE"],
            "variable_name": ["IP"],
            "category": [1],
            "category_name": ["Ind"],
            "source": ["eurostat"],
        }
    )

    mock_fetch_one.side_effect = [df1, RuntimeError("API error")]

    # Call - should not raise
    result = fetch_many(["DE_IP_001", "EA_FX_001"], registry=mock_registry)

    # Verify we got the successful one
    assert len(result) == 1
    assert result["series_id"].iloc[0] == "DE_IP_001"

    # Verify error was printed
    captured = capsys.readouterr()
    assert "SKIPPED" in captured.out
    assert "EA_FX_001" in captured.out


@patch("meu_replication.data_fetch.fetch.fetch_one")
def test_fetch_many_all_fail_raises(mock_fetch_one, mock_registry):
    """Test fetch_many raises when all series fail."""
    mock_fetch_one.side_effect = RuntimeError("API error")

    with pytest.raises(ValueError, match="No data fetched for any series"):
        fetch_many(["DE_IP_001", "EA_FX_001"], registry=mock_registry)
