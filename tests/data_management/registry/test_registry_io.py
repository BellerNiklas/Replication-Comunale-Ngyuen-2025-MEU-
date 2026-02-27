"""Tests for registry I/O and validation."""

import pandas as pd
import pytest

from meu_replication.data_management.registry.registry_io import (
    load_registry,
    validate_registry,
)


def test_load_registry_succeeds():
    """Test that load_registry successfully loads the real registry."""
    registry = load_registry()

    # Check it's a DataFrame
    assert isinstance(registry, pd.DataFrame)

    # Check it has rows
    assert len(registry) > 0

    # Check required columns exist
    required_cols = {
        "series_id",
        "source",
        "category",
        "category_name",
        "country_iso2",
        "variable_name",
        "dataset",
        "key",
        "filters_json",
        "unit_measure_filter",
        "frequency",
        "start_period",
    }
    assert required_cols.issubset(set(registry.columns))


def test_validate_registry_unique_series_id():
    """Test that duplicate series_id raises ValueError."""
    # Create DataFrame with duplicate series_id
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001", "DE_IP_001"],  # Duplicate
            "source": ["eurostat", "eurostat"],
            "category": [1, 1],
            "category_name": ["Industrial_production", "Industrial_production"],
            "country_iso2": ["DE", "DE"],
            "variable_name": ["Test variable", "Test variable"],
            "dataset": ["STS_INPR_M", "STS_INPR_M"],
            "key": ["", ""],
            "filters_json": ['{"geo": "DE"}', '{"geo": "DE"}'],
            "unit_measure_filter": ["", ""],
            "frequency": ["M", "M"],
            "start_period": ["2003-01", "2003-01"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate series_id found"):
        validate_registry(df)


def test_validate_registry_invalid_source_raises():
    """Test that invalid source value raises ValueError."""
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["invalid_source"],  # Invalid
            "category": [1],
            "category_name": ["Industrial_production"],
            "country_iso2": ["DE"],
            "variable_name": ["Test variable"],
            "dataset": ["STS_INPR_M"],
            "key": [""],
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
        }
    )

    with pytest.raises(ValueError, match="Invalid source values"):
        validate_registry(df)


def test_validate_registry_eurostat_requires_filters_json():
    """Test that Eurostat series without filters_json raises ValueError."""
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["eurostat"],
            "category": [1],
            "category_name": ["Industrial_production"],
            "country_iso2": ["DE"],
            "variable_name": ["Test variable"],
            "dataset": ["STS_INPR_M"],
            "key": [""],
            "filters_json": [""],  # Missing
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
        }
    )

    with pytest.raises(ValueError, match="Eurostat series requires 'filters_json'"):
        validate_registry(df)


def test_validate_registry_ecb_requires_key():
    """Test that ECB series without key raises ValueError."""
    df = pd.DataFrame(
        {
            "series_id": ["DE_INT_001"],
            "source": ["ecb"],
            "category": [7],
            "category_name": ["Financial"],
            "country_iso2": ["DE"],
            "variable_name": ["Test variable"],
            "dataset": ["IRS"],
            "key": [""],  # Missing
            "filters_json": [""],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
        }
    )

    with pytest.raises(ValueError, match="ECB series requires 'key'"):
        validate_registry(df)


def test_validate_registry_invalid_country_code_raises():
    """Test that invalid country_iso2 raises ValueError."""
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["eurostat"],
            "category": [1],
            "category_name": ["Industrial_production"],
            "country_iso2": ["DEU"],  # Should be 2 letters, not 3
            "variable_name": ["Test variable"],
            "dataset": ["STS_INPR_M"],
            "key": [""],
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
        }
    )

    with pytest.raises(ValueError, match="Invalid country_iso2 codes"):
        validate_registry(df)


def test_validate_registry_missing_dataset_raises():
    """Test that missing dataset field raises ValueError."""
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["eurostat"],
            "category": [1],
            "category_name": ["Industrial_production"],
            "country_iso2": ["DE"],
            "variable_name": ["Test variable"],
            "dataset": [""],  # Missing
            "key": [""],
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
        }
    )

    with pytest.raises(ValueError, match="Missing required field 'dataset'"):
        validate_registry(df)


def test_validate_registry_valid_registry_passes():
    """Test that a valid registry passes validation."""
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001", "DE_INT_001"],
            "source": ["eurostat", "ecb"],
            "category": [1, 7],
            "category_name": ["Industrial_production", "Financial"],
            "country_iso2": ["DE", "DE"],
            "variable_name": ["Test variable 1", "Test variable 2"],
            "dataset": ["STS_INPR_M", "IRS"],
            "key": ["", "M.DE.L.L40.CI.0000.EUR.N.Z"],
            "filters_json": ['{"geo": "DE"}', ""],
            "unit_measure_filter": ["", ""],
            "frequency": ["M", "M"],
            "start_period": ["2003-01", "2003-01"],
        }
    )

    # Should not raise
    validate_registry(df)
