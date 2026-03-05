import pandas as pd
import pytest

from meu_replication.registry.registry_io import (
    load_registry,
    validate_registry,
)


def test_committed_registry_matches_templates():
    """Guard: committed series_registry.csv must match expand(templates x countries).

    Prevents drift between templates and the committed registry. If this fails,
    regenerate with: pixi run python -m meu_replication.registry.expand_registry
    """
    from meu_replication.config import load_countries
    from meu_replication.registry.expand_registry import expand_registry, load_templates

    committed = load_registry()
    expanded = expand_registry(load_templates(), load_countries())

    assert len(committed) == len(expanded), (
        f"Registry has {len(committed)} rows but expansion produces {len(expanded)}. "
        "Regenerate with: pixi run python -m meu_replication.registry.expand_registry"
    )

    committed_ids = set(committed["series_id"])
    expanded_ids = set(expanded["series_id"])
    assert committed_ids == expanded_ids, (
        f"Drift detected: {committed_ids ^ expanded_ids}"
    )


def test_load_registry_returns_dataframe():
    assert isinstance(load_registry(), pd.DataFrame)


def test_load_registry_has_rows():
    assert len(load_registry()) > 0


def test_load_registry_has_required_columns():
    registry = load_registry()
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
        "transformationcode",
    }
    assert required_cols.issubset(set(registry.columns))


def test_validate_registry_unique_series_id():
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001", "DE_IP_001"],
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
            "transformationcode": [5, 5],
        }
    )
    with pytest.raises(ValueError, match="Duplicate series_id found"):
        validate_registry(df)


def test_validate_registry_invalid_source_raises():
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["invalid_source"],
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
            "transformationcode": [5],
        }
    )
    with pytest.raises(ValueError, match="Invalid source values"):
        validate_registry(df)


def test_validate_registry_eurostat_requires_filters_json():
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
            "filters_json": [""],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
            "transformationcode": [5],
        }
    )
    with pytest.raises(ValueError, match="Eurostat series requires 'filters_json'"):
        validate_registry(df)


def test_validate_registry_ecb_requires_key():
    df = pd.DataFrame(
        {
            "series_id": ["DE_INT_001"],
            "source": ["ecb"],
            "category": [7],
            "category_name": ["Financial"],
            "country_iso2": ["DE"],
            "variable_name": ["Test variable"],
            "dataset": ["IRS"],
            "key": [""],
            "filters_json": [""],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
            "transformationcode": [2],
        }
    )
    with pytest.raises(ValueError, match="ECB series requires 'key'"):
        validate_registry(df)


def test_validate_registry_invalid_country_code_raises():
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["eurostat"],
            "category": [1],
            "category_name": ["Industrial_production"],
            "country_iso2": ["DEU"],
            "variable_name": ["Test variable"],
            "dataset": ["STS_INPR_M"],
            "key": [""],
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
            "transformationcode": [5],
        }
    )
    with pytest.raises(ValueError, match="Invalid country_iso2 codes"):
        validate_registry(df)


def test_validate_registry_missing_dataset_raises():
    df = pd.DataFrame(
        {
            "series_id": ["DE_IP_001"],
            "source": ["eurostat"],
            "category": [1],
            "category_name": ["Industrial_production"],
            "country_iso2": ["DE"],
            "variable_name": ["Test variable"],
            "dataset": [""],
            "key": [""],
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
            "transformationcode": [5],
        }
    )
    with pytest.raises(ValueError, match="Missing required field 'dataset'"):
        validate_registry(df)


def test_validate_registry_missing_transformationcode_raises():
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
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
        }
    )
    with pytest.raises(ValueError, match="Registry missing required columns"):
        validate_registry(df)


def test_validate_registry_non_integer_transformationcode_raises():
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
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
            "transformationcode": [1.5],
        }
    )
    with pytest.raises(ValueError, match="Non-integer transformationcode"):
        validate_registry(df)


def test_validate_registry_unsupported_transformationcode_raises():
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
            "filters_json": ['{"geo": "DE"}'],
            "unit_measure_filter": [""],
            "frequency": ["M"],
            "start_period": ["2003-01"],
            "transformationcode": [9],
        }
    )
    with pytest.raises(ValueError, match="Unsupported transformationcode values"):
        validate_registry(df)


def test_validate_registry_valid_registry_passes():
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
            "transformationcode": [5, 2],
        }
    )
    validate_registry(df)
