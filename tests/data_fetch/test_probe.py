"""Tests for probe module (mocked to avoid network calls)."""

from unittest.mock import patch

import pandas as pd
import pytest

from template_project.data_fetch.probe import (
    _extract_template_id,
    _is_data_absence_error,
    _probe_start_period,
    probe_one,
)


# --- Fixtures ---


@pytest.fixture()
def mock_registry():
    """Create mock registry with test series for probing."""
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
                "filters_json": '{"geo": "DE", "nace_r2": "B-D", "startPeriod": 2003}',
                "unit_measure_filter": "",
                "frequency": "M",
                "start_period": "2003-01",
            },
            {
                "series_id": "DE_FIN_LOAN_001",
                "source": "ecb",
                "category": 7,
                "category_name": "Financial",
                "country_iso2": "DE",
                "variable_name": "MFI Loans",
                "dataset": "BSI",
                "key": "M.DE.N.A.A20.A.1.U6.1000.Z01.E",
                "filters_json": "",
                "unit_measure_filter": "",
                "frequency": "M",
                "start_period": "2003-01",
            },
            {
                "series_id": "DE_OECD_SENT_001",
                "source": "oecd",
                "category": 6,
                "category_name": "Sentiment",
                "country_iso2": "DE",
                "variable_name": "BTS Construction",
                "dataset": "OECD.SDD.STES,DSD_STES@DF_BTS,4.0",
                "key": "DEU.M.BCICP.PB.F.Y._Z._Z.N",
                "filters_json": "",
                "unit_measure_filter": "",
                "frequency": "M",
                "start_period": "2003-01",
            },
            {
                "series_id": "DE_NEER_001",
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
            {
                "series_id": "U2_FX_001",
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
        ]
    )


def _mock_standardized_df(n_rows: int) -> pd.DataFrame:
    """Create a mock standardized DataFrame with n_rows."""
    return pd.DataFrame(
        {
            "date": [f"2025-{i + 1:02d}" for i in range(n_rows)],
            "value": [100.0 + i for i in range(n_rows)],
            "series_id": ["TEST"] * n_rows,
            "country_iso2": ["DE"] * n_rows,
            "variable_name": ["Test"] * n_rows,
            "category": [1] * n_rows,
            "category_name": ["Test"] * n_rows,
            "source": ["test"] * n_rows,
        }
    )


# --- _extract_template_id ---


def test_extract_template_id_de():
    """Test template_id extraction for DE prefix."""
    assert _extract_template_id("DE_IP_001", "DE") == "IP_001"


def test_extract_template_id_u2():
    """Test template_id extraction for U2 prefix."""
    assert _extract_template_id("U2_FX_001", "U2") == "FX_001"


def test_extract_template_id_multi_underscore():
    """Test template_id extraction with multiple underscores."""
    assert _extract_template_id("DE_FIN_LOAN_001", "DE") == "FIN_LOAN_001"


def test_extract_template_id_no_prefix():
    """Test fallback when series_id doesn't start with country prefix."""
    assert _extract_template_id("UNKNOWN_001", "DE") == "UNKNOWN_001"


# --- _is_data_absence_error ---


def test_is_data_absence_no_data():
    """Test 'No data returned' classified as absence."""
    assert _is_data_absence_error(RuntimeError("No data returned from Eurostat"))


def test_is_data_absence_empty():
    """Test 'Empty response' classified as absence."""
    assert _is_data_absence_error(ValueError("Empty response from URL"))


def test_is_data_absence_empty_df():
    """Test 'Empty DataFrame' classified as absence."""
    assert _is_data_absence_error(RuntimeError("Empty DataFrame returned from URL"))


def test_is_data_absence_404():
    """Test '404' in error message classified as absence."""
    assert _is_data_absence_error(RuntimeError("HTTP 404 Not Found"))


def test_is_data_absence_400_bad_request():
    """Test '400 Client Error: Bad Request' classified as absence."""
    assert _is_data_absence_error(
        RuntimeError("400 Client Error: Bad Request for url: https://ec.europa.eu/eurostat/api/...")
    )


def test_is_not_absence_timeout():
    """Test timeout is NOT classified as absence."""
    assert not _is_data_absence_error(RuntimeError("Connection timed out"))


def test_is_not_absence_500():
    """Test 500 error is NOT classified as absence."""
    assert not _is_data_absence_error(RuntimeError("HTTP 500 Internal Server Error"))


def test_is_not_absence_rate_limit():
    """Test rate limit error is NOT classified as absence."""
    assert not _is_data_absence_error(RuntimeError("429 Too Many Requests"))


# --- _probe_start_period ---


def test_probe_start_period_fixed():
    """Test probe start period is fixed at 2020-01 for discontinued series."""
    result = _probe_start_period()
    assert result == "2020-01"


# --- probe_one ---


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_ok(mock_fetch, mock_registry):
    """Test probe returns 'ok' when ≥10 rows returned."""
    mock_fetch.return_value = _mock_standardized_df(12)

    result = probe_one("DE_IP_001", registry=mock_registry)

    assert result["status"] == "ok"
    assert result["rows_fetched"] == 12
    assert result["series_id"] == "DE_IP_001"
    assert result["template_id"] == "IP_001"
    assert result["country_iso2"] == "DE"
    assert result["error_kind"] == ""


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_ok_short(mock_fetch, mock_registry):
    """Test probe returns 'ok_short' when 1-9 rows returned."""
    mock_fetch.return_value = _mock_standardized_df(5)

    result = probe_one("DE_IP_001", registry=mock_registry)

    assert result["status"] == "ok_short"
    assert result["rows_fetched"] == 5


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_missing_empty_df(mock_fetch, mock_registry):
    """Test probe returns 'missing' when empty DataFrame returned."""
    mock_fetch.return_value = pd.DataFrame()

    result = probe_one("DE_IP_001", registry=mock_registry)

    assert result["status"] == "missing"
    assert result["rows_fetched"] == 0
    assert result["error_kind"] == "empty_result"


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_missing_no_data_error(mock_fetch, mock_registry):
    """Test probe returns 'missing' for 'No data returned' error."""
    mock_fetch.side_effect = RuntimeError("No data returned from Eurostat")

    result = probe_one("DE_IP_001", registry=mock_registry)

    assert result["status"] == "missing"
    assert result["rows_fetched"] == 0
    assert result["error_kind"] == "RuntimeError"


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_error_timeout(mock_fetch, mock_registry):
    """Test probe returns 'error' for timeout."""
    mock_fetch.side_effect = RuntimeError("Connection timed out after 60s")

    result = probe_one("DE_IP_001", registry=mock_registry)

    assert result["status"] == "error"
    assert result["rows_fetched"] == 0
    assert result["error_kind"] == "RuntimeError"
    assert "timed out" in result["error_message"]


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_ecb_country(mock_fetch, mock_registry):
    """Test probe works for ECB country-level series."""
    mock_fetch.return_value = _mock_standardized_df(10)

    result = probe_one("DE_FIN_LOAN_001", registry=mock_registry)

    assert result["status"] == "ok"
    assert result["template_id"] == "FIN_LOAN_001"


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_u2_ea_series(mock_fetch, mock_registry):
    """Test probe works for U2 EA-aggregate series."""
    mock_fetch.return_value = _mock_standardized_df(12)

    result = probe_one("U2_FX_001", registry=mock_registry)

    assert result["status"] == "ok"
    assert result["country_iso2"] == "U2"
    assert result["template_id"] == "FX_001"


# --- Dispatch to correct adapter ---


@patch("template_project.data_fetch.probe.standardize")
@patch("template_project.data_fetch.probe.adapters")
def test_probe_one_eurostat_calls_correct_adapter(
    mock_adapters, mock_std, mock_registry
):
    """Test Eurostat probe calls fetch_eurostat_raw."""
    mock_adapters.fetch_eurostat_raw.return_value = pd.DataFrame(
        {"geo": ["DE"], "2025-01": [100.0]}
    )
    mock_std.standardize_from_eurostat.return_value = _mock_standardized_df(1)

    probe_one("DE_IP_001", registry=mock_registry)

    mock_adapters.fetch_eurostat_raw.assert_called_once()
    call_args = mock_adapters.fetch_eurostat_raw.call_args
    # Verify filters have overridden startPeriod (2020, not 2003)
    filters = call_args[0][1]
    assert filters["startPeriod"] == 2020


@patch("template_project.data_fetch.probe.standardize")
@patch("template_project.data_fetch.probe.adapters")
def test_probe_one_ecb_calls_correct_adapter(mock_adapters, mock_std, mock_registry):
    """Test ECB probe calls fetch_ecb_raw with recent start_period."""
    mock_adapters.fetch_ecb_raw.return_value = pd.DataFrame(
        {"TIME_PERIOD": ["2025-01"], "OBS_VALUE": [100.0]}
    )
    mock_std.standardize_from_ecb_like.return_value = _mock_standardized_df(1)

    probe_one("DE_FIN_LOAN_001", registry=mock_registry)

    mock_adapters.fetch_ecb_raw.assert_called_once()
    call_args = mock_adapters.fetch_ecb_raw.call_args
    # start_period should be 2020-01 (not 2003-01)
    assert call_args.kwargs["start_period"] == "2020-01"


@patch("template_project.data_fetch.probe.standardize")
@patch("template_project.data_fetch.probe.adapters")
def test_probe_one_oecd_calls_correct_adapter(mock_adapters, mock_std, mock_registry):
    """Test OECD probe calls fetch_oecd_raw."""
    mock_adapters.fetch_oecd_raw.return_value = pd.DataFrame(
        {"TIME_PERIOD": ["2025-01"], "OBS_VALUE": [100.0]}
    )
    mock_std.standardize_from_ecb_like.return_value = _mock_standardized_df(1)

    probe_one("DE_OECD_SENT_001", registry=mock_registry)

    mock_adapters.fetch_oecd_raw.assert_called_once()


@patch("template_project.data_fetch.probe.standardize")
@patch("template_project.data_fetch.probe.adapters")
def test_probe_one_bis_calls_correct_adapter(mock_adapters, mock_std, mock_registry):
    """Test BIS probe calls fetch_bis_raw."""
    mock_adapters.fetch_bis_raw.return_value = pd.DataFrame(
        {"TIME_PERIOD": ["2025-01"], "OBS_VALUE": [100.0]}
    )
    mock_std.standardize_from_bis.return_value = _mock_standardized_df(1)

    probe_one("DE_NEER_001", registry=mock_registry)

    mock_adapters.fetch_bis_raw.assert_called_once()


# --- Error message truncation ---


@patch("template_project.data_fetch.probe._fetch_probe")
def test_probe_one_truncates_long_error(mock_fetch, mock_registry):
    """Test that error messages are truncated to 200 chars."""
    long_msg = "A" * 500
    mock_fetch.side_effect = RuntimeError(long_msg)

    result = probe_one("DE_IP_001", registry=mock_registry)

    assert len(result["error_message"]) == 200
