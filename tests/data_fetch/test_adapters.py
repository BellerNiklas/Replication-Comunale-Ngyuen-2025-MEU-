from unittest.mock import Mock, patch

import pytest
import requests

from meu_replication.data_fetch.adapters import _http_get_csv


def _response(status_code: int, text: str) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    if status_code >= 400 and status_code != 429:
        resp.raise_for_status.side_effect = requests.RequestException(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


@patch("meu_replication.data_fetch.adapters.requests.get")
def test_http_get_csv_success(mock_get):
    mock_get.return_value = _response(200, "TIME_PERIOD,OBS_VALUE\n2024-01,1.2\n")

    df = _http_get_csv("https://example.org", {"a": "b"}, retries=3)

    assert len(df) == 1
    assert list(df.columns) == ["TIME_PERIOD", "OBS_VALUE"]


@patch("meu_replication.data_fetch.adapters.sleep")
@patch("meu_replication.data_fetch.adapters.requests.get")
def test_http_get_csv_retries_then_success(mock_get, mock_sleep):
    mock_get.side_effect = [
        requests.RequestException("network fail"),
        _response(200, "TIME_PERIOD,OBS_VALUE\n2024-01,1.2\n"),
    ]

    df = _http_get_csv("https://example.org", {}, retries=2)

    assert len(df) == 1
    assert mock_get.call_count == 2
    mock_sleep.assert_called_with(2)


@patch("meu_replication.data_fetch.adapters.sleep")
@patch("meu_replication.data_fetch.adapters.requests.get")
def test_http_get_csv_handles_429_with_wait(mock_get, mock_sleep):
    mock_get.side_effect = [
        _response(429, "rate limited"),
        _response(200, "TIME_PERIOD,OBS_VALUE\n2024-01,1.2\n"),
    ]

    df = _http_get_csv("https://example.org", {}, retries=3)

    assert len(df) == 1
    # First call gets 429, second call succeeds in same attempt.
    assert mock_get.call_count == 2
    mock_sleep.assert_called_with(15)


@patch("meu_replication.data_fetch.adapters.requests.get")
def test_http_get_csv_raises_after_retries(mock_get):
    mock_get.side_effect = requests.RequestException("always failing")

    with pytest.raises(RuntimeError, match="Failed after 2 retries"):
        _http_get_csv("https://example.org", {}, retries=2)
