"""Minimal wrappers for each data source API."""

from io import StringIO
from time import sleep
from typing import Any

import eurostat
import pandas as pd
import requests


def fetch_eurostat_raw(dataset: str, filters: dict[str, Any]) -> pd.DataFrame:
    """Fetch from Eurostat, return wide DataFrame.

    Args:
        dataset: Eurostat dataset ID (e.g., "STS_INPR_M")
        filters: Dictionary of dimension filters (e.g., {"geo": "DE", "nace_r2": "B-D"})

    Returns:
        Wide format DataFrame with time columns and dimension columns.

    Raises:
        RuntimeError: If API request fails.
    """
    try:
        df = eurostat.get_data_df(dataset, filter_pars=filters)
        if df is None or df.empty:
            msg = f"No data returned from Eurostat for {dataset} with filters {filters}"
            raise ValueError(msg)
        return df
    except Exception as e:
        msg = f"Eurostat fetch failed for {dataset}: {e}"
        raise RuntimeError(msg) from e


def fetch_ecb_raw(flow: str, key: str, *, start_period: str = "2003-01") -> pd.DataFrame:
    """Fetch from ECB, return long DataFrame with TIME_PERIOD, OBS_VALUE.

    Args:
        flow: ECB flow/dataset ID (e.g., "EXR")
        key: SDMX key string (e.g., "M.USD.EUR.SP00.A")
        start_period: Start date in YYYY-MM format

    Returns:
        Long format DataFrame with TIME_PERIOD and OBS_VALUE columns.

    Raises:
        RuntimeError: If API request fails.
    """
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
    params = {
        "startPeriod": start_period,
        "format": "csvdata",
    }
    return _http_get_csv(url, params, timeout=60)


def fetch_oecd_raw(flow: str, key: str, *, start_period: str = "2003-01") -> pd.DataFrame:
    """Fetch from OECD, return long DataFrame with TIME_PERIOD, OBS_VALUE.

    Args:
        flow: OECD flow/dataset ID (e.g., "MEI")
        key: SDMX key string (e.g., "DEU.PRMNTO01.IXOB.M")
        start_period: Start date in YYYY-MM format

    Returns:
        Long format DataFrame with TIME_PERIOD and OBS_VALUE columns.

    Raises:
        RuntimeError: If API request fails.
    """
    url = f"https://sdmx.oecd.org/public/rest/data/{flow}/{key}"
    params = {
        "startPeriod": start_period,
        "format": "csv",
    }
    return _http_get_csv(url, params, timeout=60)


def fetch_bis_raw(dataset: str, key: str, *, start_period: str = "2003-01") -> pd.DataFrame:
    """Fetch from BIS, return long DataFrame with TIME_PERIOD, OBS_VALUE.

    Args:
        dataset: BIS dataset ID (e.g., "WS_EER")
        key: SDMX key string (e.g., "M.N.B.DE")
        start_period: Start date in YYYY-MM format

    Returns:
        Long format DataFrame with TIME_PERIOD and OBS_VALUE columns.

    Raises:
        RuntimeError: If API request fails.
    """
    url = f"https://stats.bis.org/api/v1/data/{dataset}/{key}"
    params = {
        "startPeriod": start_period,
        "format": "csv",
    }
    return _http_get_csv(url, params, timeout=60)


def _http_get_csv(
    url: str,
    params: dict[str, str],
    *,
    timeout: int = 30,
    retries: int = 3,
) -> pd.DataFrame:
    """HTTP GET with retry logic, return DataFrame.

    Args:
        url: API endpoint URL
        params: Query parameters
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        DataFrame parsed from CSV response.

    Raises:
        RuntimeError: If all retries fail.
    """
    last_error = None

    for attempt in range(retries):
        try:
            # Rate limiting - wait 1 second between requests
            if attempt > 0:
                sleep(2**attempt)  # Exponential backoff

            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()

            if not response.text.strip():
                msg = f"Empty response from {url}"
                raise ValueError(msg)

            # Parse CSV
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                msg = f"Empty DataFrame returned from {url}"
                raise ValueError(msg)

            return df

        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < retries - 1:
                continue
            msg = f"Failed after {retries} retries for {url}: {e}"
            raise RuntimeError(msg) from last_error

    # Should never reach here, but for type checking
    msg = f"Failed to fetch from {url}"
    raise RuntimeError(msg)
