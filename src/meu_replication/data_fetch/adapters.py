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


def fetch_oecd_bulk(
    registry: pd.DataFrame,
    countries: pd.DataFrame,
    *,
    start_period: str = "2003-01",
) -> pd.DataFrame:
    """Fetch all OECD series in 4 bulk API calls using SDMX '+' syntax.

    Makes one request per OECD dataset, combining all countries and
    indicator variants into a single key. Reduces 152 individual calls to 4.

    Args:
        registry: Full series registry DataFrame.
        countries: Countries DataFrame with source_oecd column for ISO3 codes.
        start_period: Start date in YYYY-MM format.

    Returns:
        Concatenated raw DataFrame from all 4 OECD API calls,
        with an extra 'dataset' column identifying the source flow.
    """
    oecd_reg = registry[registry.source == "oecd"]
    datasets = oecd_reg["dataset"].unique()

    all_dfs = []
    for dataset in datasets:
        bulk_key = _build_oecd_bulk_key(oecd_reg, countries, dataset)
        raw = fetch_oecd_raw(dataset, bulk_key, start_period=start_period)
        raw = raw.assign(dataset=dataset)
        all_dfs.append(raw)

    return pd.concat(all_dfs, ignore_index=True)


def _build_oecd_bulk_key(
    oecd_registry: pd.DataFrame,
    countries: pd.DataFrame,
    dataset: str,
) -> str:
    """Build SDMX bulk key with '+' syntax for one OECD dataset.

    Analyses registry keys for the dataset to identify which dimensions
    vary across series, then combines all values with '+' per dimension.

    Args:
        oecd_registry: Registry filtered to source=="oecd".
        countries: Countries DataFrame with source_oecd column.
        dataset: OECD dataset flow ID.

    Returns:
        Bulk key string, e.g. "AUT+BEL+DEU.M.BCICP.PB.C+F+G47+GTU.Y._Z._Z.N"
    """
    ds_rows = oecd_registry[oecd_registry.dataset == dataset]

    # All EA ISO3 codes from countries table
    ea_countries = countries[countries.ea_member == True]  # noqa: E712
    iso3_codes = sorted(ea_countries["source_oecd"].unique())
    country_part = "+".join(iso3_codes)

    # Parse key structure: collect unique values per dimension position
    sample_key = ds_rows.iloc[0]["key"]
    n_dims = len(sample_key.split("."))

    dim_values = []
    for i in range(n_dims):
        if i == 0:
            dim_values.append(country_part)
        else:
            unique_vals = sorted(
                ds_rows["key"].apply(lambda k, idx=i: k.split(".")[idx]).unique()
            )
            dim_values.append("+".join(unique_vals))

    return ".".join(dim_values)


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
            # Exponential backoff on retries
            if attempt > 0:
                sleep(2**attempt)

            response = requests.get(url, params=params, timeout=timeout)

            # OECD 429 handling: wait and retry with longer backoff
            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  429 rate-limited, waiting {wait}s (attempt {attempt + 1}/{retries})...")
                sleep(wait)
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
