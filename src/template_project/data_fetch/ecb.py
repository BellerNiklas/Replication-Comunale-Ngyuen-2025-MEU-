"""Functions for fetching macroeconomic data from the ECB Statistical Data Warehouse.

Fetches two groups of variables:
1. Country-specific (categories 4 and 7): MFI balance sheet items, MFI interest
   rates, lending spreads, and car registrations.
2. Euro-area level (category 8): government benchmark bond yields, money market
   rates, DJ Euro Stoxx equity indices, monetary aggregates, and bilateral exchange
   rates (Table 2 in Comunale & Nguyen 2025 appendix).

REST API: https://data-api.ecb.europa.eu/service/data/{flow}/{key}
Key lookup: https://data.ecb.europa.eu/data/datasets

Key format varies by flow:
  EXR: FREQ.CURRENCY.CURRENCY_DENOM.EXR_TYPE.EXR_SUFFIX
  FM:  FREQ.REF_AREA.CURRENCY.PROVIDER_FM.INSTRUMENT_FM.DATA_TYPE_FM
  IRS: FREQ.REF_AREA.BS_SUFFIX.BS_ITEM.BS_COUNT_SECTOR.BS_REP_SECTOR.MATURITY.CURRENCY.BS_POSITION.BS_SUFFIX2
  BSI: FREQ.REF_AREA.ADJUSTMENT.BS_REP_SECTOR.BS_ITEM.MATURITY.BS_COUNT_SECTOR.BS_INSTR.BS_POSITION.BS_SUFFIX
  MIR: FREQ.REF_AREA.BS_REP_SECTOR.MIR_PRIV_SECTOR.BS_CAT.MIRR.BS_ITEM.MATURITY_CAT.DATA_TYPE.BS_COUNT_SECTOR.CURR_DENOM.IR_BUS_COV.TRG_SECTOR.OBS_CONF
"""

from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import requests

from template_project.config import BLD, SRC

ECB_BASE_URL: str = "https://data-api.ecb.europa.eu/service/data"

# ============================================================================
# DATASET_CONFIGS
#
# Each top-level key is a descriptive label (not always the ECB flow ID).
# Required fields per config:
#   flow         – ECB SDMX flow reference (e.g. "EXR", "FM", "BSI", "MIR")
#   category     – integer category (1-7 country-specific; 8 euro-area level)
#   category_name– string label for the output filename
#   country_iso2 – ISO 3166-1 alpha-2 code, or "U2" for the euro area
#   series_prefix– prefix for auto-generated series IDs
#   variables    – list of {id, key, desc} where key is the ECB SDMX key
#                  WITHOUT the flow prefix, e.g. "M.USD.EUR.SP00.A"
#
# VERIFICATION STATUS is noted inline.  Keys marked "NEEDS VERIFY" should be
# checked at https://sdw-wsrest.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata
# before relying on them.
# ============================================================================

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # ========================================================================
    # EXCHANGE RATES (EXR) – Euro area level
    # CONFIRMED: all five keys are valid ECB SDW series.
    # ========================================================================
    "EXR": {
        "flow": "EXR",
        "category": 8,
        "category_name": "EA_Financial",
        "country_iso2": "U2",
        "series_prefix": "EA_FX",
        "variables": [
            {"id": "001", "key": "M.USD.EUR.SP00.A", "desc": "USD/EUR exchange rate"},
            {"id": "002", "key": "M.GBP.EUR.SP00.A", "desc": "GBP/EUR exchange rate"},
            {"id": "003", "key": "M.JPY.EUR.SP00.A", "desc": "JPY/EUR exchange rate"},
            {"id": "004", "key": "M.CNY.EUR.SP00.A", "desc": "CNY/EUR exchange rate"},
            {"id": "005", "key": "M.CHF.EUR.SP00.A", "desc": "CHF/EUR exchange rate"},
        ],
    },
    # ========================================================================
    # MONEY MARKET RATES (FM) – Euro area level
    # EURIBOR tenors and EONIA.  LIKELY correct; verify if any fail.
    # ========================================================================
    "FM_RATES": {
        "flow": "FM",
        "category": 8,
        "category_name": "EA_Financial",
        "country_iso2": "U2",
        "series_prefix": "EA_MMR",
        "variables": [
            {
                "id": "001",
                "key": "M.U2.EUR.RT0.BB.EURIBOR1MD_.HSTA",
                "desc": "EURIBOR 1-month (hist. close avg.)",
            },
            {
                "id": "002",
                "key": "M.U2.EUR.RT0.BB.EURIBOR3MD_.HSTA",
                "desc": "EURIBOR 3-month (hist. close avg.)",
            },
            {
                "id": "003",
                "key": "M.U2.EUR.RT0.BB.EURIBOR6MD_.HSTA",
                "desc": "EURIBOR 6-month (hist. close avg.)",
            },
            {
                "id": "004",
                "key": "M.U2.EUR.RT0.BB.EURIBOR1YD_.HSTA",
                "desc": "EURIBOR 1-year (hist. close avg.)",
            },
            {
                "id": "005",
                "key": "M.U2.EUR.RT0.BB.EONIA_D.HSTA",
                "desc": "EONIA",
            },
        ],
    },
    # ========================================================================
    # DJ EURO STOXX EQUITY INDICES (FM) – Euro area level
    # DataStream-sourced via ECB FM flow.
    # NEEDS VERIFY: DataStream instrument codes – check ECB SDW browser.
    # ========================================================================
    "FM_EQUITY": {
        "flow": "FM",
        "category": 8,
        "category_name": "EA_Financial",
        "country_iso2": "U2",
        "series_prefix": "EA_EQI",
        "variables": [
            {
                "id": "001",
                "key": "M.U2.EUR.DS.EXR.DJE0000.HSTA",
                "desc": "DJ Euro Stoxx 50",
            },
            {
                "id": "002",
                "key": "M.U2.EUR.DS.EXR.S1ESXPB.HSTA",
                "desc": "DJ Euro Stoxx Basic Materials",
            },
            {
                "id": "003",
                "key": "M.U2.EUR.DS.EXR.DJESCSU.HSTA",
                "desc": "DJ Euro Stoxx Consumer Services",
            },
            {
                "id": "004",
                "key": "M.U2.EUR.DS.EXR.DJESCFU.HSTA",
                "desc": "DJ Euro Stoxx Financials",
            },
            {
                "id": "005",
                "key": "M.U2.EUR.DS.EXR.DJESOTU.HSTA",
                "desc": "DJ Euro Stoxx Technology",
            },
            {
                "id": "006",
                "key": "M.U2.EUR.DS.EXR.S1ESHIE.HSTA",
                "desc": "DJ Euro Stoxx Healthcare",
            },
            {
                "id": "007",
                "key": "M.U2.EUR.DS.EXR.DJESIOU.HSTA",
                "desc": "DJ Euro Stoxx Industrials",
            },
            {
                "id": "008",
                "key": "M.U2.EUR.DS.EXR.DJESNOU.HSTA",
                "desc": "DJ Euro Stoxx Telecommunications",
            },
            {
                "id": "009",
                "key": "M.U2.EUR.DS.EXR.DJESNUU.HSTA",
                "desc": "DJ Euro Stoxx Utilities",
            },
        ],
    },
    # ========================================================================
    # GOVERNMENT BENCHMARK BOND YIELDS (IRS) – Euro area level
    # Monthly government benchmark yields from the ECB IRS dataset.
    # NEEDS VERIFY: maturity codes within IRS key structure.
    # ========================================================================
    "IRS_EA": {
        "flow": "IRS",
        "category": 8,
        "category_name": "EA_Financial",
        "country_iso2": "U2",
        "series_prefix": "EA_BOND",
        "variables": [
            {
                "id": "001",
                "key": "M.U2.L.L40.CI.0.EUR.2Y.Z",
                "desc": "EA 2Y government benchmark bond yield",
            },
            {
                "id": "002",
                "key": "M.U2.L.L40.CI.0.EUR.3Y.Z",
                "desc": "EA 3Y government benchmark bond yield",
            },
            {
                "id": "003",
                "key": "M.U2.L.L40.CI.0.EUR.5Y.Z",
                "desc": "EA 5Y government benchmark bond yield",
            },
            {
                "id": "004",
                "key": "M.U2.L.L40.CI.0.EUR.7Y.Z",
                "desc": "EA 7Y government benchmark bond yield",
            },
            {
                "id": "005",
                "key": "M.U2.L.L40.CI.0.EUR.N.Z",
                "desc": "EA 10Y government benchmark bond yield",
            },
            {
                "id": "006",
                "key": "M.U2.R.L40.CI.0.EUR.N.Z",
                "desc": "EA 10Y real government benchmark bond yield",
            },
        ],
    },
    # ========================================================================
    # MFI BALANCE SHEET – LOANS (BSI) – Germany country-specific
    # Outstanding loan stocks at end of period, all currencies combined.
    # Counterpart sector codes: 2300=All domestic, 2250=HH (S.14+S.15),
    # 2240=NFCs (S.11).
    # NEEDS VERIFY: counterpart codes and key structure.
    # ========================================================================
    "BSI_LOANS": {
        "flow": "BSI",
        "category": 7,
        "category_name": "Financial",
        "country_iso2": "DE",
        "series_prefix": "DE_FIN_LOAN",
        "variables": [
            {
                "id": "001",
                "key": "M.DE.N.A.A20.A.1.U2.2300.Z01.E",
                "desc": "MFI Loans – Total outstanding (all domestic sectors)",
            },
            {
                "id": "002",
                "key": "M.DE.N.A.A20.A.1.U2.2250.Z01.E",
                "desc": "MFI Loans – To households & NPISHs (S.14+S.15)",
            },
            {
                "id": "003",
                "key": "M.DE.N.A.A20.A.1.U2.2240.Z01.E",
                "desc": "MFI Loans – To non-financial corporations (S.11)",
            },
        ],
    },
    # ========================================================================
    # MFI BALANCE SHEET – DEPOSIT LIABILITIES (BSI) – Germany
    # NEEDS VERIFY: BS_ITEM for liabilities and counterpart codes.
    # ========================================================================
    "BSI_DEPOSITS": {
        "flow": "BSI",
        "category": 7,
        "category_name": "Financial",
        "country_iso2": "DE",
        "series_prefix": "DE_FIN_DEP",
        "variables": [
            {
                "id": "001",
                "key": "M.DE.N.A.L20.A.1.U2.2201.Z01.E",
                "desc": "MFI Deposit liabilities – To MFIs",
            },
            {
                "id": "002",
                "key": "M.DE.N.A.L20.A.1.U2.2240.Z01.E",
                "desc": "MFI Deposit liabilities – To NFCs (S.11)",
            },
            {
                "id": "003",
                "key": "M.DE.N.A.L20.A.1.U2.2250.Z01.E",
                "desc": "MFI Deposit liabilities – To households (S.14+S.15)",
            },
        ],
    },
    # ========================================================================
    # MFI BALANCE SHEET – DEBT SECURITIES HELD (BSI) – Germany
    # NEEDS VERIFY: BS_ITEM for debt securities and counterpart codes.
    # ========================================================================
    "BSI_DEBT": {
        "flow": "BSI",
        "category": 7,
        "category_name": "Financial",
        "country_iso2": "DE",
        "series_prefix": "DE_FIN_DSH",
        "variables": [
            {
                "id": "001",
                "key": "M.DE.N.A.A40.A.1.U2.2240.Z01.E",
                "desc": "MFI Debt securities held – NFCs (S.11)",
            },
            {
                "id": "002",
                "key": "M.DE.N.A.A40.A.1.U2.2201.Z01.E",
                "desc": "MFI Debt securities held – MFIs",
            },
            {
                "id": "003",
                "key": "M.DE.N.A.A40.A.1.U2.2210.Z01.E",
                "desc": "MFI Debt securities held – General government",
            },
        ],
    },
    # ========================================================================
    # MONETARY AGGREGATES M1 and M3 (BSI) – Euro area level
    # Index of Notional Stocks, working-day and seasonally adjusted.
    # NEEDS VERIFY: adjustment codes and M1/M3 item codes in BSI.
    # ========================================================================
    "BSI_MON_AGG": {
        "flow": "BSI",
        "category": 8,
        "category_name": "EA_Financial",
        "country_iso2": "U2",
        "series_prefix": "EA_MON",
        "variables": [
            {
                "id": "001",
                "key": "M.U2.Y.V.M10.M.I.U2.2300.Z01.E",
                "desc": "M1 – Index of Notional Stocks (SA+WDA)",
            },
            {
                "id": "002",
                "key": "M.U2.Y.V.M30.M.I.U2.2300.Z01.E",
                "desc": "M3 – Index of Notional Stocks (SA+WDA)",
            },
        ],
    },
    # ========================================================================
    # MFI INTEREST RATES ON NEW BUSINESS (MIR) – Germany country-specific
    # Includes lending spreads vs. swap rate and new-business lending/deposit
    # rates at various maturities.
    # NEEDS VERIFY: all MIR keys – structure is complex; check ECB SDW browser.
    # ========================================================================
    "MIR_DE": {
        "flow": "MIR",
        "category": 7,
        "category_name": "Financial",
        "country_iso2": "DE",
        "series_prefix": "DE_FIN_MIR",
        "variables": [
            # Lending spreads (MIR vs. swap rate at matching maturity)
            {
                "id": "SPR_001",
                "key": "M.DE.B.A2Z.A.R.A.2240.EUR.N.R.A",
                "desc": "Lending spread – NFC loans vs. swap rate",
            },
            {
                "id": "SPR_002",
                "key": "M.DE.B.A2Z.A.R.A.2250.EUR.N.R.A",
                "desc": "Lending spread – Household loans vs. swap rate",
            },
            # New-business lending rates
            {
                "id": "LEN_001",
                "key": "M.DE.B.A2Z.A.R.A.2250.EUR.N.A.A",
                "desc": "MIR – Lending for house purchase (new business)",
            },
            {
                "id": "LEN_002",
                "key": "M.DE.B.A2Z.O.R.A.2250.EUR.N.A.A",
                "desc": "MIR – Other lending, households (new business)",
            },
            # Deposit rates on new business
            {
                "id": "DEP_001",
                "key": "M.DE.B.L20.A.R.L.2300.EUR.N.A.A",
                "desc": "MIR – Deposit rate, total new business",
            },
            {
                "id": "DEP_002",
                "key": "M.DE.B.L20.O.R.L.2300.EUR.N.A.A",
                "desc": "MIR – Deposit rate, up to 1 year",
            },
            {
                "id": "DEP_003",
                "key": "M.DE.B.L20.P.R.L.2300.EUR.N.A.A",
                "desc": "MIR – Deposit rate, over 1Y up to 2Y",
            },
            {
                "id": "DEP_004",
                "key": "M.DE.B.L20.Q.R.L.2300.EUR.N.A.A",
                "desc": "MIR – Deposit rate, over 2 years",
            },
            {
                "id": "DEP_005",
                "key": "M.DE.B.L20.M.R.L.2300.EUR.N.A.A",
                "desc": "MIR – Deposit rate, up to 2 years (composite)",
            },
        ],
    },
    # ========================================================================
    # CAR REGISTRATIONS – ECB absolute value series (category 4)
    # Absolute registration counts (SA+WDA), complementing Eurostat indices.
    # NEEDS VERIFY: ECB STS flow is accessible and key format is correct.
    # Look up under ECB SDW → Statistics → Economy → Short-term statistics.
    # ========================================================================
    "STS_CARS_ECB": {
        "flow": "STS",
        "category": 4,
        "category_name": "Activity_indicators",
        "country_iso2": "DE",
        "series_prefix": "DE_ACT_CARS_ECB",
        "variables": [
            {
                "id": "001",
                "key": "M.DE.S.BCRT.PASS.3.STS.ABS.WDA",
                "desc": "Car registrations – New passenger cars (SA+WDA, absolute)",
            },
            {
                "id": "002",
                "key": "M.DE.S.BCRT.COMM.3.STS.ABS.WDA",
                "desc": "Car registrations – New commercial vehicles (SA+WDA)",
            },
            {
                "id": "003",
                "key": "M.DE.S.BCRT.HCV.3.STS.ABS.WDA",
                "desc": "Car registrations – New heavy commercial vehicles (SA+WDA)",
            },
            {
                "id": "004",
                "key": "M.DE.S.BCRT.LCV.3.STS.ABS.WDA",
                "desc": "Car registrations – New light commercial vehicles (SA+WDA)",
            },
        ],
    },
}

# ============================================================================
# Minimal single-series config used by test_fetch() to verify API access.
# ============================================================================
TEST_CONFIG: dict[str, str] = {
    "flow": "EXR",
    "key": "M.USD.EUR.SP00.A",
    "series_id": "EA_FX_001",
    "desc": "USD/EUR exchange rate (test series)",
}


def generate_all_variable_configs() -> list[dict[str, Any]]:
    """Expand DATASET_CONFIGS into a flat list of per-variable config dicts.

    Returns:
        List of dicts, each containing:
        - series_id:     Full identifier (e.g. 'EA_FX_001')
        - flow:          ECB SDMX flow (e.g. 'EXR')
        - key:           ECB SDMX key without flow prefix
        - description:   Human-readable label
        - category:      Category integer
        - category_name: Category string
        - country_iso2:  Country code or 'U2'

    """
    all_configs: list[dict[str, Any]] = []

    for _label, ds_cfg in DATASET_CONFIGS.items():
        for var in ds_cfg["variables"]:
            series_id = f"{ds_cfg['series_prefix']}_{var['id']}"
            all_configs.append(
                {
                    "series_id": series_id,
                    "flow": ds_cfg["flow"],
                    "key": var["key"],
                    "description": var["desc"],
                    "category": ds_cfg["category"],
                    "category_name": ds_cfg["category_name"],
                    "country_iso2": ds_cfg["country_iso2"],
                }
            )

    return all_configs


def fetch_ecb_series(flow: str, key: str) -> pd.DataFrame:
    """Fetch a single time series from the ECB SDW REST API.

    Args:
        flow: ECB SDMX flow reference (e.g. 'EXR', 'FM', 'BSI').
        key:  SDMX key string without flow prefix (e.g. 'M.USD.EUR.SP00.A').

    Returns:
        DataFrame as returned by the ECB API (long format with TIME_PERIOD
        and OBS_VALUE columns).

    Raises:
        ValueError: If flow or key is empty.
        RuntimeError: If the API request fails after all retries.

    """
    if not flow:
        msg = "flow cannot be empty"
        raise ValueError(msg)
    if not key:
        msg = "key cannot be empty"
        raise ValueError(msg)

    url = f"{ECB_BASE_URL}/{flow}/{key}"
    params: dict[str, str] = {"startPeriod": "2003-01", "format": "csvdata"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))
            sleep(1)  # Rate limiting
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                msg = f"Failed to fetch {flow}/{key} after {max_retries} attempts: {e}"
                raise RuntimeError(msg) from e
            sleep(2**attempt)

    msg = "Unreachable code"
    raise RuntimeError(msg)


def transform_ecb_data(
    data: pd.DataFrame,
    series_id: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Transform ECB REST API response (long format) to standardised MEU format.

    The ECB 'csvdata' format returns one row per observation with TIME_PERIOD
    and OBS_VALUE as the key columns, plus dimension columns as metadata.

    Args:
        data:      Raw DataFrame from fetch_ecb_series().
        series_id: Series identifier (e.g. 'EA_FX_001').
        metadata:  Dict containing country_iso2, variable_name, category,
                   category_name.

    Returns:
        Long-format DataFrame with columns:
        date | value | series_id | country_iso2 | variable_name |
        category | category_name

    Raises:
        ValueError: If TIME_PERIOD or OBS_VALUE columns are missing.

    """
    if "TIME_PERIOD" not in data.columns:
        msg = f"No TIME_PERIOD column in ECB response for {series_id}"
        raise ValueError(msg)
    if "OBS_VALUE" not in data.columns:
        msg = f"No OBS_VALUE column in ECB response for {series_id}"
        raise ValueError(msg)

    result = pd.DataFrame(
        {
            "date": data["TIME_PERIOD"].astype(str),
            "value": pd.to_numeric(data["OBS_VALUE"], errors="coerce"),
            "series_id": series_id,
            "country_iso2": metadata.get("country_iso2", "U2"),
            "variable_name": metadata.get("variable_name", ""),
            "category": metadata.get("category"),
            "category_name": metadata.get("category_name", ""),
        }
    )

    # Filter to sample start and sort
    result = result[result["date"] >= "2003"].sort_values("date").reset_index(drop=True)

    return result


def validate_series_data(data: pd.DataFrame, series_id: str) -> None:
    """Validate that fetched series meets minimum quality requirements.

    Args:
        data:      Transformed DataFrame to validate.
        series_id: Series identifier used in error messages.

    Raises:
        ValueError: If data is empty or all values are NaN.

    """
    if data.empty:
        msg = f"No data fetched for {series_id}"
        raise ValueError(msg)
    if "value" in data.columns and data["value"].isna().all():
        msg = f"All values are NaN for {series_id}"
        raise ValueError(msg)


def fetch_category_variables(
    category_num: int,
    all_configs: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch all variables belonging to a given category number.

    Args:
        category_num: Target category (e.g. 7 for financial, 8 for EA-level).
        all_configs:  Full flat config list from generate_all_variable_configs().

    Returns:
        Tuple (successful_data, failed_series):
        - successful_data: dict mapping series_id → transformed DataFrame
        - failed_series:   dict mapping series_id → error message

    """
    category_configs = [c for c in all_configs if c["category"] == category_num]

    successful_data: dict[str, pd.DataFrame] = {}
    failed_series: dict[str, str] = {}

    for i, cfg in enumerate(category_configs, 1):
        series_id = cfg["series_id"]
        print(f"  [{i}/{len(category_configs)}] {series_id}: {cfg['description']}")

        try:
            raw = fetch_ecb_series(cfg["flow"], cfg["key"])
            metadata = {
                "country_iso2": cfg["country_iso2"],
                "variable_name": cfg["description"],
                "category": cfg["category"],
                "category_name": cfg["category_name"],
            }
            transformed = transform_ecb_data(raw, series_id, metadata)
            validate_series_data(transformed, series_id)
            successful_data[series_id] = transformed
            print(f"    SUCCESS: {len(transformed)} rows")

        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            print(f"    FAILED: {error_msg[:100]}")
            failed_series[series_id] = error_msg

    return successful_data, failed_series


def concatenate_category_data(
    category_dfs: dict[str, pd.DataFrame],
    category_num: int,
    category_name: str,
) -> pd.DataFrame:
    """Concatenate all DataFrames for a category into a single DataFrame.

    Args:
        category_dfs:  Dict mapping series_id → DataFrame.
        category_num:  Category number (used in error message only).
        category_name: Category name (used in error message only).

    Returns:
        Concatenated DataFrame sorted by date then series_id.

    Raises:
        ValueError: If category_dfs is empty.

    """
    if not category_dfs:
        msg = f"No successful data for category {category_num} ({category_name})"
        raise ValueError(msg)

    concatenated = pd.concat(category_dfs.values(), ignore_index=True)
    return concatenated.sort_values(["date", "series_id"]).reset_index(drop=True)


def save_category_csv(
    data: pd.DataFrame,
    category_num: int,
    category_name: str,
    output_dir: Path,
) -> Path:
    """Save category data to a CSV file following project naming convention.

    Args:
        data:          DataFrame to save.
        category_num:  Category number used in the filename.
        category_name: Category name used in the filename.
        output_dir:    Directory where the CSV will be written.

    Returns:
        Path to the saved CSV file.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"category_{category_num}_{category_name}.csv"
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    return output_path


def test_fetch() -> bool:
    """Fetch a single confirmed series to verify ECB API connectivity.

    Attempts to retrieve the USD/EUR exchange rate (EXR.M.USD.EUR.SP00.A),
    which is a stable, always-available ECB SDW series.  Prints the result
    summary and returns True on success, False on failure.

    Returns:
        True if the test series fetched successfully, False otherwise.

    """
    cfg = TEST_CONFIG
    print(f"TEST: fetching {cfg['flow']}/{cfg['key']} ({cfg['desc']})")
    try:
        raw = fetch_ecb_series(cfg["flow"], cfg["key"])
        metadata = {
            "country_iso2": "U2",
            "variable_name": cfg["desc"],
            "category": 8,
            "category_name": "EA_Financial",
        }
        df = transform_ecb_data(raw, cfg["series_id"], metadata)
        validate_series_data(df, cfg["series_id"])
        print(f"  SUCCESS: {len(df)} rows, date range {df['date'].min()} – {df['date'].max()}")
        print(f"  Sample value ({df['date'].iloc[-1]}): {df['value'].iloc[-1]:.4f}")
        return True  # noqa: TRY300
    except (ValueError, RuntimeError) as e:
        print(f"  FAILED: {e}")
        return False


def fetch_all_ecb_variables(
    output_dir: Path,
    *,
    test_mode: bool = False,
) -> dict[str, Path]:
    """Fetch all ECB variables and save category CSV files.

    Args:
        output_dir: Directory where category CSV files will be saved.
        test_mode:  If True, only run test_fetch() and skip the full fetch.

    Returns:
        Dict mapping 'category_{N}' keys to saved CSV file paths.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        success = test_fetch()
        if not success:
            print("Test fetch failed – check API connectivity before running full mode.")
        return {}

    all_configs = generate_all_variable_configs()
    results: dict[str, Path] = {}
    summary_stats: dict[int, dict[str, Any]] = {}

    # Determine which category numbers exist in configs
    category_nums = sorted({c["category"] for c in all_configs})
    category_names = {
        c["category"]: c["category_name"] for c in all_configs
    }

    for category_num in category_nums:
        cat_name = category_names[category_num]
        cat_configs = [c for c in all_configs if c["category"] == category_num]
        print(
            f"\nFetching Category {category_num}: {cat_name} "
            f"({len(cat_configs)} variables)"
        )

        successful_data, failed_series = fetch_category_variables(
            category_num, all_configs
        )

        total = len(cat_configs)
        success_count = len(successful_data)
        fail_count = len(failed_series)
        summary_stats[category_num] = {
            "total": total,
            "success": success_count,
            "failed": fail_count,
            "failed_series": failed_series,
            "category_name": cat_name,
        }

        if successful_data:
            cat_df = concatenate_category_data(successful_data, category_num, cat_name)
            output_path = save_category_csv(cat_df, category_num, cat_name, output_dir)
            results[f"category_{category_num}"] = output_path
            print(
                f"\n  Category {category_num}: {success_count}/{total} succeeded "
                f"({fail_count} failed)"
            )
            print(f"    Saved: {output_path.name} ({len(cat_df):,} rows)")
        else:
            print(f"\n  Category {category_num}: all {total} variables failed!")

    # Final summary
    print("\n" + "=" * 70)
    print("ECB FETCH SUMMARY")
    print("=" * 70)
    total_vars = sum(s["total"] for s in summary_stats.values())
    total_success = sum(s["success"] for s in summary_stats.values())
    total_failed = sum(s["failed"] for s in summary_stats.values())
    print(f"\nOverall: {total_success}/{total_vars} variables fetched successfully")
    if total_failed > 0:
        print(f"\nFailed ({total_failed}):")
        for cat_num, stats in summary_stats.items():
            for sid, err in stats["failed_series"].items():
                print(f"  [{cat_num}] {sid}: {err[:80]}")

    return results


def main() -> None:
    """Fetch ECB data and save to bld/data/raw/ecb/.

    Pass --test to run only the single-series connectivity test.
    """
    import sys  # noqa: PLC0415

    output_dir = BLD / "data" / "raw" / "ecb"
    test_mode = "--test" in sys.argv

    print("ECB Statistical Data Warehouse fetch")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'TEST' if test_mode else 'FULL'}\n")

    fetch_all_ecb_variables(output_dir, test_mode=test_mode)


if __name__ == "__main__":
    main()
