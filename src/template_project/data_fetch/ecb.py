"""Functions for fetching macroeconomic data from the ECB Statistical Data Warehouse.

Fetches two groups of variables used in Comunale & Nguyen (2025):
1. Country-specific (categories 4 and 7): MFI balance sheet items, MFI interest rates,
   and car registrations.
2. Euro-area level (category 8): government benchmark bond yields, money market rates,
   DJ Euro Stoxx equity indices, monetary aggregates, and bilateral exchange rates.

REST API: https://data-api.ecb.europa.eu/service/data/{flow}/{key}
Key lookup: https://data.ecb.europa.eu/data/datasets

Key formats confirmed via live API exploration:
  EXR:  FREQ.CURRENCY.CURRENCY_DENOM.EXR_TYPE.EXR_SUFFIX
  FM:   FREQ.REF_AREA.CURRENCY.PROVIDER_FM.INSTRUMENT_FM.PROVIDER_FM_ID.DATA_TYPE_FM
  BSI:  FREQ.REF_AREA.ADJUSTMENT.BS_REP_SECTOR.BS_ITEM.MATURITY_ORIG.DATA_TYPE.
        COUNT_AREA.BS_COUNT_SECTOR.CURRENCY_TRANS.BS_SUFFIX
  MIR:  FREQ.REF_AREA.BS_REP_SECTOR.BS_ITEM.MATURITY_NOT_IRATE.DATA_TYPE_MIR.
        AMOUNT_CAT.BS_COUNT_SECTOR.CURRENCY_TRANS.IR_BUS_COV
  STS:  FREQ.REF_AREA.ADJUSTMENT.STS_CONCEPT.STS_CLASS.STS_INSTITUTION.STS_SUFFIX

Important:
  - BSI domestic counterpart area is U6 (not U2, which is euro-area cross-border).
  - BSI debt securities use item code A30 (not A40).
  - MIR has exactly 10 key dimensions.
  - FM benchmark bond yields are in the FM flow (PROVIDER_FM=4F), not the IRS flow.
  - EONIA was discontinued January 2022; it is omitted here.
  - ECB car registration STS series was last published 2022-12 (discontinued).
"""

from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import requests

from template_project.config import BLD, SRC  # noqa: F401

ECB_BASE_URL: str = "https://data-api.ecb.europa.eu/service/data"

# ============================================================================
# DATASET_CONFIGS
#
# Each top-level key is a descriptive label.
# Required fields per config:
#   flow         – ECB SDMX flow reference (e.g. "EXR", "FM", "BSI", "MIR")
#   category     – integer category (4 or 7 = country-specific; 8 = euro-area level)
#   category_name– string label for the output filename
#   country_iso2 – ISO 3166-1 alpha-2 code, or "U2" for the euro area
#   series_prefix– prefix for auto-generated series IDs
#   variables    – list of {id, key, desc}
#
# All keys marked CONFIRMED were verified against the live ECB Data Portal API.
# ============================================================================

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    # ========================================================================
    # EXCHANGE RATES (EXR) – Euro area level, category 8
    # Key: FREQ.CURRENCY.CURRENCY_DENOM.EXR_TYPE.EXR_SUFFIX
    # CONFIRMED: all five keys return 200 with 277+ monthly rows.
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
    # GOVERNMENT BENCHMARK BOND YIELDS (FM) – Euro area level, category 8
    # Key: FREQ.REF_AREA.CURRENCY.PROVIDER_FM.INSTRUMENT_FM.PROVIDER_FM_ID.DATA_TYPE_FM
    #   PROVIDER_FM=4F (ECB benchmark series)
    #   INSTRUMENT_FM=BB (benchmark bond)
    #   DATA_TYPE_FM=YLD (yield) or YLDA (real yield average)
    # CONFIRMED: all five keys return 200.
    # Note: These are in the FM flow, NOT the IRS flow.
    # ========================================================================
    "FM_BONDS": {
        "flow": "FM",
        "category": 8,
        "category_name": "EA_Financial",
        "country_iso2": "U2",
        "series_prefix": "EA_BOND",
        "variables": [
            {
                "id": "001",
                "key": "M.U2.EUR.4F.BB.U2_2Y.YLD",
                "desc": "EA 2Y government benchmark bond yield",
            },
            {
                "id": "002",
                "key": "M.U2.EUR.4F.BB.U2_3Y.YLD",
                "desc": "EA 3Y government benchmark bond yield",
            },
            {
                "id": "003",
                "key": "M.U2.EUR.4F.BB.U2_5Y.YLD",
                "desc": "EA 5Y government benchmark bond yield",
            },
            {
                "id": "004",
                "key": "M.U2.EUR.4F.BB.U2_10Y.YLD",
                "desc": "EA 10Y government benchmark bond yield",
            },
            {
                "id": "005",
                "key": "M.U2.EUR.4F.BB.R_U2_10Y.YLDA",
                "desc": "EA real 10Y government benchmark bond yield",
            },
        ],
    },
    # ========================================================================
    # MONEY MARKET RATES – EURIBOR, Real EURIBOR, EONIA (FM) – Euro area level, cat. 8
    # Key: FREQ.REF_AREA.CURRENCY.PROVIDER_FM.INSTRUMENT_FM.PROVIDER_FM_ID.DATA_TYPE_FM
    #   Nominal EURIBOR: PROVIDER_FM=RT (Refinitiv), INSTRUMENT_FM=MM
    #   Real EURIBOR 3M: PROVIDER_FM=4F (ECB), PROVIDER_FM_ID=R_EURIBOR3MD_ (R_ = real)
    #   EONIA:           PROVIDER_FM=4F (ECB), PROVIDER_FM_ID=EONIA (discontinued 2021-12)
    # CONFIRMED: all six keys return 200.
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
                "key": "M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA",
                "desc": "EURIBOR 1-month (hist. close avg.)",
            },
            {
                "id": "002",
                "key": "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
                "desc": "EURIBOR 3-month (hist. close avg.)",
            },
            {
                "id": "003",
                "key": "M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA",
                "desc": "EURIBOR 6-month (hist. close avg.)",
            },
            {
                "id": "004",
                "key": "M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA",
                "desc": "EURIBOR 1-year (hist. close avg.)",
            },
            {
                "id": "005",
                "key": "M.U2.EUR.4F.MM.R_EURIBOR3MD_.HSTA",
                "desc": "Real EURIBOR 3-month (hist. close avg.)",
            },
            {
                "id": "006",
                "key": "M.U2.EUR.4F.MM.EONIA.HSTA",
                "desc": "EONIA (hist. close avg.; discontinued 2021-12)",
            },
        ],
    },
    # ========================================================================
    # DJ EURO STOXX EQUITY INDICES (FM) – Euro area level, category 8
    # Key: FREQ.REF_AREA.CURRENCY.PROVIDER_FM.INSTRUMENT_FM.PROVIDER_FM_ID.DATA_TYPE_FM
    #   PROVIDER_FM=DS (DataStream), INSTRUMENT_FM=EI (equity/index)
    # CONFIRMED: all ten keys return 200.
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
                "key": "M.U2.EUR.DS.EI.DJES50I.HSTA",
                "desc": "DJ Euro Stoxx 50",
            },
            {
                "id": "002",
                "key": "M.U2.EUR.DS.EI.DJEURST.HSTA",
                "desc": "DJ Euro Stoxx Price Index",
            },
            {
                "id": "003",
                "key": "M.U2.EUR.DS.EI.S1ESBME.HSTA",
                "desc": "DJ Euro Stoxx Basic Materials",
            },
            {
                "id": "004",
                "key": "M.U2.EUR.DS.EI.S1ESCSE.HSTA",
                "desc": "DJ Euro Stoxx Consumer Services",
            },
            {
                "id": "005",
                "key": "M.U2.EUR.DS.EI.S1ESFNE.HSTA",
                "desc": "DJ Euro Stoxx Financials",
            },
            {
                "id": "006",
                "key": "M.U2.EUR.DS.EI.S1ESG1E.HSTA",
                "desc": "DJ Euro Stoxx Technology",
            },
            {
                "id": "007",
                "key": "M.U2.EUR.DS.EI.S1ESH1E.HSTA",
                "desc": "DJ Euro Stoxx Healthcare",
            },
            {
                "id": "008",
                "key": "M.U2.EUR.DS.EI.S1ESIDE.HSTA",
                "desc": "DJ Euro Stoxx Industrials",
            },
            {
                "id": "009",
                "key": "M.U2.EUR.DS.EI.S1EST1E.HSTA",
                "desc": "DJ Euro Stoxx Telecommunications",
            },
            {
                "id": "010",
                "key": "M.U2.EUR.DS.EI.S1ESU1E.HSTA",
                "desc": "DJ Euro Stoxx Utilities",
            },
        ],
    },
    # ========================================================================
    # MFI BALANCE SHEET – LOANS (BSI) – Germany, category 7
    # Key: FREQ.REF_AREA.ADJUSTMENT.BS_REP_SECTOR.BS_ITEM.MATURITY_ORIG.DATA_TYPE.
    #      COUNT_AREA.BS_COUNT_SECTOR.CURRENCY_TRANS.BS_SUFFIX
    #   BS_ITEM=A20 (loans), DATA_TYPE=1 (outstanding stocks), COUNT_AREA=U6 (domestic)
    # CONFIRMED: all three keys return 200.
    # Note: COUNT_AREA must be U6 (domestic), not U2 (euro-area cross-border).
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
                "key": "M.DE.N.A.A20.A.1.U6.1000.Z01.E",
                "desc": "MFI Loans – To domestic MFIs (S.12)",
            },
            {
                "id": "002",
                "key": "M.DE.N.A.A20.A.1.U6.2250.Z01.E",
                "desc": "MFI Loans – To households & NPISHs (S.14+S.15)",
            },
            {
                "id": "003",
                "key": "M.DE.N.A.A20.A.1.U6.2240.Z01.E",
                "desc": "MFI Loans – To non-financial corporations (S.11)",
            },
        ],
    },
    # ========================================================================
    # MFI BALANCE SHEET – DEPOSIT LIABILITIES (BSI) – Germany, category 7
    # BS_ITEM=L20 (deposits), DATA_TYPE=1 (stocks), COUNT_AREA=U6 (domestic)
    # CONFIRMED: all three keys return 200.
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
                "key": "M.DE.N.A.L20.A.1.U6.1000.Z01.E",
                "desc": "MFI Deposit liabilities – From domestic MFIs (S.12)",
            },
            {
                "id": "002",
                "key": "M.DE.N.A.L20.A.1.U6.2240.Z01.E",
                "desc": "MFI Deposit liabilities – From NFCs (S.11)",
            },
            {
                "id": "003",
                "key": "M.DE.N.A.L20.A.1.U6.2250.Z01.E",
                "desc": "MFI Deposit liabilities – From households (S.14+S.15)",
            },
        ],
    },
    # ========================================================================
    # MFI BALANCE SHEET – DEBT SECURITIES HELD (BSI) – Germany, category 7
    # BS_ITEM=A30 (debt securities), DATA_TYPE=1 (stocks), COUNT_AREA=U6 (domestic)
    # CONFIRMED: all three keys return 200.
    # Note: Item code is A30, not A40.
    # Sector 2200 = non-MFIs excl. general government (S.11+S.14+S.15+others).
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
                "key": "M.DE.N.A.A30.A.1.U6.2200.Z01.E",
                "desc": "MFI Debt securities held – Non-MFIs excl. general government",
            },
            {
                "id": "002",
                "key": "M.DE.N.A.A30.A.1.U6.1000.Z01.E",
                "desc": "MFI Debt securities held – MFIs (S.12)",
            },
            {
                "id": "003",
                "key": "M.DE.N.A.A30.A.1.U6.2100.Z01.E",
                "desc": "MFI Debt securities held – General government (S.13)",
            },
        ],
    },
    # ========================================================================
    # MONETARY AGGREGATES M1, M3 and Currency in Circulation (BSI) – Euro area, cat. 8
    # CONFIRMED: all three keys return 200.
    # MATURITY_ORIG=X (not M) for working-day and seasonally adjusted notional stocks.
    # L10 = currency in circulation (BS_ITEM).
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
                "key": "M.U2.Y.V.M10.X.I.U2.2300.Z01.E",
                "desc": "M1 – Index of Notional Stocks (SA+WDA)",
            },
            {
                "id": "002",
                "key": "M.U2.Y.V.M30.X.I.U2.2300.Z01.E",
                "desc": "M3 – Index of Notional Stocks (SA+WDA)",
            },
            {
                "id": "003",
                "key": "M.U2.Y.V.L10.X.I.U2.2300.Z01.E",
                "desc": "Currency in circulation – Index of Notional Stocks (SA+WDA)",
            },
        ],
    },
    # ========================================================================
    # MFI INTEREST RATES ON NEW BUSINESS (MIR) – Germany, category 7
    # Key (10 dims): FREQ.REF_AREA.BS_REP_SECTOR.BS_ITEM.MATURITY_NOT_IRATE.
    #                DATA_TYPE_MIR.AMOUNT_CAT.BS_COUNT_SECTOR.CURRENCY_TRANS.IR_BUS_COV
    # CONFIRMED: all keys return 200.
    #
    # Lending – individual instrument types:
    #   A2C = house purchase, A2D = other household lending excl. revolving
    #   Sector 2250 = households (S.14+S.15)
    #
    # Lending – cost-of-borrowing composites (A2J = A2C + A2A + A2Z):
    #   FM = up to 1 year, KM = over 1 year
    #   Sector 2230 = NFCs + households combined (S.11+S.14+S.15)
    #
    # Deposits with agreed maturity (L22), maturity buckets:
    #   A = total new business, F = up to 1Y, G = over 1Y up to 2Y,
    #   H = over 2Y, K = over 1Y (cost-of-borrowing), L = up to 2Y (aggregate)
    #   Sector 2230 = NFCs + households combined (S.11+S.14+S.15) — exact paper match.
    #
    # Not found: lending spreads vs. swap rate (no API key available).
    # ========================================================================
    "MIR_DE": {
        "flow": "MIR",
        "category": 7,
        "category_name": "Financial",
        "country_iso2": "DE",
        "series_prefix": "DE_FIN_MIR",
        "variables": [
            # Lending rates – individual instrument types
            {
                "id": "LEN_001",
                "key": "M.DE.B.A2C.A.R.A.2250.EUR.N",
                "desc": "MIR – Lending for house purchase, total maturity (new business)",
            },
            {
                "id": "LEN_002",
                "key": "M.DE.B.A2D.A.R.A.2250.EUR.N",
                "desc": "MIR – Other lending, households excl. revolving (new business)",
            },
            # Lending rates – cost-of-borrowing composites (A2J), NFC+HH (2230)
            {
                "id": "LEN_003",
                "key": "M.DE.B.A2J.FM.R.A.2230.EUR.N",
                "desc": "MIR – Cost-of-borrowing loans, up to 1 year, NFC+HH (new business)",
            },
            {
                "id": "LEN_004",
                "key": "M.DE.B.A2J.KM.R.A.2230.EUR.N",
                "desc": "MIR – Cost-of-borrowing loans, over 1 year, NFC+HH (new business)",
            },
            # Deposit rates – NFC+HH combined (S.11+S.14+S.15, sector 2230) by maturity
            {
                "id": "DEP_001",
                "key": "M.DE.B.L22.A.R.A.2230.EUR.N",
                "desc": "MIR – Deposit rate, total new business (S.11+S.14+S.15)",
            },
            {
                "id": "DEP_002",
                "key": "M.DE.B.L22.F.R.A.2230.EUR.N",
                "desc": "MIR – Deposit rate, up to 1 year (new business)",
            },
            {
                "id": "DEP_003",
                "key": "M.DE.B.L22.G.R.A.2230.EUR.N",
                "desc": "MIR – Deposit rate, over 1Y up to 2Y (new business)",
            },
            {
                "id": "DEP_004",
                "key": "M.DE.B.L22.H.R.A.2230.EUR.N",
                "desc": "MIR – Deposit rate, over 2 years (new business)",
            },
            {
                "id": "DEP_005",
                "key": "M.DE.B.L22.K.R.A.2230.EUR.N",
                "desc": "MIR – Deposit rate, over 1 year (cost-of-borrowing, new business)",
            },
            {
                "id": "DEP_006",
                "key": "M.DE.B.L22.L.R.A.2230.EUR.N",
                "desc": "MIR – Deposit rate, up to 2 years aggregate (new business)",
            },
        ],
    },
    # ========================================================================
    # CAR REGISTRATIONS (STS) – Germany, category 4
    # Key (7 dims): FREQ.REF_AREA.ADJUSTMENT.STS_CONCEPT.STS_CLASS.STS_INSTITUTION.STS_SUFFIX
    #   ADJUSTMENT=Y (SA+WDA), STS_INSTITUTION=3 (ECB), STS_SUFFIX=ABS
    # CONFIRMED: all four keys return 200.
    # Note: ECB discontinued this STS series as of 2022-12.
    #       Data is available from 2003-01 through 2022-12.
    # ========================================================================
    "STS_CARS": {
        "flow": "STS",
        "category": 4,
        "category_name": "Activity_indicators",
        "country_iso2": "DE",
        "series_prefix": "DE_ACT_CARS",
        "variables": [
            {
                "id": "001",
                "key": "M.DE.Y.CREG.PC0000.3.ABS",
                "desc": "Car registrations – New passenger cars (SA+WDA, absolute; 2003–2022)",
            },
            {
                "id": "002",
                "key": "M.DE.Y.CREG.CV0000.3.ABS",
                "desc": "Car registrations – New commercial vehicles (SA+WDA, absolute; 2003–2022)",
            },
            {
                "id": "003",
                "key": "M.DE.Y.CREG.CVH000.3.ABS",
                "desc": "Car registrations – New heavy commercial vehicles (SA+WDA, absolute; 2003–2022)",
            },
            {
                "id": "004",
                "key": "M.DE.Y.CREG.CVL000.3.ABS",
                "desc": "Car registrations – New light commercial vehicles (SA+WDA, absolute; 2003–2022)",
            },
        ],
    },
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
    """Transform ECB REST API response (long format) to standardised long format.

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

    result = (
        result[result["date"] >= "2003"]
        .dropna(subset=["value"])
        .sort_values("date")
        .reset_index(drop=True)
    )

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


def fetch_one(spec: dict[str, Any]) -> pd.DataFrame:
    """Fetch single ECB series from spec dict.

    Args:
        spec: Must contain 'flow', 'key', 'series_id',
              'description', 'category', 'category_name', 'country_iso2'

    Returns:
        Standardized long DataFrame with columns:
        date | value | series_id | country_iso2 | variable_name |
        category | category_name

    Raises:
        ValueError: If spec is missing required keys or data validation fails.
        RuntimeError: If API request fails.

    """
    raw = fetch_ecb_series(spec["flow"], spec["key"])
    metadata = {
        "variable_name": spec["description"],
        "country_iso2": spec.get("country_iso2", "U2"),
        "category": spec["category"],
        "category_name": spec["category_name"],
    }
    transformed = transform_ecb_data(raw, spec["series_id"], metadata)
    validate_series_data(transformed, spec["series_id"])
    return transformed


