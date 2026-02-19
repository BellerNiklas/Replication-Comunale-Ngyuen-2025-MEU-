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
    which is a stable, always-available ECB SDW series.

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

    category_nums = sorted({c["category"] for c in all_configs})
    category_names = {c["category"]: c["category_name"] for c in all_configs}

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
