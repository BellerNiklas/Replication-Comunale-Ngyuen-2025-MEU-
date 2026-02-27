"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()


DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

# Template groups removed - was used for smoking example
# TEMPLATE_GROUPS: tuple[str, ...] = ("marital_status", "highest_qualification")

# EA19 member countries (ISO2 codes only - U2 NOT included)
# U2 (Euro Area) is in countries.csv for code mapping, but NOT here.
# EA-aggregate series are controlled by scope=ea in templates.
MEU_COUNTRIES: tuple[str, ...] = (
    "AT",
    "BE",
    "CY",
    "DE",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "PT",
    "SI",
    "SK",
)

# Replication sample period (Comunale & Nguyen 2025: 2003-01 to 2022-12)
SAMPLE_START: str = "2003-01"
SAMPLE_END: str = "2022-12"


def load_countries():
    """Load canonical country table.

    Returns:
        DataFrame with columns: country_iso2, country_iso3, country_name,
        source_eurostat, source_ecb, source_oecd, source_bis, ea_member
    """
    import pandas as pd

    countries_path = SRC / "data_management" / "registry" / "countries.csv"
    return pd.read_csv(countries_path)
