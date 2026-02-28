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
SAMPLE_END_ALT: str = "2021-12"  # Alternative window (includes CARS_002-004)

# After differencing (codes 2/5), the first observation (2003-01) is lost.
SAMPLE_START_TRANSFORMED: str = "2003-02"

# High-correlation filter threshold (Comunale & Nguyen 2025)
HIGH_CORR_THRESHOLD: float = 0.95

# Panel variant keys (used by temporal coverage + correlation tasks)
PANEL_VARIANTS: tuple[str, ...] = (
    "panel_2003_2022_strict",
    "panel_2003_2022_cov98",
    "panel_2003_2021_strict",
    "panel_2003_2021_cov98",
)
PRIMARY_PANEL: str = "panel_2003_2022_strict_corr"


def load_countries():
    """Load canonical country table.

    Returns:
        DataFrame with columns: country_iso2, country_iso3, country_name,
        source_eurostat, source_ecb, source_oecd, source_bis, ea_member
    """
    import pandas as pd

    countries_path = SRC / "registry" / "countries.csv"
    return pd.read_csv(countries_path)
