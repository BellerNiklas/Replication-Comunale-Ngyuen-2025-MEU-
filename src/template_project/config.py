"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()


DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

# Template groups removed - was used for smoking example
# TEMPLATE_GROUPS: tuple[str, ...] = ("marital_status", "highest_qualification")

# MEU-specific configuration can be added here:
# MEU_COUNTRIES: tuple[str, ...] = ("DE",)  # Start with Germany
# MEU_CATEGORIES: tuple[str, ...] = (
#     "industrial_production",
#     "labor_market",
#     "prices",
#     "activity",
#     "trade",
#     "sentiment",
#     "financial",
# )
