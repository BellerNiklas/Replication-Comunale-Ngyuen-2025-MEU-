# Replication of Comunale & Nguyen (2025): MacroEconomic Uncertainty for the Euro Area

## Project Overview

This project replicates the MacroEconomic Uncertainty (MEU) measure for the euro area developed by Comunale and Nguyen (2025, Journal of International Money and Finance). The MEU follows the methodology of Jurado, Ludvigson, and Ng (AER 2015), measuring uncertainty as the conditional volatility of unforecastable components of macroeconomic time series.

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh/) for environment and dependency management
- Network access for API calls to Eurostat, ECB, OECD, and BIS

### Installation and Execution

```bash
pixi install              # Install all dependencies (pinned via pixi.lock)
pixi run pytask           # Run the full computational pipeline
pixi run pytest           # Run the test suite
pixi run prek             # Run pre-commit checks
```

### Expected Runtime

The full data fetch (`pixi run pytask`) takes approximately **15-30 minutes** depending on network speed and API responsiveness. The OECD bulk fetch uses batched API calls with rate-limiting delays.

## Directory Layout

```
src/meu_replication/      # Source code (hand-written, version controlled)
  config.py               # Central path definitions (SRC, BLD, ROOT)
  data_fetch/             # API adapters for Eurostat, ECB, OECD, BIS
  data_management/        # pytask tasks: fetch, probe, clean, coverage
    registry/             # Series registry and country definitions (CSV)
  analysis/               # (Placeholder for MEU estimation tasks)
  final/                  # (Placeholder for figure/table generation)
bld/                      # Generated outputs (NOT committed, safe to delete)
_build/                   # Document build outputs (NOT committed)
documents/                # Paper and presentation sources (MyST Markdown)
tests/                    # Unit and integration tests
```

## Data

The replication requires a large monthly dataset of approximately **1,330 variables** (after cleaning) covering **19 euro area countries** from **January 2003** onward.

### Data Sources and Availability

All data is fetched programmatically from public APIs. No manual downloads are required.

| Source | Access | Rate Limits | Fetch Strategy |
|--------|--------|-------------|----------------|
| Eurostat | Public REST API | Moderate | Per-country direct fetch |
| ECB SDW | Public SDMX API | Moderate (1s delay) | Per-country direct fetch |
| OECD | Public SDMX API | Strict (429 errors) | Bulk fetch via SDMX `+` syntax (4 calls) |
| BIS | Public CSV API | Moderate (1s delay) | Per-country direct fetch |

### Country-Specific Variables (up to 122 per country)

| Category | Examples |
|----------|----------|
| Industrial Production | Total industry, manufacturing, capital goods, consumer goods, energy |
| Labor Market | Employment indices, unemployment rates, hours worked, wages |
| Prices | PPI, HICP (overall, energy, food, services), import price indices |
| Activity Indicators | Car registrations, turnover indices, building permits |
| Trade | Imports and exports with world |
| Sentiment & Surveys | Economic sentiment, consumer/industrial/services confidence |
| Financial | Loans, deposits, debt securities, share prices, spreads |

### Euro Area-Level Variables (30 series)
- Government bond yields (2y, 3y, 5y, 7y, 10y)
- Money market rates (Euribor 1m, 3m, 6m, 1y; Eonia)
- Dow Jones Euro Stoxx indices (broad and sector-specific)
- Monetary aggregates (M1, M3, currency in circulation)
- Bilateral exchange rates (USD, GBP, JPY, CHF, CNY)

## References

- Jurado, K., Ludvigson, S.C., & Ng, S. (2015). Measuring Uncertainty. *American Economic Review*, 105(3), 1177-1216.
- Comunale, M., & Nguyen, A.D.M. (2025). A comprehensive MacroEconomic uncertainty measure for the euro area. *Journal of International Money and Finance*, 157, 103370. [DOI: 10.1016/j.jimonfin.2025.103370](https://doi.org/10.1016/j.jimonfin.2025.103370)
