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
pixi run pytask -n 8      # Same, but with 8 parallel workers (recommended)
pixi run pytest           # Run the test suite
pixi run prek             # Run pre-commit checks
```

### Expected Runtime

The full data fetch (`pixi run pytask`) takes approximately **15-30 minutes** depending on network speed and API responsiveness. The OECD bulk fetch uses batched API calls with rate-limiting delays.

## Directory Layout

```
src/meu_replication/      # Source code (hand-written, version controlled)
  config.py               # Central path definitions (SRC, BLD, ROOT)
  cleaning/               # Pure functions: stationarity, temporal coverage, correlation
  data_fetch/             # API adapters for Eurostat, ECB, OECD, BIS
  data_management/        # pytask tasks: fetch, probe, clean, transform, filter
  registry/               # Series registry, templates, country definitions (CSV)
  analysis/               # (Placeholder for MEU estimation tasks)
  final/                  # (Placeholder for figure/table generation)
bld/                      # Generated outputs (NOT committed, safe to delete)
_build/                   # Document build outputs (NOT committed)
documents/                # Paper and presentation sources (MyST Markdown)
tests/                    # Unit and integration tests
```

## Data

The replication requires a large monthly dataset of approximately **1,470 variables** (before cleaning) covering **19 euro area countries** from **January 2003** to **December 2022** (240 months). Each country has up to 119 country-specific series, plus 29 euro area-level (U2) financial series shared across all countries, totalling up to 148 variables per country.

### Data Sources and Availability

All data is fetched programmatically from public APIs. No manual downloads are required.

| Source | Access | Rate Limits | Fetch Strategy |
|--------|--------|-------------|----------------|
| Eurostat | Public REST API | Moderate | Per-country direct fetch |
| ECB SDW | Public SDMX API | Moderate (1s delay) | Per-country direct fetch |
| OECD | Public SDMX API | Strict (429 errors) | Bulk fetch via SDMX `+` syntax (4 calls) |
| BIS | Public CSV API | Moderate (1s delay) | Per-country direct fetch |

### Country-Specific Variables (up to 119 per country)

| Cat. | Category | # Templates | Examples |
|------|----------|-------------|----------|
| 1 | Industrial Production | 12 | Total industry, manufacturing, capital goods, consumer goods, energy |
| 2 | Labor Market | 21 | Employment indices, unemployment rate, hours worked, wages |
| 3 | Prices | 22 | PPI, HICP (overall, energy, food, services), import price indices |
| 4 | Activity Indicators | 17 | Car registrations, turnover indices, building permits, retail |
| 5 | Trade | 2 | Total imports and exports (world, million EUR) |
| 6 | Sentiment & Surveys | 12 | Economic sentiment, consumer/industrial/services/construction confidence |
| 7 | Financial | 19 | MFI loans/deposits/debt securities, interest rates, NEER, share prices |

### Euro Area-Level Variables (29 series, category 8)

| Group | # Series | Examples |
|-------|----------|----------|
| Government bond yields | 5 | 2Y, 3Y, 5Y, 10Y nominal; 10Y real |
| Money market rates | 6 | EURIBOR 1m, 3m, 6m, 1Y; real EURIBOR 3m; EONIA |
| Equity indices | 10 | DJ Euro Stoxx 50, Price Index, sector indices |
| Exchange rates | 5 | USD, GBP, JPY, CNY, CHF vs EUR |
| Monetary aggregates | 3 | M1, M3, currency in circulation |

## Data Cleaning Pipeline

The raw data undergoes three sequential cleaning stages before it is used for MEU estimation. Each stage is implemented as a separate pytask task with pure functions (no mutations), following the project's EPP functional rules.

### Stage 1: Stationarity Transformations

Following Comunale & Nguyen (2025), all series are transformed to achieve stationarity before further processing. Each variable in the series registry (`series_registry.csv`) has a `transformationcode` column specifying its transformation:

| Code | Transformation | Formula | # Templates | Applied to |
|------|---------------|---------|-------------|------------|
| 1 | No transformation | x_t | 12 | Sentiment and confidence indicators (already stationary, bounded survey data) |
| 2 | First difference | x_t - x_{t-1} | 23 | Interest rates, bond yields, unemployment rate (can take negative values) |
| 5 | Log first difference | ln(x_t) - ln(x_{t-1}) | 113 | Indices, quantities, price levels, nominal stocks, exchange rates (strictly positive) |

**Rationale for each code:**

- **Code 1** (12 templates): Survey-based diffusion indices (e.g., Economic Sentiment Indicator, Consumer Confidence) are inherently stationary and mean-reverting. They are bounded by construction and do not exhibit unit roots. Additionally, several of these indicators take negative values, which rules out log transformation.

- **Code 2** (23 templates): Interest rates, bond yields (e.g., EURIBOR, government bond yields), and the unemployment rate are persistent but already expressed in percentage points. They can take negative values (e.g., EURIBOR was -0.58% during the negative rate period), which rules out log transformation. First differencing renders them stationary.

- **Code 5** (113 templates): The majority of variables are indices (e.g., industrial production, HICP, PPI with base year = 100), nominal aggregates (e.g., MFI deposits in millions EUR), or absolute counts (e.g., car registrations). All are strictly positive. Log first differencing converts them to approximate month-on-month growth rates, removing both level trends and multiplicative scaling.

### Stage 2: Temporal Coverage Filtering

After transformation, series are filtered based on their temporal coverage within the sample period. This produces **four panel variants** from the cross-product of two sample windows and two missing-data thresholds:

| Panel | Sample period | Missing rule | Allowed missing months | Output file |
|-------|--------------|-------------|----------------------|-------------|
| 2022 strict | 2003-01 to 2022-12 (240 months) | 100% coverage | 0 | `panel_2003_2022_strict.parquet` |
| 2022 cov98 | 2003-01 to 2022-12 (240 months) | 98% coverage | 4 | `panel_2003_2022_cov98.parquet` |
| 2021 strict | 2003-01 to 2021-12 (228 months) | 100% coverage | 0 | `panel_2003_2021_strict.parquet` |
| 2021 cov98 | 2003-01 to 2021-12 (228 months) | 98% coverage | 4 | `panel_2003_2021_cov98.parquet` |

The 98% threshold allows up to `floor(0.02 * n_months)` missing observations. The 2021 alternative window includes additional car registration series (CARS_002-004) that are not available through 2022.

**Implementation:** `temporal_coverage.py` contains the pure filtering functions. For each series, the number of distinct months present in the sample period is counted. Series falling below the threshold are dropped and documented in a drop-info DataFrame.

### Stage 3: High-Correlation Filtering

Following Comunale & Nguyen (2025): "for each country, if two variables are highly correlated variables (with a correlation larger than 0.95 in absolute term), only one of them will be kept."

This step operates on the **temporal-coverage filtered** panels (after Stage 2), ensuring correlations are computed on stationary, coverage-complete data rather than raw levels with common trends. For each country independently:

1. The panel is pivoted to wide format (rows = dates, columns = series)
2. The pairwise Pearson correlation matrix is computed
3. All pairs with |correlation| > 0.95 are identified
4. A greedy maximum-independent-set algorithm selects which series to keep:
   - Series are considered in alphabetical order by `series_id`
   - A series is kept only if none of its already-kept neighbours exceed the correlation threshold
   - This maximises the number of retained variables while guaranteeing no two kept series are correlated above 0.95

The alphabetical tie-breaking rule ensures full determinism and reproducibility across runs and platforms.

**Implementation:** `high_correlation.py` contains the pure functions. The pytask task reads the filtered panel, applies the correlation filter, and writes both the filtered panel (`*_corr.parquet`) and a metadata CSV (`high_corr_drop_info.csv`) documenting every dropped series, its highest-correlation kept neighbour, and the correlation value.

**Output files** (in `bld/data/clean/`):

| File | Description |
|------|-------------|
| `macro_panel.parquet` | Combined raw panel (all sources, all countries, no filtering) |
| `panel_2003_2022_strict.parquet` | After Stage 2: strict coverage, 2022 horizon |
| `panel_2003_2022_cov98.parquet` | After Stage 2: 98% coverage, 2022 horizon |
| `panel_2003_2021_strict.parquet` | After Stage 2: strict coverage, 2021 horizon |
| `panel_2003_2021_cov98.parquet` | After Stage 2: 98% coverage, 2021 horizon |
| `panel_2003_2022_strict_corr.parquet` | After Stage 3: correlation-filtered (strict, 2022) |
| `high_corr_drop_info.csv` | Metadata: which series were dropped and why |

## References

- Jurado, K., Ludvigson, S.C., & Ng, S. (2015). Measuring Uncertainty. *American Economic Review*, 105(3), 1177-1216.
- Comunale, M., & Nguyen, A.D.M. (2025). A comprehensive MacroEconomic uncertainty measure for the euro area. *Journal of International Money and Finance*, 157, 103370. [DOI: 10.1016/j.jimonfin.2025.103370](https://doi.org/10.1016/j.jimonfin.2025.103370)
