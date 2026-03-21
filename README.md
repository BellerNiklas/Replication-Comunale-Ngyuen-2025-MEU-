# Replication of Comunale & Nguyen (2025): MacroEconomic Uncertainty for the Euro Area

## Project Overview

This repository replicates the MacroEconomic Uncertainty (MEU) measure for the euro area developed by Comunale and Nguyen (2025, *Journal of International Money and Finance*). The measure follows the Jurado, Ludvigson, and Ng (2015) framework: forecast a large panel of macroeconomic series, isolate the unforecastable component of each series, estimate the stochastic volatility of those forecast errors, and then aggregate horizon-specific uncertainty into a common euro-area measure.

The codebase currently implements the full baseline pipeline through **Milestone 5**: data preparation, factor extraction, forecast-error estimation, stochastic-volatility estimation, stochastic-volatility validation, horizon-specific uncertainty calculation, and euro-area MEU aggregation.

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh/) for environment and dependency management
- Network access for Eurostat, ECB, OECD, and BIS API calls

### Installation and Execution

```bash
pixi install              # Install the pinned environment
pixi run pytask           # Run the full pipeline
pixi run pytask-parallel  # Run all panel branches with a safer 2-worker default
pixi run pytask -k analysis_2025  # Re-run only the 2025 analysis branch
pixi run pytest           # Run the test suite
pixi run prek             # Run pre-commit checks
```

### Expected Runtime

Data fetching and cleaning typically finish much faster than the analysis stage and depend on API responsiveness. The factor and forecast-error stages are moderate local computations, while the current full-mode stochastic-volatility stage is the main runtime bottleneck because it is run in a **reference-aligned serial panel loop** within each panel branch. In practice, a full end-to-end run can therefore take **hours**, not just a short fetch/clean cycle, although `pytask-parallel` can now schedule the 2021, 2022, and 2025 analysis branches concurrently. The default Pixi task uses `2` workers because three simultaneous full `sv_y` runs can overrun Windows/R memory on this machine; if a stronger workstation handles it comfortably, you can still raise `-n` manually. The task-selection keys use short ids such as `analysis_2025` so they remain Windows-safe for the R-backed `pytask-r` tasks.

If outputs already exist, `pytask` will skip unchanged tasks and reruns will be much faster.

## Directory Layout

```text
src/meu_replication/      # Source code
  config.py               # Central path definitions
  cleaning/               # Pure cleaning helpers
  data_fetch/             # API adapters for Eurostat, ECB, OECD, BIS
  data_management/        # pytask data ingestion and cleaning tasks
  registry/               # Series registry, templates, country definitions
  analysis/               # Implemented MEU stages: factors, forecast errors, SV, validation
  final/                  # Final figures/tables stage (still a template)
bld/                      # Generated outputs (not committed)
tests/                    # Unit and integration tests
```

## Data and Preprocessing

The replication builds a large monthly macro-financial panel for the 19 euro-area countries, plus euro-area aggregate financial series. Data are fetched programmatically from **Eurostat**, **ECB SDW**, **OECD**, and **BIS**. No manual downloads are required.

The analysis input is produced by a short three-stage preprocessing pipeline:

1. **Stationarity transformations**  
   Each registry series is transformed according to the paper's transformation codes: no transform, first difference, or log first difference.
2. **Temporal coverage filtering**  
   The transformed panel is filtered into strict endpoint windows for 2021, 2022, and 2025.
3. **High-correlation filtering**  
   Within each country, highly correlated series are filtered using a deterministic alphabetical tie-break rule and a `|corr| > 0.95` threshold.

The supported analysis inputs are the correlation-filtered strict panels for:

- `bld/data/clean/panel_2003_2021_strict_corr.parquet`
- `bld/data/clean/panel_2003_2022_strict_corr.parquet`
- `bld/data/clean/panel_2003_2025_strict_corr.parquet`

A reproducible correlation-audit stage now writes pair-level and decision-level
review outputs under `bld/analysis/audits/correlation/`, plus a ranked review
report at `bld/documents/correlation_cleaning_review.md`.

Because differencing removes the first observation for transformed series, the cleaned transformed sample starts in **2003-02**.

## MEU Estimation Pipeline

Plain-language summary: MEU is built by first forecasting each macro series as well as the common factors, then treating the remaining forecast error as the unpredictable component of that series. The volatility of those unpredictable components is modeled over time, and later stages convert those one-step uncertainty objects into horizon-specific expected variances and finally into an aggregate euro-area uncertainty index.

### 1. Factor Extraction - Implemented

The cleaned long panel is pivoted once into a deterministic wide matrix, standardized, and used to estimate static factors using the Bai-Ng IC2 criterion on both `X` and `X^2`. The resulting predictor set follows the JLN-style structure used downstream.

Each full analysis branch now writes into a results-first panel directory:

- `bld/analysis/panels/panel_2003_2021_strict_corr/`
- `bld/analysis/panels/panel_2003_2022_strict_corr/`
- `bld/analysis/panels/panel_2003_2025_strict_corr/`

Within each panel, outputs are grouped as:

- `results/` for public MEU deliverables
- `diagnostics/` for validation summaries
- `artifacts/` for stage outputs consumed downstream
- `internal/` for implementation-specific caches such as raw SV intermediates

Main outputs include:

- `panel_wide.parquet`
- `series_order.parquet`
- `fhat.parquet`
- `ghat.parquet`
- `predictor_set.parquet`
- `factor_metadata.parquet`

The filenames below refer to files within one panel-specific analysis directory,
under the appropriate `results/`, `diagnostics/`, `artifacts/`, or `internal/`
subfolder.

### 2. Forecast-Error Estimation - Implemented

Each macro series is regressed on its own lags and the factor-based predictor set using MATLAB/JLN-style conventions: lag construction, Newey-West HAC standard errors, and coefficient-level hard thresholding. Separate AR forecasts are also estimated for the predictor block.

Main outputs include:

- `forecast_errors_y.parquet`
- `forecast_errors_f.parquet`
- `regression_coefs_y.parquet`
- `regression_coefs_f.parquet`
- `predictor_selection_masks.parquet`
- `forecast_metadata.parquet`

The effective downstream forecast-error sample begins in **2003-06**.

### 3. Stochastic Volatility - Implemented

The forecast-error panels are passed to **R `stochvol::svsample()`**, which estimates an AR(1) stochastic-volatility model for each target series and each predictor residual series. The current production path is intentionally aligned with the reference JLN-style R workflow: one seed per panel, deterministic series order, and full-mode posterior means plus Geweke diagnostics. A validation stage then checks split-Rhat on sentinel series, full-panel diagnostics, and fast-vs-doubled-fast stability.

Main outputs include:

- `sv_params_y.parquet`
- `sv_latent_y.parquet`
- `sv_params_f.parquet`
- `sv_latent_f.parquet`
- `sv_diagnostics.parquet`
- `sv_validation_summary.parquet`
- `sv_validation_subset_metrics.parquet`

### 4. Horizon-Specific Uncertainty Calculation - Implemented

The uncertainty stage combines the forecast-system coefficients with the stochastic-volatility outputs to compute expected variance for each series and horizon `h = 1, ..., 12`, following the JLN/Comunale-Nguyen uncertainty construction. The production implementation uses a Python IRF-based scalar variance engine aligned to the MATLAB recursion in the JLN reference files and writes the result in deterministic long format.

Main output:

- `artifacts/uncertainty/uncertainty_variance.parquet`

### 5. Euro-Area Aggregation - Implemented

The final baseline stage aggregates the series-level uncertainty objects into the euro-area MEU measure using the replication-package cross-sectional mean path, `mean(sqrt(variance))`, across all series for each date and horizon.

Main output:

- `results/euro_area/meu_ea.parquet`
- final figures and tables under `src/meu_replication/final/`

### 6. Country-Level MEU Aggregation - Implemented

Country MEUs are derived from the same common euro-area model outputs, not estimated in isolated country reruns. For each of the 19 EA member countries, the aggregation restricts to that country's retained series plus the shared `U2_FX_*` bilateral exchange-rate series and computes `mean(sqrt(variance))` per date-horizon pair.

The choice to include only the FX block as the shared common component is a project assumption. The paper text is ambiguous between "euro area-wide common variables" and "bilateral exchange rates"; the current implementation uses the narrower FX-only rule.

Main outputs (per panel, under `results/countries/`):

- `all_countries_meu.parquet` — consolidated country MEUs (`country_iso2`, `date`, `horizon`, `meu`)
- `basket_membership.parquet` — audit table showing which series enter each country's basket

## Current Status and Limitations

- The repository currently implements the baseline **Milestones 1-5** pipeline, including euro-area aggregation, on the strict 2021, 2022, and 2025 cleaned panels.
- Country-level MEUs for all 19 EA members are now implemented as a lightweight aggregation stage on top of the common model outputs. PCA aggregation and paper-style final figures/tables are still follow-on work.
- Full pipeline runtime is dominated by the reference-aligned full-mode stochastic-volatility estimation, which is still slow within each panel even though panel branches can now be scheduled in parallel.
- Public data providers can revise historical observations, so exact row counts and values may drift over time.
- The replication uses the current public-data panel available through the programmed fetch pipeline, which may differ slightly from the paper authors' original source snapshot.
- The relaxed `98%` coverage path and the generated coverage audit report have been removed; preprocessing is strict-only.

## References

- Jurado, K., Ludvigson, S. C., & Ng, S. (2015). Measuring Uncertainty. *American Economic Review*, 105(3), 1177-1216.
- Comunale, M., & Nguyen, A. D. M. (2025). A comprehensive MacroEconomic uncertainty measure for the euro area. *Journal of International Money and Finance*, 157, 103370. [DOI: 10.1016/j.jimonfin.2025.103370](https://doi.org/10.1016/j.jimonfin.2025.103370)
