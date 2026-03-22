# Replication of Comunale & Nguyen (2025): MacroEconomic Uncertainty for the Euro Area

This repository rebuilds the baseline MacroEconomic Uncertainty (MEU) pipeline
from Comunale and Nguyen (2025) for the euro area. It fetches public monthly
data, constructs strict cleaned panels, estimates factor-based forecast errors,
fits stochastic-volatility models in R, converts them into horizon-specific
uncertainty, and aggregates both euro-area and country-level MEU series.

The project is set up as a reproducible `pytask` DAG. A fresh clone starts
without generated outputs because `bld/` is build-only and ignored by Git.

## What Is Implemented

The current repository already covers the baseline MEU construction end to end:

- public-data fetching from Eurostat, ECB, OECD, and BIS
- stationarity transformations and strict endpoint-specific cleaned panels
- correlation filtering and correlation-audit outputs
- factor estimation and forecast-error estimation
- stochastic-volatility estimation and validation
- horizon-specific uncertainty for `h = 1, ..., 12`
- euro-area MEU aggregation
- country-level MEU aggregation for all 19 euro-area member countries
- final Plotly figures for availability and MEU comparisons

Supported analysis panels:

- `panel_2003_2021_strict_corr`
- `panel_2003_2022_strict_corr`
- `panel_2003_2025_strict_corr`

## Prerequisites

- [Pixi](https://pixi.sh/) for dependency and environment management
- network access for Eurostat, ECB, OECD, and BIS API calls
- enough patience for a long run: the stochastic-volatility stage is the main
  bottleneck

The pinned environment includes both Python and R dependencies, including
`stochvol`.

## Run From a Fresh Clone

Use these commands from the repository root:

```bash
pixi install
pixi run pytask
pixi run pytask-parallel
pixi run pytest
```

What each command does:

- `pixi install`: installs the default pinned environment
- `pixi run pytask`: runs the full pipeline from fetch through final figures
- `pixi run pytask-parallel`: runs the full DAG with a safer 2-worker default
- `pixi run pytest`: runs the test suite

Notes:

- `pixi run pytask` is the clearest default command for a full rebuild
- `pixi run pytask-parallel` maps to `pytask build -n 2`
- if outputs already exist, `pytask` skips unchanged tasks automatically

## Runtime Expectations

Data fetching and cleaning are usually much faster than the analysis stage. The
main bottleneck is the full-mode stochastic-volatility estimation for the
target-series residuals (`sv_y`), which runs through R and can take hours for a
fresh end-to-end rebuild.

Practical expectations on this project:

- a full rebuild can take multiple hours
- `pytask-parallel` helps by scheduling panel branches concurrently
- the 2-worker default is intentional because 3 simultaneous full SV branches
  can be too heavy on Windows/R memory
- reruns after a successful build are much faster because unchanged tasks are
  skipped

## After the Run, Look Here

The most useful outputs are:

- `bld/final/plots/`
  final Plotly figures in both HTML and PNG
- `bld/analysis/panels/<panel>/results/`
  panel-specific MEU deliverables
- `bld/analysis/audits/correlation/`
  correlation-cleaning review outputs

In particular:

- `bld/final/plots/availability/availability_overview.html`
- `bld/final/plots/meu/ea_meu_h3_by_panel.html`
- `bld/analysis/panels/panel_2003_2025_strict_corr/results/euro_area/meu_ea.parquet`
- `bld/analysis/panels/panel_2003_2025_strict_corr/results/countries/all_countries_meu.parquet`

## Selective Reruns

If you do not want to rebuild everything, target specific tasks or panels:

```bash
pixi run pytask build -k analysis_2025
pixi run pytask build -k country_analysis_2025
pixi run pytask build -k "ea_meu_h3 or country_vs_ea_2025"
```

Useful task ids include:

- `analysis_2021`, `analysis_2022`, `analysis_2025`
- `country_analysis_2021`, `country_analysis_2022`, `country_analysis_2025`
- `availability_overview`
- `ea_meu_h3`
- `country_vs_ea_2021`, `country_vs_ea_2022`, `country_vs_ea_2025`

## Project Layout

```text
src/meu_replication/
  cleaning/         cleaning helpers
  data_fetch/       source adapters for Eurostat, ECB, OECD, BIS
  data_management/  pytask fetch and cleaning stages
  registry/         country table, templates, expanded registry
  analysis/         factors, forecasts, stochastic volatility, uncertainty, MEU
  final/            final Plotly figure stage
documents/          supporting notes and project-status docs
tests/              unit and integration tests
bld/                generated outputs only; not committed
```

The generated analysis tree uses a results-first layout:

- `bld/analysis/panels/<panel>/results/`
- `bld/analysis/panels/<panel>/diagnostics/`
- `bld/analysis/panels/<panel>/artifacts/`
- `bld/analysis/panels/<panel>/internal/`

## Pipeline Summary

The project follows the JLN-style logic used in Comunale and Nguyen:

1. fetch and standardize a large monthly macro-financial panel
2. transform series to stationarity
3. filter to strict balanced endpoint windows for 2021, 2022, and 2025
4. remove highly correlated within-country series
5. estimate static factors and predictor sets
6. forecast each series and isolate forecast errors
7. estimate stochastic volatility of the forecast errors in R
8. compute horizon-specific uncertainty
9. aggregate to euro-area and country-level MEU series

Country MEUs are produced as a post-uncertainty aggregation stage from the
common panel outputs, not via separate country reruns.

## Current Status and Caveats

The core baseline pipeline is implemented, and the final plotting stage is also
implemented. What remains imperfect relative to the paper is mostly upstream:

- the public-data panel does not yet match the paper's underlying source
  snapshot exactly
- correlation cleaning still appears to remove more series than the paper's
  appendix suggests
- country-level basket construction is implemented with a documented project
  assumption for the shared common block

So the strongest remaining gap is panel composition rather than the downstream
factor, forecast, SV, uncertainty, or aggregation machinery.

## Further Reading

For short supporting notes, see:

- `documents/project_overview.md`
- `documents/what_works_well.md`

## References

- Jurado, K., Ludvigson, S. C., & Ng, S. (2015). Measuring Uncertainty.
  *American Economic Review*, 105(3), 1177-1216.
- Comunale, M., & Nguyen, A. D. M. (2025). A comprehensive MacroEconomic
  uncertainty measure for the euro area. *Journal of International Money and
  Finance*, 157, 103370.
  [DOI: 10.1016/j.jimonfin.2025.103370](https://doi.org/10.1016/j.jimonfin.2025.103370)
