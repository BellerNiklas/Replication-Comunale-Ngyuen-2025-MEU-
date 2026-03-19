# Project Overview and Next Steps

## Project Summary

This repository already implements the baseline MacroEconomic Uncertainty
(MEU) pipeline for the euro area. The core analysis workflow is now mostly in
place, from cleaned panel input through uncertainty estimation and final
aggregation. The cleaning-rule decision has now been fixed: the project uses
strict endpoint-specific panels as its supported preprocessing path.

In practical terms, the project has moved beyond building the baseline MEU
engine. The core workflow now runs on the three strict cleaned endpoints, 2021,
2022, and 2025, with panel-specific outputs under `bld/analysis/<panel_name>/`.
The next phase is about investigating the remaining correlation gap with the
paper and then broadening the outputs from euro-area series to a full set of
euro-area and country-level MEUs.

## What Has Already Been Done

The current repository already covers the main steps of the baseline
replication workflow:

- raw-data processing and variable transformations
- strict completeness filtering for the 2021, 2022, and 2025 endpoint versions
- correlation filtering within countries
- factor estimation
- forecast-error estimation
- stochastic-volatility estimation
- uncertainty computation
- baseline euro-area MEU aggregation

As a result, the repo now produces EA-wide MEUs for all three cleaned strict
panels.
The root [README](../README.md) remains the main technical entry point for the
implemented pipeline and the current analysis outputs.

## Current Open Data Questions

The completeness rule is no longer an open question. The project now uses a
strict rule that drops variables unless the full endpoint window is complete,
and the relaxed 98% coverage variant is no longer part of the supported
pipeline.

The second unresolved question is the relatively large number of
correlation-based drops compared with the paper. The appendix suggests a
dataset with about 1470 variables before the correlation screen and about 1330
after it, so this remains an important gap to investigate. The current evidence
suggests that the extra drops are concentrated in overlapping sentiment and
price indicators, which means the issue is likely not only the threshold
itself, but also the underlying variable registry and source overlap.

## Target Cleaning Structure

The cleaning pipeline should ultimately expose six key data frames built around
the three strict target endpoint windows, 2021, 2022, and 2025:

- three data frames listing the variables dropped at the completeness stage
- three fully cleaned data frames after completeness filtering and correlation
  filtering

These six outputs should become the main data foundation for the rest of the
project. They make the cleaning choices visible, keep the endpoint versions
separate, and provide a clear bridge from raw series selection to MEU
estimation.

## Next Analytical Deliverables

The cleaned panels are now wired into the full analysis DAG, and `pixi run
pytask-parallel` can schedule the 2021, 2022, and 2025 branches concurrently
with a safer two-worker default on Windows. Single-panel reruns can be targeted
with `pytask -k analysis_2025` and the analogous `analysis_2021` and
`analysis_2022` keys.

The most important next extension is now to calculate country-level MEUs
using the same method. This should not require a new uncertainty model. The
existing pipeline should be reused, with the main change happening at the final
aggregation step, where the series included in the average are restricted to a
given country instead of the full euro-area panel.

This country-level extension is the key step that turns the project from a
single baseline replication into a broader uncertainty framework.

## Priority Order

1. investigate the high correlation drops
2. compute country MEUs for all countries
3. add alternative aggregation choices if needed for the paper comparison

## Closing Note

The project already has the baseline MEU engine in place. The remaining work is
now mainly about finalizing the cleaned datasets and expanding the outputs from
one euro-area series to a full EA-and-country MEU framework built on the same
methodology.
