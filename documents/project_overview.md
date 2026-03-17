# Project Overview and Next Steps

## Project Summary

This repository already implements the baseline MacroEconomic Uncertainty
(MEU) pipeline for the euro area. The core analysis workflow is now mostly in
place, from cleaned panel input through uncertainty estimation and final
aggregation. At this stage, the main open questions are less about the MEU
method itself and more about the data-cleaning choices that define the final
input panels.

In practical terms, the project has moved beyond building the baseline MEU
engine. The next phase is about deciding which cleaned datasets should be
treated as the main specification, extending the endpoint to 2025, and then
broadening the outputs from one euro-area series to a full set of euro-area
and country-level MEUs.

## What Has Already Been Done

The current repository already covers the main steps of the baseline
replication workflow:

- raw-data processing and variable transformations
- completeness filtering for the 2021 and 2022 endpoint versions
- correlation filtering within countries
- factor estimation
- forecast-error estimation
- stochastic-volatility estimation
- uncertainty computation
- baseline euro-area MEU aggregation

As a result, the repo already produces an EA-wide MEU for a cleaned 2022 panel.
The root [README](../README.md) remains the main technical entry point for the
implemented pipeline and the current analysis outputs.

## Current Open Data Questions

The biggest unresolved question is which completeness rule should define the
main cleaned datasets. There are currently two candidate approaches:

- a strict rule that drops variables unless the full window is complete
- a relaxed rule that allows a 98% coverage version

That choice matters because it determines which final panels will feed all
later MEU calculations.

The second unresolved question is the relatively large number of
correlation-based drops compared with the paper. The appendix suggests a
dataset with about 1470 variables before the correlation screen and about 1330
after it, so this remains an important gap to investigate. The current evidence
suggests that the extra drops are concentrated in overlapping sentiment and
price indicators, which means the issue is likely not only the threshold
itself, but also the underlying variable registry and source overlap.

## Target Cleaning Structure

The cleaning pipeline should ultimately expose six key data frames built around
the three target endpoint windows, 2021, 2022, and 2025:

- three data frames listing the variables dropped at the completeness stage
- three fully cleaned data frames after completeness filtering and correlation
  filtering

These six outputs should become the main data foundation for the rest of the
project. They make the cleaning choices visible, keep the endpoint versions
separate, and provide a clear bridge from raw series selection to MEU
estimation.

## Next Analytical Deliverables

Once the cleaned panels are finalized, the next euro-area goal is to calculate
the MEU for all three fully cleaned panels:

- 2021
- 2022
- 2025

The most important next extension, however, is to calculate country-level MEUs
using the same method. This should not require a new uncertainty model. The
existing pipeline should be reused, with the main change happening at the final
aggregation step, where the series included in the average are restricted to a
given country instead of the full euro-area panel.

This country-level extension is the key step that turns the project from a
single baseline replication into a broader uncertainty framework.

## Priority Order

1. choose the main cleaning rule
2. extend the endpoint to 2025
3. investigate the high correlation drops
4. finalize the three cleaned panels
5. compute EA MEUs for all three panels
6. compute country MEUs for all countries

## Closing Note

The project already has the baseline MEU engine in place. The remaining work is
now mainly about finalizing the cleaned datasets and expanding the outputs from
one euro-area series to a full EA-and-country MEU framework built on the same
methodology.
