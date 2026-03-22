# Project Overview and Next Steps

This note is a short project-status companion to the main
[README](../README.md). The README is the canonical entry point for setup,
execution, and output locations. This file only tracks where the project stands
and what still matters most analytically.

## Current State

The repository now implements the baseline MEU pipeline from public-data fetch
through final aggregation and plotting. The supported production path is the
strict endpoint-specific workflow for the 2021, 2022, and 2025 panels.

What is already in place:

- strict cleaned panel construction
- correlation filtering and correlation-audit outputs
- factor estimation
- forecast-error estimation
- stochastic-volatility estimation and validation
- horizon-specific uncertainty
- euro-area MEU aggregation
- country-level MEU aggregation
- final availability and MEU figures

Generated analysis outputs live under `bld/analysis/panels/<panel_name>/` in a
results-first structure with `results/`, `diagnostics/`, `artifacts/`, and
`internal/` subdirectories.

## Main Open Questions

The biggest remaining gap relative to the paper is upstream panel composition,
not downstream MEU construction.

Most importantly:

- the correlation screen still appears to remove more series than the paper's
  appendix suggests
- the public-data panel is close in spirit to CN, but not yet an exact match to
  the paper's original source snapshot
- the country-level common-variable basket still relies on a documented project
  assumption

## Next Priorities

1. investigate the remaining correlation-cleaning gap
2. tighten the paper-vs-repo panel comparison
3. refine aggregation choices only if the paper comparison requires it

## Reading Order

- start with [README](../README.md) for how to run the project
- use [what_works_well.md](./what_works_well.md) for a short strengths and
  replication-quality note
