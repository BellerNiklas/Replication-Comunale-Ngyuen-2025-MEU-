"""Compute horizon-specific uncertainty from the forecast and SV artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.output_layout import AnalysisOutputLayout
from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
)
from meu_replication.analysis.sv_validation import assert_sv_validation_passed
from meu_replication.analysis.uncertainty import (
    compute_uncertainty,
    uncertainty_result_to_long_frame,
)


def _uncertainty_dependencies(layout: AnalysisOutputLayout) -> dict[str, Path]:
    return {
        "series_order": layout.factors_dir / "series_order.parquet",
        "forecast_metadata": layout.forecasts_dir / "forecast_metadata.parquet",
        "regression_coefs_y": layout.forecasts_dir / "regression_coefs_y.parquet",
        "regression_coefs_f": layout.forecasts_dir / "regression_coefs_f.parquet",
        "sv_params_y": layout.sv_dir / "sv_params_y.parquet",
        "sv_latent_y": layout.sv_dir / "sv_latent_y.parquet",
        "sv_params_f": layout.sv_dir / "sv_params_f.parquet",
        "sv_latent_f": layout.sv_dir / "sv_latent_f.parquet",
        "sv_validation_summary": layout.sv_diagnostics_dir
        / "sv_validation_summary.parquet",
    }


def _uncertainty_outputs(layout: AnalysisOutputLayout) -> dict[str, Path]:
    return {
        "uncertainty_variance": layout.uncertainty_dir / "uncertainty_variance.parquet",
    }


def run_uncertainty_stage(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    config: MEUConfig,
) -> None:
    """Execute the stage-4 uncertainty computation and persist the output."""
    assert_sv_validation_passed(depends_on["sv_validation_summary"])

    result = compute_uncertainty(
        forecast_metadata=pd.read_parquet(depends_on["forecast_metadata"]),
        series_order=pd.read_parquet(depends_on["series_order"]),
        regression_coefs_y=pd.read_parquet(depends_on["regression_coefs_y"]),
        regression_coefs_f=pd.read_parquet(depends_on["regression_coefs_f"]),
        sv_params_y=pd.read_parquet(depends_on["sv_params_y"]),
        sv_latent_y=pd.read_parquet(depends_on["sv_latent_y"]),
        sv_params_f=pd.read_parquet(depends_on["sv_params_f"]),
        sv_latent_f=pd.read_parquet(depends_on["sv_latent_f"]),
        h_max=config.h_max,
    )
    uncertainty_long = uncertainty_result_to_long_frame(result)

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    uncertainty_long.to_parquet(produces["uncertainty_variance"], index=False)

    print(
        f"[{config.panel_name}] Uncertainty stage complete: "
        f"{len(result.dates)} dates, "
        f"{len(result.series_ids)} series, "
        f"{len(result.horizons)} horizons, "
        f"{len(uncertainty_long)} rows."
    )


for _spec in ANALYSIS_PANELS:

    @pytask.task(id=_spec.task_id)
    def task_compute_uncertainty(
        depends_on: dict[str, Path] = _uncertainty_dependencies(_spec.layout),
        produces: dict[str, Path] = _uncertainty_outputs(_spec.layout),
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Run the uncertainty stage for one cleaned panel."""
        run_uncertainty_stage(
            depends_on=depends_on,
            produces=produces,
            config=MEUConfig(panel_name=panel_name),
        )
