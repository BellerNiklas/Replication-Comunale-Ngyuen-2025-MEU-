"""Estimate stochastic volatility with reference-aligned R-backed stochvol tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pytask import mark, task

from meu_replication.analysis.model_config import MEUConfig
from meu_replication.analysis.output_layout import (
    AnalysisOutputLayout,
    build_panel_output_layout,
)
from meu_replication.analysis.panel_specs import (
    ANALYSIS_PANELS,
    DEFAULT_ANALYSIS_PANEL,
)
from meu_replication.analysis.sv_r_bridge import SV_R_SCRIPT, run_sv_r_script

_CORE_SUBDIR = "_sv_r"
_PARAM_COLUMNS_GENERIC = [
    "series_position",
    "series_key",
    "mu",
    "phi",
    "sigma",
    "offset_applied",
    "adjusted_for_zero",
]
_DIAGNOSTIC_COLUMNS_GENERIC = [
    "series_type",
    "series_position",
    "series_key",
    "geweke_mu",
    "geweke_phi",
    "geweke_sigma",
    "acceptance_phi",
    "offset_applied",
    "adjusted_for_zero",
    "n_param_draws",
    "n_latent_draws",
]
_SV_DIAGNOSTIC_COLUMNS = [
    *_DIAGNOSTIC_COLUMNS_GENERIC,
    "sv_mode",
    "sv_draws",
    "sv_burnin",
    "sv_thin_para",
    "sv_thin_latent",
    "y_sample_start",
    "y_sample_end",
]


def _production_config(panel_name: str = DEFAULT_ANALYSIS_PANEL.panel_name) -> MEUConfig:
    # Production keeps the reference one-seed-per-panel behavior, even though
    # that gives up the previous chunked-Y speedup.
    return MEUConfig(panel_name=panel_name, sv_mode="full")


def _sv_public_dependencies(layout: AnalysisOutputLayout) -> dict[str, Path]:
    return {
        "forecast_metadata": layout.forecasts_dir / "forecast_metadata.parquet",
        "series_order": layout.factors_dir / "series_order.parquet",
        **_sv_core_dependencies(layout),
    }


def _sv_public_outputs(layout: AnalysisOutputLayout) -> dict[str, Path]:
    return {
        "sv_params_y": layout.sv_dir / "sv_params_y.parquet",
        "sv_latent_y": layout.sv_dir / "sv_latent_y.parquet",
        "sv_params_f": layout.sv_dir / "sv_params_f.parquet",
        "sv_latent_f": layout.sv_dir / "sv_latent_f.parquet",
        "sv_diagnostics": layout.sv_diagnostics_dir / "sv_diagnostics.parquet",
    }


def _sv_core_dir(layout: AnalysisOutputLayout) -> Path:
    return layout.sv_r_dir


def _sv_y_core_outputs(layout: AnalysisOutputLayout) -> dict[str, Path]:
    core_dir = _sv_core_dir(layout)
    return {
        "params": core_dir / "sv_params_y_core.parquet",
        "latent": core_dir / "sv_latent_y_core.parquet",
        "diagnostics": core_dir / "sv_diagnostics_y_core.parquet",
    }


def _sv_f_core_outputs(layout: AnalysisOutputLayout) -> dict[str, Path]:
    core_dir = _sv_core_dir(layout)
    return {
        "params": core_dir / "sv_params_f_core.parquet",
        "latent": core_dir / "sv_latent_f_core.parquet",
        "diagnostics": core_dir / "sv_diagnostics_f_core.parquet",
    }


def _sv_core_dependencies(layout: AnalysisOutputLayout) -> dict[str, Path]:
    y_outputs = _sv_y_core_outputs(layout)
    factor_outputs = _sv_f_core_outputs(layout)
    return {
        "sv_params_y_core": y_outputs["params"],
        "sv_latent_y_core": y_outputs["latent"],
        "sv_diagnostics_y_core": y_outputs["diagnostics"],
        "sv_params_f_core": factor_outputs["params"],
        "sv_latent_f_core": factor_outputs["latent"],
        "sv_diagnostics_f_core": factor_outputs["diagnostics"],
    }


def _panel_dependencies(layout: AnalysisOutputLayout, series_type: str) -> dict[str, Path]:
    if series_type == "y":
        return {
            "forecast_errors": layout.forecasts_dir / "forecast_errors_y.parquet",
            "series_order": layout.factors_dir / "series_order.parquet",
        }
    return {
        "forecast_errors": layout.forecasts_dir / "forecast_errors_f.parquet",
    }


def _panel_task_kwargs(
    series_type: str,
    config: MEUConfig | None = None,
) -> dict[str, object]:
    config = _production_config() if config is None else config
    seed = config.sv_seed_y if series_type == "y" else config.sv_seed_f
    options: dict[str, object] = {
        "task_kind": "panel",
        "seed": seed,
        "series_type": series_type,
        "sv_burnin": config.sv_burnin,
        "sv_draws": config.sv_draws,
        "sv_offset": config.sv_offset,
        "sv_thin_latent": config.sv_thin_latent,
        "sv_thin_para": config.sv_thin_para,
    }
    return options


def run_stochastic_volatility_stage(
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    config: MEUConfig,
) -> None:
    """Run the SV stage directly from Python for tests and checkpoints."""
    layout = _layout_from_outputs(produces)

    _run_panel_estimation(
        depends_on={
            "forecast_errors": depends_on["forecast_errors_y"],
            "series_order": depends_on["series_order"],
        },
        produces=_sv_y_core_outputs(layout),
        config=config,
        series_type="y",
    )
    _run_panel_estimation(
        depends_on={"forecast_errors": depends_on["forecast_errors_f"]},
        produces=_sv_f_core_outputs(layout),
        config=config,
        series_type="f",
    )
    _consolidate_sv_outputs(
        depends_on={
            "forecast_metadata": depends_on["forecast_metadata"],
            "series_order": depends_on["series_order"],
            **_sv_core_dependencies(layout),
        },
        produces=produces,
        config=config,
    )


def _run_panel_estimation(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    config: MEUConfig,
    series_type: str,
) -> None:
    seed = config.sv_seed_y if series_type == "y" else config.sv_seed_f
    options: dict[str, object] = {
        "task_kind": "panel",
        "seed": seed,
        "series_type": series_type,
        "sv_burnin": config.sv_burnin,
        "sv_draws": config.sv_draws,
        "sv_offset": config.sv_offset,
        "sv_thin_latent": config.sv_thin_latent,
        "sv_thin_para": config.sv_thin_para,
    }

    run_sv_r_script(depends_on=depends_on, produces=produces, options=options)


def _normalize_param_frame(
    frame: pd.DataFrame,
    *,
    key_name: str,
    position_name: str,
) -> pd.DataFrame:
    renamed = frame.loc[:, _PARAM_COLUMNS_GENERIC].rename(
        columns={"series_key": key_name, "series_position": position_name}
    )
    normalized = renamed.astype(
        {
            position_name: "int64",
            "mu": "float64",
            "phi": "float64",
            "sigma": "float64",
            "offset_applied": "float64",
            "adjusted_for_zero": "bool",
        }
    )
    normalized[key_name] = normalized[key_name].astype(str)
    return normalized.sort_values(position_name).reset_index(drop=True)


def _normalize_latent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    column_names = frame.columns.tolist()
    normalized = pd.DataFrame(
        {
            "date": frame["date"].astype(str),
            **{
                column_name: frame[column_name].astype("float64")
                for column_name in column_names[1:]
            },
        }
    )
    return normalized


def _order_latent_columns(frame: pd.DataFrame, ordered_names: list[str]) -> pd.DataFrame:
    missing_columns = [name for name in ordered_names if name not in frame.columns]
    if missing_columns:
        msg = "Y latent core output is missing expected series columns: "
        raise ValueError(msg + ", ".join(missing_columns))
    ordered_columns = ["date", *ordered_names]
    return frame.loc[:, ordered_columns]


def _normalize_diagnostics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.loc[:, _DIAGNOSTIC_COLUMNS_GENERIC].astype(
        {
            "series_position": "int64",
            "geweke_mu": "float64",
            "geweke_phi": "float64",
            "geweke_sigma": "float64",
            "acceptance_phi": "float64",
            "offset_applied": "float64",
            "adjusted_for_zero": "bool",
            "n_param_draws": "int64",
            "n_latent_draws": "int64",
        }
    )
    normalized["series_type"] = normalized["series_type"].astype(str)
    normalized["series_key"] = normalized["series_key"].astype(str)
    return normalized.sort_values(["series_type", "series_position"]).reset_index(
        drop=True
    )


def _layout_from_outputs(produces: dict[str, Path]) -> AnalysisOutputLayout:
    sv_dir = produces["sv_params_y"].parent
    panel_root = sv_dir.parent.parent
    return build_panel_output_layout(panel_root.name, panels_dir=panel_root.parent)


def _consolidate_sv_outputs(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    config: MEUConfig,
) -> None:
    forecast_metadata = pd.read_parquet(depends_on["forecast_metadata"])
    series_order = pd.read_parquet(depends_on["series_order"])
    ordered_series = (
        series_order.sort_values("series_position")["series_id"].astype(str).tolist()
    )

    params_y = _normalize_param_frame(
        pd.read_parquet(depends_on["sv_params_y_core"]),
        key_name="series_id",
        position_name="series_position",
    )
    latent_y = _order_latent_columns(
        _normalize_latent_frame(pd.read_parquet(depends_on["sv_latent_y_core"])),
        ordered_series,
    )
    diagnostics_y = _normalize_diagnostics_frame(
        pd.read_parquet(depends_on["sv_diagnostics_y_core"])
    )

    params_f = _normalize_param_frame(
        pd.read_parquet(depends_on["sv_params_f_core"]),
        key_name="predictor_name",
        position_name="predictor_position",
    )
    latent_f = _normalize_latent_frame(pd.read_parquet(depends_on["sv_latent_f_core"]))
    diagnostics_f = _normalize_diagnostics_frame(
        pd.read_parquet(depends_on["sv_diagnostics_f_core"])
    )

    diagnostics = pd.concat([diagnostics_y, diagnostics_f], ignore_index=True).assign(
        sv_mode=config.sv_mode,
        sv_draws=config.sv_draws,
        sv_burnin=config.sv_burnin,
        sv_thin_para=config.sv_thin_para,
        sv_thin_latent=config.sv_thin_latent,
        y_sample_start=str(forecast_metadata.loc[0, "y_sample_start"]),
        y_sample_end=str(forecast_metadata.loc[0, "y_sample_end"]),
    )
    diagnostics = diagnostics.loc[:, _SV_DIAGNOSTIC_COLUMNS]

    for output_path in produces.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    params_y.to_parquet(produces["sv_params_y"], index=False)
    latent_y.to_parquet(produces["sv_latent_y"], index=False)
    params_f.to_parquet(produces["sv_params_f"], index=False)
    latent_f.to_parquet(produces["sv_latent_f"], index=False)
    diagnostics.to_parquet(produces["sv_diagnostics"], index=False)

    print(
        f"[{config.panel_name}] Stochastic-volatility stage complete: "
        f"{len(params_y)} y-series, "
        f"{len(params_f)} factor series, "
        f"mode={config.sv_mode}, "
        f"draws={config.sv_draws}, burn_in={config.sv_burnin}."
    )


for _spec in ANALYSIS_PANELS:

    @task(
        name="sv_y",
        id=_spec.task_id,
        kwargs=_panel_task_kwargs("y", config=_production_config(_spec.panel_name)),
    )
    @mark.r(script=SV_R_SCRIPT)
    def sv_y(
        depends_on: dict[str, Path] = _panel_dependencies(_spec.layout, "y"),
        produces: dict[str, Path] = _sv_y_core_outputs(_spec.layout),
    ) -> None:
        """Estimate stochastic volatility for one Y residual panel."""
        pass

    @task(
        name="sv_f",
        id=_spec.task_id,
        kwargs=_panel_task_kwargs("f", config=_production_config(_spec.panel_name)),
    )
    @mark.r(script=SV_R_SCRIPT)
    def sv_f(
        depends_on: dict[str, Path] = _panel_dependencies(_spec.layout, "f"),
        produces: dict[str, Path] = _sv_f_core_outputs(_spec.layout),
    ) -> None:
        """Estimate stochastic volatility for one factor residual panel."""
        pass

    @task(id=_spec.task_id)
    def task_sv_consolidate(
        depends_on: dict[str, Path] = _sv_public_dependencies(_spec.layout),
        produces: dict[str, Path] = _sv_public_outputs(_spec.layout),
        panel_name: str = _spec.panel_name,
    ) -> None:
        """Normalize SV outputs for one cleaned panel."""
        _consolidate_sv_outputs(
            depends_on=depends_on,
            produces=produces,
            config=_production_config(panel_name),
        )
