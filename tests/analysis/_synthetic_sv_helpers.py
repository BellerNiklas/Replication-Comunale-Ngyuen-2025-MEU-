from pathlib import Path

import numpy as np
import pandas as pd


def write_synthetic_sv_outputs(
    *,
    tmp_path: Path,
    series_order_path: Path,
    regression_coefs_f_path: Path,
    forecast_errors_y_path: Path,
    forecast_errors_f_path: Path,
) -> dict[str, Path]:
    series_order = pd.read_parquet(series_order_path).sort_values("series_position")
    regression_coefs_f = pd.read_parquet(regression_coefs_f_path).sort_values(
        "predictor_position"
    )
    forecast_errors_y = pd.read_parquet(forecast_errors_y_path)
    forecast_errors_f = pd.read_parquet(forecast_errors_f_path)

    sv_outputs = {
        "sv_params_y": tmp_path / "sv_params_y.parquet",
        "sv_latent_y": tmp_path / "sv_latent_y.parquet",
        "sv_params_f": tmp_path / "sv_params_f.parquet",
        "sv_latent_f": tmp_path / "sv_latent_f.parquet",
    }

    pd.DataFrame(
        {
            "series_position": series_order["series_position"].to_numpy(),
            "series_id": series_order["series_id"].astype(str).to_numpy(),
            "mu": np.full(len(series_order), -0.25),
            "phi": np.full(len(series_order), 0.85),
            "sigma": np.full(len(series_order), 0.4),
            "offset_applied": np.zeros(len(series_order)),
            "adjusted_for_zero": np.zeros(len(series_order), dtype=bool),
        }
    ).to_parquet(sv_outputs["sv_params_y"], index=False)

    latent_y = pd.DataFrame({"date": forecast_errors_y["date"].astype(str)})
    for series_id in series_order["series_id"].astype(str):
        latent_y[series_id] = np.log(np.square(forecast_errors_y[series_id]) + 0.1)
    latent_y.to_parquet(sv_outputs["sv_latent_y"], index=False)

    pd.DataFrame(
        {
            "predictor_position": regression_coefs_f["predictor_position"].to_numpy(),
            "predictor_name": regression_coefs_f["predictor_name"].astype(str).to_numpy(),
            "mu": np.full(len(regression_coefs_f), -0.5),
            "phi": np.full(len(regression_coefs_f), 0.8),
            "sigma": np.full(len(regression_coefs_f), 0.35),
            "offset_applied": np.zeros(len(regression_coefs_f)),
            "adjusted_for_zero": np.zeros(len(regression_coefs_f), dtype=bool),
        }
    ).to_parquet(sv_outputs["sv_params_f"], index=False)

    latent_f = pd.DataFrame({"date": forecast_errors_f["date"].astype(str)})
    predictor_names = regression_coefs_f["predictor_name"].astype(str).tolist()
    for predictor_name in predictor_names:
        latent_f[predictor_name] = np.log(np.square(forecast_errors_f[predictor_name]) + 0.1)
    latent_f.to_parquet(sv_outputs["sv_latent_f"], index=False)

    return sv_outputs
