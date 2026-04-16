"""Helpers for applying stationarity transformations (Comunale & Nguyen 2025).

Each series in the macro panel has a transformationcode from the registry:
    Code 1: x_t           (no transformation - sentiment indicators)
    Code 2: x_t - x_{t-1} (first difference - rates, yields, unemployment,
                           and stock variables that can legitimately hit zero)
    Code 5: ln(x_t) - ln(x_{t-1}) (log first difference - indices, quantities, prices)

Differencing (codes 2 and 5) loses the first observation per series.
"""

import warnings

import numpy as np
import pandas as pd

SUPPORTED_CODES: set[int] = {1, 2, 5}


def _transform_series(values: pd.Series, code: int) -> pd.Series:
    """Apply one transformation code to a sorted value series.

    Args:
        values: Numeric series sorted by date.
        code: Transformation code (1, 2, or 5).

    Returns:
        Transformed series (same length; first obs is NaN for codes 2/5).

    Raises:
        ValueError: If code is not in {1, 2, 5}.
    """
    if code == 1:
        return values
    if code == 2:
        return values.diff()
    if code == 5:
        log_vals = np.log(values.where(values > 0))
        return log_vals.diff()
    msg = f"Unsupported transformation code: {code}. Expected one of {SUPPORTED_CODES}."
    raise ValueError(msg)


def build_transform_map(registry: pd.DataFrame) -> pd.DataFrame:
    """Extract series_id -> transformationcode mapping from registry.

    Args:
        registry: Full series registry with transformationcode column.

    Returns:
        DataFrame with columns [series_id, transformationcode].

    Raises:
        ValueError: If registry contains unsupported transformation codes.
    """
    required = {"series_id", "transformationcode"}
    missing = required - set(registry.columns)
    if missing:
        msg = f"Registry missing columns: {missing}"
        raise ValueError(msg)

    tmap = pd.DataFrame(
        {
            "series_id": registry["series_id"],
            "transformationcode": registry["transformationcode"].astype(int),
        }
    )

    bad_codes = set(tmap["transformationcode"].unique()) - SUPPORTED_CODES
    if bad_codes:
        msg = f"Unsupported transformation codes in registry: {bad_codes}"
        raise ValueError(msg)

    return tmap


def apply_stationarity_transforms(
    panel: pd.DataFrame,
    transform_map: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-series stationarity transformations to the macro panel.

    Args:
        panel: Long-format panel with columns [date, value, series_id,
            country_iso2, variable_name, category, category_name, source].
            Must be sorted by (series_id, date).
        transform_map: DataFrame with columns [series_id, transformationcode].

    Returns:
        Transformed panel with same schema as input plus column
        'transformationcode'. Rows where the transformation produces NaN
        (first observation after differencing) are dropped.
    """
    if panel.empty:
        empty_cols = [*panel.columns, "transformationcode"]
        empty_panel = pd.DataFrame(columns=pd.Index(empty_cols))
        return empty_panel

    merged = panel.merge(transform_map, on="series_id", how="left")

    unmatched = merged["transformationcode"].isna()
    if unmatched.any():
        n = unmatched.sum()
        ids = merged.loc[unmatched, "series_id"].unique()[:5]
        warnings.warn(
            f"{n} rows have no transformationcode (series: {ids.tolist()}). "
            "These rows will be dropped.",
            stacklevel=2,
        )
        merged = merged[~unmatched].copy()

    merged = merged.sort_values(["series_id", "date"]).reset_index(drop=True)

    transformed_parts = []
    for _, g in merged.groupby("series_id", sort=False):
        code = int(g["transformationcode"].iloc[0])
        transformed_parts.append(
            _transform_series(g["value"].reset_index(drop=True), code)
        )
    transformed_values = pd.concat(transformed_parts, ignore_index=True)

    transformed_panel = pd.DataFrame(
        {
            "date": merged["date"].values,
            "value": transformed_values.values,
            "series_id": merged["series_id"].values,
            "country_iso2": merged["country_iso2"].values,
            "variable_name": merged["variable_name"].values,
            "category": merged["category"].values,
            "category_name": merged["category_name"].values,
            "source": merged["source"].values,
            "transformationcode": merged["transformationcode"].astype(int).values,
        }
    )
    transformed_panel = transformed_panel.dropna(subset=["value"]).reset_index(
        drop=True
    )
    return transformed_panel
