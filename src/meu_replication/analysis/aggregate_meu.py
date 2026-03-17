"""Baseline EA-wide MEU aggregation for Milestone 5."""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_UNCERTAINTY_COLUMNS = ["date", "series_id", "horizon", "variance"]


def aggregate_ea_meu(
    uncertainty_variance: pd.DataFrame,
    series_order: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate series-level uncertainty to the baseline EA-wide MEU."""
    ordered_series_ids = _ordered_series_ids(series_order)
    validated = _validate_uncertainty_panel(
        uncertainty_variance=uncertainty_variance,
        ordered_series_ids=ordered_series_ids,
    )

    aggregated = (
        validated.assign(meu=np.sqrt(validated["variance"].to_numpy(dtype=np.float64)))
        .groupby(["date", "horizon"], as_index=False, sort=True, observed=True)["meu"]
        .mean()
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    aggregated["date"] = aggregated["date"].astype(str)
    aggregated["horizon"] = aggregated["horizon"].astype("int16")
    aggregated["meu"] = aggregated["meu"].astype("float64")
    return aggregated.loc[:, ["date", "horizon", "meu"]]


def _ordered_series_ids(series_order: pd.DataFrame) -> tuple[str, ...]:
    required_columns = {"series_position", "series_id"}
    missing = required_columns.difference(series_order.columns)
    if missing:
        msg = f"series_order is missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    if series_order.empty:
        msg = "series_order cannot be empty."
        raise ValueError(msg)
    if series_order["series_position"].duplicated().any():
        msg = "series_order cannot contain duplicated series_position values."
        raise ValueError(msg)
    if series_order["series_id"].astype(str).duplicated().any():
        msg = "series_order cannot contain duplicated series_id values."
        raise ValueError(msg)

    ordered = series_order.sort_values("series_position").reset_index(drop=True).copy()
    expected_positions = np.arange(len(ordered), dtype=np.int64)
    positions = ordered["series_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        msg = "series_order positions must be consecutive and zero-based."
        raise ValueError(msg)

    return tuple(ordered["series_id"].astype(str).tolist())


def _validate_uncertainty_panel(
    *,
    uncertainty_variance: pd.DataFrame,
    ordered_series_ids: tuple[str, ...],
) -> pd.DataFrame:
    if uncertainty_variance.columns.tolist() != _REQUIRED_UNCERTAINTY_COLUMNS:
        msg = (
            "uncertainty_variance must have exactly columns "
            f"{_REQUIRED_UNCERTAINTY_COLUMNS}."
        )
        raise ValueError(msg)
    if uncertainty_variance.empty:
        msg = "uncertainty_variance cannot be empty."
        raise ValueError(msg)

    validated = uncertainty_variance.copy()
    validated["date"] = validated["date"].astype(str)
    validated["series_id"] = validated["series_id"].astype(str)
    validated["horizon"] = validated["horizon"].astype("int16")
    validated["variance"] = validated["variance"].astype("float64")

    if not np.isfinite(validated["variance"]).all():
        msg = "uncertainty_variance must contain only finite variance values."
        raise ValueError(msg)
    if (validated["variance"] <= 0.0).any():
        msg = "uncertainty_variance must contain strictly positive variances."
        raise ValueError(msg)
    if validated.duplicated(["date", "horizon", "series_id"]).any():
        msg = "uncertainty_variance cannot contain duplicated date-horizon-series rows."
        raise ValueError(msg)

    unexpected_series_ids = sorted(set(validated["series_id"]) - set(ordered_series_ids))
    if unexpected_series_ids:
        msg = (
            "uncertainty_variance contains unexpected series_id values: "
            f"{unexpected_series_ids[:5]}"
        )
        raise ValueError(msg)

    validated["series_id"] = pd.Categorical(
        validated["series_id"],
        categories=list(ordered_series_ids),
        ordered=True,
    )

    expected_series_count = len(ordered_series_ids)
    group_sizes = validated.groupby(["date", "horizon"], observed=True).size()
    if not group_sizes.eq(expected_series_count).all():
        msg = (
            "Each date-horizon group in uncertainty_variance must contain exactly "
            "one row per persisted series."
        )
        raise ValueError(msg)

    ordered_panel = validated.sort_values(["date", "horizon", "series_id"]).reset_index(
        drop=True,
    )
    repeated_codes = np.tile(
        np.arange(expected_series_count, dtype=np.int16),
        len(ordered_panel) // expected_series_count,
    )
    if not np.array_equal(
        ordered_panel["series_id"].cat.codes.to_numpy(dtype=np.int16),
        repeated_codes,
    ):
        msg = (
            "Each date-horizon group in uncertainty_variance must match the "
            "persisted series order exactly."
        )
        raise ValueError(msg)

    return ordered_panel
