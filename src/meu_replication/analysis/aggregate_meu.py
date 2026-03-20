"""EA-wide and country-level MEU aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meu_replication.config import MEU_COUNTRIES

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


# ---------------------------------------------------------------------------
# Country-level MEU aggregation
# ---------------------------------------------------------------------------

_BASKET_REQUIRED_COLUMNS = {
    "series_id",
    "series_position",
    "country_iso2",
    "source",
    "category_name",
}

_MAX_FX_SERIES = 5


def _validate_full_panel(
    uncertainty_variance: pd.DataFrame,
    series_order: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the full uncertainty panel against the persisted series order."""
    ordered_series_ids = _ordered_series_ids(series_order)
    return _validate_uncertainty_panel(
        uncertainty_variance=uncertainty_variance,
        ordered_series_ids=ordered_series_ids,
    )


def build_country_baskets(
    series_order: pd.DataFrame,
    countries: tuple[str, ...] = MEU_COUNTRIES,
) -> dict[str, pd.DataFrame]:
    """Build per-country basket metadata from the panel series order.

    Each basket contains:
    - all rows where ``country_iso2`` matches the country (``basket_component="own"``)
    - all rows where ``series_id`` starts with ``U2_FX_`` (``basket_component="common_fx"``)

    This is a project assumption: the paper text is ambiguous between
    "euro area-wide common variables" and "bilateral exchange rates."
    The current implementation uses only the FX block as the shared component.
    """
    missing = _BASKET_REQUIRED_COLUMNS.difference(series_order.columns)
    if missing:
        msg = f"series_order is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    fx_mask = series_order["series_id"].astype(str).str.startswith("U2_FX_")
    fx_block = series_order.loc[fx_mask].copy()

    if fx_block.empty:
        msg = "series_order contains no U2_FX_* series for the shared FX block."
        raise ValueError(msg)
    if len(fx_block) > _MAX_FX_SERIES:
        msg = (
            f"series_order contains {len(fx_block)} U2_FX_* series, "
            f"but at most {_MAX_FX_SERIES} are expected."
        )
        raise ValueError(msg)

    baskets: dict[str, pd.DataFrame] = {}
    for country in countries:
        own_mask = series_order["country_iso2"].astype(str) == country
        own_block = series_order.loc[own_mask].copy()
        if own_block.empty:
            msg = (
                f"Country {country!r} has zero own-series rows in series_order "
                f"after filtering."
            )
            raise ValueError(msg)

        own_block = own_block.assign(basket_component="own")
        fx_part = fx_block.assign(basket_component="common_fx")

        basket = (
            pd.concat([own_block, fx_part], ignore_index=True)
            .sort_values("series_position")
            .reset_index(drop=True)
        )
        baskets[country] = basket

    return baskets


def build_basket_membership(
    baskets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate per-country baskets into a single audit table.

    The ``country_iso2`` column in the output reflects the basket country
    (the dict key), not the original series_order value. This means
    ``U2_FX_*`` rows appear once per country basket, each tagged with
    that country's ISO2 code.
    """
    _MEMBERSHIP_COLUMNS = [
        "country_iso2",
        "series_id",
        "basket_component",
        "series_position",
        "source",
        "category_name",
    ]

    parts: list[pd.DataFrame] = []
    for country_code, basket_df in sorted(baskets.items()):
        part = basket_df.copy()
        part["country_iso2"] = country_code
        parts.append(part[_MEMBERSHIP_COLUMNS])

    membership = pd.concat(parts, ignore_index=True)
    membership["basket_component"] = pd.Categorical(
        membership["basket_component"],
        categories=["own", "common_fx"],
        ordered=False,
    )
    return membership


def aggregate_country_meu(
    uncertainty_variance: pd.DataFrame,
    series_order: pd.DataFrame,
    countries: tuple[str, ...] = MEU_COUNTRIES,
) -> pd.DataFrame:
    """Aggregate series-level uncertainty to country-level MEUs.

    Validates the full panel once, builds country baskets, and computes
    ``mean(sqrt(variance))`` over each country's basket for every
    date-horizon pair.
    """
    validated = _validate_full_panel(uncertainty_variance, series_order)
    baskets = build_country_baskets(series_order, countries=countries)

    parts: list[pd.DataFrame] = []
    for country, basket_df in sorted(baskets.items()):
        basket_series = set(basket_df["series_id"].astype(str))
        country_panel = validated.loc[
            validated["series_id"].isin(basket_series)
        ].copy()

        agg = (
            country_panel.assign(
                meu=np.sqrt(country_panel["variance"].to_numpy(dtype=np.float64)),
            )
            .groupby(["date", "horizon"], as_index=False, sort=True, observed=True)[
                "meu"
            ]
            .mean()
        )
        agg["country_iso2"] = country
        parts.append(agg)

    result = pd.concat(parts, ignore_index=True)
    result["date"] = result["date"].astype(str)
    result["horizon"] = result["horizon"].astype("int16")
    result["meu"] = result["meu"].astype("float64")
    return (
        result.loc[:, ["country_iso2", "date", "horizon", "meu"]]
        .sort_values(["country_iso2", "date", "horizon"])
        .reset_index(drop=True)
    )
