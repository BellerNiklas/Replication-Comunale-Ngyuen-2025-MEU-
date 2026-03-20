import numpy as np
import pandas as pd
import pytest

from meu_replication.analysis.aggregate_meu import (
    aggregate_country_meu,
    aggregate_ea_meu,
    build_basket_membership,
    build_country_baskets,
)
from meu_replication.analysis.task_aggregate_country_meu import (
    run_country_meu_aggregation_stage,
)
from meu_replication.config import MEU_COUNTRIES


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _series_order(
    *,
    include_bond: bool = True,
    extra_fx: int = 0,
) -> pd.DataFrame:
    """Minimal series_order with two countries, FX, and optionally a bond."""
    rows = [
        (0, "AT_IP_001", "AT", "eurostat", "Industrial_production"),
        (1, "AT_IP_002", "AT", "eurostat", "Industrial_production"),
        (2, "DE_CPI_001", "DE", "eurostat", "Prices"),
        (3, "DE_CPI_002", "DE", "eurostat", "Prices"),
        (4, "U2_FX_001", "U2", "ecb", "EA_Financial"),
        (5, "U2_FX_002", "U2", "ecb", "EA_Financial"),
    ]
    if include_bond:
        rows.append((6, "U2_BOND_001", "U2", "ecb", "EA_Financial"))
    for i in range(extra_fx):
        pos = len(rows)
        rows.append((pos, f"U2_FX_0{i + 3:02d}", "U2", "ecb", "EA_Financial"))
    return pd.DataFrame(
        rows,
        columns=["series_position", "series_id", "country_iso2", "source",
                 "category_name"],
    )


def _uncertainty_panel(series_order: pd.DataFrame) -> pd.DataFrame:
    """Dense uncertainty panel for two dates, two horizons."""
    series_ids = series_order["series_id"].tolist()
    rng = np.random.default_rng(42)
    records = []
    for date in ["2020-01", "2020-02"]:
        for horizon in [1, 2]:
            for sid in series_ids:
                records.append({
                    "date": date,
                    "series_id": sid,
                    "horizon": horizon,
                    "variance": float(rng.uniform(0.1, 5.0)),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# build_country_baskets
# ---------------------------------------------------------------------------


class TestBuildCountryBaskets:

    def test_basket_includes_own_series_and_fx(self):
        so = _series_order()
        baskets = build_country_baskets(so, countries=("AT", "DE"))

        at_ids = set(baskets["AT"]["series_id"])
        assert "AT_IP_001" in at_ids
        assert "AT_IP_002" in at_ids
        assert "U2_FX_001" in at_ids
        assert "U2_FX_002" in at_ids

        de_ids = set(baskets["DE"]["series_id"])
        assert "DE_CPI_001" in de_ids
        assert "DE_CPI_002" in de_ids
        assert "U2_FX_001" in de_ids
        assert "U2_FX_002" in de_ids

    def test_basket_excludes_non_fx_u2(self):
        so = _series_order(include_bond=True)
        baskets = build_country_baskets(so, countries=("AT",))
        at_ids = set(baskets["AT"]["series_id"])
        assert "U2_BOND_001" not in at_ids

    def test_basket_component_labels(self):
        so = _series_order()
        baskets = build_country_baskets(so, countries=("AT",))
        basket = baskets["AT"]
        own = basket.loc[basket["basket_component"] == "own", "series_id"].tolist()
        fx = basket.loc[basket["basket_component"] == "common_fx", "series_id"].tolist()
        assert set(own) == {"AT_IP_001", "AT_IP_002"}
        assert set(fx) == {"U2_FX_001", "U2_FX_002"}

    def test_basket_sorted_by_series_position(self):
        so = _series_order()
        baskets = build_country_baskets(so, countries=("AT",))
        positions = baskets["AT"]["series_position"].tolist()
        assert positions == sorted(positions)

    def test_error_on_missing_country(self):
        so = _series_order()
        with pytest.raises(ValueError, match="XX"):
            build_country_baskets(so, countries=("AT", "XX"))

    def test_error_on_empty_fx_block(self):
        so = pd.DataFrame({
            "series_position": [0, 1],
            "series_id": ["AT_IP_001", "AT_IP_002"],
            "country_iso2": ["AT", "AT"],
            "source": ["eurostat", "eurostat"],
            "category_name": ["IP", "IP"],
        })
        with pytest.raises(ValueError, match="no U2_FX_"):
            build_country_baskets(so, countries=("AT",))

    def test_error_on_too_many_fx_series(self):
        so = _series_order(include_bond=False, extra_fx=5)
        # 2 base FX + 5 extra = 7, exceeds _MAX_FX_SERIES=5
        with pytest.raises(ValueError, match="at most 5"):
            build_country_baskets(so, countries=("AT",))


# ---------------------------------------------------------------------------
# build_basket_membership
# ---------------------------------------------------------------------------


class TestBuildBasketMembership:

    def test_membership_country_iso2_is_basket_key(self):
        so = _series_order()
        baskets = build_country_baskets(so, countries=("AT", "DE"))
        membership = build_basket_membership(baskets)

        fx_rows = membership.loc[membership["series_id"] == "U2_FX_001"]
        assert set(fx_rows["country_iso2"]) == {"AT", "DE"}

    def test_membership_has_expected_columns(self):
        so = _series_order()
        baskets = build_country_baskets(so, countries=("AT",))
        membership = build_basket_membership(baskets)
        assert membership.columns.tolist() == [
            "country_iso2", "series_id", "basket_component",
            "series_position", "source", "category_name",
        ]

    def test_membership_basket_component_is_categorical(self):
        so = _series_order()
        baskets = build_country_baskets(so, countries=("AT",))
        membership = build_basket_membership(baskets)
        assert membership["basket_component"].dtype.name == "category"
        assert list(membership["basket_component"].cat.categories) == [
            "own", "common_fx",
        ]


# ---------------------------------------------------------------------------
# aggregate_country_meu
# ---------------------------------------------------------------------------


class TestAggregateCountryMeu:

    def test_matches_manual_mean_sqrt_variance(self):
        so = _series_order()
        uv = _uncertainty_panel(so)
        result = aggregate_country_meu(uv, so, countries=("AT", "DE"))

        # Manual calculation for AT, date=2020-01, horizon=1
        at_basket_ids = {"AT_IP_001", "AT_IP_002", "U2_FX_001", "U2_FX_002"}
        subset = uv.loc[
            (uv["series_id"].isin(at_basket_ids))
            & (uv["date"] == "2020-01")
            & (uv["horizon"] == 1)
        ]
        expected_meu = np.sqrt(subset["variance"].to_numpy(dtype=np.float64)).mean()
        actual = result.loc[
            (result["country_iso2"] == "AT")
            & (result["date"] == "2020-01")
            & (result["horizon"] == 1),
            "meu",
        ].iloc[0]
        np.testing.assert_almost_equal(actual, expected_meu)

    def test_output_schema_and_sorting(self):
        so = _series_order()
        uv = _uncertainty_panel(so)
        result = aggregate_country_meu(uv, so, countries=("DE", "AT"))

        assert result.columns.tolist() == [
            "country_iso2", "date", "horizon", "meu",
        ]
        assert result["horizon"].dtype == np.int16
        assert result["meu"].dtype == np.float64
        # Sorted by country, date, horizon
        assert result.iloc[0]["country_iso2"] == "AT"

    def test_row_count(self):
        so = _series_order()
        uv = _uncertainty_panel(so)
        result = aggregate_country_meu(uv, so, countries=("AT", "DE"))
        n_countries = 2
        n_dates = 2
        n_horizons = 2
        assert len(result) == n_countries * n_dates * n_horizons

    def test_deterministic_on_repeated_calls(self):
        so = _series_order()
        uv = _uncertainty_panel(so)
        r1 = aggregate_country_meu(uv, so, countries=("AT", "DE"))
        r2 = aggregate_country_meu(uv, so, countries=("AT", "DE"))
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Cross-check: EA MEU ≠ average of country MEUs
# ---------------------------------------------------------------------------


def test_ea_meu_differs_from_average_of_country_meus():
    """When non-FX U2_* series exist, EA and avg(country) must differ.

    The EA aggregate uses ALL series (including U2_BOND_*), while country
    baskets include only own-country + U2_FX_*. The two should not match.
    """
    so = _series_order(include_bond=True)
    uv = _uncertainty_panel(so)

    ea = aggregate_ea_meu(uv, so)
    country = aggregate_country_meu(uv, so, countries=("AT", "DE"))

    # Average country MEUs across countries for each date-horizon
    avg_country = (
        country.groupby(["date", "horizon"], as_index=False)["meu"].mean()
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )

    # They should NOT be equal because the EA uses U2_BOND_001 too
    assert not np.allclose(
        ea["meu"].to_numpy(), avg_country["meu"].to_numpy()
    ), "EA MEU should differ from average of country MEUs when non-FX U2 series exist"


# ---------------------------------------------------------------------------
# Task-level test with synthetic parquet inputs
# ---------------------------------------------------------------------------


def _full_series_order() -> pd.DataFrame:
    """Series order covering all 19 MEU_COUNTRIES, FX, and one excluded bond."""
    rows = []
    pos = 0
    for country in MEU_COUNTRIES:
        rows.append((pos, f"{country}_VAR_001", country, "eurostat", "Macro"))
        pos += 1
    for i in range(1, 3):
        rows.append((pos, f"U2_FX_00{i}", "U2", "ecb", "EA_Financial"))
        pos += 1
    rows.append((pos, "U2_BOND_001", "U2", "ecb", "EA_Financial"))
    return pd.DataFrame(
        rows,
        columns=["series_position", "series_id", "country_iso2", "source",
                 "category_name"],
    )


def test_run_country_meu_aggregation_stage(tmp_path):
    so = _full_series_order()
    uv = _uncertainty_panel(so)
    so.to_parquet(tmp_path / "series_order.parquet", index=False)
    uv.to_parquet(tmp_path / "uncertainty_variance.parquet", index=False)

    produces = {
        "all_countries_meu": tmp_path / "countries" / "all_countries_meu.parquet",
        "basket_membership": tmp_path / "countries" / "basket_membership.parquet",
    }

    run_country_meu_aggregation_stage(
        depends_on={
            "uncertainty_variance": tmp_path / "uncertainty_variance.parquet",
            "series_order": tmp_path / "series_order.parquet",
        },
        produces=produces,
        panel_name="test_panel",
    )

    assert produces["all_countries_meu"].exists()
    assert produces["basket_membership"].exists()

    meu = pd.read_parquet(produces["all_countries_meu"])
    membership = pd.read_parquet(produces["basket_membership"])

    n_countries = 19
    n_dates = 2
    n_horizons = 2
    assert len(meu) == n_countries * n_dates * n_horizons
    assert meu["country_iso2"].nunique() == n_countries

    # Membership: FX rows appear once per country basket
    fx_rows = membership.loc[membership["series_id"] == "U2_FX_001"]
    assert len(fx_rows) == n_countries
    assert (fx_rows["basket_component"] == "common_fx").all()

    # Bond rows must NOT appear in any basket
    assert "U2_BOND_001" not in membership["series_id"].values
