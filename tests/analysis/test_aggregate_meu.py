import numpy as np
import pandas as pd
import pytest

from meu_replication.analysis.aggregate_meu import aggregate_ea_meu


def test_aggregate_ea_meu_matches_mean_of_square_root_variances():
    series_order = pd.DataFrame(
        {
            "series_position": [0, 1, 2],
            "series_id": ["A", "B", "C"],
        }
    )
    uncertainty = pd.DataFrame(
        {
            "date": ["2020-01", "2020-01", "2020-01", "2020-01", "2020-01", "2020-01"],
            "series_id": ["A", "B", "C", "A", "B", "C"],
            "horizon": [1, 1, 1, 2, 2, 2],
            "variance": [1.0, 4.0, 9.0, 4.0, 9.0, 16.0],
        }
    )

    result = aggregate_ea_meu(uncertainty, series_order)

    expected = pd.DataFrame(
        {
            "date": ["2020-01", "2020-01"],
            "horizon": pd.Series([1, 2], dtype="int16"),
            "meu": pd.Series([2.0, 3.0], dtype="float64"),
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_aggregate_ea_meu_rejects_missing_series_within_group():
    series_order = _series_order()
    uncertainty = _uncertainty_panel().iloc[:-1].copy()

    with pytest.raises(ValueError, match="exactly one row per persisted series"):
        aggregate_ea_meu(uncertainty, series_order)


def test_aggregate_ea_meu_rejects_duplicated_series_within_group():
    series_order = _series_order()
    uncertainty = pd.concat(
        [_uncertainty_panel(), _uncertainty_panel().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicated date-horizon-series rows"):
        aggregate_ea_meu(uncertainty, series_order)


def test_aggregate_ea_meu_rejects_unexpected_series_id():
    series_order = _series_order()
    uncertainty = _uncertainty_panel().copy()
    uncertainty.loc[0, "series_id"] = "Z"

    with pytest.raises(ValueError, match="unexpected series_id"):
        aggregate_ea_meu(uncertainty, series_order)


@pytest.mark.parametrize("bad_variance", [0.0, -1.0, np.inf, np.nan])
def test_aggregate_ea_meu_rejects_non_positive_or_non_finite_variance(bad_variance):
    series_order = _series_order()
    uncertainty = _uncertainty_panel().copy()
    uncertainty.loc[0, "variance"] = bad_variance

    expected_match = (
        "finite variance values"
        if not np.isfinite(bad_variance)
        else "strictly positive variances"
    )
    with pytest.raises(ValueError, match=expected_match):
        aggregate_ea_meu(uncertainty, series_order)


def _series_order() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_position": [0, 1],
            "series_id": ["A", "B"],
        }
    )


def _uncertainty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01", "2020-01", "2020-02", "2020-02"],
            "series_id": ["A", "B", "A", "B"],
            "horizon": [1, 1, 1, 1],
            "variance": [1.0, 4.0, 9.0, 16.0],
        }
    )
