import numpy as np
import pandas as pd
import pytest

from meu_replication.analysis.factor_estimation import (
    estimate_predictor_set,
    extract_static_factors,
    prepare_analysis_panel,
    select_factor_count,
    standardize_columns,
)


def _make_balanced_panel() -> pd.DataFrame:
    dates = ["2020-03", "2020-01", "2020-02", "2020-04", "2020-05", "2020-06"]
    series_values = {
        "B_SER": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "A_SER": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "C_SER": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0],
    }

    rows = []
    for series_id, values in series_values.items():
        for date, value in zip(dates, values):
            rows.append(
                {
                    "date": date,
                    "value": value,
                    "series_id": series_id,
                    "country_iso2": "DE",
                    "variable_name": f"name_{series_id}",
                    "category": 1,
                    "category_name": "test",
                    "source": "test",
                    "transformationcode": 1,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_analysis_panel_sorts_rows_and_columns():
    wide, series_order, date_index = prepare_analysis_panel(_make_balanced_panel())

    assert list(wide.index) == [
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        "2020-05",
        "2020-06",
    ]
    assert list(wide.columns) == ["A_SER", "B_SER", "C_SER"]
    assert list(series_order["series_id"]) == ["A_SER", "B_SER", "C_SER"]
    assert date_index["date"].tolist() == list(wide.index)
    assert date_index["date_position"].tolist() == [0, 1, 2, 3, 4, 5]


def test_prepare_analysis_panel_raises_for_unbalanced_panel():
    panel = _make_balanced_panel()
    result = panel[
        ~((panel["series_id"] == "A_SER") & (panel["date"] == "2020-06"))
    ].copy()

    with pytest.raises(ValueError, match="balanced"):
        prepare_analysis_panel(result)


def test_standardize_columns_matches_ddof_one():
    x = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    standardized, means, stds = standardize_columns(x)

    np.testing.assert_allclose(means, np.array([2.0, 4.0]))
    np.testing.assert_allclose(stds, np.array([1.0, 2.0]))
    np.testing.assert_allclose(standardized.mean(axis=0), np.zeros(2))
    np.testing.assert_allclose(standardized.std(axis=0, ddof=1), np.ones(2))


def test_select_factor_count_on_rank_one_matrix():
    time_factor = np.arange(1.0, 13.0).reshape(-1, 1)
    loadings = np.array([[1.0, 2.0, 3.0]])
    x = time_factor @ loadings
    standardized, _, _ = standardize_columns(x)

    assert select_factor_count(standardized, kmax=3, ic_criterion=2) == 1


def test_extract_static_factors_handles_t_less_than_n():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 8))
    standardized, _, _ = standardize_columns(x)

    fhat, loadings, eigenvalues = extract_static_factors(standardized, n_factors=2)

    assert fhat.shape == (5, 2)
    assert loadings.shape == (8, 2)
    assert eigenvalues.shape == (5,)


def test_extract_static_factors_produces_orthogonal_scores():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(14, 4))
    standardized, _, _ = standardize_columns(x)

    fhat, _, _ = extract_static_factors(standardized, n_factors=2)
    gram = fhat.T @ fhat / standardized.shape[0]

    np.testing.assert_allclose(
        gram - np.diag(np.diag(gram)),
        np.zeros((2, 2)),
        atol=1e-6,
    )


def test_estimate_predictor_set_has_expected_shapes():
    rng = np.random.default_rng(123)
    latent = np.linspace(-2.0, 2.0, 20).reshape(-1, 1)
    loadings = np.array([[1.0, 0.8, -0.5, 1.2]])
    x = latent @ loadings + 0.05 * rng.normal(size=(20, 4))

    result = estimate_predictor_set(x, kmax=4, ic_criterion=2)

    assert result.factor_count >= 1
    assert result.squared_factor_count >= 1
    assert result.fhat.shape == (20, result.factor_count)
    assert result.ghat.shape == (20, result.squared_factor_count)
    assert result.predictor_set.shape == (20, result.factor_count + 2)
    np.testing.assert_allclose(
        result.predictor_set[:, result.factor_count],
        result.fhat[:, 0] ** 2,
    )
    np.testing.assert_allclose(
        result.predictor_set[:, result.factor_count + 1],
        result.ghat[:, 0],
    )
