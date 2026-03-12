import numpy as np

from meu_replication.analysis.forecast_errors import (
    generate_forecast_errors,
    mlags,
)


def test_mlags_matches_matlab_column_ordering():
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    result = mlags(x, k=2)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 10.0, 0.0, 0.0],
            [2.0, 20.0, 1.0, 10.0],
            [3.0, 30.0, 2.0, 20.0],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_generate_forecast_errors_trims_dates_and_zero_fills_excluded_coefficients():
    rng = np.random.default_rng(42)
    n_obs = 80
    dates = [
        f"{year}-{month:02d}" for year in range(2020, 2027) for month in range(1, 13)
    ][:n_obs]

    predictor = np.zeros(n_obs)
    series = np.zeros((n_obs, 2))
    for idx in range(2, n_obs):
        predictor[idx] = (
            0.6 * predictor[idx - 1] - 0.1 * predictor[idx - 2] + rng.normal(scale=0.3)
        )
        series[idx, 0] = (
            0.5 * series[idx - 1, 0] + 1.8 * predictor[idx - 1] + rng.normal(scale=0.2)
        )
        series[idx, 1] = (
            0.4 * series[idx - 1, 1] + 1.5 * predictor[idx - 1] + rng.normal(scale=0.2)
        )

    result = generate_forecast_errors(
        standardized_y=series,
        predictor_values=predictor.reshape(-1, 1),
        dates=dates,
        predictor_names=["F1"],
        py=4,
        pz=2,
        pf=4,
        threshold=2.575,
    )

    assert result.y_dates[0] == dates[4]
    assert result.f_dates[0] == dates[4]
    assert result.y_residuals.shape == (76, 2)
    assert result.f_residuals.shape == (76, 1)
    assert result.selection_mask.shape == (2, 2)
    assert result.selection_mask.shape[1] == 2 * 1
    assert result.selection_mask[:, 0].all()
    excluded = ~result.selection_mask
    predictor_beta_block = result.y_coefficients[:, 5:]
    assert np.all(predictor_beta_block[excluded] == 0.0)
