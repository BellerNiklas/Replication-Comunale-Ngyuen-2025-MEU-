import numpy as np
import pytest

from meu_replication.analysis.newey_west import (
    compute_newey_west_bandwidth,
    newey_west,
)

EXPECTED_BANDWIDTH = 4


def test_compute_newey_west_bandwidth_matches_jln_rule():
    assert compute_newey_west_bandwidth(239) == EXPECTED_BANDWIDTH


def test_newey_west_matches_ols_when_nlag_zero():
    x_core = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 1.5],
            [5.0, 1.0],
        ]
    )
    x = np.column_stack((np.ones(len(x_core)), x_core))
    beta_true = np.array([1.5, -0.25, 0.8])
    y = x @ beta_true + np.array([0.2, -0.1, 0.0, 0.05, -0.05, 0.1])

    result = newey_west(y, x, nlag=0)
    beta_ols, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    np.testing.assert_allclose(result.beta, beta_ols)
    np.testing.assert_allclose(result.resid, y - x @ beta_ols)
    assert result.covariance.shape == (3, 3)


def test_newey_west_rejects_underdetermined_regression():
    y = np.array([1.0, 2.0, 3.0])
    x = np.eye(3)

    with pytest.raises(ValueError, match="more observations"):
        newey_west(y, x, nlag=0)
