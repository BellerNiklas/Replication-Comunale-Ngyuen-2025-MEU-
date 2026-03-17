import numpy as np
import pandas as pd

from meu_replication.analysis.uncertainty import (
    _companion_recursion_scalar_uncertainty,
    build_combined_companion_matrix,
    build_predictor_companion_matrix,
    build_series_transition_blocks,
    compute_irf_squares,
    compute_scalar_uncertainty_from_irf,
    expected_variance,
)


def test_expected_variance_matches_direct_jln_formula():
    alpha = 0.2
    beta = 0.5
    tau2 = 0.3
    x_t = np.array([-1.0, 0.0, 1.0])
    h = 3

    expected = np.exp(
        alpha * (1.0 + 0.5 + 0.25)
        + 0.5 * tau2 * (1.0 + 0.25 + 0.0625)
        + np.power(beta, h) * x_t
    )

    result = expected_variance(alpha, beta, tau2, x_t, h)

    np.testing.assert_allclose(result, expected)


def test_expected_variance_is_stable_near_unit_root():
    alpha = 0.1
    beta = 1.0 - 1e-12
    tau2 = 0.2
    x_t = np.array([0.5, -0.25])
    h = 12

    expected = np.exp(
        alpha * np.sum(np.power(beta, np.arange(h)))
        + 0.5 * tau2 * np.sum(np.power(beta**2, np.arange(h)))
        + np.power(beta, h) * x_t
    )

    result = expected_variance(alpha, beta, tau2, x_t, h)

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected)


def test_build_predictor_companion_matrix_matches_matlab_layout():
    regression_coefs_f = pd.DataFrame(
        {
            "predictor_position": [0, 1],
            "predictor_name": ["F1", "G1"],
            "const": [0.0, 0.0],
            "lag1": [0.1, 0.2],
            "lag2": [0.3, 0.4],
            "lag3": [0.5, 0.6],
        }
    )

    result = build_predictor_companion_matrix(regression_coefs_f, pf=3)
    expected = np.array(
        [
            [0.1, 0.0, 0.3, 0.0, 0.5, 0.0],
            [0.0, 0.2, 0.0, 0.4, 0.0, 0.6],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_build_series_transition_blocks_preserves_exact_coefficient_order():
    y_coefficients = pd.Series(
        {
            "const": 0.0,
            "y_lag1": 0.7,
            "y_lag2": -0.2,
            "y_lag3": 0.1,
            "F1_lag1": 1.0,
            "F2_lag1": 2.0,
            "G1_lag1": 3.0,
            "F1_lag2": 4.0,
            "F2_lag2": 5.0,
            "G1_lag2": 6.0,
        }
    )

    lambda_matrix, phiy = build_series_transition_blocks(
        y_coefficients,
        predictor_names=["F1", "F2", "G1"],
        py=3,
        pz=2,
        pf=4,
    )

    expected_lambda = np.zeros((3, 12))
    expected_lambda[0, :6] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected_phiy = np.array(
        [
            [0.7, -0.2, 0.1],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    np.testing.assert_allclose(lambda_matrix, expected_lambda)
    np.testing.assert_allclose(phiy, expected_phiy)


def test_irf_engine_matches_companion_recursion_for_ar_only_system():
    phi = np.array(
        [
            [0.5, -0.2],
            [1.0, 0.0],
        ]
    )
    shock_variance_paths = [
        np.array([[1.2], [0.8]]),
        np.array([[1.4], [0.9]]),
        np.array([[1.6], [1.0]]),
    ]

    irf_squares = compute_irf_squares(
        phi,
        y_position=0,
        shock_positions=[0],
        h_max=len(shock_variance_paths),
    )
    irf_result = compute_scalar_uncertainty_from_irf(
        irf_squares=irf_squares,
        shock_variance_paths=shock_variance_paths,
    )
    recursion_result = _companion_recursion_scalar_uncertainty(
        phi=phi,
        y_position=0,
        shock_positions=[0],
        shock_variance_paths=shock_variance_paths,
    )

    np.testing.assert_allclose(irf_result, recursion_result)


def test_irf_engine_matches_companion_recursion_for_predictor_plus_y_system():
    regression_coefs_f = pd.DataFrame(
        {
            "predictor_position": [0, 1],
            "predictor_name": ["F1", "G1"],
            "const": [0.0, 0.0],
            "lag1": [0.4, 0.2],
            "lag2": [0.1, -0.1],
        }
    )
    phif = build_predictor_companion_matrix(regression_coefs_f, pf=2)
    lambda_matrix, phiy = build_series_transition_blocks(
        pd.Series(
            {
                "const": 0.0,
                "y_lag1": 0.3,
                "y_lag2": -0.2,
                "F1_lag1": 0.5,
                "G1_lag1": -0.4,
            }
        ),
        predictor_names=["F1", "G1"],
        py=2,
        pz=1,
        pf=2,
    )
    phi = build_combined_companion_matrix(
        phif=phif,
        lambda_matrix=lambda_matrix,
        phiy=phiy,
    )
    shock_variance_paths = [
        np.array([[0.8, 0.6, 1.2], [0.7, 0.5, 1.1]]),
        np.array([[0.9, 0.7, 1.3], [0.8, 0.6, 1.2]]),
        np.array([[1.0, 0.8, 1.4], [0.9, 0.7, 1.3]]),
    ]
    y_position = 4
    shock_positions = [0, 1, y_position]

    irf_squares = compute_irf_squares(
        phi,
        y_position=y_position,
        shock_positions=shock_positions,
        h_max=len(shock_variance_paths),
    )
    irf_result = compute_scalar_uncertainty_from_irf(
        irf_squares=irf_squares,
        shock_variance_paths=shock_variance_paths,
    )
    recursion_result = _companion_recursion_scalar_uncertainty(
        phi=phi,
        y_position=y_position,
        shock_positions=shock_positions,
        shock_variance_paths=shock_variance_paths,
    )

    np.testing.assert_allclose(irf_result, recursion_result)
