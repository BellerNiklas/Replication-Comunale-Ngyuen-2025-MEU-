import pytest

from meu_replication.analysis.model_config import MEUConfig

EXPECTED_KMAX = 20
EXPECTED_IC = 2
EXPECTED_LAG = 4
EXPECTED_PZ = 2
EXPECTED_H_MAX = 12
EXPECTED_FAST_DRAWS = 5_000
EXPECTED_FAST_BURNIN = 5_000
EXPECTED_FAST_THIN = 2
EXPECTED_VALIDATION_CHAINS = 4
EXPECTED_VALIDATION_SUBSET = 12
EXPECTED_FULL_DRAWS = 50_000
EXPECTED_FULL_BURNIN = 50_000
EXPECTED_FULL_THIN = 10
EXPECTED_Y_SEED = 0
EXPECTED_F_SEED = 1_000


def test_meu_config_defaults_match_milestone_one():
    config = MEUConfig()
    assert config.panel_name == "panel_2003_2022_strict_corr"
    assert config.kmax == EXPECTED_KMAX
    assert config.ic_criterion == EXPECTED_IC
    assert config.py == EXPECTED_LAG
    assert config.pz == EXPECTED_PZ
    assert config.pf == EXPECTED_LAG
    assert config.threshold == pytest.approx(2.575)
    assert config.h_max == EXPECTED_H_MAX
    assert config.sv_mode == "fast"
    assert config.forecast_lag_order == EXPECTED_LAG


def test_meu_config_fast_mode_properties():
    config = MEUConfig(sv_mode="fast")
    assert config.sv_draws == EXPECTED_FAST_DRAWS
    assert config.sv_burnin == EXPECTED_FAST_BURNIN
    assert config.sv_thin_para == EXPECTED_FAST_THIN
    assert config.sv_thin_latent == EXPECTED_FAST_THIN
    assert config.sv_validation_chains == EXPECTED_VALIDATION_CHAINS
    assert config.sv_validation_subset_size == EXPECTED_VALIDATION_SUBSET
    assert config.sv_rhat_threshold == pytest.approx(1.05)
    assert config.sv_mu_rhat_threshold == pytest.approx(1.02)
    assert config.sv_geweke_threshold == pytest.approx(2.5)
    assert config.sv_geweke_p90_threshold == pytest.approx(5.0)
    assert config.sv_stability_relative_tolerance == pytest.approx(0.20)
    assert config.sv_seed_y == EXPECTED_Y_SEED
    assert config.sv_seed_f == EXPECTED_F_SEED


def test_meu_config_full_mode_properties():
    config = MEUConfig(sv_mode="full")
    assert config.sv_draws == EXPECTED_FULL_DRAWS
    assert config.sv_burnin == EXPECTED_FULL_BURNIN
    assert config.sv_thin_para == EXPECTED_FULL_THIN
    assert config.sv_thin_latent == EXPECTED_FULL_THIN


def test_meu_config_is_frozen():
    assert MEUConfig.__dataclass_params__.frozen is True
