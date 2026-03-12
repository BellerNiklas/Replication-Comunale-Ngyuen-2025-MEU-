import pytest

from meu_replication.analysis.model_config import MEUConfig


def test_meu_config_defaults_match_milestone_one():
    config = MEUConfig()
    assert config.panel_name == "panel_2003_2022_strict_corr"
    assert config.kmax == 20
    assert config.ic_criterion == 2
    assert config.py == 4
    assert config.pz == 2
    assert config.pf == 4
    assert config.threshold == pytest.approx(2.575)
    assert config.h_max == 12
    assert config.sv_mode == "fast"
    assert config.forecast_lag_order == 4


def test_meu_config_fast_mode_properties():
    config = MEUConfig(sv_mode="fast")
    assert config.sv_draws == 5_000
    assert config.sv_burnin == 2_000
    assert config.sv_thin_para == 1
    assert config.sv_thin_latent == 1


def test_meu_config_full_mode_properties():
    config = MEUConfig(sv_mode="full")
    assert config.sv_draws == 50_000
    assert config.sv_burnin == 50_000
    assert config.sv_thin_para == 10
    assert config.sv_thin_latent == 10


def test_meu_config_is_frozen():
    assert MEUConfig.__dataclass_params__.frozen is True
