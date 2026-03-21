"""Configuration for the MEU analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SVMode = Literal["fast", "full"]


@dataclass(frozen=True)
class MEUConfig:
    """Frozen configuration for the baseline MEU pipeline."""

    panel_name: str = "panel_2003_2022_strict_corr"

    # Factor estimation
    kmax: int = 20
    ic_criterion: int = 2

    # Forecast equations
    py: int = 4
    pz: int = 2
    pf: int = 4
    threshold: float = 2.575

    # Uncertainty horizons
    h_max: int = 12

    # Stochastic volatility
    sv_mode: SVMode = "fast"
    sv_draws_fast: int = 5_000
    sv_burnin_fast: int = 5_000
    sv_thin_para_fast: int = 2
    sv_thin_latent_fast: int = 2
    sv_draws_full: int = 50_000
    sv_burnin_full: int = 50_000
    sv_thin_para_full: int = 10
    sv_thin_latent_full: int = 10
    sv_offset: float = 1e-5
    sv_seed_y: int = 0
    sv_seed_f: int = 1_000
    sv_seed: int = 42
    sv_validation_chains: int = 4
    sv_validation_subset_size: int = 12
    sv_rhat_threshold: float = 1.05
    sv_mu_rhat_threshold: float = 1.02
    sv_geweke_threshold: float = 2.6
    sv_geweke_p90_threshold: float = 5.0
    sv_stability_relative_tolerance: float = 0.20

    # Parallelism
    num_workers: int = -1

    @property
    def forecast_lag_order(self) -> int:
        """Maximum lag order used in the forecast stage."""
        return max(self.py, self.pz)

    @property
    def sv_draws(self) -> int:
        """Number of retained MCMC draws."""
        return self.sv_draws_full if self.sv_mode == "full" else self.sv_draws_fast

    @property
    def sv_burnin(self) -> int:
        """Number of burn-in iterations."""
        return self.sv_burnin_full if self.sv_mode == "full" else self.sv_burnin_fast

    @property
    def sv_thin_para(self) -> int:
        """Parameter thinning interval."""
        return (
            self.sv_thin_para_full if self.sv_mode == "full" else self.sv_thin_para_fast
        )

    @property
    def sv_thin_latent(self) -> int:
        """Latent-state thinning interval."""
        return (
            self.sv_thin_latent_full
            if self.sv_mode == "full"
            else self.sv_thin_latent_fast
        )
