"""Core emission and economic damage models for carbon risk simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmissionsModelConfig:
    """Configuration for annual emissions projection dynamics."""

    baseline_emissions: float = 5_000_000.0
    trend_mean: float = 0.01
    trend_std: float = 0.01
    shock_std: float = 0.02
    shock_distribution: str = "normal"


@dataclass(frozen=True)
class CostModelConfig:
    """Configuration for translating emissions into economic damages."""

    damage_factor_mean: float = 40.0
    damage_factor_std: float = 8.0


def _sample_shock(rng: np.random.Generator, std: float, distribution: str) -> float:
    """Sample an annual stochastic shock applied to emissions growth."""
    if distribution == "normal":
        return float(rng.normal(0.0, std))
    if distribution == "lognormal":
        return float(np.exp(rng.normal(-0.5 * std**2, std)) - 1.0)
    raise ValueError("shock_distribution must be either 'normal' or 'lognormal'")


def project_emissions_path(
    years: int,
    reduction_fraction: float,
    config: EmissionsModelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Project one stochastic emissions path over a number of years."""
    if years <= 0:
        raise ValueError("years must be positive")
    if not 0.0 <= reduction_fraction <= 1.0:
        raise ValueError("reduction_fraction must be between 0 and 1")

    emissions = np.empty(years, dtype=float)
    emissions[0] = config.baseline_emissions

    for t in range(1, years):
        trend = rng.normal(config.trend_mean, config.trend_std)
        shock = _sample_shock(rng, config.shock_std, config.shock_distribution)
        annual_change = trend + shock - reduction_fraction
        emissions[t] = max(0.0, emissions[t - 1] * (1.0 + annual_change))

    return emissions


def estimate_economic_costs(
    emissions_path: np.ndarray,
    cost_config: CostModelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate yearly economic cost from emissions with uncertain damage factor."""
    damage_factor = max(0.0, float(rng.normal(cost_config.damage_factor_mean, cost_config.damage_factor_std)))
    return emissions_path * damage_factor
