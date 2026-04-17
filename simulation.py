"""Monte Carlo simulation engine for carbon and climate-economic risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from models import CostModelConfig, EmissionsModelConfig, estimate_economic_costs, project_emissions_path


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for Monte Carlo simulation runs."""

    years: int = 20
    n_simulations: int = 10_000
    reduction_percent: float = 20.0
    seed: int = 42
    emissions_model: EmissionsModelConfig = EmissionsModelConfig()
    cost_model: CostModelConfig = CostModelConfig()


def run_scenario(sim_config: SimulationConfig, reduction_percent: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Run one emissions reduction scenario and return simulation matrices."""
    reduction_fraction = reduction_percent / 100.0

    emissions = np.empty((sim_config.n_simulations, sim_config.years), dtype=float)
    costs = np.empty_like(emissions)

    for i in range(sim_config.n_simulations):
        path = project_emissions_path(sim_config.years, reduction_fraction, sim_config.emissions_model, rng)
        emissions[i] = path
        costs[i] = estimate_economic_costs(path, sim_config.cost_model, rng)

    return {"emissions": emissions, "costs": costs}


def run_monte_carlo(sim_config: SimulationConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """Run baseline and intervention scenarios."""
    seed_sequence = np.random.SeedSequence(sim_config.seed)
    base_seed, intervention_seed = seed_sequence.spawn(2)

    baseline_rng = np.random.default_rng(base_seed)
    intervention_rng = np.random.default_rng(intervention_seed)

    baseline = run_scenario(sim_config, reduction_percent=0.0, rng=baseline_rng)
    intervention = run_scenario(sim_config, reduction_percent=sim_config.reduction_percent, rng=intervention_rng)

    return {"baseline": baseline, "intervention": intervention}


def summarize_paths(paths: np.ndarray) -> pd.DataFrame:
    """Summarize simulation paths by year with central interval."""
    return pd.DataFrame(
        {
            "year": np.arange(1, paths.shape[1] + 1),
            "mean": np.mean(paths, axis=0),
            "p05": np.percentile(paths, 5, axis=0),
            "p95": np.percentile(paths, 95, axis=0),
        }
    )


def compare_scenarios(results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Compare baseline and intervention outcomes for emissions and costs."""
    baseline_emissions = results["baseline"]["emissions"].sum(axis=1)
    intervention_emissions = results["intervention"]["emissions"].sum(axis=1)

    baseline_costs = results["baseline"]["costs"].sum(axis=1)
    intervention_costs = results["intervention"]["costs"].sum(axis=1)

    return {
        "mean_total_emissions_baseline": float(np.mean(baseline_emissions)),
        "mean_total_emissions_intervention": float(np.mean(intervention_emissions)),
        "emissions_reduction_mean": float(np.mean(baseline_emissions - intervention_emissions)),
        "emissions_variance_baseline": float(np.var(baseline_emissions)),
        "emissions_variance_intervention": float(np.var(intervention_emissions)),
        "mean_total_cost_baseline": float(np.mean(baseline_costs)),
        "mean_total_cost_intervention": float(np.mean(intervention_costs)),
        "cost_reduction_mean": float(np.mean(baseline_costs - intervention_costs)),
    }
