"""Focused tests for Monte Carlo simulation behavior."""

import unittest

from simulation import SimulationConfig, compare_scenarios, run_monte_carlo


class SimulationTestCase(unittest.TestCase):
    def test_output_shapes_match_configuration(self) -> None:
        config = SimulationConfig(years=12, n_simulations=300, reduction_percent=15.0, seed=7)
        results = run_monte_carlo(config)

        self.assertEqual(results["baseline"]["emissions"].shape, (300, 12))
        self.assertEqual(results["intervention"]["emissions"].shape, (300, 12))
        self.assertEqual(results["baseline"]["costs"].shape, (300, 12))

    def test_intervention_reduces_average_total_emissions(self) -> None:
        config = SimulationConfig(years=20, n_simulations=500, reduction_percent=20.0, seed=9)
        results = run_monte_carlo(config)
        metrics = compare_scenarios(results)

        self.assertGreater(metrics["mean_total_emissions_baseline"], metrics["mean_total_emissions_intervention"])
        self.assertGreater(metrics["mean_total_cost_baseline"], metrics["mean_total_cost_intervention"])


if __name__ == "__main__":
    unittest.main()
