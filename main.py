"""CLI entry point for the Nairobi Carbon Risk Simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models import CostModelConfig, EmissionsModelConfig
from simulation import SimulationConfig, compare_scenarios, run_monte_carlo, summarize_paths
from visualization import plot_final_outcome_histogram, plot_mean_with_interval


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Carbon Risk Simulator (Monte Carlo)")
    parser.add_argument("--years", type=int, default=20, help="Projection horizon in years")
    parser.add_argument("--simulations", type=int, default=10_000, help="Number of Monte Carlo simulations")
    parser.add_argument("--reduction", type=float, default=20.0, help="Annual emissions reduction percentage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--baseline-emissions", type=float, default=5_000_000.0, help="Starting annual emissions")
    parser.add_argument("--trend-mean", type=float, default=0.01, help="Mean annual emissions trend")
    parser.add_argument("--trend-std", type=float, default=0.01, help="Standard deviation of annual trend")
    parser.add_argument("--shock-std", type=float, default=0.02, help="Standard deviation for stochastic shocks")
    parser.add_argument(
        "--shock-distribution",
        choices=["normal", "lognormal"],
        default="normal",
        help="Shock distribution type",
    )
    parser.add_argument("--damage-factor-mean", type=float, default=40.0, help="Mean economic damage per unit emission")
    parser.add_argument("--damage-factor-std", type=float, default=8.0, help="Std. dev. of economic damage factor")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for generated plots and exports")
    parser.add_argument("--export-csv", action="store_true", help="Export yearly scenario summaries to CSV")
    return parser.parse_args()


def main() -> None:
    """Run simulation, print key metrics, and generate visual outputs."""
    args = parse_args()

    emissions_model = EmissionsModelConfig(
        baseline_emissions=args.baseline_emissions,
        trend_mean=args.trend_mean,
        trend_std=args.trend_std,
        shock_std=args.shock_std,
        shock_distribution=args.shock_distribution,
    )
    cost_model = CostModelConfig(
        damage_factor_mean=args.damage_factor_mean,
        damage_factor_std=args.damage_factor_std,
    )
    config = SimulationConfig(
        years=args.years,
        n_simulations=args.simulations,
        reduction_percent=args.reduction,
        seed=args.seed,
        emissions_model=emissions_model,
        cost_model=cost_model,
    )

    results = run_monte_carlo(config)
    comparison = compare_scenarios(results)

    baseline_summary = summarize_paths(results["baseline"]["emissions"])
    intervention_summary = summarize_paths(results["intervention"]["emissions"])

    output_dir = Path(args.output_dir)
    trajectory_path = plot_mean_with_interval(
        baseline=results["baseline"]["emissions"],
        intervention=results["intervention"]["emissions"],
        output_dir=output_dir,
    )
    histogram_path = plot_final_outcome_histogram(
        baseline=results["baseline"]["emissions"],
        intervention=results["intervention"]["emissions"],
        output_dir=output_dir,
    )

    if args.export_csv:
        summary_df = pd.DataFrame(
            {
                "year": baseline_summary["year"],
                "baseline_mean": baseline_summary["mean"],
                "baseline_p05": baseline_summary["p05"],
                "baseline_p95": baseline_summary["p95"],
                "intervention_mean": intervention_summary["mean"],
                "intervention_p05": intervention_summary["p05"],
                "intervention_p95": intervention_summary["p95"],
            }
        )
        summary_df.to_csv(output_dir / "yearly_summary.csv", index=False)

    print("Carbon Risk Simulation Results (Nairobi default)")
    print(f"- Simulations: {args.simulations:,}")
    print(f"- Years: {args.years}")
    print(f"- Intervention reduction: {args.reduction:.1f}%")
    print(f"- Mean total emissions (baseline): {comparison['mean_total_emissions_baseline']:,.2f}")
    print(f"- Mean total emissions (intervention): {comparison['mean_total_emissions_intervention']:,.2f}")
    print(f"- Mean emissions reduced: {comparison['emissions_reduction_mean']:,.2f}")
    print(f"- Emissions variance baseline: {comparison['emissions_variance_baseline']:,.2f}")
    print(f"- Emissions variance intervention: {comparison['emissions_variance_intervention']:,.2f}")
    print(f"- Mean total cost (baseline): {comparison['mean_total_cost_baseline']:,.2f}")
    print(f"- Mean total cost (intervention): {comparison['mean_total_cost_intervention']:,.2f}")
    print(f"- Mean cost reduced: {comparison['cost_reduction_mean']:,.2f}")
    print(f"- Saved line/CI plot: {trajectory_path}")
    print(f"- Saved final outcome histogram: {histogram_path}")


if __name__ == "__main__":
    main()
