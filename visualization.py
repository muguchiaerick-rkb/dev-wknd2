"""Visualization utilities for carbon risk simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_mean_with_interval(
    baseline: np.ndarray,
    intervention: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Plot mean emissions and 5th-95th percentile interval over time."""
    output_path = _ensure_output_dir(output_dir) / "emissions_trajectory.png"
    years = np.arange(1, baseline.shape[1] + 1)

    plt.figure(figsize=(10, 5))

    for label, matrix, color in [
        ("Baseline", baseline, "#1f77b4"),
        ("Intervention", intervention, "#2ca02c"),
    ]:
        mean = np.mean(matrix, axis=0)
        p05 = np.percentile(matrix, 5, axis=0)
        p95 = np.percentile(matrix, 95, axis=0)

        plt.plot(years, mean, label=f"{label} mean", color=color)
        plt.fill_between(years, p05, p95, color=color, alpha=0.2, label=f"{label} 5-95%")

    plt.title("Nairobi Emissions Projection: Baseline vs Intervention")
    plt.xlabel("Year")
    plt.ylabel("Emissions (metric tons)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_final_outcome_histogram(
    baseline: np.ndarray,
    intervention: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Plot histogram of final-year emissions for scenario comparison."""
    output_path = _ensure_output_dir(output_dir) / "final_year_histogram.png"

    plt.figure(figsize=(10, 5))
    plt.hist(baseline[:, -1], bins=40, alpha=0.5, label="Baseline", color="#1f77b4")
    plt.hist(intervention[:, -1], bins=40, alpha=0.5, label="Intervention", color="#2ca02c")
    plt.title("Distribution of Final-Year Emissions")
    plt.xlabel("Final-year emissions (metric tons)")
    plt.ylabel("Frequency")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
