from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .nsga2 import NSGA2Result


def plot_pareto(result: NSGA2Result, save_path: Path | None = None) -> None:
    population = result.population
    objectives = result.objectives
    pareto_idx = result.pareto_front

    pareto_decisions = population[pareto_idx]
    pareto_objectives = objectives[pareto_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(objectives[:, 0], objectives[:, 1], color="lightgray", s=20, label="Solutions")
    ax.scatter(
        pareto_objectives[:, 0],
        pareto_objectives[:, 1],
        color="red",
        s=40,
        label="Pareto front",
    )
    ax.set_xlabel("S (lateral area)")
    ax.set_ylabel("T (total area)")
    ax.set_title("Objective space")
    ax.legend()

    ax = axes[1]
    ax.scatter(population[:, 0], population[:, 1], color="lightgray", s=20, label="Population")
    ax.scatter(
        pareto_decisions[:, 0],
        pareto_decisions[:, 1],
        color="red",
        s=40,
        label="Pareto set",
    )
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Height h")
    ax.set_title("Decision space")
    ax.legend()

    fig.suptitle("NSGA-II on Cone design problem")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()
