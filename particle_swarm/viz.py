"""Vizualizačné pomôcky pre Particle Swarm Optimization."""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from visualize import plot_surface_3d, mark_best
from .pso import PSOResult


Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


def plot_pso_search(
    objective: Objective,
    bounds: Bounds,
    result: PSOResult,
    elev: int = 35,
    azim: int = -60,
    cmap: str = "viridis",
    heatmap_cmap: str = "coolwarm",
    heatmap_grid: int = 250,
) -> plt.Figure:
    """Vykreslí 3D povrch a 2D heatmapu s trajektóriou častíc."""
    if result.position_history.shape[2] != 2:
        raise ValueError("Vizualizácia je dostupná iba pre 2D problémy.")

    fig = plt.figure(figsize=(12, 5))
    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    plot_surface_3d(objective, bounds, grid_size=160, elev=elev, azim=azim, ax=ax_surface)

    history = result.position_history
    pop_size = history.shape[1]

    flat_positions = history.reshape(-1, 2)
    generations = np.repeat(np.arange(history.shape[0]), pop_size)
    z_values = np.apply_along_axis(objective, 1, flat_positions)

    scatter = ax_surface.scatter(
        flat_positions[:, 0],
        flat_positions[:, 1],
        z_values,
        c=generations,
        cmap=cmap,
        s=15,
        alpha=0.35,
    )

    last_positions = history[-1]
    last_values = np.apply_along_axis(objective, 1, last_positions)
    ax_surface.scatter(
        last_positions[:, 0],
        last_positions[:, 1],
        last_values,
        color="black",
        s=40,
        alpha=0.7,
        marker="o",
        edgecolor="white",
        linewidths=0.6,
    )

    mark_best(ax_surface, result.best_point, objective, label=f"best={result.best_value:.4f}")
    ax_surface.set_title("PSO priebeh na 3D povrchu", fontsize=11)

    cbar = fig.colorbar(scatter, ax=ax_surface, pad=0.12, shrink=0.65)
    cbar.set_label("Iterácia")

    # 2D heatmapa s trajektóriou častíc
    ax_heat = fig.add_subplot(1, 2, 2)
    x_lin = np.linspace(bounds[0][0], bounds[0][1], heatmap_grid)
    y_lin = np.linspace(bounds[1][0], bounds[1][1], heatmap_grid)
    X, Y = np.meshgrid(x_lin, y_lin)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.apply_along_axis(objective, 1, grid_points).reshape(X.shape)

    heat = ax_heat.contourf(X, Y, Z, levels=60, cmap=heatmap_cmap)
    norm = Normalize(vmin=0, vmax=max(1, history.shape[0] - 1))
    cmap_obj = plt.get_cmap(cmap)

    # Trajektória každého jedinca v rovine XY
    particle_colors = cmap_obj(np.linspace(0, 1, pop_size))
    for particle_idx, color in enumerate(particle_colors):
        path = history[:, particle_idx, :]
        ax_heat.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            alpha=0.55,
            linewidth=1.2,
        )

    ax_heat.scatter(
        flat_positions[:, 0],
        flat_positions[:, 1],
        c=cmap_obj(norm(generations)),
        s=10,
        alpha=0.25,
        edgecolors="none",
    )

    ax_heat.scatter(
        last_positions[:, 0],
        last_positions[:, 1],
        color="black",
        s=35,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.6,
    )

    ax_heat.scatter(
        [result.best_point[0]],
        [result.best_point[1]],
        color="yellow",
        s=110,
        marker="*",
        edgecolors="black",
        linewidths=0.7,
        zorder=5,
    )

    ax_heat.set_title("PSO heatmapa a trajektórie", fontsize=11)
    ax_heat.set_xlabel("x1")
    ax_heat.set_ylabel("x2")
    ax_heat.set_aspect("equal", adjustable="box")

    heat_cbar = fig.colorbar(heat, ax=ax_heat, pad=0.02, shrink=0.85)
    heat_cbar.set_label("f(x)")

    fig.tight_layout()
    return fig
