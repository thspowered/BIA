from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (potrebné pre 3D projekciu)

# Alias typov pre čitateľnosť
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


def plot_surface_3d(objective: Objective, bounds: Bounds, grid_size: int = 200, elev: int = 35, azim: int = -60):
    """Vypočíta a vykreslí 3D povrch funkcie nad 2D mriežkou v daných hraniciach."""
    x_lin = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y_lin = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(X)
    for i in range(grid_size):
        pts = np.stack([X[i], Y[i]], axis=1)
        Z[i] = np.apply_along_axis(objective, 1, pts)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    # Nižšia opacita (alpha) pre lepšiu čitateľnosť trajektórie a značiek
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap="coolwarm", linewidth=0.2, alpha=0.4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    return fig, ax


def overlay_trajectory(ax, trajectory: np.ndarray, objective: Objective | None = None, every: int = 1):
    """Prekreslí na povrch trajektóriu (poradie navštívených bodov)."""
    xyz = trajectory[:, :2]
    if objective is not None:
        z = np.apply_along_axis(objective, 1, xyz)
    else:
        z = np.arange(xyz.shape[0])
    ax.plot(xyz[::every, 0], xyz[::every, 1], z[::every], marker=".", color="k", linewidth=0.5, markersize=3, alpha=0.5)
    if objective is not None:
        best_idx = int(np.argmin(z))
        ax.scatter([xyz[best_idx, 0]], [xyz[best_idx, 1]], [z[best_idx]], color="black", s=40)
    return ax


def mark_best(ax, point: np.ndarray, objective: Objective, label: str | None = None, color: str = "black"):
    """Zvýrazní najlepší bod hviezdou a pridá čitateľný popis s hodnotou f(x)."""
    x, y = float(point[0]), float(point[1])
    z = float(objective(np.array([x, y])))
    ax.scatter([x], [y], [z], color=color, s=140, marker="*", edgecolor="white", linewidths=0.8, zorder=10)
    if label is None:
        label = f"best={z:.4f}"
    ax.text(x, y, z, f"  {label}", color=color, fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
    return ax
