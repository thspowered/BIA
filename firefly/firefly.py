"""Firefly algorithm for continuous optimization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from differential_evolution.de import _validate_bounds


Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class FireflyConfig:
    population_size: int = 2
    iterations: int = 1
    alpha: float = 0.3            # náhodný krok (0..1)
    beta0: float = 1.0            # počiatočná príťažlivosť
    gamma: float = 1.0            # absorpčný koeficient pre vzdialenosť
    seed: int | None = None


@dataclass
class FireflyResult:
    best_point: np.ndarray
    best_value: float
    positions_history: np.ndarray 
    fitness_history: np.ndarray    
    best_fitness_curve: np.ndarray


def firefly_optimize(objective: Objective, bounds: Bounds, config: FireflyConfig | None = None) -> FireflyResult:
    """Spustí Firefly algoritmus pre hľadanie minima."""
    cfg = config or FireflyConfig()

    if cfg.population_size < 2:
        raise ValueError("population_size musí byť aspoň 2.")
    if cfg.iterations <= 0:
        raise ValueError("iterations musí byť kladné číslo.")
    if cfg.alpha < 0:
        raise ValueError("alpha musí byť nezáporná.")
    if cfg.beta0 <= 0:
        raise ValueError("beta0 musí byť kladné.")
    if cfg.gamma < 0:
        raise ValueError("gamma musí byť nezáporná.")

    low, high = _validate_bounds(bounds)
    dims = len(bounds)
    span = high - low

    rng = np.random.default_rng(cfg.seed)
    positions = rng.uniform(low, high, size=(cfg.population_size, dims))
    fitness = np.apply_along_axis(objective, 1, positions)

    positions_hist = [positions.copy()]
    fitness_hist = [fitness.copy()]

    best_idx = int(np.argmin(fitness))
    best_point = positions[best_idx].copy()
    best_value = float(fitness[best_idx])
    best_curve = [best_value]

    for _ in range(cfg.iterations):
        for i in range(cfg.population_size):
            xi = positions[i].copy()
            fi = fitness[i]

            for j in range(cfg.population_size):
                if fitness[j] < fi:
                    diff = positions[j] - xi
                    r = np.linalg.norm(diff)
                    beta = cfg.beta0 / (1.0 + r)
                    xi += beta * diff

            # náhodný krok s Gaussovským šumom
            noise = rng.normal(0.0, 1.0, size=dims)
            xi += cfg.alpha * noise * span

            xi = np.clip(xi, low, high)
            positions[i] = xi

        fitness = np.apply_along_axis(objective, 1, positions)

        positions_hist.append(positions.copy())
        fitness_hist.append(fitness.copy())

        current_best_idx = int(np.argmin(fitness))
        current_best_value = float(fitness[current_best_idx])
        if current_best_value < best_value:
            best_value = current_best_value
            best_point = positions[current_best_idx].copy()
        best_curve.append(best_value)

    return FireflyResult(
        best_point=best_point,
        best_value=best_value,
        positions_history=np.stack(positions_hist),
        fitness_history=np.stack(fitness_hist),
        best_fitness_curve=np.array(best_curve),
    )
