"""Teaching-Learning Based Optimization (TLBO)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from differential_evolution.de import _validate_bounds


Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class TLBOConfig:
    population_size: int = 30
    iterations: int = 200
    seed: int | None = None


@dataclass
class TLBOResult:
    best_point: np.ndarray
    best_value: float
    population_history: np.ndarray
    fitness_history: np.ndarray
    best_fitness_curve: np.ndarray


def tlbo_optimize(objective: Objective, bounds: Bounds, config: TLBOConfig | None = None) -> TLBOResult:
    cfg = config or TLBOConfig()

    if cfg.population_size < 2:
        raise ValueError("population_size musí byť aspoň 2.")
    if cfg.iterations <= 0:
        raise ValueError("iterations musí byť kladné číslo.")

    low, high = _validate_bounds(bounds)
    dims = len(bounds)

    rng = np.random.default_rng(cfg.seed)
    population = rng.uniform(low, high, size=(cfg.population_size, dims))
    fitness = np.apply_along_axis(objective, 1, population)

    pop_history = [population.copy()]
    fit_history = [fitness.copy()]

    best_idx = int(np.argmin(fitness))
    best_point = population[best_idx].copy()
    best_value = float(fitness[best_idx])
    best_curve = [best_value]

    for _ in range(cfg.iterations):
        mean = np.mean(population, axis=0)
        teacher_idx = int(np.argmin(fitness))
        teacher = population[teacher_idx]
        Tf = rng.integers(1, 3)  # 1 alebo 2

        # Teacher phase
        for i in range(cfg.population_size):
            r = rng.random(dims)
            difference = r * (teacher - Tf * mean)
            candidate = population[i] + difference
            candidate = np.clip(candidate, low, high)
            candidate_value = float(objective(candidate))
            if candidate_value < fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_value

        # Learner phase
        for i in range(cfg.population_size):
            j = i
            while j == i:
                j = rng.integers(0, cfg.population_size)
            r = rng.random(dims)
            if fitness[j] < fitness[i]:
                candidate = population[i] + r * (population[j] - population[i])
            else:
                candidate = population[i] + r * (population[i] - population[j])
            candidate = np.clip(candidate, low, high)
            candidate_value = float(objective(candidate))
            if candidate_value < fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_value

        pop_history.append(population.copy())
        fit_history.append(fitness.copy())

        current_best_idx = int(np.argmin(fitness))
        current_best_value = float(fitness[current_best_idx])
        if current_best_value < best_value:
            best_value = current_best_value
            best_point = population[current_best_idx].copy()
        best_curve.append(best_value)

    return TLBOResult(
        best_point=best_point,
        best_value=best_value,
        population_history=np.stack(pop_history),
        fitness_history=np.stack(fit_history),
        best_fitness_curve=np.array(best_curve),
    )
