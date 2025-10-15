"""Optimalizátor Differential Evolution (DE/rand/1/bin).

Implementácia kopíruje pseudokód z prednášky:
    - populácia reálnych vektorov sa vyvíja pomocou diferenciálnej mutácie
    - mutačný vektor v = x_r1 + F * (x_r2 - x_r3)
    - binomálne kríženie s garanciou aspoň jednej mutantnej zložky
    - chamtivý výber medzi cieľovým vektorom a trial vektorom

Funkcia ukladá kompletnú históriu populácie, aby bolo možné neskôr vizualizovať
proces hľadania (napr. na povrchovom grafe v 2D).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


# Typ aliasy pre čitateľnosť
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class DEConfig:
    """Konfigurácia pre Differential Evolution."""

    population_size: int = 20  # NP
    generations: int = 50      # G_max
    differential_weight: float = 0.5  # F
    crossover_rate: float = 0.5       # CR
    seed: int | None = None


@dataclass
class DEResult:
    """Výsledok DE optimalizácie."""

    best_point: np.ndarray
    best_value: float
    population_history: np.ndarray  # shape: (generations + 1, NP, dims)
    fitness_history: np.ndarray     # shape: (generations + 1, NP)
    best_fitness_per_generation: np.ndarray  # monotónna krivka najlepšej hodnoty


def _validate_bounds(bounds: Bounds) -> Tuple[np.ndarray, np.ndarray]:
    if len(bounds) == 0:
        raise ValueError("Hranice musia obsahovať aspoň jeden pár (min, max).")
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    if np.any(high <= low):
        raise ValueError("Každé horné ohraničenie musí byť väčšie ako dolné.")
    return low, high


def differential_evolution(objective: Objective, bounds: Bounds, config: DEConfig | None = None) -> DEResult:
    """Spustí Differential Evolution pre hľadanie minima."""
    cfg = config or DEConfig()

    if cfg.population_size < 4:
        raise ValueError("population_size musí byť aspoň 4 (potrebujeme 3 mutantov a cieľ).")
    if not (0.0 <= cfg.crossover_rate <= 1.0):
        raise ValueError("crossover_rate musí ležať v intervale <0, 1>.")
    if cfg.generations <= 0:
        raise ValueError("generations musí byť kladný počet.")

    low, high = _validate_bounds(bounds)
    dims = len(bounds)

    rng = np.random.default_rng(cfg.seed)

    # Inicializuj populáciu v rámci zadaných hraníc
    population = rng.uniform(low, high, size=(cfg.population_size, dims))
    fitness = np.apply_along_axis(objective, 1, population)

    pop_history = [population.copy()]
    fit_history = [fitness.copy()]
    best_idx = int(np.argmin(fitness))
    best_point = population[best_idx].copy()
    best_value = float(fitness[best_idx])
    best_curve = [best_value]

    for _ in range(cfg.generations):
        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(cfg.population_size):
            # Výber troch rôznych indexov (okrem targetu i)
            candidates = np.delete(np.arange(cfg.population_size), i)
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            mutant = x_r1 + cfg.differential_weight * (x_r2 - x_r3)

            # Binomálne kríženie s garanciou aspoň jednej mutantnej zložky
            crossover_mask = rng.random(dims) < cfg.crossover_rate
            crossover_mask[rng.integers(dims)] = True
            trial = np.where(crossover_mask, mutant, population[i])

            # Ošetri hranice
            trial = np.clip(trial, low, high)
            trial_fitness = float(objective(trial))

            # Chamtivý výber: prijmi rovnako dobré alebo lepšie riešenie
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        population = new_population
        fitness = new_fitness

        pop_history.append(population.copy())
        fit_history.append(fitness.copy())

        gen_best_idx = int(np.argmin(fitness))
        gen_best_value = float(fitness[gen_best_idx])
        if gen_best_value < best_value:
            best_value = gen_best_value
            best_point = population[gen_best_idx].copy()
        best_curve.append(best_value)

    return DEResult(
        best_point=best_point,
        best_value=best_value,
        population_history=np.stack(pop_history),
        fitness_history=np.stack(fit_history),
        best_fitness_per_generation=np.array(best_curve),
    )
