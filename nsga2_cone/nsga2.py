"""NSGA-II implementation for the cone multi-objective problem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Bounds = Tuple[Tuple[float, float], Tuple[float, float]]  # (r_bounds, h_bounds)


@dataclass
class NSGA2Config:
    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    eta_c: float = 20.0  # SBX distribution index
    eta_m: float = 20.0  # polynomial mutation index
    target_volume: float | None = None
    penalty_weight: float = 1e6
    seed: int | None = None


@dataclass
class NSGA2Result:
    population: np.ndarray            # shape: (pop_size, 2)
    objectives: np.ndarray            # shape: (pop_size, 2)
    pareto_front: np.ndarray          # indices of rank-0 solutions
    history: List[np.ndarray]         # objective values per generation


def cone_objectives(individuals: np.ndarray) -> np.ndarray:
    """Compute lateral surface area S and total area T for each (r, h)."""
    r = individuals[:, 0]
    h = individuals[:, 1]
    s = np.sqrt(r * r + h * h)
    S = np.pi * r * s
    T = np.pi * r * (r + s)
    return np.column_stack([S, T])


def cone_volume(individuals: np.ndarray) -> np.ndarray:
    r = individuals[:, 0]
    h = individuals[:, 1]
    return (np.pi / 3.0) * r * r * h


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    population_size = objectives.shape[0]
    S = [[] for _ in range(population_size)]
    domination_counts = np.zeros(population_size, dtype=int)
    ranks = np.zeros(population_size, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def crowding_distance(front: List[int], objectives: np.ndarray) -> np.ndarray:
    if not front:
        return np.array([])
    distances = np.zeros(len(front))
    for m in range(objectives.shape[1]):
        values = objectives[front, m]
        sorted_idx = np.argsort(values)
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
        min_val = values[sorted_idx[0]]
        max_val = values[sorted_idx[-1]]
        if max_val == min_val:
            continue
        for i in range(1, len(front) - 1):
            prev_val = values[sorted_idx[i - 1]]
            next_val = values[sorted_idx[i + 1]]
            distances[sorted_idx[i]] += (next_val - prev_val) / (max_val - min_val)
    return distances


def tournament_selection(
    population: np.ndarray,
    objectives: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    i, j = rng.integers(0, len(population), size=2)
    if ranks[i] < ranks[j]:
        return population[i]
    if ranks[j] < ranks[i]:
        return population[j]
    return population[i] if crowding[i] > crowding[j] else population[j]


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta_c: float,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1)):
        if rng.random() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                x1 = min(parent1[i], parent2[i])
                x2 = max(parent1[i], parent2[i])
                rand = rng.random()
                beta = 1.0 + (2.0 * (x1 - low[i]) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta_c + 1.0)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                beta = 1.0 + (2.0 * (high[i] - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta_c + 1.0)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                c1 = np.clip(c1, low[i], high[i])
                c2 = np.clip(c2, low[i], high[i])
                if rng.random() <= 0.5:
                    child1[i] = c2
                    child2[i] = c1
                else:
                    child1[i] = c1
                    child2[i] = c2
    return child1, child2


def polynomial_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    eta_m: float,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mutant = individual.copy()
    for i in range(len(mutant)):
        if rng.random() < mutation_rate:
            delta1 = (mutant[i] - low[i]) / (high[i] - low[i])
            delta2 = (high[i] - mutant[i]) / (high[i] - low[i])
            rand = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            mutant[i] += deltaq * (high[i] - low[i])
            mutant[i] = np.clip(mutant[i], low[i], high[i])
    return mutant


def _evaluate(population: np.ndarray, cfg: NSGA2Config) -> Tuple[np.ndarray, np.ndarray]:
    actual = cone_objectives(population)
    if cfg.target_volume is None:
        return actual, actual
    volumes = cone_volume(population)
    denom = max(abs(cfg.target_volume), 1e-9)
    penalties = cfg.penalty_weight * np.abs(volumes - cfg.target_volume) / denom
    penalized = actual + penalties[:, None]
    return actual, penalized


def nsga2_cone(bounds: Bounds, config: NSGA2Config | None = None) -> NSGA2Result:
    cfg = config or NSGA2Config()
    low = np.array([bounds[0][0], bounds[1][0]], dtype=float)
    high = np.array([bounds[0][1], bounds[1][1]], dtype=float)
    rng = np.random.default_rng(cfg.seed)

    population = rng.uniform(low, high, size=(cfg.population_size, 2))
    objectives_actual, objectives_eval = _evaluate(population, cfg)
    history = [objectives_actual.copy()]

    for _ in range(cfg.generations):
        ranks = np.zeros(cfg.population_size, dtype=int)
        crowding = np.zeros(cfg.population_size, dtype=float)
        fronts = fast_non_dominated_sort(objectives_eval)
        for front_idx, front in enumerate(fronts):
            dist = crowding_distance(front, objectives_eval)
            for local_idx, individual_idx in enumerate(front):
                ranks[individual_idx] = front_idx
                crowding[individual_idx] = dist[local_idx]

        offspring = []
        while len(offspring) < cfg.population_size:
            parent1 = tournament_selection(population, objectives_eval, ranks, crowding, rng)
            parent2 = tournament_selection(population, objectives_eval, ranks, crowding, rng)
            if rng.random() < cfg.crossover_rate:
                child1, child2 = sbx_crossover(parent1, parent2, cfg.eta_c, low, high, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            child1 = polynomial_mutation(child1, cfg.mutation_rate, cfg.eta_m, low, high, rng)
            child2 = polynomial_mutation(child2, cfg.mutation_rate, cfg.eta_m, low, high, rng)
            offspring.append(child1)
            if len(offspring) < cfg.population_size:
                offspring.append(child2)

        offspring = np.array(offspring)
        combined = np.vstack([population, offspring])
        combined_actual, combined_eval = _evaluate(combined, cfg)

        fronts = fast_non_dominated_sort(combined_eval)
        new_population = []
        new_actual = []
        new_eval = []
        for front in fronts:
            if len(new_population) + len(front) > cfg.population_size:
                dist = crowding_distance(front, combined_eval)
                sorted_front = sorted(
                    zip(front, dist),
                    key=lambda item: (item[1]),
                    reverse=True,
                )
                for idx, _ in sorted_front:
                    if len(new_population) < cfg.population_size:
                        new_population.append(combined[idx])
                        new_actual.append(combined_actual[idx])
                        new_eval.append(combined_eval[idx])
                    else:
                        break
                break
            else:
                for idx in front:
                    new_population.append(combined[idx])
                    new_actual.append(combined_actual[idx])
                    new_eval.append(combined_eval[idx])

        population = np.array(new_population)
        objectives_actual = np.array(new_actual)
        objectives_eval = np.array(new_eval)
        history.append(objectives_actual.copy())

    final_fronts = fast_non_dominated_sort(objectives_eval)
    pareto_front = np.array(final_fronts[0], dtype=int)

    return NSGA2Result(
        population=population,
        objectives=objectives_actual,
        pareto_front=pareto_front,
        history=history,
    )
