"""Implementácia SOMA All-to-One (Self-Organizing Migrating Algorithm)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from differential_evolution.de import _validate_bounds


# Alias typov pre prehľadnosť
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class SOMAConfig:
    """Konfigurácia parametrov pre SOMA All-to-One."""

    population_size: int = 20  # pop_size
    iterations: int = 100      # M_max
    prt: float = 0.4           # pravdepodobnosť prenosu jednotlivých dimenzií
    path_length: float = 3.0   # dĺžka migrácie (multiplikátor rozdielu)
    step: float = 0.11         # krok pozdĺž cesty
    seed: int | None = None


@dataclass
class SOMAResult:
    """Výsledok SOMA optimalizácie."""

    best_point: np.ndarray
    best_value: float
    leader_history: np.ndarray        # shape: (iterations + 1, dims)
    population_history: np.ndarray    # shape: (iterations + 1, pop_size, dims)
    fitness_history: np.ndarray       # shape: (iterations + 1, pop_size)
    best_fitness_per_iteration: np.ndarray


def _generate_prt_mask(rng: np.random.Generator, dims: int, prt: float) -> np.ndarray:
    """Vytvor binárnu masku dimenzií, ktoré sa budú migrovať."""
    mask = rng.random(dims) < prt
    if not mask.any():
        # zabezpeč, aby sa aspoň jedna dimenzia zúčastnila migrácie
        mask[rng.integers(dims)] = True
    return mask


def soma_all_to_one(objective: Objective, bounds: Bounds, config: SOMAConfig | None = None) -> SOMAResult:
    """Spustí SOMA v režime All-to-One."""
    cfg = config or SOMAConfig()

    # Základná validácia vstupných parametrov algoritmu
    if cfg.population_size < 2:
        raise ValueError("population_size musí byť minimálne 2.")
    if cfg.iterations <= 0:
        raise ValueError("iterations musí byť kladné.")
    if not (0 <= cfg.prt <= 1):
        raise ValueError("prt musí ležať v intervale <0, 1>.")
    if cfg.path_length <= 0:
        raise ValueError("path_length musí byť kladná hodnota.")
    if cfg.step <= 0:
        raise ValueError("step musí byť kladná hodnota.")

    # Prevod hraníc na vektory (low, high) a príprava RNG
    low, high = _validate_bounds(bounds)
    dims = len(bounds)
    rng = np.random.default_rng(cfg.seed)

    # Inicializácia populácie náhodne v hraniciach a vyhodnotenie fitnes
    population = rng.uniform(low, high, size=(cfg.population_size, dims))
    fitness = np.apply_along_axis(objective, 1, population)

    # Určenie aktuálneho lídra (najlepšieho jedinca)
    best_idx = int(np.argmin(fitness))
    leader = population[best_idx].copy()
    leader_value = float(fitness[best_idx])

    # História pre vizualizáciu a analýzu priebehu
    leader_history = [leader.copy()]
    pop_history = [population.copy()]
    fit_history = [fitness.copy()]
    best_curve = [leader_value]

    # Hlavný migračný cyklus (M_max iterácií)
    for _ in range(cfg.iterations):
        # Pracovné kópie populácie (budú obsahovať výsledky migrácie)
        new_population = population.copy()
        new_fitness = fitness.copy()

        # Zafixuj nového lídra v tejto iterácii – líder sám nemigruje
        leader_idx = int(np.argmin(fitness))
        leader = population[leader_idx].copy()
        leader_value = float(fitness[leader_idx])

        for i in range(cfg.population_size):
            if i == leader_idx:
                continue  # líder nemigruje

            # PRT maska určuje, ktoré dimenzie migranta sa budú hýbať
            prt_mask = _generate_prt_mask(rng, dims, cfg.prt)
            # Diskretizované kroky pozdĺž úsečky smerom k lídrovi
            t_values = np.arange(cfg.step, cfg.path_length + cfg.step, cfg.step)

            # Najlepší bod, ktorý si migrant doteraz našiel (na svojej ceste)
            best_point_i = population[i].copy()
            best_value_i = float(fitness[i])

            for t in t_values:
                # Kandidát: posun od migranta k lídrovi podľa t a PRT masky
                trial = population[i] + (leader - population[i]) * t * prt_mask
                # Orezanie do hraníc problému
                trial = np.clip(trial, low, high)
                trial_value = float(objective(trial))

                if trial_value < best_value_i:
                    # Ak je nový bod lepší, migrant si ho zapamätá
                    best_value_i = trial_value
                    best_point_i = trial.copy()

            # Po prejdení celej cesty si migrant ponechá najlepší nájdený bod
            new_population[i] = best_point_i
            new_fitness[i] = best_value_i

        # Ukončenie iterácie – celá populácia sa nahradí vylepšenými jedincami
        population = new_population
        fitness = new_fitness

        # Zaznamenaj nového lídra a históriu pre vizualizáciu
        best_idx = int(np.argmin(fitness))
        leader = population[best_idx].copy()
        leader_value = float(fitness[best_idx])

        leader_history.append(leader.copy())
        pop_history.append(population.copy())
        fit_history.append(fitness.copy())
        best_curve.append(leader_value)

    # Zostav návratovú štruktúru s kompletnou históriou
    return SOMAResult(
        best_point=leader,
        best_value=leader_value,
        leader_history=np.stack(leader_history),
        population_history=np.stack(pop_history),
        fitness_history=np.stack(fit_history),
        best_fitness_per_iteration=np.array(best_curve),
    )
