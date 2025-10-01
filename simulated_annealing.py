"""Simulated Annealing optimizer.

Algoritmus začína s vysokou teplotou a postupne sa ochladzuje,
čo umožňuje akceptáciu horších riešení na začiatku hľadania, aby sa uniklo
lokálnym optimám. Vráti najlepší nájdený bod a trajektóriu hľadania
na vizualizáciu.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

# Typ aliasy pre prehľadnosť
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class SAResult:
    # Najlepší nájdený bod (parametre)
    best_point: np.ndarray
    # Hodnota cieľovej funkcie v najlepšom bode
    best_value: float
    # Celá trajektória navštívených bodov
    trajectory: np.ndarray  # shape: (iterations, n_dimensions)
    # Hodnoty cieľovej funkcie pre všetky body v trajektórii
    values: np.ndarray      # shape: (iterations,)
    # Teploty v každej iterácii
    temperatures: np.ndarray  # shape: (iterations,)


def clip_to_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Oseká vektor na dané hranice po jednotlivých súradniciach."""
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(x, low, high)


def simulated_annealing(
    objective: Objective,
    bounds: Bounds,
    seed: int | None = None,
    iterations: int = 1000,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
    step_std: float = 0.1,
) -> SAResult:
    """Simulated Annealing algoritmus pre hľadanie minima.

    Postup:
    1) Začni v náhodnom bode s vysokou teplotou
    2) V každom kroku vygeneruj náhodného suseda
    3) Ak je lepší, prijmi ho
    4) Ak je horší, prijmi ho s pravdepodobnosťou exp(-ΔE/T)
    5) Zníž teplotu a pokračuj
    6) Zastav keď teplota klesne pod minimum
    """
    rng = np.random.default_rng(seed)

    # Začiatok v náhodnom realizovateľnom bode
    start = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    current = start.copy()
    current_val = float(objective(current))

    # Inicializácia trajektórie
    trajectory = [current.copy()]
    values = [current_val]
    temperatures = []

    # Najlepší bod zatiaľ
    best_point = current.copy()
    best_value = current_val

    temperature = initial_temperature

    for iteration in range(iterations):
        # Zníž teplotu
        temperature = max(min_temperature, temperature * cooling_rate)
        temperatures.append(temperature)

        # Ak teplota klesla pod minimum, zastav
        if temperature <= min_temperature:
            break

        # Vygeneruj náhodného suseda
        noise = rng.normal(loc=0.0, scale=step_std, size=len(bounds))
        candidate = current + noise
        candidate = clip_to_bounds(candidate, bounds)
        candidate_val = float(objective(candidate))

        # Rozhodni či prijať kandidáta
        delta_e = candidate_val - current_val
        
        if delta_e < 0:
            # Lepší kandidát - vždy prijmi
            current = candidate
            current_val = candidate_val
        else:
            # Horší kandidát - prijmi s pravdepodobnosťou exp(-ΔE/T)
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current = candidate
                current_val = candidate_val

        # Pridaj do trajektórie
        trajectory.append(current.copy())
        values.append(current_val)

        # Aktualizuj najlepší bod
        if current_val < best_value:
            best_point = current.copy()
            best_value = current_val

    return SAResult(
        best_point=best_point,
        best_value=best_value,
        trajectory=np.array(trajectory),
        values=np.array(values),
        temperatures=np.array(temperatures)
    )
