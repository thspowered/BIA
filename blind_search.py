"""Blind (random) search optimizer.

The algorithm samples random points uniformly within the provided bounds and
keeps the incumbent with the best objective value. It returns the best point
found and the search trajectory for visualization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

# Typ aliasy pre prehľadnosť
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class SearchResult:
    # Najlepší nájdený bod (parametre)
    best_point: np.ndarray
    # Hodnota cieľovej funkcie v najlepšom bode
    best_value: float
    # Celá trajektória navštívených bodov (t.j. všetky náhodné vzorky)
    trajectory: np.ndarray  # shape: (iterations, n_dimensions)
    # Hodnoty cieľovej funkcie pre všetky body v trajektórii
    values: np.ndarray      # shape: (iterations,)


def uniform_random_in_bounds(rng: np.random.Generator, bounds: Bounds, n: int = 1) -> np.ndarray:
    """Vygeneruje n bodov rovnomerne z intervalu pre každú súradnicu.

    Každá súradnica je samostatne vzorkovaná z <low, high> pre danú os.
    """
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(low, high, size=(n, len(bounds)))


def blind_search(objective: Objective, bounds: Bounds, iterations: int = 5000, seed: int | None = None) -> SearchResult:
    """Slepé (náhodné) hľadanie minima.

    - náhodne vygeneruje 'iterations' bodov v hraniciach
    - vyhodnotí cieľovú funkciu vo všetkých bodoch
    - vráti najlepší bod a kompletnú trajektóriu na vizualizáciu
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    # Náhodná trajektória – každý riadok je jeden bod
    trajectory = uniform_random_in_bounds(rng, bounds, n=iterations)
    # Vyhodnotenia funkcie pre všetky body
    values = np.apply_along_axis(objective, 1, trajectory)

    # Index globálne najnižšej hodnoty medzi vzorkami
    best_idx = int(np.argmin(values))
    best_point = trajectory[best_idx].copy()
    best_value = float(values[best_idx])

    return SearchResult(best_point=best_point, best_value=best_value, trajectory=trajectory, values=values)
