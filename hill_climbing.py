from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

# Alias typov – cieľová funkcia a hranice
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class HillClimbResult:
    # Najlepší nájdený bod
    best_point: np.ndarray
    # Hodnota funkcie v najlepšom bode
    best_value: float
    # Trajektória navštívených bodov v poradí
    trajectory: np.ndarray  # visited points in order
    # Zaznamenané hodnoty f(x) pre body v trajektórii
    values: np.ndarray


def clip_to_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Oseká vektor na dané hranice po jednotlivých súradniciach."""
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(x, low, high)


def hill_climb(
    objective: Objective,
    bounds: Bounds,
    seed: int | None = None,
    iterations: int = 200,
    neighbors_per_step: int = 20,
    step_std: float = 0.1,
) -> HillClimbResult:
    """Jednoduchý Hill Climbing s gaussovskými susedmi.

    Postup:
    1) Začni v náhodnom bode v rámci hraníc.
    2) V každom kroku vygeneruj N susedov ~ N(0, step_std) okolo aktuálneho bodu.
    3) Vyber najlepšieho suseda; ak zlepšuje, presuň sa naň, inak skonči.
    """
    rng = np.random.default_rng(seed)

    # Začiatok v náhodnom realizovateľnom bode
    start = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    current = start.copy()
    current_val = float(objective(current))

    traj = [current.copy()]
    vals = [current_val]

    for _ in range(iterations):
        # Návrh susedov okolo current pomocou Gaussovho šumu
        noise = rng.normal(loc=0.0, scale=step_std, size=(neighbors_per_step, len(bounds)))
        proposals = current + noise
        proposals = np.apply_along_axis(lambda v: clip_to_bounds(v, bounds), 1, proposals)

        # Vyhodnoť všetkých susedov a nájdi najlepšieho
        neigh_vals = np.apply_along_axis(objective, 1, proposals)
        best_idx = int(np.argmin(neigh_vals))
        best_candidate = proposals[best_idx]
        best_val = float(neigh_vals[best_idx])

        # Posun len pri zlepšení, inak stop (lokálne minimum)
        if best_val < current_val:
            current = best_candidate
            current_val = best_val
            traj.append(current.copy())
            vals.append(current_val)
        else:
            break

    return HillClimbResult(best_point=current, best_value=current_val, trajectory=np.array(traj), values=np.array(vals))
