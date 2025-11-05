"""Ant Colony Optimization (ACO) pre úlohu TSP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


Tour = List[int]
DistanceMatrix = np.ndarray
OnIterationCallback = Callable[[int, Tour, float], None]


@dataclass
class ACOConfig:
    """Konfigurácia parametrov ACO."""

    num_ants: int = 20
    iterations: int = 200
    alpha: float = 1.0          # váha feromónu
    beta: float = 2.0           # váha heuristiky (viditeľnosť = 1 / d)
    evaporation: float = 0.5    # koeficient vyparovania ρ
    q: float = 1.0              # množstvo feromónu uloženého (Q / L)
    initial_pheromone: float = 1.0
    seed: int | None = None


@dataclass
class ACOResult:
    """Výsledok behu ACO."""

    best_tour: Tour
    best_length: float
    best_history: List[float]  # najlepšie dĺžky v jednotlivých iteráciách


class AntColonyTSP:
    """ACO algoritmus pre TSP pracujúci s maticou vzdialeností."""

    def __init__(self, distance_matrix: DistanceMatrix, config: ACOConfig | None = None):
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix musí byť kvadratická matica.")
        self.cfg = config or ACOConfig()
        self.dmat = distance_matrix.astype(np.float64)
        self.n = self.dmat.shape[0]

        self.rng = np.random.default_rng(self.cfg.seed)

        # Maticu viditeľnosti η = 1 / d, pričom diagonála 0.
        self.visibility = np.zeros_like(self.dmat)
        with np.errstate(divide="ignore"):
            inv = 1.0 / self.dmat
        inv[np.isinf(inv)] = 0.0
        np.fill_diagonal(inv, 0.0)
        self.visibility = inv

    def _probabilities(self, current: int, allowed: Sequence[int], pheromone: np.ndarray) -> np.ndarray:
        """Vypočíta pravdepodobnosti výberu ďalšieho mesta pre aktuálnu mravčiu."""
        tau = np.power(pheromone[current, allowed], self.cfg.alpha)
        eta = np.power(self.visibility[current, allowed], self.cfg.beta)
        desirability = tau * eta
        total = np.sum(desirability)
        if total <= 0.0 or np.isnan(total):
            # fallback: uniformné rozdelenie
            return np.full(len(allowed), 1.0 / len(allowed))
        return desirability / total

    def _construct_tour(self, start: int, pheromone: np.ndarray) -> Tour:
        """Postaví trasu pre jednu mravčiu z daného štartu."""
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)
        current = start

        while unvisited:
            allowed = list(unvisited)
            probs = self._probabilities(current, allowed, pheromone)
            next_idx = int(self.rng.choice(allowed, p=probs))
            tour.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx

        return tour

    def _tour_length(self, tour: Tour) -> float:
        total = 0.0
        for i in range(self.n):
            a = tour[i]
            b = tour[(i + 1) % self.n]
            total += float(self.dmat[a, b])
        return total

    def run(self, on_iteration: OnIterationCallback | None = None) -> ACOResult:
        """Spustí ACO a vracia najlepšiu nájdenú trasu."""
        cfg = self.cfg
        pheromone = np.full((self.n, self.n), cfg.initial_pheromone, dtype=np.float64)
        np.fill_diagonal(pheromone, 0.0)

        best_tour: Tour | None = None
        best_length = float("inf")
        best_history: List[float] = []

        for iteration in range(cfg.iterations):
            tours: List[Tour] = []
            lengths: List[float] = []

            for ant_idx in range(cfg.num_ants):
                start_city = ant_idx % self.n
                tour = self._construct_tour(start_city, pheromone)
                length = self._tour_length(tour)
                tours.append(tour)
                lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()

            # Vyparovanie feromónu
            pheromone *= (1.0 - cfg.evaporation)

            # Ukladanie feromónu podľa kvality trás
            for tour, length in zip(tours, lengths):
                deposit = cfg.q / length if length > 0 else 0.0
                for i in range(self.n):
                    a = tour[i]
                    b = tour[(i + 1) % self.n]
                    pheromone[a, b] += deposit
                    pheromone[b, a] += deposit

            best_history.append(best_length)

            if on_iteration is not None and best_tour is not None:
                on_iteration(iteration, best_tour, best_length)

        if best_tour is None:
            raise RuntimeError("ACO nevyprodukovalo žiadnu trasu.")

        return ACOResult(best_tour=best_tour, best_length=best_length, best_history=best_history)
