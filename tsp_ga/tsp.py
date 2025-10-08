from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Reprezentácia mesta v 2D priestore
@dataclass(frozen=True)
class City:
    label: str
    x: float
    y: float


def generate_cities(num_cities: int, width: int = 220, height: int = 220, seed: int | None = None) -> List[City]:
    """Vygeneruje náhodné mestá v obdĺžniku [10, width-10] x [10, height-10].

    - num_cities: počet miest
    - seed: voliteľné semienko pre reprodukovateľnosť
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    labels = [chr(ord('A') + i) if i < 26 else f"C{i}" for i in range(num_cities)]
    xs = np.random.uniform(10, width - 10, size=num_cities)
    ys = np.random.uniform(10, height - 10, size=num_cities)
    return [City(labels[i], float(xs[i]), float(ys[i])) for i in range(num_cities)]


def build_distance_matrix(cities: List[City]) -> np.ndarray:
    """Vypočíta Euklidovskú maticu vzdialeností medzi všetkými dvojicami miest."""
    n = len(cities)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cities[i].x - cities[j].x
            dy = cities[i].y - cities[j].y
            d = math.hypot(dx, dy)
            m[i, j] = d
            m[j, i] = d
    return m


def tour_length(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Vráti dĺžku Hamiltonovho okruhu definovaného permutáciou indexov `tour`.

    Okruh je uzavretý: pridáva sa hrana z posledného mesta späť do prvého.
    """
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += float(distance_matrix[a, b])
    return total


def pretty_path(cities: List[City], tour: List[int]) -> List[Tuple[float, float]]:
    """Pomocná funkcia pre vizualizáciu – súradnice v poradí podľa `tour`.

    Vracia zoznam bodov s návratom na počiatočný bod, aby sa dal vykresliť uzavretý okruh.
    """
    return [(cities[i].x, cities[i].y) for i in tour] + [(cities[tour[0]].x, cities[tour[0]].y)]


def nearest_neighbor_tour(distance_matrix: np.ndarray, start: int = 0) -> List[int]:
    """Greedy heuristika najbližšieho suseda.

    Začneme v `start`, vždy vyberieme najbližšie ešte nenavštívené mesto.
    Výsledok je permutácia vrcholov – približné riešenie TSP.
    """
    n = distance_matrix.shape[0]
    unvisited = set(range(n))
    tour: List[int] = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        # vyber najbližšie ešte nenavštívené mesto
        nxt = min(unvisited, key=lambda j: float(distance_matrix[current, j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


