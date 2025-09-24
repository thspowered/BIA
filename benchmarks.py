"""Collection of continuous optimization benchmark functions.

All functions accept a NumPy array of shape (n_dimensions,) and return a
scalar float. They are vectorized where possible using NumPy operations.

The module also provides convenient defaults for search bounds.
"""
from __future__ import annotations

from math import pi
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


# Typ aliasy pre prehľadnosť
BenchmarkFunction = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


# ------------------------- Definície funkcií ------------------------- #

def sphere(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sum(x * x))


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * pi) -> float:
    x = np.asarray(x)
    n = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x * x) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return float(term1 + term2 + a + np.e)


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x)
    n = x.size
    return float(10 * n + np.sum(x * x - 10 * np.cos(2 * pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0))


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x)
    n = x.size
    sum_term = np.sum(x * x) / 4000.0
    indices = np.arange(1, n + 1, dtype=float)
    prod_term = np.prod(np.cos(x / np.sqrt(indices)))
    return float(1.0 + sum_term - prod_term)


def schwefel(x: np.ndarray) -> float:
    x = np.asarray(x)
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x: np.ndarray) -> float:
    x = np.asarray(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)


def michalewicz(x: np.ndarray, m: int = 10) -> float:
    x = np.asarray(x)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * (np.sin(i * x * x / pi) ** (2 * m))))


def zakharov(x: np.ndarray) -> float:
    x = np.asarray(x)
    i = np.arange(1, x.size + 1, dtype=float)
    s1 = np.sum(x * x)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2 * s2 + s2 ** 4)


# --------------------------- Odporúčané domény ----------------------- #
# Referencia: https://www.sfu.ca/~ssurjano/optimization.html
# 1D intervaly pre každú funkciu; používajú sa na zostavenie 2D hraníc pre vykresľovanie
RECOMMENDED_DOMAINS: Dict[str, Tuple[float, float]] = {
    # Veľa lokálnych miním
    "ackley": (-32.768, 32.768),
    "griewank": (-600.0, 600.0),
    "levy": (-10.0, 10.0),
    "rastrigin": (-5.12, 5.12),
    "schwefel": (-500.0, 500.0),
    # Misa (bowl-shaped)
    "sphere": (-5.12, 5.12),
    # Doskovitý tvar (plate-shaped)
    "zakharov": (-5.0, 10.0),
    # Údolie (valley-shaped)
    "rosenbrock": (-2.048, 2.048),
    # Strmé hrany/pády
    "michalewicz": (0.0, pi),
}

# Zostav 2D predvolené hranice – opakuj 1D interval pre os x1 aj x2
DEFAULT_BOUNDS: Dict[str, Bounds] = {
    name: [RECOMMENDED_DOMAINS[name], RECOMMENDED_DOMAINS[name]] for name in RECOMMENDED_DOMAINS
}


def get_bounds_for_dimensions(bounds: Bounds, n_dimensions: int) -> Bounds:
    """Opakuje zadané 2D hranice na požadovanú dimenziu.

    Ak sa dĺžka `bounds` už rovná `n_dimensions`, vracia pôvodné hranice.
    """
    if len(bounds) == n_dimensions:
        return bounds
    if len(bounds) == 0:
        raise ValueError("Hranice musia obsahovať aspoň jeden (nízky, vysoký) pár.")
    return list(bounds)[:1] * n_dimensions
