"""Optimalizátor Particle Swarm Optimization (PSO) so zotrvačnosťou."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from differential_evolution.de import _validate_bounds


# Typ aliasy pre prehľadnosť
Objective = Callable[[np.ndarray], float]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class PSOConfig:
    """Konfigurácia pre PSO."""

    population_size: int = 15   # veľkosť roja (popsize)
    iterations: int = 50        # M_max
    inertia_weight: float = 0.7  # w
    cognitive_coef: float = 2.0  # c1
    social_coef: float = 2.0     # c2
    velocity_clamp: float | None = None  # max |v| pre každú dimenziu (symetricky)
    seed: int | None = None


@dataclass
class PSOResult:
    """Výsledok PSO optimalizácie."""

    best_point: np.ndarray
    best_value: float
    position_history: np.ndarray   # tvar: (iterácie + 1, popsize, dims)
    velocity_history: np.ndarray   # tvar: (iterácie + 1, popsize, dims)
    fitness_history: np.ndarray    # tvar: (iterácie + 1, popsize)
    best_fitness_per_iteration: np.ndarray  # najlepšia hodnota po každej iterácii


def particle_swarm_optimization(objective: Objective, bounds: Bounds, config: PSOConfig | None = None) -> PSOResult:
    """Spustí Particle Swarm Optimization s váhou zotrvačnosti."""
    cfg = config or PSOConfig()

    if cfg.population_size < 2:
        raise ValueError("population_size musí byť aspoň 2.")
    if cfg.iterations <= 0:
        raise ValueError("iterations musí byť kladné číslo.")
    if cfg.inertia_weight < 0:
        raise ValueError("inertia_weight (w) musí byť nezáporné.")
    if cfg.cognitive_coef < 0 or cfg.social_coef < 0:
        raise ValueError("cognitive_coef a social_coef musia byť nezáporné.")

    low, high = _validate_bounds(bounds)
    dims = len(bounds)

    rng = np.random.default_rng(cfg.seed)

    # Inicializuj pozície častíc rovnomerne v hraniciach
    positions = rng.uniform(low, high, size=(cfg.population_size, dims))
    # Počiatočné rýchlosti z menšieho rozsahu (10 % šírky intervalu v každej dimenzii)
    velocity_span = (high - low) * 0.1
    velocities = rng.uniform(-velocity_span, velocity_span, size=(cfg.population_size, dims))

    fitness = np.apply_along_axis(objective, 1, positions)
    personal_best_positions = positions.copy()
    personal_best_values = fitness.copy()

    best_idx = int(np.argmin(personal_best_values))
    global_best_point = personal_best_positions[best_idx].copy()
    global_best_value = float(personal_best_values[best_idx])

    pos_history = [positions.copy()]
    vel_history = [velocities.copy()]
    fit_history = [fitness.copy()]
    best_curve = [global_best_value]

    for _ in range(cfg.iterations):
        r1 = rng.random(size=(cfg.population_size, dims))  # náhodné čísla pre kognitívnu zložku
        r2 = rng.random(size=(cfg.population_size, dims))  # náhodné čísla pre sociálnu zložku

        # KOGNITÍVNA ZLOŽKA - ťahá časticu k jej osobnej najlepšej pozícii
        cognitive_term = cfg.cognitive_coef * r1 * (personal_best_positions - positions)
        
        # SOCIÁLNA ZLOŽKA - ťahá časticu k globálnej najlepšej pozícii
        social_term = cfg.social_coef * r2 * (global_best_point - positions)

        # AKTUALIZÁCIA RÝCHLOSTI - PSO rovnica
        velocities = cfg.inertia_weight * velocities + cognitive_term + social_term

        # OBRANIE RÝCHLOSTI - zabráni príliš rýchlemu pohybu častíc
        if cfg.velocity_clamp is not None:
            max_abs = abs(cfg.velocity_clamp)  # maximálna absolútna hodnota rýchlosti
            velocities = np.clip(velocities, -max_abs, max_abs)  # obmedzenie rýchlosti

        # AKTUALIZÁCIA POZÍCIE - častice sa posunú podľa svojej rýchlosti
        positions = positions + velocities
        # UDRŽANIE V HRANICIACH
        positions = np.clip(positions, low, high)

        # VÝPOČET FITNESS - hodnotenie nových pozícií častíc
        fitness = np.apply_along_axis(objective, 1, positions)

        # AKTUALIZÁCIA OSOBNÝCH NAJLEPŠÍCH POZÍCIÍ
        improved = fitness < personal_best_values  
        if np.any(improved):  
            personal_best_positions[improved] = positions[improved]  
            personal_best_values[improved] = fitness[improved]  

        # AKTUALIZÁCIA GLOBÁLNEJ NAJLEPŠEJ POZÍCIE
        best_idx = int(np.argmin(personal_best_values))  # nájdi index najlepšej častice
        if personal_best_values[best_idx] < global_best_value:  # ak je lepšia ako globálna
            global_best_value = float(personal_best_values[best_idx])  # aktualizuj globálnu hodnotu
            global_best_point = personal_best_positions[best_idx].copy()  # aktualizuj globálnu pozíciu

        # ULOŽENIE HISTÓRIE 
        pos_history.append(positions.copy())  # pozície všetkých častíc
        vel_history.append(velocities.copy())  # rýchlosti všetkých častíc
        fit_history.append(fitness.copy())  # fitness všetkých častíc
        best_curve.append(global_best_value)  # najlepšia hodnota v tejto iterácii

    return PSOResult(
        best_point=global_best_point,
        best_value=global_best_value,
        position_history=np.stack(pos_history),
        velocity_history=np.stack(vel_history),
        fitness_history=np.stack(fit_history),
        best_fitness_per_iteration=np.array(best_curve),
    )
