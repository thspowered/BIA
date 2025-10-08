from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


FitnessFn = Callable[[List[int]], float]


# Konfigurácia GA (veľkosť populácie, počty generácií, pravdepodobnosti operátorov)
@dataclass
class GAConfig:
    population_size: int = 150
    generations: int = 400
    tournament_size: int = 4
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elitism: int = 2
    seed: int | None = None


class GeneticAlgorithmTSP:
    def __init__(self, num_genes: int, fitness_fn: FitnessFn, config: GAConfig):
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        self.num_genes = num_genes
        self.fitness_fn = fitness_fn
        self.cfg = config

    # --- inicializácia populácie ---
    def _random_individual(self) -> List[int]:
        genome = list(range(self.num_genes))
        random.shuffle(genome)
        return genome

    def _init_population(self) -> List[List[int]]:
        return [self._random_individual() for _ in range(self.cfg.population_size)]

    # --- hodnotenie jedincov ---
    def _evaluate(self, population: List[List[int]]) -> List[float]:
        # Minimalizačný problém: fitness je dĺžka okruhu (čím menšie, tým lepšie)
        return [self.fitness_fn(ind) for ind in population]

    # --- selekcia: turnaj ---
    def _tournament(self, population: List[List[int]], scores: List[float]) -> List[int]:
        k = self.cfg.tournament_size
        participants = random.sample(range(len(population)), k)
        best = min(participants, key=lambda idx: scores[idx])
        return population[best][:]

    # --- kríženie: prefix z rodiča A + zvyšok v poradí z rodiča B (bez duplicít) ---
    def _crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.cfg.crossover_rate:
            return p1[:], p2[:]
        n = self.num_genes
        cut = random.randint(1, n - 1)  # veľkosť prefixu prevzatého z rodiča A

        def prefix_merge(parent_a: List[int], parent_b: List[int]) -> List[int]:
            # prefix z A ponecháme, zvyšné mestá doplníme v poradí z B
            prefix = parent_a[:cut]
            remainder = [g for g in parent_b if g not in prefix]
            return prefix + remainder

        return prefix_merge(p1, p2), prefix_merge(p2, p1)

    # --- mutácia: výmena (swap) dvoch náhodných pozícií ---
    def _mutate(self, genome: List[int]) -> None:
        if random.random() > self.cfg.mutation_rate:
            return
        i, j = random.sample(range(self.num_genes), 2)
        genome[i], genome[j] = genome[j], genome[i]

    def evolve(self, on_generation: Callable[[int, List[int], float], None] | None = None) -> Tuple[List[int], float]:
        # inicializácia a zistenie zatiaľ najlepšieho jedinca
        pop = self._init_population()
        scores = self._evaluate(pop)
        best_idx = int(np.argmin(scores))
        best, best_score = pop[best_idx][:], float(scores[best_idx])

        for gen in range(self.cfg.generations):
            next_pop: List[List[int]] = []

            # elitarizmus: prekopíruj najlepších priamo do ďalšej populácie
            elites_idx = list(np.argsort(scores))[: self.cfg.elitism]
            for ei in elites_idx:
                next_pop.append(pop[ei][:])

            # tvorba potomkov cez selekciu -> crossover -> mutácia
            while len(next_pop) < self.cfg.population_size:
                p1 = self._tournament(pop, scores)
                p2 = self._tournament(pop, scores)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                if len(next_pop) < self.cfg.population_size:
                    next_pop.append(c1)
                self._mutate(c2)
                if len(next_pop) < self.cfg.population_size:
                    next_pop.append(c2)

            pop = next_pop
            scores = self._evaluate(pop)
            gen_best_idx = int(np.argmin(scores))
            gen_best, gen_best_score = pop[gen_best_idx][:], float(scores[gen_best_idx])
            if gen_best_score < best_score:
                best, best_score = gen_best, gen_best_score
            if on_generation is not None:
                on_generation(gen, best, best_score)

        return best, best_score


