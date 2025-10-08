from __future__ import annotations

import argparse
from typing import List

import numpy as np

from .ga import GAConfig, GeneticAlgorithmTSP
from .tsp import generate_cities, build_distance_matrix, tour_length, nearest_neighbor_tour
from .viz import LiveTSPPlot


def main() -> None:
    # CLI rozhranie – spúšťacie parametre pre GA a vizualizáciu
    parser = argparse.ArgumentParser(description='Genetic Algorithm for TSP with live visualization')

    # Jediný zdroj defaultov: použijeme GAConfig, aby sa hodnoty neudržiavali duplicitne
    default_cfg = GAConfig()

    parser.add_argument('--num-cities', type=int, default=30)
    parser.add_argument('--pop', type=int, default=default_cfg.population_size)
    parser.add_argument('--gens', type=int, default=default_cfg.generations)
    parser.add_argument('--mutation', type=float, default=default_cfg.mutation_rate)
    parser.add_argument('--crossover', type=float, default=default_cfg.crossover_rate)
    parser.add_argument('--elitism', type=int, default=default_cfg.elitism)
    parser.add_argument('--seed', type=int, default=default_cfg.seed if default_cfg.seed is not None else 42)
    parser.add_argument('--runs', type=int, default=1, help='repeat GA multiple times for stats')
    parser.add_argument('--no-viz', action='store_true')
    args = parser.parse_args()

    # Náhodné mestá a ich matica vzdialeností
    cities = generate_cities(args.num_cities, seed=args.seed)
    dmat = build_distance_matrix(cities)

    def fitness(genome: List[int]) -> float:
        return tour_length(genome, dmat)

    # Konfigurácia GA podľa zadaných (alebo defaultných) parametrov
    cfg = GAConfig(
        population_size=args.pop,
        generations=args.gens,
        crossover_rate=args.crossover,
        mutation_rate=args.mutation,
        elitism=args.elitism,
        seed=args.seed,
    )

    # Základná heuristika (baseline): najbližší sused na porovnanie kvality GA
    nn_tour = nearest_neighbor_tour(dmat, start=0)
    nn_len = tour_length(nn_tour, dmat)
    print(f'Nearest-neighbor baseline length: {nn_len:.3f}')

    results = []
    for r in range(args.runs):
        # Pri viacerých behoch ľahko meníme seed, aby boli behy odlišné
        if args.runs > 1:
            cfg.seed = (args.seed + r) if args.seed is not None else None
        ga = GeneticAlgorithmTSP(num_genes=len(cities), fitness_fn=fitness, config=cfg)

        plot = None
        # Vizualizáciu spustíme len pri poslednom behu (aby sa neotváralo viac okien)
        if not args.no_viz and r == args.runs - 1:
            plot = LiveTSPPlot(cities)

        def on_gen(gen: int, best: List[int], best_score: float) -> None:
            # priebežné prekreslenie aktuálne najlepšieho riešenia
            if plot is not None and gen % 1 == 0:
                plot.update(best, title_suffix=f'gen {gen}, length {best_score:.2f}')

        best, best_score = ga.evolve(on_generation=on_gen if not args.no_viz and plot is not None else None)
        results.append((best, best_score))

        if plot is not None:
            plot.update(best, title_suffix=f'final length {best_score:.2f}')
            plot.show()

    # Zhrnutie viacerých behov: minimum, priemer, smerodajná odchýlka
    best_overall = min(results, key=lambda x: x[1])
    print(f'Best GA tour length: {best_overall[1]:.3f} (improvement vs NN: {(nn_len - best_overall[1]):.3f})')
    print('Best tour order:', best_overall[0])


if __name__ == '__main__':
    main()


