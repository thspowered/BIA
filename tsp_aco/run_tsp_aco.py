from __future__ import annotations

import argparse
from dataclasses import replace
from typing import List

import numpy as np

from tsp_ga.tsp import (
    generate_cities,
    build_distance_matrix,
    tour_length,
    nearest_neighbor_tour,
)
from .aco import ACOConfig, AntColonyTSP
from .viz import LiveACOPlot


def main() -> None:
    parser = argparse.ArgumentParser(description="Ant Colony Optimization pre TSP s vizualizáciou.")

    default_cfg = ACOConfig()
    parser.add_argument("--num-cities", type=int, default=30)
    parser.add_argument("--ants", type=int, default=default_cfg.num_ants)
    parser.add_argument("--iters", type=int, default=default_cfg.iterations)
    parser.add_argument("--alpha", type=float, default=default_cfg.alpha)
    parser.add_argument("--beta", type=float, default=default_cfg.beta)
    parser.add_argument("--rho", type=float, default=default_cfg.evaporation)
    parser.add_argument("--q", type=float, default=default_cfg.q)
    parser.add_argument("--pheromone", type=float, default=default_cfg.initial_pheromone)
    parser.add_argument("--seed", type=int, default=default_cfg.seed if default_cfg.seed is not None else 42)
    parser.add_argument("--runs", type=int, default=1, help="Počet nezávislých behov (pre štatistiku).")
    parser.add_argument("--draw-every", type=int, default=1, help="Prekresliť vizualizáciu každých N iterácií.")
    parser.add_argument("--no-viz", action="store_true", help="Zakáže živú vizualizáciu.")
    args = parser.parse_args()

    cities = generate_cities(args.num_cities, seed=args.seed)
    dmat = build_distance_matrix(cities)

    # baseline: najbližší sused
    nn_tour = nearest_neighbor_tour(dmat, start=0)
    nn_len = tour_length(nn_tour, dmat)
    print(f"Nearest-neighbor baseline length: {nn_len:.3f}")

    results: List[tuple[List[int], float]] = []

    base_cfg = ACOConfig(
        num_ants=args.ants,
        iterations=args.iters,
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.rho,
        q=args.q,
        initial_pheromone=args.pheromone,
        seed=args.seed,
    )

    for run_idx in range(args.runs):
        cfg = replace(base_cfg, seed=args.seed + run_idx if args.seed is not None else None)
        colony = AntColonyTSP(dmat, config=cfg)

        plot = None
        if not args.no_viz and run_idx == args.runs - 1:
            plot = LiveACOPlot(cities)

        def on_iter(iteration: int, best_tour: List[int], best_length: float) -> None:
            if plot is not None and iteration % max(1, args.draw_every) == 0:
                plot.update(best_tour, title_suffix=f"iter {iteration}, length {best_length:.2f}")

        result = colony.run(on_iteration=on_iter if plot is not None else None)
        results.append((result.best_tour, result.best_length))

        if plot is not None:
            plot.update(result.best_tour, title_suffix=f"final length {result.best_length:.2f}")
            plot.show()

        print(f"Run {run_idx + 1}/{args.runs}: best length {result.best_length:.3f}")

    best_overall = min(results, key=lambda x: x[1])
    improvement = nn_len - best_overall[1]
    print(f"Best ACO tour length: {best_overall[1]:.3f} (improvement vs NN: {improvement:.3f})")
    print("Best tour order:", best_overall[0])


if __name__ == "__main__":
    main()
