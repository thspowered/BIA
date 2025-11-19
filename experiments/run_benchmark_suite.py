from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, List

import numpy as np
from openpyxl import Workbook

from benchmarks import (
    DEFAULT_BOUNDS,
    ackley,
    griewank,
    levy,
    michalewicz,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    zakharov,
    get_bounds_for_dimensions,
)
from differential_evolution.de import DEConfig, differential_evolution
from particle_swarm.pso import PSOConfig, particle_swarm_optimization
from soma_all_to_one.soma import SOMAConfig, soma_all_to_one
from firefly.firefly import FireflyConfig, firefly_optimize
from tlbo.tlbo import TLBOConfig, tlbo_optimize


Benchmark = Callable[[np.ndarray], float]


FUNCTIONS = {
    "sphere": sphere,
    "ackley": ackley,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "griewank": griewank,
    "schwefel": schwefel,
    "levy": levy,
    "michalewicz": michalewicz,
    "zakharov": zakharov,
}

ALGORITHM_NAMES = ["DE", "PSO", "SOMA", "FA", "TLBO"]


def run_de(objective: Benchmark, bounds, pop: int, iters: int, seed: int) -> float:
    cfg = DEConfig(
        population_size=pop,
        generations=iters,
        differential_weight=0.5,
        crossover_rate=0.5,
        seed=seed,
    )
    return differential_evolution(objective, bounds, cfg).best_value


def run_pso(objective: Benchmark, bounds, pop: int, iters: int, seed: int) -> float:
    cfg = PSOConfig(
        population_size=pop,
        iterations=iters,
        inertia_weight=0.7,
        cognitive_coef=2.0,
        social_coef=2.0,
        seed=seed,
    )
    return particle_swarm_optimization(objective, bounds, cfg).best_value


def run_soma(objective: Benchmark, bounds, pop: int, iters: int, seed: int) -> float:
    cfg = SOMAConfig(
        population_size=pop,
        iterations=iters,
        prt=0.4,
        path_length=3.0,
        step=0.11,
        seed=seed,
    )
    return soma_all_to_one(objective, bounds, cfg).best_value


def run_firefly(objective: Benchmark, bounds, pop: int, iters: int, seed: int) -> float:
    cfg = FireflyConfig(population_size=pop, iterations=iters, alpha=0.3, beta0=1.0, gamma=1.0, seed=seed)
    return firefly_optimize(objective, bounds, cfg).best_value


def run_tlbo(objective: Benchmark, bounds, pop: int, iters: int, seed: int) -> float:
    cfg = TLBOConfig(population_size=pop, iterations=iters, seed=seed)
    return tlbo_optimize(objective, bounds, cfg).best_value


RUNNERS = {
    "DE": run_de,
    "PSO": run_pso,
    "SOMA": run_soma,
    "FA": run_firefly,
    "TLBO": run_tlbo,
}


def export_to_xlsx(results: Dict[str, Dict[str, List[float]]], out_path: Path) -> None:
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    for func_name, algo_results in results.items():
        ws = wb.create_sheet(title=func_name[:31].capitalize())
        header = ["Experiment"] + ALGORITHM_NAMES
        ws.append(header)

        num_experiments = len(next(iter(algo_results.values())))
        for exp_idx in range(num_experiments):
            row = [exp_idx + 1]
            for algo in ALGORITHM_NAMES:
                row.append(algo_results[algo][exp_idx])
            ws.append(row)

        mean_row = ["Mean"]
        std_row = ["Std Dev"]
        for algo in ALGORITHM_NAMES:
            values = algo_results[algo]
            mean_row.append(mean(values))
            std_row.append(pstdev(values))
        ws.append(mean_row)
        ws.append(std_row)

    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark suite for TLBO vs other algorithms.")
    parser.add_argument("--dims", type=int, default=30, help="Number of dimensions (D).")
    parser.add_argument("--pop", type=int, default=30, help="Population size NP.")
    parser.add_argument("--experiments", type=int, default=30, help="Number of experiments per function.")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per algorithm run.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--out", type=Path, default=Path("tlbo_comparison.xlsx"), help="Output XLSX file.")
    args = parser.parse_args()

    results: Dict[str, Dict[str, List[float]]] = {
        func_name: {algo: [] for algo in ALGORITHM_NAMES} for func_name in FUNCTIONS
    }

    for func_name, func in FUNCTIONS.items():
        bounds2d = DEFAULT_BOUNDS[func_name]
        bounds = get_bounds_for_dimensions(bounds2d, args.dims)

        def objective(vec: np.ndarray) -> float:
            return func(vec)

        for algo in ALGORITHM_NAMES:
            runner = RUNNERS[algo]
            for exp in range(args.experiments):
                seed = args.seed + exp
                best_value = runner(objective, bounds, args.pop, args.iterations, seed)
                results[func_name][algo].append(best_value)
                print(
                    f"{func_name.upper():>10} | {algo:4} | run {exp + 1:02d}/{args.experiments} -> best {best_value:.6f}"
                )

    out_path = Path("img") / args.out if args.out.parent == Path(".") else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_xlsx(results, out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
