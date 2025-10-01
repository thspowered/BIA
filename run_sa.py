from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from benchmarks import (
    sphere,
    ackley,
    rastrigin,
    rosenbrock,
    griewank,
    schwefel,
    levy,
    michalewicz,
    zakharov,
    DEFAULT_BOUNDS,
    get_bounds_for_dimensions,
)
from simulated_annealing import simulated_annealing
from visualize import plot_surface_3d, overlay_trajectory, mark_best


# Mapovanie názvu funkcie -> implementácia funkcie
NAME_TO_FUNC = {
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



def create_all_grid(iters, seed, elev, azim, save_path, initial_temp, cooling_rate, min_temp, step_std):
    """Vytvorí 3x3 grid so všetkými funkciami pre porovnanie Simulated Annealing."""
    fig = plt.figure(figsize=(18, 15))
    functions = list(NAME_TO_FUNC.keys())

    for i, func_name in enumerate(functions):
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        func = NAME_TO_FUNC[func_name]
        bounds2d = DEFAULT_BOUNDS[func_name]
        bounds = get_bounds_for_dimensions(bounds2d, 2)

        # Objektív danej funkcie
        def objective(v: np.ndarray) -> float:
            return func(v)

        # Spusti Simulated Annealing s danými parametrami
        result = simulated_annealing(
            objective, bounds, 
            seed=seed, iterations=iters, 
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            min_temperature=min_temp,
            step_std=step_std
        )

        # Povrch + trajektória využitím helperov z visualize.py
        _, ax_sub = plot_surface_3d(objective, bounds, grid_size=120, elev=elev, azim=azim, ax=ax)
        overlay_trajectory(ax_sub, result.trajectory, objective, every=max(1, iters // 2000), points_only=False)
        mark_best(ax_sub, result.best_point, objective)

        ax_sub.set_title(f"{func_name.capitalize()}\n(best={result.best_value:.4f})", fontsize=10)
        ax_sub.set_xlabel("x1", fontsize=8)
        ax_sub.set_ylabel("x2", fontsize=8)
        ax_sub.set_zlabel("f(x)", fontsize=8)

    fig.suptitle(f"Simulated Annealing on All Benchmark Functions (iters={iters}, seed={seed})", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved grid to {save_path}")


def main():
    # CLI argumenty pre Simulated Annealing
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, nargs="?")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--initial-temp", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--min-temp", type=float, default=0.01)
    parser.add_argument("--step-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elev", type=int, default=35)
    parser.add_argument("--azim", type=int, default=-60)
    parser.add_argument("--save", type=Path, default=Path("figure_sa.png"))

    args = parser.parse_args()
    if not args.function:
        parser.error("Function name is required. Use 'all' for grid.")

    # Príprava výstupného adresára ./img
    project_root = Path(__file__).resolve().parent
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Režim: grid so všetkými funkciami
    if args.function.lower() == "all":
        filename = args.save.name if args.save.name else "all_functions_sa.png"
        if Path(filename).suffix == "":
            filename = f"{filename}.png"
        out_path = img_dir / filename
        create_all_grid(
            args.iters, args.seed, args.elev, args.azim, out_path,
            args.initial_temp, args.cooling_rate, args.min_temp, args.step_std
        )
        return

    # Režim: jedna konkrétna funkcia
    key = args.function.strip().lower()
    if key not in NAME_TO_FUNC:
        raise SystemExit(f"Unknown function '{args.function}'. Available: {sorted(NAME_TO_FUNC.keys())} or 'all'")

    func = NAME_TO_FUNC[key]
    bounds2d = DEFAULT_BOUNDS[key]
    bounds = get_bounds_for_dimensions(bounds2d, 2)

    # Objektív
    def objective(v: np.ndarray) -> float:
        return func(v)

    # Spusti Simulated Annealing
    result = simulated_annealing(
        objective, bounds, 
        seed=args.seed, iterations=args.iters,
        initial_temperature=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temperature=args.min_temp,
        step_std=args.step_std
    )

    # Vykresli povrch a trajektóriu pomocou visualize.py
    fig, ax = plot_surface_3d(objective, bounds, elev=args.elev, azim=args.azim)
    overlay_trajectory(ax, result.trajectory, objective, every=max(1, args.iters // 2000), points_only=False)
    mark_best(ax, result.best_point, objective)
    fig.suptitle(f"Simulated Annealing on {key} (best={result.best_value:.4f})")

    # Uloženie vždy do ./img
    filename = args.save.name if args.save.name else f"{key}_sa.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
