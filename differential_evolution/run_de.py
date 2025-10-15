from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
from .de import DEConfig, differential_evolution
from .viz import plot_de_search


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Differential Evolution (DE/rand/1/bin) on benchmark functions.")
    parser.add_argument("function", type=str, help=f"One of: {', '.join(sorted(NAME_TO_FUNC))}")
    parser.add_argument("--dims", type=int, default=2, help="Number of dimensions (visualization supports only 2).")
    parser.add_argument("--pop", type=int, default=20, help="Population size NP.")
    parser.add_argument("--gens", type=int, default=50, help="Number of generations G_max.")
    parser.add_argument("--f", type=float, default=0.5, help="Differential weight F.")
    parser.add_argument("--cr", type=float, default=0.5, help="Crossover rate CR.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save", type=Path, default=Path("de.png"), help="Output filename (stored under ./img).")
    parser.add_argument("--elev", type=int, default=35, help="3D view elevation.")
    parser.add_argument("--azim", type=int, default=-60, help="3D view azimuth.")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization (just print the best result).")

    args = parser.parse_args()

    key = args.function.strip().lower()
    if key not in NAME_TO_FUNC:
        raise SystemExit(f"Unknown function '{args.function}'. Available: {sorted(NAME_TO_FUNC.keys())}")

    func = NAME_TO_FUNC[key]
    bounds2d = DEFAULT_BOUNDS[key]
    bounds = get_bounds_for_dimensions(bounds2d, args.dims)

    def objective(v: np.ndarray) -> float:
        return func(v)

    config = DEConfig(
        population_size=args.pop,
        generations=args.gens,
        differential_weight=args.f,
        crossover_rate=args.cr,
        seed=args.seed,
    )

    result = differential_evolution(objective, bounds, config=config)

    print(f"Best value: {result.best_value:.6f}")
    print(f"Best point: {np.array2string(result.best_point, precision=4)}")

    if args.no_plot or args.dims != 2:
        if args.dims != 2:
            print("Visualization is only available for 2D problems. Rerun with --dims 2 to generate the figure.")
        return

    fig = plot_de_search(objective, bounds, result, elev=args.elev, azim=args.azim)

    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = args.save.name if args.save.name else f"{key}_de.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
