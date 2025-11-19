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
from .tlbo import TLBOConfig, tlbo_optimize
from .viz import plot_tlbo_search


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
    parser = argparse.ArgumentParser(description="Teaching-Learning Based Optimization (TLBO).")
    parser.add_argument("function", type=str, help=f"Jedna z: {', '.join(sorted(NAME_TO_FUNC))}")
    parser.add_argument("--dims", type=int, default=2, help="Počet dimenzií (vizualizácia len pre 2D).")
    parser.add_argument("--pop", type=int, default=30, help="Veľkosť triedy (population size).")
    parser.add_argument("--iters", type=int, default=200, help="Počet generácií.")
    parser.add_argument("--seed", type=int, default=42, help="Náhodné semienko.")
    parser.add_argument("--save", type=Path, default=Path("tlbo.png"), help="Názov výstupného obrázka.")
    parser.add_argument("--elev", type=int, default=35, help="3D elevácia.")
    parser.add_argument("--azim", type=int, default=-60, help="3D azimut.")
    parser.add_argument("--no-plot", action="store_true", help="Nevytváraj vizualizáciu.")
    args = parser.parse_args()

    key = args.function.strip().lower()
    if key not in NAME_TO_FUNC:
        raise SystemExit(f"Neznáma funkcia '{args.function}'.")

    func = NAME_TO_FUNC[key]
    bounds2d = DEFAULT_BOUNDS[key]
    bounds = get_bounds_for_dimensions(bounds2d, args.dims)

    def objective(v: np.ndarray) -> float:
        return func(v)

    cfg = TLBOConfig(population_size=args.pop, iterations=args.iters, seed=args.seed)
    result = tlbo_optimize(objective, bounds, config=cfg)

    print(f"Najlepšia hodnota: {result.best_value:.6f}")
    print(f"Najlepší bod: {np.array2string(result.best_point, precision=4)}")

    if args.no_plot or args.dims != 2:
        if args.dims != 2:
            print("Vizualizácia je dostupná len pre 2D (nastav --dims 2).")
        return

    fig = plot_tlbo_search(objective, bounds, result, elev=args.elev, azim=args.azim)

    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = args.save.name if args.save.name else f"{key}_tlbo.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Obrázok uložený do {out_path}")


if __name__ == "__main__":
    main()
