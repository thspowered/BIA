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
from .firefly import FireflyConfig, firefly_optimize
from .viz import plot_firefly_search


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
    parser = argparse.ArgumentParser(description="Firefly algorithm on benchmark functions.")
    parser.add_argument("function", type=str, help=f"Jedna z: {', '.join(sorted(NAME_TO_FUNC))}")
    parser.add_argument("--dims", type=int, default=2, help="Počet dimenzií (vizualizácia podporuje iba 2).")
    parser.add_argument("--pop", type=int, default=3, help="Počet svätojánskych mušiek.")
    parser.add_argument("--iters", type=int, default=50, help="Počet generácií.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Amplitúda náhodného kroku α.")
    parser.add_argument("--beta0", type=float, default=1.0, help="Počiatočná príťažlivosť β0.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Absorpčný koeficient γ.")
    parser.add_argument("--seed", type=int, default=42, help="Náhodné semienko.")
    parser.add_argument("--save", type=Path, default=Path("firefly.png"), help="Výstupný obrázok (uložený do ./img).")
    parser.add_argument("--elev", type=int, default=35, help="Elevácia 3D grafu.")
    parser.add_argument("--azim", type=int, default=-60, help="Azimut 3D grafu.")
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

    cfg = FireflyConfig(
        population_size=args.pop,
        iterations=args.iters,
        alpha=args.alpha,
        beta0=args.beta0,
        gamma=args.gamma,
        seed=args.seed,
    )

    result = firefly_optimize(objective, bounds, config=cfg)

    print(f"Najlepšia nájdená hodnota: {result.best_value:.6f}")
    print(f"Najlepší bod: {np.array2string(result.best_point, precision=4)}")

    if args.no_plot or args.dims != 2:
        if args.dims != 2:
            print("Vizualizáciu je možné zobraziť iba pre 2D úlohy (nastav --dims 2).")
        return

    fig = plot_firefly_search(objective, bounds, result, elev=args.elev, azim=args.azim)

    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = args.save.name if args.save.name else f"{key}_firefly.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Obrázok uložený do {out_path}")


if __name__ == "__main__":
    main()
