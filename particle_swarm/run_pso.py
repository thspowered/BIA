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
from .pso import PSOConfig, particle_swarm_optimization
from .viz import plot_pso_search


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
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization (PSO) so zotrvačnosťou.")
    parser.add_argument("function", type=str, help=f"Dostupné funkcie: {', '.join(sorted(NAME_TO_FUNC))}")
    parser.add_argument("--dims", type=int, default=2, help="Počet dimenzií (vizualizácia podporuje iba 2).")
    parser.add_argument("--pop", type=int, default=15, help="Veľkosť roja (popsize).")
    parser.add_argument("--iters", type=int, default=50, help="Počet iterácií M_max.")
    parser.add_argument("--w", type=float, default=0.7, help="Váha zotrvačnosti (inertia weight).")
    parser.add_argument("--c1", type=float, default=2.0, help="Kognitívny koeficient c1.")
    parser.add_argument("--c2", type=float, default=2.0, help="Sociálny koeficient c2.")
    parser.add_argument("--vclamp", type=float, default=None, help="Maximálna absolútna rýchlosť (voliteľné).")
    parser.add_argument("--seed", type=int, default=42, help="Náhodné semienko.")
    parser.add_argument("--save", type=Path, default=Path("pso.png"), help="Výstupný súbor (uložený do ./img).")
    parser.add_argument("--elev", type=int, default=35, help="Uhol elevácie 3D grafu.")
    parser.add_argument("--azim", type=int, default=-60, help="Azimut 3D grafu.")
    parser.add_argument("--no-plot", action="store_true", help="Preskoč vizualizáciu (iba vypíš najlepšie riešenie).")

    args = parser.parse_args()

    key = args.function.strip().lower()
    if key not in NAME_TO_FUNC:
        raise SystemExit(f"Neznáma funkcia '{args.function}'. Dostupné: {sorted(NAME_TO_FUNC.keys())}")

    func = NAME_TO_FUNC[key]
    bounds2d = DEFAULT_BOUNDS[key]
    bounds = get_bounds_for_dimensions(bounds2d, args.dims)

    def objective(v: np.ndarray) -> float:
        return func(v)

    cfg = PSOConfig(
        population_size=args.pop,
        iterations=args.iters,
        inertia_weight=args.w,
        cognitive_coef=args.c1,
        social_coef=args.c2,
        velocity_clamp=args.vclamp,
        seed=args.seed,
    )

    result = particle_swarm_optimization(objective, bounds, config=cfg)

    print(f"Najlepšia hodnota: {result.best_value:.6f}")
    print(f"Najlepší bod: {np.array2string(result.best_point, precision=4)}")

    if args.no_plot or args.dims != 2:
        if args.dims != 2:
            print("Vizualizácia je k dispozícii iba pre 2D problémy. Spusť s --dims 2.")
        return

    fig = plot_pso_search(objective, bounds, result, elev=args.elev, azim=args.azim)

    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = args.save.name if args.save.name else f"{key}_pso.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Obrázok uložený do {out_path}")


if __name__ == "__main__":
    main()
