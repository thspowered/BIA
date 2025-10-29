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
from .soma import SOMAConfig, soma_all_to_one
from .viz import plot_soma_search


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
    parser = argparse.ArgumentParser(description="SOMA All-to-One na benchmark funkciách.")
    parser.add_argument("function", type=str, help=f"Dostupné: {', '.join(sorted(NAME_TO_FUNC))}")
    parser.add_argument("--dims", type=int, default=2, help="Počet dimenzií (vizualizácia podporuje iba 2).")
    parser.add_argument("--pop", type=int, default=20, help="Veľkosť populácie (pop_size).")
    parser.add_argument("--iters", type=int, default=100, help="Počet migrácií (M_max).")
    parser.add_argument("--prt", type=float, default=0.4, help="Pravdepodobnosť prenosu dimenzie (PRT).")
    parser.add_argument("--path", type=float, default=3.0, help="PathLength.")
    parser.add_argument("--step", type=float, default=0.11, help="Krok pozdĺž cesty.")
    parser.add_argument("--seed", type=int, default=42, help="Náhodné semienko.")
    parser.add_argument("--save", type=Path, default=Path("soma.png"), help="Výstupný súbor (uložený do ./img).")
    parser.add_argument("--elev", type=int, default=35, help="Uhol elevácie 3D grafu.")
    parser.add_argument("--azim", type=int, default=-60, help="Azimut 3D grafu.")
    parser.add_argument("--no-plot", action="store_true", help="Preskoč vizualizáciu (iba vypíš najlepší bod).")

    args = parser.parse_args()

    key = args.function.strip().lower()
    if key not in NAME_TO_FUNC:
        raise SystemExit(f"Neznáma funkcia '{args.function}'. Dostupné: {sorted(NAME_TO_FUNC.keys())}")

    func = NAME_TO_FUNC[key]
    bounds2d = DEFAULT_BOUNDS[key]
    bounds = get_bounds_for_dimensions(bounds2d, args.dims)

    def objective(v: np.ndarray) -> float:
        return func(v)

    cfg = SOMAConfig(
        population_size=args.pop,
        iterations=args.iters,
        prt=args.prt,
        path_length=args.path,
        step=args.step,
        seed=args.seed,
    )

    result = soma_all_to_one(objective, bounds, config=cfg)

    print(f"Najlepšia hodnota: {result.best_value:.6f}")
    print(f"Najlepší bod: {np.array2string(result.best_point, precision=4)}")

    if args.no_plot or args.dims != 2:
        if args.dims != 2:
            print("Vizualizáciu je možné zobraziť iba pre 2D problémy (spusť s --dims 2).")
        return

    fig = plot_soma_search(objective, bounds, result, elev=args.elev, azim=args.azim)

    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = args.save.name if args.save.name else f"{key}_soma.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Obrázok uložený do {out_path}")


if __name__ == "__main__":
    main()
