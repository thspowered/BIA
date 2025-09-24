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
from blind_search import blind_search
from visualize import plot_surface_3d, overlay_trajectory, mark_best


# Mapovanie názvu funkcie -> samotná implementácia
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


def create_all_functions_grid(iters: int, seed: int, elev: int, azim: int, save_path: Path):
    """Vytvorí 3x3 grid všetkých funkcií s trajektóriou Blind Search.

    - pre každú funkciu použije jej odporúčané hranice
    - spustí slepé hľadanie a vykreslí povrch + trajektóriu + najlepší bod
    """
    fig = plt.figure(figsize=(18, 15))
    
    functions = list(NAME_TO_FUNC.keys())
    
    for i, func_name in enumerate(functions):
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        
        func = NAME_TO_FUNC[func_name]
        bounds2d = DEFAULT_BOUNDS[func_name]
        bounds = get_bounds_for_dimensions(bounds2d, 2)
        
        # Uzavretie objektívu pre aktuálnu funkciu
        def make_objective(f):
            def objective(v: np.ndarray) -> float:
                return f(v)
            return objective
        
        objective = make_objective(func)
        
        # Spusti Blind Search
        result = blind_search(objective, bounds, iterations=iters, seed=seed)
        
        # Vypočítaj povrch pre mriežku a vykresli
        x_lin = np.linspace(bounds[0][0], bounds[0][1], 100)
        y_lin = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        Z = np.zeros_like(X)
        for j in range(100):
            pts = np.stack([X[j], Y[j]], axis=1)
            Z[j] = np.apply_along_axis(objective, 1, pts)
        
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap="coolwarm", linewidth=0.2, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        
        # Trajektória + označenie najlepšieho bodu
        xyz = result.trajectory[:, :2]
        z = np.apply_along_axis(objective, 1, xyz)
        every = max(1, iters // 1000)
        ax.plot(xyz[::every, 0], xyz[::every, 1], z[::every], marker=".", color="k", linewidth=0.5, markersize=2, alpha=0.7)
        mark_best(ax, result.best_point, objective)
        
        ax.set_title(f"{func_name.capitalize()}\n(best={result.best_value:.4f})", fontsize=10)
        ax.set_xlabel("x1", fontsize=8)
        ax.set_ylabel("x2", fontsize=8)
        ax.set_zlabel("f(x)", fontsize=8)
    
    fig.suptitle(f"Blind Search on All Benchmark Functions (iters={iters}, seed={seed})", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved grid to {save_path}")


def main():
    # CLI argumenty pre ovládanie experimentu a vizualizácie
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, nargs="?")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=Path("figure.png"))
    parser.add_argument("--elev", type=int, default=35)
    parser.add_argument("--azim", type=int, default=-60)

    args = parser.parse_args()

    if not args.function:
        parser.error("Function name is required. Use 'all' for grid of all functions.")

    # Režim: vykresli grid všetkých funkcií
    if args.function.lower() == "all":
        project_root = Path(__file__).resolve().parent
        img_dir = project_root / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        filename = args.save.name if args.save.name else "all_functions_grid.png"
        if Path(filename).suffix == "":
            filename = f"{filename}.png"
        out_path = img_dir / filename
        
        create_all_functions_grid(args.iters, args.seed, args.elev, args.azim, out_path)
        return

    # Režim: jedna konkrétna funkcia
    key = args.function.strip().lower()
    if key not in NAME_TO_FUNC:
        raise SystemExit(f"Unknown function '{args.function}'. Available: {sorted(NAME_TO_FUNC.keys())} or 'all'")

    func = NAME_TO_FUNC[key]
    bounds2d = DEFAULT_BOUNDS[key]
    bounds = get_bounds_for_dimensions(bounds2d, args.dims)

    # Objektív pre danú funkciu
    def objective(v: np.ndarray) -> float:
        return func(v)

    # Spusti Blind Search
    result = blind_search(objective, bounds, iterations=args.iters, seed=args.seed)

    # Pre 2D vykresli povrch + trajektóriu + najlepší bod
    if args.dims != 2:
        print(f"Best value: {result.best_value:.6f} at {result.best_point}")
        print("Visualization available only for 2D. Rerun with --dims 2 to plot.")
        return

    fig, ax = plot_surface_3d(objective, bounds, elev=args.elev, azim=args.azim)
    overlay_trajectory(ax, result.trajectory, objective, every=max(1, args.iters // 2000))
    mark_best(ax, result.best_point, objective)
    fig.suptitle(f"Blind Search on {key} (best={result.best_value:.4f})")
    fig.tight_layout()

    # Uloženie vždy do ./img
    project_root = Path(__file__).resolve().parent
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = args.save.name if args.save.name else f"{key}.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename

    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
