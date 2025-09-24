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
from hill_climbing import hill_climb
from visualize import mark_best


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


def plot_surface_with_path(objective, bounds, path, elev, azim, title):
    """Vykreslí 3D povrch funkcie a ponad neho cestu Hill Climbingu."""
    x_lin = np.linspace(bounds[0][0], bounds[0][1], 120)
    y_lin = np.linspace(bounds[1][0], bounds[1][1], 120)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        pts = np.stack([X[i], Y[i]], axis=1)
        Z[i] = np.apply_along_axis(objective, 1, pts)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    # Jednotný vzhľad povrchu s nižšou opacitou
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap="coolwarm", linewidth=0.2, alpha=0.4)
    ax.view_init(elev=elev, azim=azim)

    # Trajektória vybraných bodov počas hľadania
    z_path = np.apply_along_axis(objective, 1, path)
    ax.plot(path[:, 0], path[:, 1], z_path, color="k", linewidth=1.5, marker=".", markersize=3)

    # Označenie najlepšieho bodu
    best_idx = int(np.argmin(z_path))
    best_point = path[best_idx]
    mark_best(ax, best_point, objective)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    fig.tight_layout()
    return fig


def create_all_grid(iters, seed, elev, azim, save_path, neighbors, std):
    """Vytvorí 3x3 grid so všetkými funkciami pre porovnanie Hill Climbingu."""
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

        # Spusti Hill Climbing s danými parametrami
        result = hill_climb(objective, bounds, seed=seed, iterations=iters, neighbors_per_step=neighbors, step_std=std)

        # Povrch
        x_lin = np.linspace(bounds[0][0], bounds[0][1], 100)
        y_lin = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        Z = np.zeros_like(X)
        for j in range(100):
            pts = np.stack([X[j], Y[j]], axis=1)
            Z[j] = np.apply_along_axis(objective, 1, pts)
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap="coolwarm", linewidth=0.2, alpha=0.4)
        ax.view_init(elev=elev, azim=azim)

        # Cesta + najlepší bod
        path = result.trajectory
        z = np.apply_along_axis(objective, 1, path)
        ax.plot(path[:, 0], path[:, 1], z, color="k", linewidth=1.5, marker=".", markersize=3)
        mark_best(ax, result.best_point, objective)

        ax.set_title(f"{func_name.capitalize()}\n(best={result.best_value:.4f})", fontsize=10)
        ax.set_xlabel("x1", fontsize=8)
        ax.set_ylabel("x2", fontsize=8)
        ax.set_zlabel("f(x)", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved grid to {save_path}")


def main():
    # CLI argumenty pre Hill Climbing (počet krokov, susedov, veľkosť kroku, ...)
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, nargs="?")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elev", type=int, default=35)
    parser.add_argument("--azim", type=int, default=-60)
    parser.add_argument("--save", type=Path, default=Path("figure_hc.png"))

    args = parser.parse_args()
    if not args.function:
        parser.error("Function name is required. Use 'all' for grid.")

    # Príprava výstupného adresára ./img
    project_root = Path(__file__).resolve().parent
    img_dir = project_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Režim: grid so všetkými funkciami
    if args.function.lower() == "all":
        filename = args.save.name if args.save.name else "all_functions_hc.png"
        if Path(filename).suffix == "":
            filename = f"{filename}.png"
        out_path = img_dir / filename
        create_all_grid(args.iters, args.seed, args.elev, args.azim, out_path, args.neighbors, args.std)
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

    # Spusti Hill Climbing
    result = hill_climb(objective, bounds, seed=args.seed, iterations=args.iters, neighbors_per_step=args.neighbors, step_std=args.std)

    # Vykresli povrch a trajektóriu
    fig = plot_surface_with_path(objective, bounds, result.trajectory, args.elev, args.azim, f"Hill Climb on {key} (best={result.best_value:.4f})")

    # Uloženie vždy do ./img
    filename = args.save.name if args.save.name else f"{key}_hc.png"
    if Path(filename).suffix == "":
        filename = f"{filename}.png"
    out_path = img_dir / filename
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
