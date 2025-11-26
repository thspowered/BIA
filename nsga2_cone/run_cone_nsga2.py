from __future__ import annotations

import argparse

from pathlib import Path

from .nsga2 import NSGA2Config, nsga2_cone
from .viz import plot_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="NSGA-II for the cone design problem.")
    parser.add_argument("--pop", type=int, default=120, help="Population size.")
    parser.add_argument("--gens", type=int, default=200, help="Number of generations.")
    parser.add_argument("--crossover", type=float, default=0.9, help="Crossover probability.")
    parser.add_argument("--mutation", type=float, default=0.2, help="Mutation probability per variable.")
    parser.add_argument("--eta-c", type=float, default=20.0, help="SBX distribution index.")
    parser.add_argument("--eta-m", type=float, default=20.0, help="Polynomial mutation index.")
    parser.add_argument("--volume", type=float, default=None, help="Fix cone volume to this value (optional).")
    parser.add_argument("--penalty", type=float, default=1e6, help="Penalty weight for volume constraint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save", type=Path, default=Path("nsga2_cone.png"), help="Where to save the plot.")
    args = parser.parse_args()

    cfg = NSGA2Config(
        population_size=args.pop,
        generations=args.gens,
        crossover_rate=args.crossover,
        mutation_rate=args.mutation,
        eta_c=args.eta_c,
        eta_m=args.eta_m,
        target_volume=args.volume,
        penalty_weight=args.penalty,
        seed=args.seed,
    )

    bounds = ((1e-3, 10.0), (1e-3, 20.0))
    result = nsga2_cone(bounds, cfg)

    pareto_points = result.population[result.pareto_front]
    pareto_objs = result.objectives[result.pareto_front]
    print(f"Found {len(pareto_points)} Pareto-optimal solutions.")
    print("Example Pareto point (r, h, S, T):")
    for (r, h), (S, T) in zip(pareto_points[:5], pareto_objs[:5]):
        print(f"  r={r:.3f}, h={h:.3f}, S={S:.3f}, T={T:.3f}")

    out_path = Path("img") / args.save if args.save.parent == Path(".") else args.save
    plot_pareto(result, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
