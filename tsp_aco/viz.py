from __future__ import annotations

from typing import List

from tsp_ga.viz import LiveTSPPlot
from tsp_ga.tsp import City


class LiveACOPlot(LiveTSPPlot):
    """Live vizualizácia TSP, prispôsobená pre Ant Colony Optimization."""

    def __init__(self, cities: List[City]):
        super().__init__(cities)
        self.ax.set_title("TSP via Ant Colony Optimization")

    def update(self, tour: List[int], cities: List[City] | None = None, title_suffix: str | None = None) -> None:
        # Využi pôvodnú logiku na prekreslenie cesty, ale uprav titulok na ACO
        super().update(tour, cities=cities, title_suffix=None)
        if title_suffix:
            self.ax.set_title(f"TSP via Ant Colony Optimization — {title_suffix}")
        else:
            self.ax.set_title("TSP via Ant Colony Optimization")
