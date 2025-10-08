from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt

from .tsp import City


class LiveTSPPlot:
    def __init__(self, cities: List[City]):
        # základné vykreslenie bodov miest a príprava línie pre trasu
        self.cities = cities
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        xs = [c.x for c in cities]
        ys = [c.y for c in cities]
        self.ax.scatter(xs, ys, c='red')
        for c in cities:
            self.ax.text(c.x + 2, c.y + 2, c.label, fontsize=8)
        self.line, = self.ax.plot([], [], '-o', color='red', alpha=0.6)
        self.ax.set_xlim(min(xs) - 10, max(xs) + 10)
        self.ax.set_ylim(min(ys) - 10, max(ys) + 10)
        self.ax.set_title('TSP via Genetic Algorithm')
        self.fig.tight_layout()

    def update(self, tour: List[int], cities: List[City] | None = None, title_suffix: str | None = None) -> None:
        # aktualizácia spojnice podľa aktuálne najlepšej permutácie `tour`
        cs = cities or self.cities
        path_x = [cs[i].x for i in tour] + [cs[tour[0]].x]
        path_y = [cs[i].y for i in tour] + [cs[tour[0]].y]
        self.line.set_data(path_x, path_y)
        if title_suffix:
            self.ax.set_title(f'TSP via Genetic Algorithm — {title_suffix}')
        plt.pause(0.001)

    def show(self) -> None:
        # blokujúce zobrazenie okna s grafom po skončení evolúcie
        plt.show()


