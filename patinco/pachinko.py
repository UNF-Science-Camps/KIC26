"""Pachinko (Galton board) simulation core.

Pure-stdlib model shared by the browser app (``pachinko_browser.html``)
and the notebook (``pachinko_notebook.ipynb``).

A coin is dropped onto the apex peg. At every peg it bounces left or
right with equal probability (the 50/50 chance), and after ``rows``
bounces it lands in one of ``rows + 1`` bins — a Binomial(rows, 1/2)
distribution.

Vertices are addressed as ``(row, k)``: ``row`` is the peg row counted
from the apex (``row == rows`` is the bin slot the coin lands in) and
``k`` is the position index within that row (0 = leftmost).
"""

from __future__ import annotations

import math
import random

__all__ = ["Coin", "PachinkoBoard"]


class Coin:
    """One coin travelling down the board.

    ``path`` is the list of vertices ``(row, k)`` the coin has visited,
    starting with the apex ``(0, 0)``.
    """

    def __init__(self):
        self.row = 0
        self.k = 0
        self.path = [(0, 0)]
        self.landed = False
        self.bin = None

    @property
    def last_vertex(self):
        """The most recent vertex the coin has moved to."""
        return self.path[-1]

    @property
    def last_move(self):
        """``'L'`` or ``'R'`` for the most recent bounce, ``None`` before the first."""
        if len(self.path) < 2:
            return None
        return "R" if self.path[-1][1] > self.path[-2][1] else "L"


class PachinkoBoard:
    def __init__(self, rows: int = 10, seed=None):
        self.rows = rows
        self.rng = random.Random(seed)
        self.bins = [0] * (rows + 1)
        self.coin: Coin | None = None
        self.total = 0
        # Traffic per edge, for the heatmap: (row, k, dk) -> count, meaning
        # a bounce from vertex (row, k) to (row + 1, k + dk), dk in {0, 1}.
        self.edge_counts: dict[tuple[int, int, int], int] = {}

    # -- simulation ----------------------------------------------------

    def drop(self) -> Coin:
        """Start a new coin at the apex and make it the active coin."""
        self.coin = Coin()
        return self.coin

    def bounce(self):
        """Advance the active coin one bounce.

        Returns ``'L'`` or ``'R'``, or ``None`` if there is no coin in
        flight. Landing (reaching ``row == rows``) updates the bins.
        """
        c = self.coin
        if c is None or c.landed:
            return None
        go_right = self.rng.random() < 0.5
        edge = (c.row, c.k, 1 if go_right else 0)
        self.edge_counts[edge] = self.edge_counts.get(edge, 0) + 1
        c.row += 1
        if go_right:
            c.k += 1
        c.path.append((c.row, c.k))
        if c.row == self.rows:
            c.landed = True
            c.bin = c.k
            self.bins[c.k] += 1
            self.total += 1
        return "R" if go_right else "L"

    def finish(self):
        """Run the active coin all the way down. Returns the coin."""
        while self.coin is not None and not self.coin.landed:
            self.bounce()
        return self.coin

    def drop_many(self, n: int = 100):
        """Drop ``n`` coins instantly (no active coin is left behind)."""
        for _ in range(n):
            self.drop()
            self.finish()
        self.coin = None

    def reset(self):
        """Clear the bins, the edge heatmap and any active coin."""
        self.bins = [0] * (self.rows + 1)
        self.coin = None
        self.total = 0
        self.edge_counts = {}

    # -- probabilities & geometry -------------------------------------

    def bin_probability(self, k: int) -> float:
        """Theoretical probability of landing in bin ``k``: C(rows, k) / 2**rows."""
        return math.comb(self.rows, k) / 2 ** self.rows

    def next_options(self):
        """The two vertices the active coin can bounce to next.

        Returns ``((row+1, k), (row+1, k+1))`` — each reached with
        probability 1/2 — or ``None`` if no coin is in flight.
        """
        c = self.coin
        if c is None or c.landed:
            return None
        return ((c.row + 1, c.k), (c.row + 1, c.k + 1))

    @staticmethod
    def vertex_xy(row: int, k: int):
        """Unit-space coordinates of a vertex: x centred on 0, y = row (downward)."""
        return (k - row / 2.0, float(row))


if __name__ == "__main__":
    # Quick sanity demo: drop 2000 coins and print the distribution.
    board = PachinkoBoard(rows=10, seed=1)
    board.drop_many(2000)
    peak = max(board.bins)
    for k, count in enumerate(board.bins):
        bar = "#" * round(40 * count / peak)
        print(f"bin {k:2d}  {count:5d}  ({board.bin_probability(k):6.2%} expected)  {bar}")
