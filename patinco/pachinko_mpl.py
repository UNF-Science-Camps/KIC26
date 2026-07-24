"""Matplotlib renderer for the pachinko board (used by the notebook).

``draw_board(board)`` returns a figure showing the pegs, the active
coin, the last edge it moved along with the last vertex highlighted
(``full_path=True`` draws the whole travelled path), the 50/50 arrows
at the next decision point, and the bin histogram with expected
percentages.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from pachinko import PachinkoBoard

# Light-mode palette (see README — reference dataviz palette).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAR = "#2a78d6"      # bin histogram (single series)
COIN = "#eb6834"     # coin + travelled path
ARROW = "#1baf7a"    # 50/50 arrows
LAST = "#e34948"     # last-vertex highlight

BIN_GAP = 0.6        # vertical gap between peg field and histogram (rows units)
BAR_MAX_H = 2.6      # tallest bar, in rows units

# Sequential blue ramp for the edge heatmap (light -> dark = few -> many)
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]


def _status_line(board: PachinkoBoard) -> str:
    c = board.coin
    if c is None:
        return f"No coin in flight — {board.total} coin(s) dropped"
    if c.landed:
        return (f"Landed in bin {c.bin} "
                f"(expected share {board.bin_probability(c.bin):.1%}) — "
                f"{board.total} coin(s) dropped")
    move = {"L": "← left", "R": "→ right", None: "dropped on apex"}[c.last_move]
    r, k = c.last_vertex
    return (f"Bounce {c.row}/{board.rows} · last move: {move} · "
            f"now at vertex (row {r}, pos {k})")


def draw_board(board: PachinkoBoard, show_pct: bool = True,
               full_path: bool = False, heatmap: bool = False,
               figsize=(6.4, 7.6)):
    """Render the board state. Returns the matplotlib Figure."""
    rows = board.rows
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")

    # Heatmap of all edges travelled so far (under everything else)
    if heatmap:
        if board.edge_counts:
            peak_e = max(board.edge_counts.values())
            for (r, k, dk), n in board.edge_counts.items():
                t = (n / peak_e) ** 0.5      # sqrt keeps rare edges visible
                x1, y1 = board.vertex_xy(r, k)
                x2, y2 = board.vertex_xy(r + 1, k + dk)
                ax.plot([x1, x2], [y1, y2], color=SEQ[round(t * (len(SEQ) - 1))],
                        lw=1.2 + 2.4 * t, zorder=1.5, solid_capstyle="round")
        # few <-> many legend, top-left corner
        lx, ly, lw_, lh = -rows / 2 - 0.2, 0.15, 1.5, 0.16
        for i, color in enumerate(SEQ):
            ax.add_patch(Rectangle((lx + lw_ * i / len(SEQ), ly),
                                   lw_ / len(SEQ) * 1.05, lh, color=color, zorder=3))
        ax.text(lx - 0.12, ly + lh / 2, "few", color=MUTED, fontsize=8,
                ha="right", va="center", zorder=3)
        ax.text(lx + lw_ + 0.12, ly + lh / 2, "many", color=MUTED, fontsize=8,
                ha="left", va="center", zorder=3)

    # Pegs
    for r in range(rows):
        for k in range(r + 1):
            x, y = board.vertex_xy(r, k)
            ax.add_patch(Circle((x, y), 0.11, color=MUTED, zorder=3))

    coin = board.coin

    # Last edge in red; the full travelled path only when toggled on
    if coin is not None and len(coin.path) > 1:
        pts = [board.vertex_xy(r, k) for r, k in coin.path]
        xs, ys = zip(*pts)
        if full_path:
            ax.plot(xs, ys, color=COIN, lw=2, alpha=0.55, zorder=2)
        ax.plot(xs[-2:], ys[-2:], color=LAST, lw=2.6, zorder=2)
    if coin is not None:
        lx, ly = board.vertex_xy(*coin.last_vertex)
        ax.add_patch(Circle((lx, ly), 0.24, fill=False, color=LAST, lw=2, zorder=4))
        ax.add_patch(Circle((lx, ly - 0.28), 0.16, color=COIN, ec=INK, lw=0.8, zorder=5))

    # 50/50 arrows to the two possible next vertices
    if show_pct and coin is not None and not coin.landed:
        cx, cy = board.vertex_xy(*coin.last_vertex)
        for (nr, nk), side in zip(board.next_options(), (-1, 1)):
            tx, ty = board.vertex_xy(nr, nk)
            ax.annotate("", xy=(tx, ty - 0.18), xytext=(cx + 0.12 * side, cy + 0.12),
                        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2), zorder=4)
            mx, my = (cx + tx) / 2 + 0.42 * side, (cy + ty) / 2
            ax.text(mx, my, "50%", color=INK, fontsize=10, zorder=5,
                    ha="right" if side < 0 else "left", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))

    # Bins: separators, bars, count labels, expected percentages
    top = rows + BIN_GAP
    base = top + BAR_MAX_H + 0.35
    for i in range(rows + 2):
        x = (i - 0.5) - rows / 2.0
        ax.plot([x, x], [top, base], color=GRID, lw=1, zorder=1)
    ax.plot([-rows / 2 - 0.5, rows / 2 + 0.5], [base, base], color=BASELINE, lw=1.2, zorder=1)

    peak = max(board.bins) or 1
    # Hide the count still animating in the browser version; here bins are final.
    for k, count in enumerate(board.bins):
        x, _ = board.vertex_xy(rows, k)
        if count:
            h = BAR_MAX_H * count / peak
            ax.add_patch(Rectangle((x - 0.36, base - h), 0.72, h,
                                   color=BAR, zorder=2))
            ax.text(x, base - h - 0.14, str(count), color=INK_2, fontsize=9,
                    ha="center", va="bottom", zorder=3)
        if show_pct:
            ax.text(x, base + 0.32, f"{board.bin_probability(k):.1%}",
                    color=MUTED, fontsize=8, ha="center", va="top", zorder=3)

    ax.set_title(_status_line(board), color=INK_2, fontsize=10, loc="left", pad=10)
    ax.set_xlim(-rows / 2 - 0.9, rows / 2 + 0.9)
    ax.set_ylim(base + (0.95 if show_pct else 0.3), -0.9)  # inverted: apex on top
    fig.tight_layout()
    return fig
