"""make_assignment_tree.py — non-ESH tariff-class assignment tree.

Draws the sklearn-style assignment tree as a clean house-style diagram:
each decision node shows only its **criterion**, each leaf shows only the
**resulting class name** (without the sklearn `samples = …` / `value = […]` lines).

Style follows `manuscript_style.py` — blue/grey palette
only, Arial, no embedded figure title, white background, dpi 130, tight bbox.

Run:
    python make_assignment_tree.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Import the house-style helper module (palette + rcParams + save()).
_HERE = Path(__file__).resolve().parent
import manuscript_style as ms  # noqa: E402

OUT = _HERE.parent / "outputs" / "ARLP_Clustering" / "notebook_figures" / "assignment_tree_non_esh.png"

# Node text size scale (a diagram, not a data chart, so no data axes).
FS_CRITERION = 16   # decision-node criterion
FS_CLASS = 17       # leaf class name
FS_EDGE = 14        # Yes/No branch labels


def node(ax, x, y, w, h, text, *, leaf):
    """Draw one rounded box centred at (x, y). Leaves are navy-filled with
    white text; decision nodes are a light lavender fill with navy text."""
    fc = ms.NAVY if leaf else ms.LAVENDER
    tc = ms.WHITE if leaf else ms.NAVY
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=2.0, edgecolor=ms.NAVY, facecolor=fc, mutation_aspect=1, zorder=3,
    ))
    ax.text(x, y, text, ha="center", va="center", color=tc, zorder=5,
            fontsize=FS_CLASS if leaf else FS_CRITERION,
            fontweight="bold" if leaf else "normal")
    return (x, y, w, h)


def edge(ax, parent, child, label):
    """Arrow from the bottom of the parent box to the top of the child box,
    with a Yes/No branch label on a small white patch for legibility."""
    px, py, _, ph = parent
    cx, cy, _, ch = child
    a = (px, py - ph / 2)
    b = (cx, cy + ch / 2)
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=20, lw=2.2,
        color=ms.NAVY, shrinkA=1, shrinkB=1, zorder=2,
    ))
    lx, ly = (a[0] + b[0]) / 2, a[1] + (b[1] - a[1]) * 0.45
    ax.text(lx, ly, label, ha="center", va="center", fontsize=FS_EDGE,
            color=ms.NAVY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc=ms.WHITE, ec="none"))


def main():
    ms.apply_house_rcparams()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Nodes (positions mirror the sklearn tree layout: Class C top-right,
    # the dwelling split feeding Class A / Class B along the bottom).
    root = node(ax, 5.2, 8.6, 5.5, 1.3, "Consumption > 6 MWh/y ?", leaf=False)
    dwelling = node(ax, 3.0, 5.0, 3.5, 1.3, "Dwelling = SFH ?", leaf=False)
    class_c = node(ax, 8.9, 5.0, 2.1, 1.1, "Class C", leaf=True)
    class_a = node(ax, 1.5, 1.5, 2.1, 1.1, "Class A", leaf=True)
    class_b = node(ax, 5.0, 1.5, 2.1, 1.1, "Class B", leaf=True)

    # Edges (branch answers).
    edge(ax, root, dwelling, "No")        # <= 6 MWh/y -> dwelling split
    edge(ax, root, class_c, "Yes")        # > 6 MWh/y  -> Class C
    edge(ax, dwelling, class_a, "No")     # MFH        -> Class A
    edge(ax, dwelling, class_b, "Yes")    # SFH        -> Class B

    ms.save(fig, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
