"""make_assignment_tree_full.py — full tariff-class assignment tree (all 5 classes).

Companion to make_assignment_tree.py (non-ESH only). Shows the whole split:
the first routing variable is heating fuel (ESH vs non-ESH), then a consumption
split on each stratum, plus the dwelling split inside the non-ESH branch.

Each decision node shows only its criterion; each leaf only the assigned class.
Same house style (blue/grey palette, Arial, no embedded title, dpi 130).

Run:
    python make_assignment_tree_full.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
import manuscript_style as ms  # noqa: E402

# Reuse the exact node/edge drawing helpers from the non-ESH figure.
from make_assignment_tree import node, edge  # noqa: E402

OUT = _HERE.parent / "outputs" / "ARLP_Clustering" / "notebook_figures" / "assignment_tree_full.png"


def main():
    ms.apply_house_rcparams()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-0.2, 16.0)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Root: heating-fuel split. Non-ESH stratum drawn on the left, ESH on the
    # right, so the leaves read A, B, C, ESH-1, ESH-2 from left to right.
    root = node(ax, 8.6, 9.0, 5.0, 1.3, "Electric space heating ?", leaf=False)

    # Level 1: one consumption split per stratum.
    non_split = node(ax, 5.0, 6.2, 5.4, 1.3, "Consumption > 6 MWh/y ?", leaf=False)
    esh_split = node(ax, 12.25, 6.2, 5.4, 1.3, "Consumption > 7.2 MWh/y ?", leaf=False)

    # Non-ESH stratum (left): dwelling split + Class C leaf.
    dwelling = node(ax, 3.0, 3.4, 3.7, 1.2, "Dwelling = SFH ?", leaf=False)
    class_c = node(ax, 7.0, 3.4, 2.3, 1.1, "Class C", leaf=True)
    class_a = node(ax, 1.5, 0.7, 2.3, 1.1, "Class A", leaf=True)
    class_b = node(ax, 4.5, 0.7, 2.3, 1.1, "Class B", leaf=True)

    # ESH stratum (right) leaves.
    esh1 = node(ax, 10.3, 3.4, 3.2, 1.1, "Class ESH-1", leaf=True)
    esh2 = node(ax, 14.2, 3.4, 3.2, 1.1, "Class ESH-2", leaf=True)

    # Edges (branch answers).
    edge(ax, root, non_split, "No")        # no ESH   -> non-ESH stratum (left)
    edge(ax, root, esh_split, "Yes")       # has ESH  -> ESH stratum (right)
    edge(ax, non_split, dwelling, "No")    # <= 6 MWh/y   -> dwelling split
    edge(ax, non_split, class_c, "Yes")    # > 6 MWh/y    -> Class C
    edge(ax, dwelling, class_a, "No")      # MFH -> Class A
    edge(ax, dwelling, class_b, "Yes")     # SFH -> Class B
    edge(ax, esh_split, esh1, "No")        # <= 7.2 MWh/y -> ESH-1
    edge(ax, esh_split, esh2, "Yes")       # > 7.2 MWh/y  -> ESH-2

    ms.save(fig, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
