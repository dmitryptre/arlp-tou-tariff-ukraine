# make_price_signals_grid.py
# ---------------------------------------------------------------------------
# The article's price-signal figure: the off-peak / peak / shoulder price
# signals in one 3x2 grid, in the house blue/grey palette.
#
#   rows = TOU zone (off-peak / peak / shoulder)
#   cols = season   (winter / summer)
#   two bars per class: F → P3 and E2 → P3 (labels from _common). For the peak and shoulder zones
#   the two baselines coincide, so those bars are equal by construction.
#
# Values come straight from two_stage_decomposition.csv (Delta r / r^0).
#
# Run from code/ :  python make_price_signals_grid.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import transition_label
from matplotlib.patches import Patch

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
OUT13 = VAULT / "outputs" / "ARLP_Clustering" / "notebook_only" / "layer13_peed_dr"
CSV = OUT13 / "two_stage_decomposition.csv"
OUT_PATH = OUT13 / "price_signals_grid.png"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
SLATE = "#676C80"
LIGHT = "#D2D7DE"

FS_TITLE = 17
FS_LABEL = 15
FS_TICK = 12
FS_LEGEND = 15

CLUSTERS = [0, 1, 2, 3, 4]
CLASS_LABELS = ["A", "B", "C", "ESH-1", "ESH-2"]
# row -> (zone label, flat-baseline column, e2z-baseline column)
ROWS = [("Off-peak", "du_offpeak_flat3z", "du_offpeak_2z3z"),
        ("Peak", "du_peak", "du_peak"),
        ("Shoulder", "du_shoulder", "du_shoulder")]
SEASONS = ["winter", "summer"]


def main():
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    x = np.arange(len(CLUSTERS))
    w = 0.38

    for r, (zone, flat_col, e2z_col) in enumerate(ROWS):
        for c, sn in enumerate(SEASONS):
            ax = axes[r, c]
            sub = df[df["season"] == sn].set_index("cluster").loc[CLUSTERS]
            flat = sub[flat_col].to_numpy() * 100.0
            e2z = sub[e2z_col].to_numpy() * 100.0
            ax.bar(x - w / 2, flat, w, color=LIGHT, edgecolor=SLATE, linewidth=0.7, zorder=3)
            ax.bar(x + w / 2, e2z, w, color=NAVY, zorder=3)
            ax.axhline(0, color=SLATE, lw=1.1, zorder=2)
            if r == 0:  # season labels the column, at the very top only
                ax.set_title(sn.capitalize(), fontsize=FS_TITLE, color=NAVY)
            ax.set_xticks(x)
            ax.set_xticklabels(CLASS_LABELS, fontsize=FS_TICK)
            ax.tick_params(axis="y", labelsize=FS_TICK)
            ax.grid(True, axis="y", alpha=0.25)
            if c == 0:  # zone identifies the row, prepended to the y-axis label
                ax.set_ylabel(f"{zone}   " + r"$\Delta r / r^{0}$ (%)", fontsize=FS_LABEL)

    handles = [
        Patch(facecolor=LIGHT, edgecolor=SLATE, linewidth=0.7, label=transition_label("flat", "p3z")),
        Patch(facecolor=NAVY, edgecolor="none", label=transition_label("e2z", "p3z")),
    ]
    fig.tight_layout(rect=(0, 0.045, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.008))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
