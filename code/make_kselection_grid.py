# make_kselection_grid.py
# ---------------------------------------------------------------------------
# The article's k-selection figure: the three validity-index sweeps
# (Silhouette, Davies-Bouldin, Calinski-Harabasz) and the gap statistic in ONE
# 2x2 grid, in the house blue/grey palette.
#
#   (a) Silhouette (maximize)      (b) Davies-Bouldin (minimize)
#   (c) Calinski-Harabasz (max.)   (d) Gap statistic (+-1 SE)
#
# Design:
#   * strict blue/grey palette, enlarged fonts, single shared bottom legend;
#   * the gap panel shows the main specification only (the "PCA-aligned
#     sensitivity check" is not drawn);
#   * a slate dashed line marks the 1-SE-selected k = 3 on every panel;
#   * no in-figure footer or arrowed callout (the caption carries them).
#
# Values come straight from the analysis CSVs; nothing is recomputed.
#
# Run from code/ :  python make_kselection_grid.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
FAM_DIR = VAULT / "outputs" / "ARLP_Clustering" / "non_esh" / "lens2_families"
KSEL_DIR = VAULT / "outputs" / "ARLP_Clustering" / "non_esh" / "lens2_families_kselection"
OUT_PATH = KSEL_DIR / "kselection_grid.png"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
SLATE = "#676C80"

FS_TITLE = 17
FS_LABEL = 15
FS_TICK = 12
FS_LEGEND = 15

LINE = dict(color=NAVY, lw=2.4, marker="o", ms=7,
            markerfacecolor=NAVY, markeredgecolor=NAVY, solid_capstyle="round")
XLABEL = "number of clusters, $k$"


def kstar_line(ax, kstar):
    ax.axvline(kstar, color=SLATE, ls="--", lw=1.8, zorder=2)


def style(ax, title):
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel(XLABEL, fontsize=FS_LABEL)
    ax.set_xticks(range(2, 8))
    ax.set_xlim(1.7, 7.3)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)


def main():
    val = pd.read_csv(FAM_DIR / "validity_indices.csv")
    val = val[val["k"] >= 2].sort_values("k")
    gap = pd.read_csv(KSEL_DIR / "gap_statistic.csv")
    gap = gap[gap["k"] >= 2].sort_values("k")
    kstar = int(gap.loc[gap["decision_flag"] == "k_star_1se", "k"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    (axA, axB), (axC, axD) = axes

    axA.plot(val["k"], val["silhouette"], zorder=4, **LINE)
    kstar_line(axA, kstar)
    style(axA, "(a) Silhouette (maximize)")

    axB.plot(val["k"], val["davies_bouldin"], zorder=4, **LINE)
    kstar_line(axB, kstar)
    style(axB, "(b) Davies–Bouldin (minimize)")

    axC.plot(val["k"], val["calinski_harabasz"], zorder=4, **LINE)
    kstar_line(axC, kstar)
    style(axC, "(c) Calinski–Harabasz (maximize)")

    axD.errorbar(gap["k"], gap["gap_k"], yerr=gap["s_k"], zorder=4,
                 color=NAVY, lw=2.4, marker="o", ms=7, capsize=5,
                 elinewidth=1.6, capthick=1.6)
    kstar_line(axD, kstar)
    style(axD, "(d) Gap statistic (±1 SE)")

    handles = [
        Line2D([0], [0], label="index / statistic value", **LINE),
        Line2D([0], [0], color=SLATE, ls="--", lw=1.8,
               label=f"selected $k$ = {kstar} (1-SE rule)"),
    ]
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.008))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"kstar = {kstar}")


if __name__ == "__main__":
    main()
