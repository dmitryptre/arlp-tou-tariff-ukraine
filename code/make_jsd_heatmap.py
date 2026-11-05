# make_jsd_heatmap.py
# ---------------------------------------------------------------------------
# The article's Jensen-Shannon distance heatmap of the 16 non-ESH ARLP weight
# vectors, in the house blue/grey palette.
#
# Plain format -- Ward-clustered ordering, no dendrogram, no embedded title --
# with two design choices that make the map informative:
#   * the near-saturated MAXIMUM-distance plateau (224 of 240 off-diagonal
#     cells all sit at JSD ~= 0.83) carries no information but would dominate
#     the map as solid navy. Those cells are shown by diagonal HATCHING on white
#     (house rule: emphasise/denote with hatching, not a colour), with a legend
#     swatch defining it as the maximum distance.
#   * with the plateau removed from the colour scale, the colormap is rescaled
#     to the informative range (0 .. ~0.25) so the low-distance city/rural pairs
#     get real colour contrast instead of all reading as pale.
#
# Run from code/ :  python make_jsd_heatmap.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
LENS1_DIR = VAULT / "outputs" / "ARLP_Clustering" / "non_esh" / "lens1_compositional"
CSV_PATH = LENS1_DIR / "jsd_distance_matrix.csv"
OUT_PATH = LENS1_DIR / "jsd_heatmap_recolored.png"

# House blue/grey sequential ramp (low distance = light, high = navy).
HOUSE_SEQ = LinearSegmentedColormap.from_list(
    "house_bluegrey",
    ["#FFFFFF", "#D2D7DE", "#5C7999", "#44586C", "#1A2C3C"],
)
C_NAVY = "#1A2C3C"       # hatch stroke + dark end of the informative ramp

# Cells with JSD at/above this threshold are the saturated maximum-distance
# plateau: uninformative, so they are hatched rather than colour-filled. The
# informative pairs top out well below it (max ~0.23), so any cut in between
# works; 0.80 sits cleanly in the empty gap before the ~0.83 plateau.
HATCH_MIN = 0.80
plt.rcParams["hatch.linewidth"] = 1.3

# Enlarged house type scale.
FS_TICK = 12
FS_ANNOT = 15
FS_CBAR = 12
FS_LABEL = 15


def main():
    jsd = pd.read_csv(CSV_PATH, index_col=0)

    # Ward's ordering on the condensed distances -> the clustered leaf order
    # that places each city/rural pair adjacent (05-13, 11-19, 09-17, ...).
    order = leaves_list(linkage(squareform(jsd.values, checks=False), method="ward"))
    jsd = jsd.iloc[order, order]
    data = jsd.to_numpy()
    n = data.shape[0]

    # Split the map: the saturated maximum plateau (hatched) vs the informative
    # cells (colour-filled). Rescale the colour ramp to the informative range so
    # the low-distance pairs span the full palette instead of the pale end.
    hatch_mask = data >= HATCH_MIN
    plateau = float(data.max())
    info_max = float(data[(data > 0) & (~hatch_mask)].max())
    vmax_info = float(np.ceil(info_max / 0.05) * 0.05)   # 0.23 -> 0.25

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        jsd,
        cmap=HOUSE_SEQ,
        vmin=0.0,
        vmax=vmax_info,
        mask=hatch_mask,             # leave the max-plateau cells unpainted
        annot=False,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.6},
        ax=ax,
    )
    ax.set_facecolor("white")

    # Diagonal hatching on every max-plateau cell (denotes the saturated maximum;
    # facecolor none lets the white background show between the navy strokes).
    for i, j in zip(*np.where(hatch_mask)):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor=C_NAVY, hatch="////", lw=0.0, zorder=2))

    # Manual cell labels for informative + diagonal cells, text colour flipped to
    # white on the darker fills so every value stays legible.
    norm = plt.Normalize(0.0, vmax_info)
    for i in range(n):
        for j in range(n):
            if hatch_mask[i, j]:
                continue
            v = data[i, j]
            r, g, b, _ = HOUSE_SEQ(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                    fontsize=FS_ANNOT, color="white" if lum < 0.5 else C_NAVY, zorder=3)

    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS_CBAR)
    cbar.set_label("Jensen–Shannon distance", size=FS_LABEL)

    # Legend defining the hatch as the maximum distance (drawn beneath the map,
    # below the rotated x tick labels).
    hatch_patch = Patch(facecolor="white", edgecolor=C_NAVY, hatch="////",
                        label=f"Maximum distance (≈ {plateau:.2f})")
    ax.legend(handles=[hatch_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.11), frameon=False, fontsize=FS_ANNOT)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"informative vmax={vmax_info}  plateau={plateau:.3f}  hatched={int(hatch_mask.sum())} cells")
    print(f"clustered order: {list(jsd.index)}")


if __name__ == "__main__":
    main()
