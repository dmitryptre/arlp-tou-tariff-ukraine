# make_pca_maps_combined.py
# ---------------------------------------------------------------------------
# The article's PCA maps: the non-ESH family PCA map (3 clusters) and the ESH
# ARLP PCA map in a SINGLE horizontally stacked (1x2) diagram, in the house
# blue/grey palette.
#
#   (a) Non-ESH families — 8 family scores with the k=3 cluster ellipses,
#       in strict house tones.
#   (b) ESH ARLPs        — 3 ARLP scores (too few points for meaningful
#       cluster ellipses).
#
# Coordinates come straight from the PCA score CSVs the analysis already writes,
# so nothing is recomputed. Cluster membership is assigned by nearest k=3 medoid
# (06_14 / 09_17 / 07_15), reproducing the k=3 family grouping of the article.
#
# Run from code/ :  python make_pca_maps_combined.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
OUT_ROOT = VAULT / "outputs" / "ARLP_Clustering"
NONESH_DIR = OUT_ROOT / "non_esh" / "lens2_families"
ESH_DIR = OUT_ROOT / "esh" / "lens2_tou_features"
OUT_PATH = OUT_ROOT / "tou_annotation" / "pca_maps_combined.png"

# ── House palette (blue/grey only) ─────────────────────────────────────────
NAVY = "#1A2C3C"
SLATE = "#676C80"
BLUEGRAY = "#5C7999"
LIGHT = "#D2D7DE"

POINT_FILL = BLUEGRAY
POINT_EDGE = NAVY
ELLIPSE_COLOR = NAVY

FS_TITLE = 17
FS_LABEL = 15
FS_TICK = 12
FS_ANNOT = 12

# k=3 non-ESH medoids (as reported in the article) -> nearest-medoid assignment
# reproduces the k=3 family clusters.
NONESH_MEDOIDS = ["FAM_06_14", "FAM_09_17", "FAM_07_15"]


def load_scores(csv_path, id_col):
    df = pd.read_csv(csv_path)
    return df.set_index(id_col)[["PC1", "PC2"]]


def variance_labels(csv_path):
    v = pd.read_csv(csv_path).set_index("component")["explained_variance_ratio"]
    return f"PC1 ({v['PC1'] * 100:.1f}%)", f"PC2 ({v['PC2'] * 100:.1f}%)"


def assign_by_medoid(scores, medoids):
    med = scores.loc[medoids].to_numpy()
    labels = {}
    for name, row in scores.iterrows():
        d = np.linalg.norm(med - row.to_numpy(), axis=1)
        labels[name] = medoids[int(d.argmin())]
    return labels


def draw_cluster_region(ax, pts, n_std=2.2):
    """Unfilled house-tone region around a cluster: covariance ellipse for
    >=3 points, an oriented oval spanning a 2-point cluster, a small circle
    for a singleton."""
    if len(pts) == 1:
        ax.add_patch(Circle(pts[0], radius=0.55, fill=False,
                            edgecolor=ELLIPSE_COLOR, ls="--", lw=1.8, zorder=2))
        return
    if len(pts) == 2:
        # A covariance ellipse is degenerate (rank-1) for two points, so build
        # the oval directly: major axis along the pair, with end + side padding.
        p0, p1 = pts[0], pts[1]
        center = (p0 + p1) / 2.0
        d = p1 - p0
        dist = float(np.hypot(*d))
        angle = np.degrees(np.arctan2(d[1], d[0]))
        ax.add_patch(Ellipse(center, dist + 1.1, 1.25, angle=angle, fill=False,
                            edgecolor=ELLIPSE_COLOR, ls="--", lw=1.8, zorder=2))
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=angle, fill=False,
                         edgecolor=ELLIPSE_COLOR, ls="--", lw=1.8, zorder=2))


def scatter_panel(ax, scores, *, title, xlabel, ylabel, clusters=None,
                  label_dx=0.10):
    ax.axhline(0, color=LIGHT, lw=1.2, zorder=0)
    ax.axvline(0, color=LIGHT, lw=1.2, zorder=0)

    if clusters is not None:
        for members in clusters.values():
            draw_cluster_region(ax, scores.loc[members].to_numpy())

    ax.scatter(scores["PC1"], scores["PC2"], s=150, facecolor=POINT_FILL,
               edgecolor=POINT_EDGE, linewidth=1.3, zorder=4)
    span = scores["PC1"].max() - scores["PC1"].min()
    for name, row in scores.iterrows():
        ax.annotate(name, (row["PC1"], row["PC2"]),
                    xytext=(row["PC1"] + label_dx + 0.02 * span, row["PC2"]),
                    va="center", ha="left", fontsize=FS_ANNOT, color=NAVY)

    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.20)


def main():
    nonesh = load_scores(NONESH_DIR / "pca_family_scores_6d.csv", "family_id")
    esh = load_scores(ESH_DIR / "pca_arlp_scores_6d.csv", "ARLP")
    nonesh_xlab, nonesh_ylab = variance_labels(NONESH_DIR / "pca_explained_variance_6d.csv")
    esh_xlab, esh_ylab = variance_labels(ESH_DIR / "pca_arlp_explained_variance_6d.csv")

    medoid_of = assign_by_medoid(nonesh, NONESH_MEDOIDS)
    clusters = {m: [n for n, mm in medoid_of.items() if mm == m] for m in NONESH_MEDOIDS}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(19, 8))
    scatter_panel(axL, nonesh, title="(a) Non-ESH families",
                  xlabel=nonesh_xlab, ylabel=nonesh_ylab, clusters=clusters)
    # ESH k=2 partition: ARLP_02 and ARLP_03 form one cluster (ARLP_01 is on
    # its own). Draw the oval around the {02, 03} pair, matching panel (a).
    esh_clusters = {"pair_02_03": ["ARLP_02", "ARLP_03"]}
    scatter_panel(axR, esh, title="(b) ESH ARLPs",
                  xlabel=esh_xlab, ylabel=esh_ylab, clusters=esh_clusters)

    # Roomy limits so right-edge labels are not clipped.
    axL.set_xlim(-5.2, 4.6)
    axL.set_ylim(-2.0, 2.5)
    axR.set_xlim(-3.4, 3.6)
    # Match panel (a)'s Y-axis range and 0.5 grid step so the two read consistently.
    axR.set_ylim(-2.0, 2.5)
    axR.set_yticks(np.arange(-2.0, 2.6, 0.5))

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    for m, members in clusters.items():
        print(f"  cluster [{m}]: {sorted(members)}")


if __name__ == "__main__":
    main()
