# make_class_fingerprints.py
# ---------------------------------------------------------------------------
# The article's 5-class shape-feature radar "fingerprints", in the house
# blue/grey palette.
#
# Same data and min-max-per-axis normalisation as notebook Cell 3C.9; only the
# design is set here: uniform navy outline + light-blue-gray fill on every panel,
# enlarged fonts, no embedded suptitle (caption carries the name), and a compact
# house-styled abbreviation legend.
#
# Values come straight from class_fingerprints_5class_preferred_features.csv.
#
# Run from code/ :  python make_class_fingerprints.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
NB_FIG = VAULT / "outputs" / "ARLP_Clustering" / "notebook_figures"
CSV = NB_FIG / "class_fingerprints_5class_preferred_features.csv"
OUT_PATH = NB_FIG / "class_fingerprints_5class_preferred_house.png"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
LIGHT = "#D2D7DE"
SLATE = "#676C80"

FS_TITLE = 18
FS_AXIS = 14
FS_LEGEND = 13
FS_HEAD = 13

METRICS = ["load_factor", "base_load_share", "peak_width_80pct", "modal_peak_hour",
           "ramp_p95_norm", "cv_daily_energy", "energy_share_winter", "tou_peak_share"]
LABELS = ["LF", "Base", "PkW", "PkHr", "Ramp", "CVd", "Wint", "TOU-P"]
ORDER = ["A", "B", "C", "ESH-1", "ESH-2"]

# Compact abbreviation legend (ABBR = full name); definitions live in the caption.
LEGEND = [
    ("LF", "load factor"), ("Base", "baseload"), ("PkW", "peak width"),
    ("PkHr", "peak hour"), ("Ramp", "ramp P95"), ("CVd", "CV daily energy"),
    ("Wint", "winter share"), ("TOU-P", "TOU peak share"),
]


def main():
    df = pd.read_csv(CSV, index_col=0).loc[ORDER, METRICS]

    # Min–max normalise per axis across the five classes.
    vmin, vmax = df.min(axis=0), df.max(axis=0)
    span = (vmax - vmin).replace(0.0, 1.0)
    norm = (df - vmin) / span

    angles = np.linspace(0.0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(22, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[2.55, 1.0], hspace=0.10, wspace=0.34,
                          left=0.04, right=0.98, top=0.94, bottom=0.04)

    for i, cls in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, i], projection="polar")
        vals = norm.loc[cls].tolist()
        vals = vals + vals[:1]
        ax.plot(angles_closed, vals, color=NAVY, lw=2.4, marker="o", ms=5, zorder=3)
        ax.fill(angles_closed, vals, color=LIGHT, alpha=0.55, zorder=2)
        ax.set_xticks(angles)
        ax.set_xticklabels(LABELS, fontsize=FS_AXIS)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(cls, fontsize=FS_TITLE, fontweight="bold", pad=16, color=NAVY)
        ax.grid(alpha=0.35)

    # Compact abbreviation legend row (two columns of four).
    lax = fig.add_subplot(gs[1, :])
    lax.axis("off")
    half = 4
    for j, (abbr, full) in enumerate(LEGEND[:half]):
        lax.text(0.06, 0.92 - j * 0.20, f"{abbr} = {full}", ha="left", va="top",
                 fontsize=FS_LEGEND, color=SLATE, transform=lax.transAxes)
    for j, (abbr, full) in enumerate(LEGEND[half:]):
        lax.text(0.54, 0.92 - j * 0.20, f"{abbr} = {full}", ha="left", va="top",
                 fontsize=FS_LEGEND, color=SLATE, transform=lax.transAxes)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
