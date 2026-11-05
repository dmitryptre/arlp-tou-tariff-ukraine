# make_loadshift_grid.py
# ---------------------------------------------------------------------------
# The article's whole-system load-shift figure: the F → P3 / E2 → P3 /
# combined load-shift scenarios in one 3x2 grid, in the house palette.
#
#   rows = response channel (F → P3 / E2 → P3 / combined; labels from _common)
#   cols = season           (winter / summer)
#   each panel: aggregate MW vs hour-ending, the no-response baseline (dotted)
#   plus the p_t = 0.10 .. 1.00 response curves (light-to-navy ramp); the TOU
#   peak zone is diagonally hatched and the off-peak zone lightly filled.
#
# Curves are rebuilt from _system_sweep_arrays.npz (persisted by Cell 13.8x for
# exactly this "regenerate standalone" purpose) — nothing is recomputed.
#
# Run from code/ :  python make_loadshift_grid.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
from _common import transition_label
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
OUT13 = VAULT / "outputs" / "ARLP_Clustering" / "notebook_only" / "layer13_peed_dr"
NPZ = OUT13 / "_system_sweep_arrays.npz"
OUT_PATH = OUT13 / "loadshift_grid.png"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
SLATE = "#676C80"
LIGHT = "#D2D7DE"
LAVENDER = "#F5F2F6"

FS_TITLE = 17
FS_LABEL = 15
FS_TICK = 11
FS_LEGEND = 14

plt.rcParams["hatch.linewidth"] = 1.2

# Light-blue-gray -> blue-gray -> navy ramp for the response intensity p_t.
PT_CMAP = LinearSegmentedColormap.from_list("pt_house", ["#C4CDD8", "#5C7999", NAVY])

HOUR_ORDER = list(range(1, 24)) + [0]   # arr index -> hour-ending 1..24
X_HE = np.arange(1, 25)
SEASONS = ["winter", "summer"]


def pt_label(p):
    return "$p_t$ = 1.00 (full)" if p == 1.0 else f"$p_t$ = {p:.2f}"


def main():
    z = np.load(NPZ, allow_pickle=True)
    S0 = float(z["S0"])
    PEN = [float(p) for p in z["PENETRATIONS"]]
    period = list(z["period"])   # zone token per 0-based hour 0..23

    def L0(sn):
        return z[f"L0_{sn}"]

    def Lpflat(p, sn):
        return z[f"Lpflat_{sn}_{p}"]

    def Lp2z(p, sn):
        return z[f"Lp2z_{sn}_{p}"]

    def sys_flat(p, sn):
        return S0 * L0(sn) + (1.0 - S0) * Lpflat(p, sn)

    def sys_2z(p, sn):
        return (1.0 - S0) * L0(sn) + S0 * Lp2z(p, sn)

    def sys_comb(p, sn):
        return (1.0 - S0) * Lpflat(p, sn) + S0 * Lp2z(p, sn)

    ROWS = [(transition_label("flat", "p3z"), sys_flat), (transition_label("e2z", "p3z"), sys_2z),
            ("Combined", sys_comb)]
    pt_colors = [PT_CMAP(t) for t in np.linspace(0.18, 1.0, len(PEN))]

    def shade_zones(ax):
        for c in range(24):
            zt = period[(c + 1) % 24]
            if zt == "peak":
                ax.axvspan(c, c + 1, facecolor=LAVENDER, edgecolor=NAVY,
                           hatch="////", linewidth=0.0, alpha=0.6, zorder=0)
            elif zt == "off-peak":
                ax.axvspan(c, c + 1, facecolor=LIGHT, edgecolor="none",
                           alpha=0.5, zorder=0)

    fig, axes = plt.subplots(3, 2, figsize=(19, 16))
    for r, (label, fn) in enumerate(ROWS):
        for c, sn in enumerate(SEASONS):
            ax = axes[r, c]
            shade_zones(ax)
            ax.plot(X_HE, np.asarray(L0(sn))[HOUR_ORDER], color=SLATE, lw=2.2, ls=":",
                    zorder=3)
            for p, color in zip(PEN, pt_colors):
                ax.plot(X_HE, np.asarray(fn(p, sn))[HOUR_ORDER], color=color, lw=2.4,
                        zorder=4)
            if r == 0:  # season labels the column at the very top only
                ax.set_title(sn.capitalize(), fontsize=FS_TITLE, color=NAVY)
            ax.set_xlim(1, 24)
            ax.set_xticks(range(2, 25, 2))
            ax.tick_params(axis="both", labelsize=FS_TICK, labelbottom=True)
            ax.grid(True, axis="y", alpha=0.22)
            if r == 2:
                ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
            if c == 0:  # response channel identifies the row, before the MW unit
                ax.set_ylabel(f"{label}   MW", fontsize=FS_LABEL)

    handles = [Line2D([0], [0], color=SLATE, lw=2.2, ls=":", label="$p_t$ = 0.00 (baseline)")]
    handles += [Line2D([0], [0], color=col, lw=2.4, label=pt_label(p))
                for p, col in zip(PEN, pt_colors)]
    handles += [
        Patch(facecolor=LAVENDER, edgecolor=NAVY, hatch="////", label="TOU peak zone"),
        Patch(facecolor=LIGHT, edgecolor="none", label="TOU off-peak zone"),
    ]
    fig.tight_layout(rect=(0, 0.06, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
