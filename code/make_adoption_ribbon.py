# make_adoption_ribbon.py
# ---------------------------------------------------------------------------
# Two-channel adoption ribbon + winter load morph, in the house blue/grey
# palette + Arial.
#
#   top    : adoption ribbon — smart-meter share, the E2 → P3 migration band,
#            the new-meter F → P3 cohort band, and the total effective p_t
#            (tariff labels from _common).
#   bottom : aggregate winter load shape at 2030 / 2035 / 2040; the peak-zone
#            reduction is shown by diagonal hatching, the off-peak fill lightly.
#
# Ribbon data from penetration_forecast.csv; load morph rebuilt from the
# _system_sweep_arrays.npz baseline + full-response deltas (Cell 13.16 logic).
#
# Run from code/ :  python make_adoption_ribbon.py
# ---------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import TARIFF_ID, transition_label
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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
CSV = OUT13 / "penetration_forecast.csv"
NPZ = OUT13 / "_system_sweep_arrays.npz"
OUT_PATH = OUT13 / "adoption_ribbon_main.png"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
SLATE = "#676C80"
BLUEGRAY = "#5C7999"
LIGHT = "#D2D7DE"
LAVENDER = "#F5F2F6"

FS_LABEL = 15
FS_TICK = 12
FS_PANEL = 15
FS_LEGEND = 12
FS_ANNOT = 12

plt.rcParams["hatch.linewidth"] = 1.1

ANCHORS = [2030, 2035, 2040]
DEVICES_2024_M = 3.29   # ~3.29 M smart meters at S0 (2024)

HE = np.arange(1, 25)


def to_he(v24):
    return np.concatenate([v24[1:], v24[:1]])


def main():
    pf = pd.read_csv(CSV)
    YEARS = pf["year"].to_numpy(float)
    SMART = pf["smart_meter_share_S"].to_numpy(float)
    S0 = float(pf["S0"].iloc[0])
    P_2Z = pf["p_2z_to_3z"].to_numpy(float)
    P_FLAT = pf["p_flat_to_3z"].to_numpy(float)
    P_TOTAL = pf["p_total"].to_numpy(float)

    z = np.load(NPZ, allow_pickle=True)
    L0 = z["L0_winter"]
    dL_2Z = z["Lp2z_winter_1.0"] - L0        # full-response 2z->p3z delta
    dL_FLAT = z["Lpflat_winter_1.0"] - L0     # full-response flat->p3z delta

    def Lyear(yr):
        p2z = float(np.interp(yr, YEARS, P_2Z))
        pfl = float(np.interp(yr, YEARS, P_FLAT))
        return L0 + p2z * dL_2Z + pfl * dL_FLAT

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, height_ratios=[1.5, 1.15], hspace=0.55, wspace=0.20)

    # ── Top: adoption ribbon ────────────────────────────────────────────────
    axR = fig.add_subplot(gs[0, :])
    axR.fill_between(YEARS, 0, P_2Z, facecolor=LAVENDER, edgecolor=SLATE,
                     linewidth=0.5, alpha=0.9, zorder=1)
    axR.fill_between(YEARS, P_2Z, P_TOTAL, facecolor=BLUEGRAY, alpha=0.30, zorder=1)
    axR.plot(YEARS, SMART, color=SLATE, lw=2.0, ls="-.", zorder=3)
    axR.axhline(S0, color=SLATE, lw=1.0, ls=":", alpha=0.8, zorder=2)
    axR.plot(YEARS, P_TOTAL, color=NAVY, lw=3.0, zorder=4)
    for yr in ANCHORS:
        p_t = float(np.interp(yr, YEARS, P_TOTAL))
        axR.axvline(yr, color=SLATE, lw=0.6, alpha=0.4, zorder=1)
        axR.annotate(f"p={p_t:.2f}", xy=(yr, p_t), xytext=(yr, p_t + 0.055),
                     ha="center", fontsize=FS_ANNOT, color=NAVY)
    axR.annotate(f"$S_0\\approx{S0:.2f}$ (~{DEVICES_2024_M:.2f} M devices, 2024)",
                 xy=(YEARS[0], S0), xytext=(YEARS[0] + 0.3, S0 + 0.11),
                 fontsize=FS_ANNOT, color=SLATE,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=SLATE))
    axR.set_xlim(YEARS.min(), YEARS.max())
    axR.set_ylim(0, 1.05)
    axR.set_xticks(ANCHORS)
    axR.tick_params(axis="both", labelsize=FS_TICK)
    axR.set_ylabel("share of households", fontsize=FS_LABEL)
    axR.set_xlabel("year", fontsize=FS_LABEL)
    axR.grid(alpha=0.22)

    ribbon_legend = [
        Line2D([0], [0], color=SLATE, lw=2.0, ls="-.", label=r"smart-meter share $S_t$"),
        Patch(facecolor=LAVENDER, edgecolor=SLATE, linewidth=0.5,
              label=transition_label("e2z", "p3z") + r" migration $\beta_t S_0$"),
        Patch(facecolor=BLUEGRAY, alpha=0.30, edgecolor="none",
              label="new-meter " + transition_label("flat", "p3z") + " cohort"),
        Line2D([0], [0], color=NAVY, lw=3.0, label=f"total effective {TARIFF_ID['p3z']} response " + r"$p_t$"),
    ]
    # One shared legend below the top panel (matching the bottom-panel legend
    # and the rest of the manuscript figures) instead of inside the plot area.
    axR.legend(handles=ribbon_legend, loc="upper center", bbox_to_anchor=(0.5, -0.14),
               fontsize=FS_LEGEND, frameon=False, ncol=4)

    # ── Bottom: winter load morph at the anchor years ───────────────────────
    axB0 = None
    for col, yr in enumerate(ANCHORS):
        ax = (fig.add_subplot(gs[1, col]) if axB0 is None
              else fig.add_subplot(gs[1, col], sharey=axB0))
        if axB0 is None:
            axB0 = ax
        p_t = float(np.interp(yr, YEARS, P_TOTAL))
        base = to_he(L0)
        Ly = to_he(Lyear(yr))
        ax.fill_between(HE, base, Ly, where=Ly < base, interpolate=True,
                        facecolor=LAVENDER, hatch="////", edgecolor=NAVY, linewidth=0.0, zorder=2)
        ax.fill_between(HE, base, Ly, where=Ly >= base, interpolate=True,
                        facecolor=LIGHT, edgecolor="none", alpha=0.9, zorder=2)
        ax.plot(HE, base, color=SLATE, lw=1.4, ls="--", zorder=3)
        ax.plot(HE, Ly, color=NAVY, lw=2.2, zorder=4)
        ax.set_title(f"{yr}  (p = {p_t:.2f})", fontsize=FS_PANEL, color=NAVY)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlim(1, 24)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
        ax.grid(alpha=0.2)
        if col == 0:
            ax.set_ylabel("aggregate load (MW)", fontsize=FS_LABEL)
        else:
            ax.tick_params(labelleft=False)

    morph_legend = [
        Line2D([0], [0], color=SLATE, lw=1.4, ls="--", label="no-response baseline"),
        Line2D([0], [0], color=NAVY, lw=2.2, label="combined two-channel"),
        Patch(facecolor=LAVENDER, edgecolor=NAVY, hatch="////", label="peak-zone reduction"),
        Patch(facecolor=LIGHT, edgecolor="none", label="off-peak increase"),
    ]
    fig.legend(handles=morph_legend, loc="lower center", ncol=4, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.03, 1, 1.0))
    fig.subplots_adjust(hspace=0.55)  # keep room for the top panel's legend
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
