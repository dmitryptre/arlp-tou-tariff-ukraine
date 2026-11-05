# make_participation_trajectory.py
# ---------------------------------------------------------------------------
# The article's participation-trajectory figure:
#   (a) the PEED-independent participation trajectories; p_t is the
#       "effective participation share"; "response intensity" is reserved
#       for lambda;
#   (b-d) the empirical distributions
#       of the change in regulated PSO compensation, dV_t^PSO, in 2030, 2035 and
#       2040: the 12 values of the designed scenarios that the allocation does
#       not change (R filled, NR open; proportional markers);
#   (e-g) the same for the change in the supplier balance, dPi_t, all 24
#       scenarios (R filled, NR open; proportional allocation navy circles,
#       valley filling blue-grey squares).
# Both relative to the modelling-window baseline (peed_pso_levels.csv).
#
# Data: layer13_peed_dr/penetration_forecast.csv (cell 59; the plotted values
# are asserted against it) and layer13_peed_sensitivity/peed_rollout_retailer.csv
# (written by peed_rollout_retailer.py). Written to
# layer13_peed_sensitivity/participation_trajectory.png.
#
# Run from code/ :  python make_participation_trajectory.py
# (after peed_rollout_retailer.py)
# ---------------------------------------------------------------------------
import _peed_fig_common as fc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peed_rollout_retailer as rp
from _common import transition_label
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 15, 12, 12, 12
ANCHORS = [2030, 2035, 2040]
S0_LABEL_X = 2030.4   # the S0 label sits on its dotted line, right of the 2030 guide, clear of S_t and p_t
DEVICES_2024_M = 3.29
DPI = 180
# lower panels (b-g)
ALLOC = {"proportional": {"color": fc.NAVY, "marker": "o", "label": "proportional"},
         "valley_fill": {"color": fc.BLUEGRAY, "marker": "s", "label": "valley filling"}}
MS, MS_LEG, MEW = 90, 9.5, 1.6
# one fixed x range per row, asserted to contain every value: the range of the three checkpoints
# widened by 10 % of its span on each side, then outward to multiples of the tick step
# (dV -2.44..17.94 -> -8..20; dPi 1.18..18.45 -> -4..24)
ROWS = ({"col": "dV_bn", "first": "b", "xlim": (-8, 20), "xstep": 4,
         "xlabel": "$\\Delta V^{\\mathrm{PSO}}_t$ (₴ bn/year)", "allocations": ("proportional",)},
        {"col": "dPi_bn", "first": "e", "xlim": (-4, 24), "xstep": 4,
         "xlabel": "$\\Delta\\Pi_t$ (₴ bn/year)", "allocations": ("proportional", "valley_fill")})
ROW_Y = (0.395, 0.105)
ROW_H = 0.194


def main(out=None):
    folder = fc.out_dir(out)
    csv = fc.find_vault() / "outputs" / "ARLP_Clustering" / "notebook_only" / "layer13_peed_dr" / "penetration_forecast.csv"
    pf = pd.read_csv(csv)
    years = pf["year"].to_numpy(float)
    smart, s0 = pf["smart_meter_share_S"].to_numpy(float), float(pf["S0"].iloc[0])
    p2z, pfl, ptot = (pf[c].to_numpy(float) for c in ("p_2z_to_3z", "p_flat_to_3z", "p_total"))
    assert np.allclose(p2z + pfl, ptot, rtol=0, atol=1e-12)
    fig, ax = plt.subplots(figsize=(16, 17))
    ax.set_position([0.075, 0.715, 0.90, 0.255])
    ax.fill_between(years, 0, p2z, facecolor=fc.LAVENDER, edgecolor=fc.SLATE, linewidth=0.5, alpha=0.9, zorder=1)
    ax.fill_between(years, p2z, ptot, facecolor=fc.BLUEGRAY, alpha=0.30, zorder=1)
    ax.plot(years, smart, color=fc.SLATE, lw=2.0, ls="-.", zorder=3)
    ax.axhline(s0, color=fc.SLATE, lw=1.0, ls=":", alpha=0.8, zorder=2)
    ax.plot(years, ptot, color=fc.NAVY, lw=3.0, zorder=4)
    for yr in ANCHORS:
        p_t = float(pf.loc[pf.year == yr, "p_total"].iloc[0])
        ax.axvline(yr, color=fc.SLATE, lw=0.6, alpha=0.4, zorder=1)
        last = yr == years.max()
        ax.annotate(f"$p_t$ = {p_t:.2f}", xy=(yr, p_t), xytext=(yr - 0.15 if last else yr + 0.15, p_t - 0.075),
                    ha="right" if last else "left", fontsize=FS_ANNOT, color=fc.NAVY)
    ax.text(S0_LABEL_X, s0 + 0.012, f"$S_0\\approx{s0:.2f}$ (~{DEVICES_2024_M:.2f} M devices, 2024)", fontsize=FS_ANNOT,
            color=fc.SLATE, ha="left", va="bottom", zorder=5)
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(range(int(years.min()), int(years.max()) + 1, 5)))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_ylabel("share of households", fontsize=FS_LABEL)
    ax.set_xlabel("year", fontsize=FS_LABEL)
    ax.grid(alpha=0.22)
    ax.set_title("(a) Meter deployment and effective tariff participation", loc="left", fontsize=15, pad=12, color=fc.NAVY)
    legend = [Line2D([0], [0], color=fc.SLATE, lw=2.0, ls="-.", label=r"smart-meter share $S_t$"),
              Patch(facecolor=fc.LAVENDER, edgecolor=fc.SLATE, linewidth=0.5,
                    label=transition_label("e2z", "p3z") + r" migration $\beta_t S_0$"),
              Patch(facecolor=fc.BLUEGRAY, alpha=0.30, edgecolor="none", label="new-meter " + transition_label("flat", "p3z") + " cohort"),
              Line2D([0], [0], color=fc.NAVY, lw=3.0, label=r"total effective participation share $p_t$")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=FS_LEGEND, frameon=False, ncol=4)
    distributions(fig, fc.read(folder, rp.OUT_NAME))
    fc.save(fig, folder / "participation_trajectory.png", dpi=DPI)


def _row_data(roll, year, spec):
    """The scenarios of one panel, sorted for the cumulative count; dV_t is checked equal under P and VF."""
    d = roll[roll.year == year]
    assert len(d) == 24 and d.scenario_id.is_unique
    assert set(d.allocation) == set(ALLOC) and set(d.energy_treatment) == {"R", "NR"}
    if spec["allocations"] == ("proportional",):
        key = ["peed_matrix", "peed_scale", "energy_treatment"]
        p = d[d.allocation == "proportional"].set_index(key).sort_index()
        v = d[d.allocation == "valley_fill"].set_index(key).sort_index()
        assert np.allclose(p[spec["col"]], v[spec["col"]], rtol=0, atol=1e-9), (year, "dV differs between P and VF")
        d = d[d.allocation == "proportional"]
    return d.sort_values([spec["col"], "scenario_id"]).reset_index(drop=True)


def distributions(fig, roll):
    """Panels (b-d) and (e-g): the cumulative counts of dV_t^PSO and dPi_t at the three checkpoints."""
    for r, spec in enumerate(ROWS):
        x0, x1 = spec["xlim"]
        for idx, year in enumerate(ANCHORS):
            ax = fig.add_axes([0.075 + idx * 0.31, ROW_Y[r], 0.28, ROW_H])
            d = _row_data(roll, year, spec)
            n = len(d)
            xx = d[spec["col"]].to_numpy(float)
            assert x0 < xx.min() and xx.max() < x1, (spec["col"], year, "outside the x range", xx.min(), xx.max())
            ranks = np.arange(1, n + 1)
            ax.step(np.r_[x0, xx, x1], np.r_[0, ranks, n], where="post", color=fc.MID, lw=1.1, zorder=1)
            for j, row in d.iterrows():
                st = ALLOC[row.allocation]
                ax.scatter(row[spec["col"]], j + 1, s=MS, marker=st["marker"],
                           facecolors=st["color"] if row.energy_treatment == "R" else "white",
                           edgecolors=st["color"], linewidths=MEW, zorder=3)
            ax.axvline(0, color=fc.SLATE, ls="--", lw=1, zorder=0)
            ax.set_xlim(x0, x1)
            ax.set_ylim(0, n + 1)
            ax.set_xticks(np.arange(x0, x1 + 1e-9, spec["xstep"]))
            ax.set_yticks(np.arange(0, n + 1, 4 if n > 12 else 2))
            ax.tick_params(labelsize=11)
            ax.grid(alpha=0.18)
            if idx == 0:
                ax.set_ylabel("Cumulative scenario count", fontsize=13)
            else:
                ax.tick_params(labelleft=False)
            if idx == 1:
                ax.set_xlabel(spec["xlabel"], fontsize=13)
            letter = chr(ord(spec["first"]) + idx)
            ax.set_title(f"({letter}) {year}  |  $p_t$ ≈ {d.p_total.iloc[0]:.2f}\n"
                         + f"{int((xx > 0).sum())}/{n} positive; median {np.median(xx):+.2f}",
                         fontsize=13, color=fc.NAVY, pad=12, linespacing=1.5)
            assert len(ax.collections) == n
    fig.text(0.075, ROW_Y[0] + ROW_H + 0.043, "Empirical distributions of the designed scenarios", fontsize=15,
             color=fc.NAVY)

    def mk(marker, color, filled):
        return Line2D([], [], marker=marker, ls="none", markersize=MS_LEG, markeredgewidth=MEW,
                      markerfacecolor=color if filled else "white", markeredgecolor=color)

    # one entry per marker, built from ALLOC so that the key matches the plotted markers
    handles, labels = [], []
    for al in ("proportional", "valley_fill"):
        for t in ("R", "NR"):
            handles.append(mk(ALLOC[al]["marker"], ALLOC[al]["color"], t == "R"))
            labels.append(f"{t} + {ALLOC[al]['label']}")
    fig.legend(handles, labels, handlelength=1.2, handletextpad=0.5, loc="lower center", bbox_to_anchor=(0.525, 0.012),
               ncol=4, frameon=False, fontsize=12, columnspacing=3.0)


if __name__ == "__main__":
    main()
