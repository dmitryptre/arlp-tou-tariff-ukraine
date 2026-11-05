# make_peed_profiles_methods.py
# ---------------------------------------------------------------------------
# Summer profiles by energy treatment and within-zone allocation (a figure of
# the article). Combined P3 at p_t = 1, lambda = 1.00.
#
#   cols = PEED specification (F11-SR, F11-LR, A19); rows = energy treatment (R, NR)
#   lines: baseline (dotted), proportional allocation (solid), valley filling
#   (dashed; an illustrative counterfactual allocation, not a behavioural estimate)
#   each maximum is marked with MW and hour-ending (on a tie, the first tied hour
#   in clock order from hour-ending 24, as in peed_sensitivity.peak_metrics); one
#   y-axis for all six panels.
#
# Profiles from _peed_sensitivity_arrays.npz; maxima and hours asserted against
# peed_sensitivity_peak.csv.
#
# Run from code/ :  python make_peed_profiles_methods.py
# ---------------------------------------------------------------------------
import _peed_fig_common as fc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 17, 15, 12, 14, 11
STYLE = {"proportional": {"color": fc.NAVY, "lw": 2.4, "ls": "-"}, "valley_fill": {"color": fc.BLUEGRAY, "lw": 2.2, "ls": "--"}}
LABEL = {"proportional": "proportional allocation", "valley_fill": "valley filling (illustrative counterfactual allocation)"}
TREAT = {"R": "R: daily energy rescaled", "NR": "NR: no rescale"}


def main(out=None):
    folder = fc.out_dir(out)
    z = np.load(folder / "_peed_sensitivity_arrays.npz")
    peak = fc.read(folder, "peed_sensitivity_peak.csv")
    period = [str(x) for x in z["period"]]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True, sharey=True)
    for r, t in enumerate(("R", "NR")):
        for c, w in enumerate(fc.MATRICES):
            ax = axes[r, c]
            fc.shade_zones(ax, period)
            qs = {}
            for al in ("proportional", "valley_fill"):
                sid = fc.scenario_id(w, 1.0, t, al)
                base, comb = fc.combined(z, sid, "P3", "summer")
                q = fc.peak_row(peak, sid, "combined P3", "summer")
                assert abs(comb.max() - q.max_MW) < 1e-6, (sid,)
                assert abs(comb[q.hour_ending % 24] - q.max_MW) < 1e-6, (sid, q.hour_ending)
                ax.plot(fc.X_HE, comb[fc.HOUR_ORDER], zorder=4, **STYLE[al])
                ax.plot([q.hour_ending], [q.max_MW], marker="o", ms=8, mfc="white", mec=STYLE[al]["color"], mew=1.8, zorder=5)
                qs[al] = q
            qp, qv = qs["proportional"], qs["valley_fill"]
            same = qp.hour_ending == qv.hour_ending and abs(qp.max_MW - qv.max_MW) < 0.5
            labels = ([("P = VF", qp)] if same else [("P", qp), ("VF", qv)])
            same_hour = not same and qp.hour_ending == qv.hour_ending      # A19 NR: both maxima at h-e 7
            for k, (tag, q) in enumerate(labels):
                he = int(q.hour_ending)
                right = he >= 16
                dx, dy = (-10 if right else 10), 8 - 20 * k
                if same_hour and k == 1:
                    dx, dy = 16, -44                                       # below right, clear of the falling curves
                ax.annotate(f"{tag}: {q.max_MW:,.0f} MW, h-e {he}", (he, q.max_MW), xytext=(dx, dy),
                            textcoords="offset points", ha="right" if right else "left", fontsize=FS_ANNOT, color=fc.NAVY,
                            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}, zorder=6)
            ax.plot(fc.X_HE, base[fc.HOUR_ORDER], color=fc.SLATE, lw=2.0, ls=":", zorder=3)
            if r == 0:
                ax.set_title(w, fontsize=FS_TITLE, color=fc.NAVY)
            ax.set_xlim(0.5, 24.5)
            ax.set_xticks(range(2, 25, 2))
            ax.tick_params(axis="both", labelsize=FS_TICK, labelbottom=True)
            ax.grid(True, axis="y", alpha=0.22)
            if r == 1:
                ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
            if c == 0:
                ax.set_ylabel(f"{TREAT[t]}\nMW", fontsize=FS_LABEL)
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo, hi + 0.06 * (hi - lo))                # headroom for the labels (shared y)
    handles = [Line2D([0], [0], color=fc.SLATE, lw=2.0, ls=":", label="no response (baseline)")]
    handles += [Line2D([0], [0], label=LABEL[al], **STYLE[al]) for al in ("proportional", "valley_fill")]
    handles += [Line2D([0], [0], marker="o", ms=8, mfc="white", mec=fc.NAVY, mew=1.8, ls="none", label="summer maximum"),
                Patch(facecolor=fc.LAVENDER, edgecolor=fc.NAVY, hatch="////", label="TOU peak zone"),
                Patch(facecolor=fc.LIGHT, edgecolor="none", label="TOU off-peak zone")]
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fc.save(fig, folder / "peed_profiles_methods.png")


if __name__ == "__main__":
    main()
