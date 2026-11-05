# make_peed_source_loadshift.py
# ---------------------------------------------------------------------------
# Source-specific load response (the article's load-shift figure by PEED
# specification).
#
#   rows = season (winter / summer); cols = PEED specification (F11-SR, F11-LR, A19)
#   each panel: combined P3 aggregate load at p_t = 1 (cohort-weighted F -> P3 and
#   E2 -> P3), the no-response baseline (dotted), lambda = 1.00 (solid) and
#   lambda = 0.25 (dashed). Energy treatment R. Proportional allocation: the
#   computational display specification, not a reference or preferred case.
#   Valley filling: the same with valley-filling allocation.
#
# x-axis: as in the representative-load-shapes figure, each hour-ending h at the
# middle of its hour (h - 0.5), the zones on whole hours (evening peak 20-22),
# the axis 0-24 with ticks every 2 h.
#
# Combined figure: the proportional and the valley-filling figure as one figure,
# panel (a) proportional allocation over panel (b) valley filling, one legend
# at the bottom, one y-scale per season in both panels.
#
# Profiles from _peed_sensitivity_arrays.npz; the maxima are asserted against
# peed_sensitivity_peak.csv. Nothing is recomputed.
#
# Run from code/ :  python make_peed_source_loadshift.py
# ---------------------------------------------------------------------------
import _peed_fig_common as fc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 17, 15, 12, 14
STYLE = {1.0: {"color": fc.NAVY, "lw": 2.4, "ls": "-"}, 0.25: {"color": fc.BLUEGRAY, "lw": 2.2, "ls": "--"}}
FIGURES = (("proportional", "peed_source_loadshift.png"),       # proportional allocation alone
           ("valley_fill", "peed_source_loadshift_vf.png"))     # valley filling alone
COMBINED = "peed_source_loadshift_panels.png"                   # both: (a) proportional over (b) valley filling
PANELS = (("proportional", "(a) Proportional allocation"), ("valley_fill", "(b) Valley filling"))


def main(out=None):
    folder = fc.out_dir(out)
    z = np.load(folder / "_peed_sensitivity_arrays.npz")
    peak = fc.read(folder, "peed_sensitivity_peak.csv")
    ylims = None
    for allocation, name in FIGURES:
        ylims = draw(z, peak, allocation, folder / name, ylims)
    draw_combined(z, peak, folder / COMBINED)


def panel(ax, z, peak, period, allocation, sn, w):
    """One season x specification panel: baseline (dotted), lambda = 1.00 (solid) and 0.25 (dashed) on the zones."""
    fc.shade_zones(ax, period, on_hours=True)
    for lam in (1.0, 0.25):
        sid = fc.scenario_id(w, lam, "R", allocation)
        base, comb = fc.combined(z, sid, "P3", sn)
        q = fc.peak_row(peak, sid, "combined P3", sn)
        assert abs(comb.max() - q.max_MW) < 1e-6 and abs(base.max() - q.base_max_MW) < 1e-6, (sid, sn)
        ax.plot(fc.X_MID, comb[fc.HOUR_ORDER], zorder=4, **STYLE[lam])
    ax.plot(fc.X_MID, base[fc.HOUR_ORDER], color=fc.SLATE, lw=2.0, ls=":", zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.tick_params(axis="both", labelsize=FS_TICK, labelbottom=True)
    ax.grid(True, axis="y", alpha=0.22)


def legend_handles():
    return [Line2D([0], [0], color=fc.SLATE, lw=2.0, ls=":", label="no response (baseline)"),
            Line2D([0], [0], label=fc.LAMBDA_LABEL[1.0], **STYLE[1.0]),
            Line2D([0], [0], label=fc.LAMBDA_LABEL[0.25], **STYLE[0.25]),
            Patch(facecolor=fc.LAVENDER, edgecolor=fc.NAVY, hatch="////", label="TOU peak zone"),
            Patch(facecolor=fc.LIGHT, edgecolor="none", label="TOU off-peak zone")]


def draw(z, peak, allocation, path, ylims=None):
    """Draw one figure; ylims (one pair per row) fixes the y-axes, so that the valley-filling figure reads on the
    scale of the proportional one."""
    period = [str(x) for x in z["period"]]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True, sharey="row")
    for r, sn in enumerate(fc.SEASONS):
        for c, w in enumerate(fc.MATRICES):
            ax = axes[r, c]
            panel(ax, z, peak, period, allocation, sn, w)
            if r == 0:
                ax.set_title(w, fontsize=FS_TITLE, color=fc.NAVY)
            if r == 1:
                ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
            if c == 0:
                ax.set_ylabel(f"{sn.capitalize()}   MW", fontsize=FS_LABEL)
    if ylims is not None:
        for r, lim in enumerate(ylims):
            axes[r, 0].set_ylim(lim)
    used = [axes[r, 0].get_ylim() for r in range(len(fc.SEASONS))]
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fc.save(fig, path)
    return used


def draw_combined(z, peak, path):
    """Both allocations in one figure: (a) proportional allocation over (b) valley filling, one legend at the bottom.

    Each season keeps one y-scale in both panels (the union of the two autoscaled ranges)."""
    period = [str(x) for x in z["period"]]
    fig = plt.figure(figsize=(19, 22), layout="constrained")
    grids = []
    for sub, (allocation, label) in zip(fig.subfigures(len(PANELS), 1, hspace=0.02), PANELS):
        axes = sub.subplots(2, 3, sharex=True, sharey="row")
        for r, sn in enumerate(fc.SEASONS):
            for c, w in enumerate(fc.MATRICES):
                panel(axes[r, c], z, peak, period, allocation, sn, w)
                if r == 0:
                    axes[r, c].set_title(w, fontsize=FS_TITLE, color=fc.NAVY)
                if c == 0:
                    axes[r, c].set_ylabel(f"{sn.capitalize()}   MW", fontsize=FS_LABEL)
        sub.suptitle(label, x=0.01, ha="left", fontsize=FS_TITLE, color=fc.NAVY)
        grids.append(axes)
    for ax in grids[-1][-1]:
        ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
    for r, sn in enumerate(fc.SEASONS):
        lims = [g[r, 0].get_ylim() for g in grids]
        lim = (min(lo for lo, _ in lims), max(hi for _, hi in lims))
        for g in grids:
            g[r, 0].set_ylim(lim)
        print(f"{sn}: y-range {lim[0]:.0f}-{lim[1]:.0f} MW (panel ranges {[(round(lo), round(hi)) for lo, hi in lims]})")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=3, fontsize=FS_LEGEND, frameon=False)
    fc.save(fig, path)


if __name__ == "__main__":
    main()
