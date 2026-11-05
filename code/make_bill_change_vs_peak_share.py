# make_bill_change_vs_peak_share.py
# ---------------------------------------------------------------------------
# Supplementary figure of the article: the passive F -> P3 bill
# change of each class-EULP pair against the deviation of its annual peak-zone
# share from its class aggregate (pp), case T, five class panels. Grey dots:
# every eligible EULP (unweighted positions); open circles: area proportional
# to the weight W_cj (largest circle of each panel fixed). The 0 lines; no
# fitted line. Read from 02_class_eulp_units.csv only; the plotted points are
# asserted against it.
#
# Run from code/ :  python -X utf8 make_bill_change_vs_peak_share.py [--out DIR]
# ---------------------------------------------------------------------------
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import _peed_fig_common as fc
import eulp_bill_distribution as bd
import eulp_hierarchy as eh
import manuscript_style as ms

S_MAX = 260.0      # pt^2, the largest circle of a panel
DPI = 300


def build_figure(units, case="T"):
    ms.apply_house_rcparams()
    u = units[units.case == case]
    fig = Figure(figsize=(190 / 25.4, 120 / 25.4), dpi=DPI)
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)
    n_pts = 0
    for k, c in enumerate(eh.CLASS_ORDER):
        ax = axs.flat[k]
        d = u[u["class"] == c]
        x, y, w = d.dev_annual_peak_pp.to_numpy(), d.dbill_pct.to_numpy(), d.W_kHH.to_numpy()
        dots = ax.scatter(x, y, s=2.0, color=fc.LIGHT, lw=0, zorder=2)
        circ = ax.scatter(x, y, s=S_MAX * w / w.max(), facecolors="none", edgecolors=fc.NAVY, lw=0.6, zorder=3)
        assert np.array_equal(np.asarray(dots.get_offsets()), np.column_stack([x, y]))
        assert np.array_equal(np.asarray(circ.get_offsets()), np.column_stack([x, y]))
        n_pts += len(x)
        ax.axhline(0, color=fc.SLATE, lw=0.7, zorder=1)
        ax.axvline(0, color=fc.SLATE, lw=0.7, ls=":", zorder=1)
        ax.set_title(f"{c if c.startswith('ESH') else 'Class ' + c} · {bd.format_count(len(d))} EULPs",
                     fontsize=9, color=fc.NAVY, loc="left", fontweight="bold")
        ax.tick_params(labelsize=7.5, length=2.5, width=0.5, colors=fc.NAVY)
        ax.grid(True, alpha=0.22, lw=0.5)
        ax.set_axisbelow(True)
        if k >= 2:
            ax.set_xlabel("Annual peak-zone share minus\nclass aggregate, pp", fontsize=8, color=fc.NAVY,
                          linespacing=1.1)
        if k % 3 == 0:
            ax.set_ylabel("Annual bill change, %", fontsize=8, color=fc.NAVY)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color(fc.SLATE)
    leg = axs.flat[5]
    leg.axis("off")
    leg.legend(handles=[Line2D([0], [0], marker="o", ls="", color=fc.LIGHT, ms=3, label="EULP (unweighted position)"),
                        Line2D([0], [0], marker="o", ls="", mfc="none", mec=fc.NAVY, ms=8,
                               label="area proportional to the weight (per panel)"),
                        Line2D([0], [0], color=fc.SLATE, lw=0.7, label="0")],
               loc="center", fontsize=8, frameon=False)
    leg.text(0.5, 0.12, "Case T: ESH threshold per EULP.\nPositive change = bill increase.", ha="center",
             fontsize=7.5, color=fc.SLATE, transform=leg.transAxes)
    fig.tight_layout()
    return fig, n_pts


def main(out=None):
    f = Path(out) if out is not None else eh.find_vault() / "outputs" / "ARLP_Clustering" / "eulp_bill_distribution"
    units = pd.read_csv(f / "02_class_eulp_units.csv")
    fig, n = build_figure(units)
    ok = n == int((units.case == "T").sum())
    png = f / "fig_bill_change_vs_peak_share.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print(f"wrote {png.name}; {n} points; ok {ok}")
    assert ok, "peak-share figure: the plotted points do not cover the case-T rows of file 02"
    return {"n_points": n, "ok": ok}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
