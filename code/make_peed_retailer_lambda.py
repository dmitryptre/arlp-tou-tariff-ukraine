# make_peed_retailer_lambda.py
# ---------------------------------------------------------------------------
# The article's response-intensity figure (method: PSO compensation and
# supplier balance): the change in regulated PSO compensation,
# dV^PSO (upper row), and the change in the supplier balance, dPi (lower row),
# of P3 and E3 against the PEED response intensity lambda in [0, 1], relative
# to the modelling-window baseline.
#
#   two rows, one column per PEED specification; R solid, NR dashed;
#   proportional allocation only (thin valley-filling curves in the lower row
#   are not legible; the valley-filling values are tabulated in the article and
#   are in the CSV); zero line. A small
#   down triangle marks the first lambda at which the baseload block falls below
#   half of b0 in a season (flag M). Points on the curves are not
#   named scenarios. main(valley_filling=True, path=...) draws a variant
#   with the thin valley-filling curves, kept for comparison; the notebook
#   draws the default.
#
# Values from peed_sensitivity_lambda_sweep.csv (UAH M; shown in UAH bn). The
# crossings (value = 0; P3 = E3) are not drawn; they are recomputed here and
# asserted equal to those of the CSV.
#
# Run from code/ :  python make_peed_retailer_lambda.py
# ---------------------------------------------------------------------------
import _peed_fig_common as fc
import matplotlib.pyplot as plt
import numpy as np
import peed_sensitivity as ps
from matplotlib.lines import Line2D

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 17, 14, 12, 13
COLOR = {"P3": fc.NAVY, "E3": fc.BLUEGRAY}
LS = {"R": "-", "NR": "--"}
ROWS = (("dV_PSO", "dV_PSO_MUAH", "$\\Delta V^{\\mathrm{PSO}}$, ₴ bn per year"),
        ("dPi", "dPi_MUAH", "$\\Delta\\Pi$, ₴ bn per year"))
M_MARK = {"marker": "v", "markersize": 10, "ls": "none", "zorder": 5}


def _true(series):
    """A boolean column read back from the CSV (True/False, possibly as text)."""
    return series.astype(str).str.lower().eq("true").to_numpy()


def _assert_crossing(cr, w, t, metric, kind, scheme, x):
    hit = ((cr.sweep_id == ps.sweep_id(w, t, "proportional")) & (cr.metric == metric) & (cr.kind == kind)
           & (cr.scheme == scheme) & np.isclose(cr.lambda_crossing.astype(float), round(x, 2)))
    assert hit.any(), (w, t, metric, kind, scheme, x)


def main(out=None, valley_filling=False, path=None):
    """The response-intensity figure. valley_filling=True draws the variant with the thin valley-filling curves where
    the vf_rule records of peed_sensitivity_lambda_sweep.csv allow them (kept for comparison); path overrides the
    output file (default: peed_retailer_lambda.png in the output folder)."""
    folder = fc.out_dir(out)
    sw = fc.read(folder, "peed_sensitivity_lambda_sweep.csv")
    pts, cr, vf = sw[sw.record == "point"], sw[sw.record == "crossing"], sw[sw.record == "vf_rule"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), sharex=True, sharey="row")
    n_marks = 0
    for r, (metric, col, ylabel) in enumerate(ROWS):
        for c, w in enumerate(fc.MATRICES):
            ax = axes[r, c]
            for t in ("R", "NR"):
                g = pts[pts.sweep_id == ps.sweep_id(w, t, "proportional")]
                gv = pts[pts.sweep_id == ps.sweep_id(w, t, "valley_fill")]
                draw_vf = False
                if valley_filling:
                    rule = vf[(vf.peed_matrix == w) & (vf.energy_treatment == t) & (vf.metric == metric)]
                    assert len(rule) == 1, (w, t, metric)
                    draw_vf = bool(_true(rule.vf_curve_in_figure)[0])
                    assert not (metric == "dV_PSO" and draw_vf), "dV^PSO must not depend on the allocation"
                series = {}
                for sch in ("P3", "E3"):
                    s = g[g.scheme == sch].sort_values("peed_scale")
                    lam, y = s.peed_scale.to_numpy(float), s[col].to_numpy(float)
                    series[sch] = (lam, y)
                    ax.plot(lam, y / 1e3, color=COLOR[sch], ls=LS[t], lw=2.4, zorder=4)
                    if draw_vf:
                        sv = gv[gv.scheme == sch].sort_values("peed_scale")
                        ax.plot(sv.peed_scale, sv[col] / 1e3, color=COLOR[sch], ls=LS[t], lw=1.0, alpha=0.8, zorder=3)
                    if metric == "dPi":
                        flag = _true(s.flag_M)
                        if flag.any():
                            i = int(np.argmax(flag))
                            ax.plot(lam[i], y[i] / 1e3, color=COLOR[sch], **M_MARK)
                            n_marks += 1
                    for x in ps.crossing(lam, y):
                        _assert_crossing(cr, w, t, metric, "direction value = 0", sch, x)
                lam, yp = series["P3"]
                for x in ps.crossing(lam, yp - series["E3"][1]):
                    _assert_crossing(cr, w, t, metric, "equality P3 = E3", "P3 vs E3", x)
            ax.axhline(0, color=fc.SLATE, lw=1.0, zorder=2)
            ax.grid(True, alpha=0.22)
            ax.tick_params(axis="both", labelsize=FS_TICK)
            ax.set_xlim(-0.02, 1.03)
            if r == 0:
                ax.set_title(w, fontsize=FS_TITLE, color=fc.NAVY)
            else:
                ax.set_xlabel("PEED response intensity λ", fontsize=FS_LABEL)
        axes[r, 0].set_ylabel(ylabel, fontsize=FS_LABEL)
    handles = [Line2D([0], [0], color=COLOR["P3"], lw=2.4, label="P3, proposed 3z tariff"),
               Line2D([0], [0], color=COLOR["E3"], lw=2.4, label="E3, existing 3z tariff"),
               Line2D([0], [0], color=fc.SLATE, lw=2.4, ls="-", label="R: energy rescaled"),
               Line2D([0], [0], color=fc.SLATE, lw=2.4, ls="--", label="NR: no rescale")]
    if valley_filling:
        handles.append(Line2D([0], [0], color=fc.SLATE, lw=1.0, label="thin: valley filling"))
    handles.append(Line2D([0], [0], color=fc.SLATE, label="baseload below half of $b_0$ from here", **M_MARK))
    fig.tight_layout(rect=(0, 0.06, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fc.save(fig, path if path is not None else folder / "peed_retailer_lambda.png")
    return n_marks


if __name__ == "__main__":
    main()
