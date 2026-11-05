# make_bill_change_histograms.py
# ---------------------------------------------------------------------------
# The histogram figure of the passive changes: distribution of passive annual
# bill changes F -> P3 of the synthetic HHs behind each class. 3 x 2 weighted
# histograms (A, B, C / ESH-1, ESH-2, All classes), case T, on the common bins
# of 05_histogram_bins.csv, with the six metrics of 04_distribution_summary.csv
# under each panel. Nothing is recomputed: every printed number is formatted
# from the CSV files by the formatters of eulp_bill_distribution and written,
# with the text actually rendered, to 10_figure_source.csv (D6). The y-axes
# follow eulp_bill_distribution.yaxis_rule (D10).
#
# layout="table" (the three within-class figures of the article; drawn by
# run_eulp_bill_response.py): the two rows of histograms directly one under the
# other, the x-axis label only under the bottom middle panel, a legend without
# the 0% line and without "(<0)" / "(≥0)", no text inside the axes (no bar
# label, no edge-bin labels), and one table under the panels: the six metrics
# and the mode (the midpoint of the tallest bin, in the row above the median)
# as rows, the six panels as columns; also the weighted mean change of the HHs
# with a change >= 0 and of those with a change < 0, each under its share row
# (columns mean_ge0_pct, mean_lt0_pct of the summary frame,
# 03_response_summary.csv). The default layout="blocks" is the histogram
# figure described above (fig_bill_change_histograms.png).
#
# Run from code/ :  python -X utf8 make_bill_change_histograms.py [--out DIR]
# ---------------------------------------------------------------------------
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

import _peed_fig_common as fc
import eulp_bill_distribution as bd
import eulp_hierarchy as eh
import manuscript_style as ms

GRID = (("A", "B", "C"), ("ESH-1", "ESH-2", "All"))
# panel title: the name in bold, then ": n EULPs" in regular type (no separator, no subtitle)
TITLE = {"A": "Class A", "B": "Class B", "C": "Class C", "ESH-1": "ESH-1", "ESH-2": "ESH-2", "All": "All classes"}
METRICS = (("P5, bill change, %", "p05_pct", "pct"), ("Median, %", "p50_pct", "pct"),
           ("P95, bill change, %", "p95_pct", "pct"),
           ("HH with non-negative bill change, % (≥0)", "share_ge0_pct", "share"),
           ("HH with bill decrease, % (<0)", "share_lt0_pct", "share"),
           ("Largest EULP weight, %", "largest_unit_weight_pct", "share"))
X_LABEL = "Annual bill change, %"
Y_LABEL = "Weighted share of synthetic HHs, %"
# no edge-bin entry: the edge bins hold < 0.05 % of the weight and cannot be seen; the figure note explains them
LEGEND = ("Bill decrease (<0)", "Non-negative bill change (≥0)", "0%")
FS = {"title": 9.0, "sub": 7.0, "label": 8.0, "tick": 7.5, "stat": 7.5, "legend": 8.0, "bar": 7.5, "edge": 7.0}
MIN_PT = 7.0
DPI = 400
FIG_W = 190 / 25.4                      # 7.48 in
MAX_H = 185 / 25.4
# layout in inches
LEFT, PITCH, AX_W = 0.66, 2.29, 1.86
BLOCK_PAD_L, BLOCK_PAD_R = 0.30, 0.0      # values end at the right spine; 0.13 in between blocks
AX_H, TOP_PAD, TITLE_H = 1.45, 0.06, (0.24, 0.24)
BELOW_AX, LINE_H, RULE_GAP, ROW_GAP, LEGEND_H = 0.37, 7.5 * 1.32 / 72, 0.05, 0.13, 0.30
VALUE_GAP = 0.08
# layout="table": no "(<0)" / "(≥0)" in the legend and the table, no 0% legend entry; the top row keeps only
# its tick labels below the axes; the legend, then the table, follow the bottom row
LEGEND_TABLE = ("Bill decrease", "Non-negative bill change")
BELOW_TICKS, TABLE_GAP = 0.18, 0.06
# the mode of the table: the midpoint of the tallest bin of the drawn histogram, in the row above the median
MODE_LABEL = "Mode, %"
# the two mean rows of the table: the weighted mean change within the >= 0 / < 0 group, each under its share row;
# "—" for an empty group (NaN in the summary frame)
MEAN_ROWS = (("Mean increase of these HH, %", "mean_ge0_pct"), ("Mean decrease of these HH, %", "mean_lt0_pct"))
TABLE_ROWS = (METRICS[0][0], MODE_LABEL, METRICS[1][0], METRICS[2][0], METRICS[3][0], MEAN_ROWS[0][0], METRICS[4][0],
              MEAN_ROWS[1][0], METRICS[5][0])
EMPTY = "—"


def _table_label(label):
    return label.replace(" (≥0)", "").replace(" (<0)", "")


def modal_bin(b):
    """The tallest bin of one panel's histogram rows (one maximum, not an edge bin) and its midpoint."""
    top = b[b.share_pct == b.share_pct.max()]
    assert len(top) == 1, f"{len(top)} bins share the largest weight: the mode is not unique"
    e = top.iloc[0]
    assert not (e.is_edge_low or e.is_edge_high), "the tallest bin is an edge bin: no midpoint"
    return e, (float(e.left_pct) + float(e.right_pct)) / 2


def folder(out=None):
    return Path(out) if out is not None else eh.find_vault() / "outputs" / "ARLP_Clustering" / "eulp_bill_distribution"


def _title_default(p, row):
    n = int(row["n_units"])
    return f": {bd.format_count(n)} " + ("class–EULP pairs" if p == "All" else "EULPs")


def displays(summary, bins, case="T", title_fmt=None, note=True, in_axes=True, mode=False, means=False):
    """The data strings of the histogram figure from files 04 / 05 with the formatters of eulp_bill_distribution:
    one row per (panel, metric). title_fmt(panel, summary row) -> the regular part of the title; note=False skips the
    class-B note strings (both for the figures of run_eulp_bill_response.py; the defaults give the histogram figure).
    in_axes=False skips the strings drawn inside the axes (the bar label and the edge-bin labels; layout="table").
    mode=True adds the mode, the midpoint of the tallest bin (layout="table"). means=True adds the two MEAN_ROWS
    (layout="table")."""
    s = summary[(summary.case == case) & (summary.weighting == "weighted") & (summary.weight_basis == "H_normalized")]
    s = s.set_index("panel")
    rows = []
    for p in bd.PANELS:
        n = int(s.at[p, "n_units"])
        # the data string is the regular part of the title, ": n EULPs"; the bold name TITLE[p] is not a data string
        rows.append({"panel": p, "metric": "title", "value": n,
                     "display": (title_fmt or _title_default)(p, s.loc[p]),
                     "source_file": "04_distribution_summary.csv", "source_column": "n_units", "rule": "count"})
        for label, col, kind in METRICS:
            v = float(s.at[p, col])
            rows.append({"panel": p, "metric": label, "value": v,
                         "display": bd.format_pct(v, unit=False) if kind == "pct" else bd.format_share(v, unit=False),
                         "source_file": "04_distribution_summary.csv", "source_column": col,
                         "rule": "format_pct (sign, U+2212, |x| < 0.05 -> 0.0)" if kind == "pct"
                         else "format_share (<0.1, >99.9)"})
        b = bins[(bins.case == case) & (bins.panel == p) & (bins.weighting == "weighted")]
        if mode:
            e, mid = modal_bin(b)
            rows.append({"panel": p, "metric": MODE_LABEL, "value": mid, "display": bd.format_pct(mid, unit=False),
                         "source_file": "histogram bins", "source_column": "left_pct, right_pct of the bin with max share_pct",
                         "rule": "midpoint of the tallest bin; format_pct (sign, U+2212, |x| < 0.05 -> 0.0)"})
        if means:
            for label, col in MEAN_ROWS:
                v = float(s.at[p, col])
                rows.append({"panel": p, "metric": label, "value": v,
                             "display": EMPTY if np.isnan(v) else bd.format_pct(v, unit=False),
                             "source_file": "03_response_summary.csv", "source_column": col,
                             "rule": "format_pct (sign, U+2212, |x| < 0.05 -> 0.0); NaN (empty group) -> —"})
        if not in_axes:
            continue
        top = float(b.share_pct.max())
        if bool(b.bar_label.iloc[0]):
            rows.append({"panel": p, "metric": "bar_label", "value": top, "display": bd.format_share(top),
                         "source_file": "05_histogram_bins.csv", "source_column": "share_pct (max)",
                         "rule": "format_share + %"})
        for side, flag in (("low", "is_edge_low"), ("high", "is_edge_high")):
            e = b[b[flag]].iloc[0]
            if e.share_pct > 0:
                edge = e.right_pct if side == "low" else e.left_pct
                # the low bin holds the values strictly below L: "<"
                rows.append({"panel": p, "metric": f"edge_{side}_label", "value": edge,
                             "display": ("<" if side == "low" else "≥") + bd.format_edge(edge) + "%",
                             "source_file": "05_histogram_bins.csv", "source_column": "range_L_pct / range_U_pct",
                             "rule": "format_edge"})
                if e.share_pct >= 0.05:
                    rows.append({"panel": p, "metric": f"edge_{side}_share", "value": float(e.share_pct),
                                 "display": bd.format_share(e.share_pct), "source_file": "05_histogram_bins.csv",
                                 "source_column": "share_pct (edge bin)", "rule": "format_share + %"})
    out = pd.DataFrame(rows).assign(where="figure")
    return pd.concat([out, note_strings(s)], ignore_index=True) if note else out


def note_strings(s):
    """The data strings of the class-B sentence of the histogram figure's note: its weighted P5, median and P95 are
    one EULP's change, printed 0.0 in the panel. Not drawn in the figure (`where = note`)."""
    b = s.loc["B"]
    assert b.p05_pct == b.p50_pct == b.p95_pct, "class B: P5, median and P95 differ; the note sentence needs revision"
    return pd.DataFrame([
        {"panel": "B", "metric": "note_B_percentiles_fine", "value": float(b.p50_pct),
         "display": bd.format_pct_fine(b.p50_pct), "source_file": "04_distribution_summary.csv",
         "source_column": "p50_pct (= p05_pct = p95_pct)", "rule": "format_pct_fine (3 decimals)", "where": "note"},
        {"panel": "B", "metric": "note_B_largest_weight", "value": float(b.largest_unit_weight_pct),
         "display": bd.format_share(b.largest_unit_weight_pct), "source_file": "04_distribution_summary.csv",
         "source_column": "largest_unit_weight_pct", "rule": "format_share + %", "where": "note"},
        {"panel": "B", "metric": "note_B_decrease_share", "value": float(b.share_lt0_pct),
         "display": bd.format_share(b.share_lt0_pct), "source_file": "04_distribution_summary.csv",
         "source_column": "share_lt0_pct", "rule": "format_share + %", "where": "note"}])


def _wrap(text, width_in, measure):
    """Greedy word wrap of a label to a width in inches (the label is never shortened)."""
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and measure(trial) > width_in:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    return lines


def build_figure(summary, bins, case="T", x_tick_step=None, title_fmt=None, note=True, layout="blocks"):
    """The histogram figure from the two CSV frames. Returns (fig, disp, info): disp is file 10 without the rendered
    text. x_tick_step: None = `bd.x_tick_step(w)`, 1 pp at w = 0.5; another value only for a variant, never for the
    canonical histogram figure. title_fmt, note: see `displays()` (the figures of run_eulp_bill_response.py use the
    same grid). layout: "blocks" = the statistics under each panel (fig_bill_change_histograms.png); "table" = one
    table under the panels (the three within-class figures of the article; see the header)."""
    assert layout in ("blocks", "table"), layout
    table = layout == "table"
    ms.apply_house_rcparams()
    disp = displays(summary, bins, case, title_fmt=title_fmt, note=note, in_axes=not table, mode=table, means=table)
    dsp = {(r.panel, r.metric): r.display for r in disp.itertuples()}
    bw = bins[(bins.case == case) & (bins.weighting == "weighted")]
    rule = bw.iloc[0]
    w, L, U = float(rule.width_pp), float(rule.range_L_pct), float(rule.range_U_pct)

    fig = Figure(figsize=(FIG_W, 6.5), dpi=DPI)
    FigureCanvasAgg(fig)
    ren = fig.canvas.get_renderer()

    def measure(s, size=FS["stat"]):
        t = fig.text(0, 0, s, fontsize=size)
        width = t.get_window_extent(ren).width / DPI
        t.remove()
        return width

    block_w = AX_W + BLOCK_PAD_L + BLOCK_PAD_R
    v_w = max(measure(dsp[(p, m[0])]) for p in bd.PANELS for m in METRICS)
    if table:
        wrapped = {label: [_table_label(label)] for label in TABLE_ROWS}
        block_h = RULE_GAP + (len(TABLE_ROWS) + 1) * LINE_H + 2 * RULE_GAP  # rule, header, rule, metrics, rule
        row_h = [TITLE_H[0] + AX_H + BELOW_TICKS, TITLE_H[1] + AX_H + BELOW_AX]
        fig_h = TOP_PAD + row_h[0] + ROW_GAP + row_h[1] + LEGEND_H + TABLE_GAP + block_h
    else:
        wrapped = {m[0]: _wrap(m[0], block_w - v_w - VALUE_GAP, measure) for m in METRICS}
        n_lines = sum(len(v) for v in wrapped.values())
        block_h = RULE_GAP + n_lines * LINE_H
        row_h = [TITLE_H[r] + AX_H + BELOW_AX + block_h for r in (0, 1)]
        fig_h = TOP_PAD + row_h[0] + ROW_GAP + row_h[1] + LEGEND_H
    assert fig_h <= MAX_H + 1e-9, f"figure height {fig_h:.2f} in > 185 mm"
    fig.set_size_inches(FIG_W, fig_h)

    def fy(y_in_from_top):
        return 1 - y_in_from_top / fig_h

    info = {"panels": {}, "fig_h_in": fig_h, "wrapped_labels": {k: v for k, v in wrapped.items() if len(v) > 1}}
    axes = {}
    y0 = TOP_PAD
    for r, row in enumerate(GRID):
        for c, p in enumerate(row):
            x0 = LEFT + c * PITCH
            ax_top = y0 + TITLE_H[r]
            ax = fig.add_axes([x0 / FIG_W, fy(ax_top + AX_H), AX_W / FIG_W, AX_H / fig_h])
            axes[p] = ax
            b = bw[bw.panel == p].sort_values("bin_index")
            for e in b.itertuples():
                if e.share_pct <= 0:
                    continue
                edge = bool(e.is_edge_low or e.is_edge_high)
                col = fc.MID if e.right_pct <= 0 else fc.NAVY
                ax.bar(e.left_pct, e.share_pct, width=w, align="edge", color=col, edgecolor="white", linewidth=0.3,
                       hatch="////" if edge else None, zorder=2)
            ax.axvline(0, color=fc.SLATE, lw=0.9, zorder=3)
            ya = bd.yaxis_rule(b.share_pct.max())
            assert ya["tick_ceiling_pct"] == float(b.tick_ceiling_pct.iloc[0]), (p, "tick ceiling")
            assert ya["y_top_pct"] == float(b.y_top_pct.iloc[0]), (p, "y_top")
            y_top = ya["y_top_pct"]
            ax.set_ylim(0, y_top)
            ax.set_yticks(ya["ticks"])
            ax.set_yticklabels([f"{t:g}" for t in ya["ticks"]])
            ax.set_xlim(L - w, U + w)
            step = x_tick_step or bd.x_tick_step(w)
            ticks = [k * step for k in range(int(np.ceil(L / step)), int(np.floor(U / step)) + 1)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([bd.format_edge(t) for t in ticks])
            ax.tick_params(axis="both", labelsize=FS["tick"], length=2.5, width=0.5, pad=2, colors=fc.NAVY)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_color(fc.SLATE)
            ax.grid(True, axis="y", alpha=0.22, lw=0.5)
            ax.set_axisbelow(True)
            if not table or (r, c) == (1, 1):            # table: only under the bottom middle panel
                ax.set_xlabel(X_LABEL, fontsize=FS["label"], labelpad=2, color=fc.NAVY)
            if c == 0:
                ax.set_ylabel(Y_LABEL.replace(" of synthetic", " of\nsynthetic"), fontsize=FS["label"], labelpad=3,
                              color=fc.NAVY, linespacing=1.1)
            name = fig.text(x0 / FIG_W, fy(y0 + 0.17), TITLE[p], fontsize=FS["title"], fontweight="bold",
                            color=fc.NAVY, ha="left", va="baseline")
            x_count = (name.get_window_extent(ren).x1 + 1.0) / fig.bbox.width   # 1 px gap: no bbox overlap (D10)
            fig.text(x_count, fy(y0 + 0.17), dsp[(p, "title")], fontsize=FS["title"], fontweight="normal",
                     color=fc.NAVY, ha="left", va="baseline", gid=f"{p}|title")
            # the value label of a bar above 50 % (the table layout draws no text inside the axes: `displays` gives
            # neither this label nor the edge-bin labels)
            lab = None
            if (p, "bar_label") in dsp:
                e = b.loc[b.share_pct.idxmax()]
                lab = ax.text(e.left_pct + w / 2, e.share_pct + 0.01 * y_top, dsp[(p, "bar_label")], ha="center",
                              va="bottom", fontsize=FS["bar"], color=fc.NAVY, gid=f"{p}|bar_label", zorder=4,
                              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
            for side, flag, ha, xx in (("low", "is_edge_low", "left", L - w), ("high", "is_edge_high", "right", U + w)):
                if (p, f"edge_{side}_label") in dsp:
                    ax.text(xx, 0.97 * y_top, dsp[(p, f"edge_{side}_label")], ha=ha, va="top", fontsize=FS["edge"],
                            color=fc.SLATE, gid=f"{p}|edge_{side}_label", zorder=4)
                    if (p, f"edge_{side}_share") in dsp:
                        ax.text(xx, 0.97 * y_top - 0.115 * y_top, dsp[(p, f"edge_{side}_share")], ha=ha, va="top",
                                fontsize=FS["edge"], color=fc.SLATE, gid=f"{p}|edge_{side}_share", zorder=4)
            # the statistics block (blocks layout; the table layout draws one table after the loop)
            if not table:
                bx0, bx1 = (x0 - BLOCK_PAD_L) / FIG_W, (x0 + AX_W + BLOCK_PAD_R) / FIG_W
                yb = ax_top + AX_H + BELOW_AX
                fig.add_artist(Line2D([bx0, bx1], [fy(yb), fy(yb)], color=fc.LIGHT, lw=0.4))
                yy = yb + RULE_GAP
                for label, _col, _k in METRICS:
                    for li, line in enumerate(wrapped[label]):
                        yy += LINE_H
                        fig.text(bx0, fy(yy - 0.03), line, fontsize=FS["stat"], color=fc.SLATE, ha="left",
                                 va="baseline")
                    fig.text(bx1, fy(yy - 0.03), dsp[(p, label)], fontsize=FS["stat"], color=fc.NAVY, ha="right",
                             va="baseline", gid=f"{p}|{label}")
            info["panels"][p] = {"y_top_pct": y_top, "tick_ceiling_pct": ya["tick_ceiling_pct"],
                                 "tallest_bar_pct": ya["tallest_bar_pct"], "label": lab}
        y0 += row_h[r] + ROW_GAP
    if table:
        _draw_table(fig, dsp, y0 - ROW_GAP, fy, measure)
    else:
        handles = [Patch(facecolor=fc.MID, edgecolor="white", label=LEGEND[0]),
                   Patch(facecolor=fc.NAVY, edgecolor="white", label=LEGEND[1]),
                   Line2D([0], [0], color=fc.SLATE, lw=0.9, label=LEGEND[2])]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS["legend"], frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.4, handletextpad=0.5)
    _fit_bar_labels(fig, axes, info, ren)
    return fig, disp, info


def _draw_table(fig, dsp, y_in, fy, measure):
    """layout="table": the legend (centred under the middle column, like the x-axis label), then one table: the
    metrics as rows, the panels in the order of GRID as columns; the values keep the gids of the blocks layout, so
    D6 checks them in the same way. y_in: the bottom of the bottom row, in inches from the top."""
    x_mid = (LEFT + PITCH + AX_W / 2) / FIG_W
    handles = [Patch(facecolor=fc.MID, edgecolor="white", label=LEGEND_TABLE[0]),
               Patch(facecolor=fc.NAVY, edgecolor="white", label=LEGEND_TABLE[1])]
    fig.legend(handles=handles, loc="center", ncol=2, fontsize=FS["legend"], frameon=False,
               bbox_to_anchor=(x_mid, fy(y_in + LEGEND_H / 2)), handlelength=1.4, columnspacing=1.4,
               handletextpad=0.5, borderaxespad=0.0, borderpad=0.0)
    tx0, tx1 = LEFT - BLOCK_PAD_L, LEFT + 2 * PITCH + AX_W
    lab_w = max(measure(_table_label(label)) for label in TABLE_ROWS)
    panels = [p for row in GRID for p in row]
    col_w = (tx1 - tx0 - lab_w - VALUE_GAP) / len(panels)
    x_right = [tx0 + lab_w + VALUE_GAP + (j + 1) * col_w for j in range(len(panels))]

    def rule(y):
        fig.add_artist(Line2D([tx0 / FIG_W, tx1 / FIG_W], [fy(y), fy(y)], color=fc.LIGHT, lw=0.4))

    yy = y_in + LEGEND_H + TABLE_GAP
    rule(yy)
    yy += RULE_GAP + LINE_H
    for p, xr in zip(panels, x_right):
        fig.text(xr / FIG_W, fy(yy - 0.03), TITLE[p], fontsize=FS["stat"], color=fc.NAVY, ha="right", va="baseline")
    yy += RULE_GAP
    rule(yy)
    for label in TABLE_ROWS:
        yy += LINE_H
        fig.text(tx0 / FIG_W, fy(yy - 0.03), _table_label(label), fontsize=FS["stat"], color=fc.SLATE, ha="left",
                 va="baseline")
        for p, xr in zip(panels, x_right):
            fig.text(xr / FIG_W, fy(yy - 0.03), dsp[(p, label)], fontsize=FS["stat"], color=fc.NAVY, ha="right",
                     va="baseline", gid=f"{p}|{label}")
    yy += RULE_GAP
    rule(yy)


def _fit_bar_labels(fig, axes, info, ren):
    """A bar label must lie inside its axes: raise y_top by 0.05 h up to 1.40 h, else stop."""
    for p, d in info["panels"].items():
        lab, ax = d["label"], axes[p]
        if lab is None:
            continue
        h = d["tallest_bar_pct"]
        y_top = d["y_top_pct"]
        while True:
            fig.canvas.draw()
            bb, ab = lab.get_window_extent(ren), ax.get_window_extent(ren)
            if bb.y1 <= ab.y1 and bb.x0 >= ab.x0 and bb.x1 <= ab.x1:
                break
            y_top += 0.05 * h
            assert y_top <= 1.40 * h + 1e-9, f"{p}: the bar label does not fit below 1.40 h"
            ax.set_ylim(0, y_top)
            lab.set_y(h + 0.01 * y_top)
        d["y_top_final_pct"] = y_top
        d["raised"] = y_top != d["y_top_pct"]


def check_rendered(fig, disp):
    """D6 and D10 on the rendered figure. Returns (disp with rendered_text, d6_ok, d10_ok, detail)."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    by_gid = {t.get_gid(): t.get_text() for t in texts if t.get_gid()}
    disp = disp.copy()
    disp["rendered_text"] = [by_gid.get(f"{r.panel}|{r.metric}", "") if r.where == "figure" else ""
                             for r in disp.itertuples()]
    fig_rows = disp[disp["where"] == "figure"]
    d6 = bool((fig_rows.rendered_text == fig_rows.display).all()) and len(by_gid) == len(fig_rows)
    small = [t.get_text() for t in texts if t.get_fontsize() < MIN_PT - 1e-9]
    boxes = [(t.get_text(), t.get_window_extent(ren)) for t in texts]
    overlaps = [(a[0], b[0]) for a, b in itertools.combinations(boxes, 2) if a[1].overlaps(b[1])]
    ok_ticks = all(t <= 100 for ax in fig.axes for t in ax.get_yticks())
    inside = True
    for ax in fig.axes:
        ab = ax.get_window_extent(ren)
        for t in ax.texts:
            bb = t.get_window_extent(ren)
            inside &= bb.x0 >= ab.x0 - 0.5 and bb.x1 <= ab.x1 + 0.5 and bb.y1 <= ab.y1 + 0.5 and bb.y0 >= ab.y0 - 0.5
        for patch in ax.patches:
            inside &= patch.get_height() <= ax.get_ylim()[1]
    d10 = ok_ticks and inside and not overlaps and not small
    detail = (f"ticks <= 100: {ok_ticks}; texts and bars inside axes: {inside}; overlaps: {len(overlaps)} "
              f"{overlaps[:3]}; text below 7 pt: {small[:3]}")
    return disp, d6, d10, detail


def main(out=None):
    f = folder(out)
    summary = pd.read_csv(f / "04_distribution_summary.csv")
    bins = pd.read_csv(f / "05_histogram_bins.csv")
    fig, disp, info = build_figure(summary, bins)
    disp, d6, d10, detail = check_rendered(fig, disp)
    raised = {p: d["y_top_final_pct"] for p, d in info["panels"].items() if d.get("raised")}
    if raised:   # the final y_top of a raised panel is written back to file 05 (05_histogram_bins.csv)
        for p, v in raised.items():
            sel = (bins.case == "T") & (bins.panel == p) & (bins.weighting == "weighted")
            bins.loc[sel, "y_top_pct"] = v
        bins.to_csv(f / "05_histogram_bins.csv", index=False, lineterminator="\n", encoding="utf-8")
    png = f / "fig_bill_change_histograms.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    disp.to_csv(f / "10_figure_source.csv", index=False, lineterminator="\n", encoding="utf-8")
    from PIL import Image
    size = Image.open(png).size
    print(f"wrote {png.name} {size[0]} x {size[1]} px; D6 {d6}; D10 {d10} ({detail})")
    assert d6, "D6: a rendered string differs from the display of file 10"
    assert d10, f"D10: {detail}"
    w = float(bins.width_pp.iloc[0])
    return {"n_strings": int((disp["where"] == "figure").sum()), "n_note_strings": int((disp["where"] == "note").sum()),
            "x_tick_step_pp": bd.x_tick_step(w), "d6_ok": d6, "d10_ok": d10, "d10_detail": detail, "png_px": size,
            "fig_h_in": round(info["fig_h_in"], 3), "wrapped_labels": info["wrapped_labels"], "y_top_raised": raised,
            "y_top_pct": {p: d["y_top_pct"] for p, d in info["panels"].items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
