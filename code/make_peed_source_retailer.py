# make_peed_source_retailer.py
# ---------------------------------------------------------------------------
# Retailer-side components by PEED specification (not a figure of the article;
# kept in the outputs): P3 and E3, lambda = 1.00 and 0.25, at p_t = 1. Bars: the passive step (PEED-independent, the same in every group),
# the retailer response (minus the customer-bill response), the external-cost
# saving S^ext_Q (DAM energy and network tariffs, bill volume basis) and their
# sum N^ret, the change in retailer contribution under the market-procurement
# counterfactual. The diamond marks N^RE, the retail-equivalent net position.
# Both are changes relative to the baseline, not levels or profits.
# Computational display specification: R, proportional allocation.
#
# Values from peed_sensitivity_retailer.csv (UAH M; shown in UAH bn), checked
# against the identity N^ret = passive + response + S^ext_Q.
#
# Run from code/ :  python make_peed_source_retailer.py
# ---------------------------------------------------------------------------
import _peed_fig_common as fc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_VALUE = 17, 15, 12, 13, 10
BARS = [("passive_MUAH", "passive step (PEED-independent)", {"facecolor": fc.LIGHT, "edgecolor": fc.SLATE, "hatch": ".."}),
        ("response_MUAH", "retailer response (bill response, retailer sign)", {"facecolor": fc.BLUEGRAY, "edgecolor": fc.BLUEGRAY}),
        ("S_ext_Q_MUAH", "external-cost saving $S^{ext}_Q$ (DAM energy + network)", {"facecolor": fc.LAVENDER, "edgecolor": fc.NAVY, "hatch": "////"}),
        ("net_ret_MUAH", "$N^{ret}$: " + fc.N_RET_LABEL, {"facecolor": fc.NAVY, "edgecolor": fc.NAVY})]
GROUPS = [("P3", 1.0), ("P3", 0.25), ("E3", 1.0), ("E3", 0.25)]


def main(out=None):
    folder = fc.out_dir(out)
    f5 = fc.read(folder, "peed_sensitivity_retailer.csv")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5), sharey=True)
    width, gap = 0.19, 0.04
    for ax, w in zip(axes, fc.MATRICES):
        for g, (sch, lam) in enumerate(GROUPS):
            sid = fc.scenario_id(w, lam, "R", "proportional")
            row = f5[(f5.scenario_id == sid) & (f5.scheme == sch)]
            assert len(row) == 1, (sid, sch)
            row = row.iloc[0]
            assert abs(row.passive_MUAH + row.response_MUAH + row.S_ext_Q_MUAH - row.net_ret_MUAH) < 1e-6, (sid, sch)
            for b, (col, _, st) in enumerate(BARS):
                x = g + (b - 1.5) * (width + gap / 2)
                v = row[col] / 1e3
                ax.bar(x, v, width=width, linewidth=0.9, zorder=3, **st)
                if col == "net_ret_MUAH":
                    vre = row.net_MUAH / 1e3
                    y = max(v, vre, 0.0) if v >= 0 else min(v, vre, 0.0)
                    ax.annotate(fc.fmt_bn(v), (x, y), xytext=(0, 7 if v >= 0 else -17), textcoords="offset points",
                                ha="center", fontsize=FS_VALUE, color=fc.NAVY)
            xr = g + 1.5 * (width + gap / 2)
            ax.plot(xr, row.net_MUAH / 1e3, marker="D", ms=8, mfc="white", mec=fc.NAVY, mew=1.6, ls="none", zorder=5)
        ax.axhline(0, color=fc.SLATE, lw=1.0, zorder=2)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{s}\nλ = {lam:.2f}" for s, lam in GROUPS], fontsize=FS_TICK)
        ax.set_title(w, fontsize=FS_TITLE, color=fc.NAVY)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.grid(True, axis="y", alpha=0.22, zorder=0)
    axes[0].set_ylabel("change relative to the baseline, ₴ bn per year", fontsize=FS_LABEL)
    handles = [Patch(label=lab, linewidth=0.9, **st) for _, lab, st in BARS]
    handles.append(Line2D([0], [0], marker="D", ms=8, mfc="white", mec=fc.NAVY, mew=1.6, ls="none",
                          label="$N^{RE}$: " + fc.N_RE_LABEL))
    fig.tight_layout(rect=(0, 0.13, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fc.save(fig, folder / "peed_source_retailer.png")


if __name__ == "__main__":
    main()
