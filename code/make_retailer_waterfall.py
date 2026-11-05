# make_retailer_waterfall.py
# ---------------------------------------------------------------------------
# System-side accounting waterfall (P3 vs E3), in the house blue/grey
# palette + Arial.
#
# The bar values are read from retailer_waterfall_values.csv, which the
# autonomous notebook writes: UAH million excl. VAT, on the canonical ESH
# split; shown here in UAH bn.
#
# Run from code/ :  python make_retailer_waterfall.py
# ---------------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from _common import TARIFF_ID, TARIFF_NAME
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
OUT_PATH = OUT13 / "retailer_waterfall_split.png"
CSV = OUT13 / "retailer_waterfall_values.csv"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
BLUEGRAY = "#5C7999"
LIGHT = "#D2D7DE"
LAVENDER = "#F5F2F6"
WHITE = "#FFFFFF"
SLATE = "#676C80"
SUBTOTAL = "#202C3B"   # dark navy for net / subtotal bars

plt.rcParams["hatch.linewidth"] = 1.1

FS_TITLE = 25
FS_LABEL = 18
FS_TICK = 15
FS_VALUE = 17
FS_LEGEND = 16

MINUS = "−"  # proper minus sign


def fmt_bn(v, suffix=""):
    """UAH bn with sign, e.g. +₴3.2 bn."""
    return f"{'+' if v >= 0 else MINUS}₴{abs(v):,.1f} bn{suffix}"


def build_panels(csv=CSV):
    """Each step: (x-label, kind, value-label, y0, y1); the bar spans [y0, y1], in UAH bn.
    kind: 'increase' (retailer gain), 'decrease' (retailer loss), 'net' (subtotal)."""
    df = pd.read_csv(csv).set_index("scheme") / 1000.0          # UAH million -> UAH bn
    kind = lambda v: "increase" if v >= 0 else "decrease"
    panels = {}
    for scheme in ("p3z", "e3z"):
        r = df.loc[scheme]
        p, s, dam = r.passive_MUAH, r.load_shift_MUAH, r.dam_saving_MUAH
        tot, net = p + s, p + s + dam
        assert abs(tot - r.total_retail_MUAH) < 1e-9 and abs(net - r.net_MUAH) < 1e-9, scheme
        panels[scheme] = [
            ("Passive", kind(p), fmt_bn(p), 0, p),
            ("Load shift", kind(s), fmt_bn(s), p, tot),
            ("Total retail\nrevenue", "net", fmt_bn(tot), 0, tot),
            ("DAM-based\nprocurement\nsavings", kind(dam), fmt_bn(dam), tot, net),
            ("Net retailer\nposition", "net", fmt_bn(net, "/yr"), 0, net),
        ]
    return panels


# increase = pale flat fill (retailer gain); decrease = navy diagonal hatch on
# white (retailer loss); net = solid dark-navy subtotal.
FACE = {"increase": LAVENDER, "decrease": WHITE, "net": SUBTOTAL}
EDGE = {"increase": NAVY, "decrease": NAVY, "net": NAVY}
HATCH = {"increase": None, "decrease": "////", "net": None}


def panel_title(scheme):
    """Publication name of a scheme key: 'p3z' -> 'Proposed 3z tariff (P3)'."""
    name = TARIFF_NAME[scheme]
    return f"{name[0].upper()}{name[1:]} ({TARIFF_ID[scheme]})"


def draw_panel(ax, steps, title, pad):
    for i, (_lab, kind, _v, y0, y1) in enumerate(steps):
        ax.bar(i, abs(y1 - y0), bottom=min(y0, y1), width=0.62,
               color=FACE[kind], edgecolor=EDGE[kind], hatch=HATCH[kind],
               linewidth=0.9, zorder=3)
    # Waterfall connectors: dashed line at each bar's result level to the next.
    for i in range(len(steps) - 1):
        y = steps[i][4]
        ax.plot([i + 0.31, i + 1 - 0.31], [y, y], color=SLATE, ls="--", lw=1.0, zorder=2)
    # Value labels: gains and positive nets above the top, losses and negative nets below the bottom.
    for i, (_lab, kind, vlab, y0, y1) in enumerate(steps):
        if kind == "increase" or (kind == "net" and y1 >= 0):
            ax.annotate(vlab, (i, max(y0, y1)), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FS_VALUE, fontweight="bold", color=NAVY)
        else:
            ax.annotate(vlab, (i, min(y0, y1)), xytext=(0, -6), textcoords="offset points",
                        ha="center", va="top", fontsize=FS_VALUE, fontweight="bold", color=NAVY)

    ax.axhline(0, color=SLATE, lw=1.3, zorder=2)
    ax.set_title(title, fontsize=FS_TITLE, color=NAVY, fontweight="bold", loc="left")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(True, axis="y", alpha=0.22)
    ax.margins(x=0.06)


def main():
    panels = build_panels()
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    for ax, (scheme, steps) in zip(axes, panels.items()):
        draw_panel(ax, steps, panel_title(scheme), pad=40)
    axes[0].set_ylabel("System-side impact (₴ bn per year)", fontsize=FS_LABEL)
    # Headroom so the value labels are not clipped.
    lo = min(min(y0, y1) for steps in panels.values() for *_, y0, y1 in steps)
    hi = max(max(y0, y1) for steps in panels.values() for *_, y0, y1 in steps)
    span = hi - lo
    axes[0].set_ylim(lo - 0.10 * span, hi + 0.10 * span)

    handles = [
        Patch(facecolor=LAVENDER, edgecolor=NAVY, label="Increase (retailer gain)"),
        Patch(facecolor=WHITE, edgecolor=NAVY, hatch="////", label="Decrease (retailer loss)"),
        Patch(facecolor=SUBTOTAL, edgecolor=NAVY, label="Net / subtotal position"),
    ]
    fig.tight_layout(rect=(0, 0.06, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
