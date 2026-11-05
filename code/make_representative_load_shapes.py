# make_representative_load_shapes.py
# ---------------------------------------------------------------------------
# Unified "Representative load shapes" figure.
#
# ONE 5x3 grid of:
#   * 8 non-ESH household-weighted family rep-days (FAM_04_12 .. FAM_11_19)
#   * 3 ESH ARLP rep-days                        (ARLP_01 .. ARLP_03)
#   * 4 DAM market panels                        (DAM 2021 .. 2024, load + retail price)
#
# The per-panel DATA is reproduced 1:1 from the notebook cells 2.6b.1 / 2.6b.2 /
# 2.6c.2 (families_nonesh_grid / esh_grid / DAM_2021_2024_grid). Only the DESIGN
# is set here:
#   1. every left Y-axis is a clean 0..1 (no legend headroom above 1);
#   2. one combined figure titled "Representative load shapes";
#   3. 3 panels per row (5 rows);
#   4. the DAM panels are titled "DAM 2021" .. "DAM 2024" (calendar years);
#   5. DAM 2021-2024 are shown; the retail-equivalent price is the DAM
#      workbook column gross_price_uah_per_mwh (DAM price + transmission +
#      distribution), nominal UAH/kWh excl. VAT;
#   6. strict blue/grey palette only -- the peak TOU zone is shown by diagonal
#      hatching, not by a colour fill;
#   7. a single shared legend at the very bottom (no per-panel legends);
#   8. no per-panel numeric values in the legend;
#   9. enlarged fonts for titles, axis labels/ticks and the legend;
#  10. bolder lines, no markers.
#  11. palette restricted to the supplied blue/grey table swatches.
#  12. every DAM panel's right axis ends at the DAM 2024 upper limit, with whole
#      UAH/kWh tick labels (axis formatting only; the plotted values are unaffected).
#
# Run from code/ :  python make_representative_load_shapes.py
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

# House font: Arial (falls back to DejaVu Sans if unavailable).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


# ── Locate the root folder and shared helpers ──────────────────────────────
def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


VAULT = find_vault()
INPUTS = VAULT / "inputs"
CODE_DIR = VAULT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from _common import TOU_PERIODS_1BASED, TOU_ZONES, tou_chart_features  # noqa: E402

OUT_DIR = VAULT / "outputs" / "ARLP_Clustering" / "tou_annotation"
OUT_PATH = OUT_DIR / "representative_load_shapes_grid.png"
DAM_PATH = VAULT / "inputs" / "DAM_2019-2025_v05_Full.xlsx"
DAM_YEARS = (2021, 2022, 2023, 2024)
DAM_RIGHT_AXIS_REF_YEAR = 2024   # every DAM panel's right axis ends where this year's does

# ── Palette (supplied blue/grey table swatches only) ───────────────────────
C_NAVY = "#1A2C3C"   # dark navy  -> representative load curve, peak hatch
C_SLATE = "#676C80"  # slate-gray -> standard-deviation line
C_BLUEGRAY = "#5C7999"  # blue-gray -> load-factor line
C_BODY = "#44586C"   # dark blue body text -> DAM retail-price curve
F_OFF = "#EAECF0"    # light neutral grey -> off-peak zone fill (matches table banding)
F_SEMI = "#F5F2F6"   # very light lavender-gray -> semi/peak zone fill

# Enlarged type scale (item 9 above).
FS_TITLE = 17
FS_LABEL = 15
FS_TICK = 12
FS_LEGEND = 15

plt.rcParams["hatch.linewidth"] = 1.3

# hour-ending (1..24) -> TOU zone token
_ZONE_OF_HOUR = {h: z for z in TOU_ZONES for h in TOU_PERIODS_1BASED[z]}


def _zone_spans():
    """Contiguous [x0, x1] runs of a single TOU zone across hour-endings 1..24.

    Hour-ending h occupies the clock interval [h-1, h], so a run of equal-zone
    hours collapses to one rectangle -> no internal seams inside a zone band.
    """
    runs, x0, cur = [], 0, _ZONE_OF_HOUR[1]
    for h in range(2, 25):
        z = _ZONE_OF_HOUR[h]
        if z != cur:
            runs.append((x0, h - 1, cur))
            x0, cur = h - 1, z
    runs.append((x0, 24, cur))
    return runs


_ZONE_SPANS = _zone_spans()


# ── Data reconstruction (matches notebook cells exactly) ───────────────────
def load_source():
    load_df = pd.read_csv(INPUTS / "ARLPs(Segments)_Hourly_Load.csv")
    load_df.columns = [c.replace("-", "_") for c in load_df.columns]
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])
    hours = load_df["timestamp"].dt.hour.values          # 0..23, end-of-hour
    hh = (pd.read_csv(INPUTS / "arlp_household_counts.csv")
            .set_index("ARLP")["H_kHH"].to_dict())
    return load_df, hours, hh


FAM_NON_ESH = [
    ("FAM_04_12", "ARLP_04", "ARLP_12"),
    ("FAM_05_13", "ARLP_05", "ARLP_13"),
    ("FAM_06_14", "ARLP_06", "ARLP_14"),
    ("FAM_07_15", "ARLP_07", "ARLP_15"),
    ("FAM_08_16", "ARLP_08", "ARLP_16"),
    ("FAM_09_17", "ARLP_09", "ARLP_17"),
    ("FAM_10_18", "ARLP_10", "ARLP_18"),
    ("FAM_11_19", "ARLP_11", "ARLP_19"),
]
ESH_ARLPS = ["ARLP_01", "ARLP_02", "ARLP_03"]


def family_repdays(load_df, hours, hh):
    out = {}
    for fid, c, r in FAM_NON_ESH:
        Hc, Hr = float(hh.get(c, 0.0)), float(hh.get(r, 0.0))
        if Hc + Hr <= 0.0:
            Hc, Hr = 1.0, 1.0                             # equal-weight fallback
        fam8760 = (Hc * load_df[c].to_numpy() + Hr * load_df[r].to_numpy()) / (Hc + Hr)
        out[fid] = (pd.Series(fam8760).groupby(hours).mean()
                    .reindex(range(24)).to_numpy())       # 0-based (idx0 = HE 24)
    return out


def esh_repdays(load_df, hours):
    rep = (load_df[ESH_ARLPS].assign(_h=hours)
           .groupby("_h")[ESH_ARLPS].mean().reindex(range(24)))
    return {name: rep[name].to_numpy() for name in ESH_ARLPS}  # 0-based


def read_dam():
    dam = pd.read_excel(DAM_PATH, sheet_name="DAM", engine="openpyxl",
                        usecols=["Timestamp", "settlement_hour",
                                 "gross_price_uah_per_mwh", "volume_buy_mwh"])
    dam["Timestamp"] = pd.to_datetime(dam["Timestamp"])
    return dam


def dam_year(dam, year):
    """Calendar year by the hour-ending Timestamp (as dataset exports 3 and 4)."""
    sub = dam.loc[dam["Timestamp"].dt.year == year]
    grp = sub.groupby("settlement_hour")
    load_1 = grp["volume_buy_mwh"].mean().reindex(range(1, 25)).to_numpy()   # 1-based
    retail_1 = grp["gross_price_uah_per_mwh"].mean().reindex(range(1, 25)).to_numpy() / 1000.0   # UAH/kWh
    return load_1, retail_1


def _retail_pad(retail_1):
    """Headroom of the right axis: 12 % of the panel's own price range."""
    return (retail_1.max() - retail_1.min()) * 0.12 or 0.01


# ── Per-panel painter ──────────────────────────────────────────────────────
def paint_panel(ax, load_24h, *, title, hour_ending, retail_1=None, retail_top=None,
                show_xname=True, show_yname=True):
    feat = tou_chart_features(load_24h, hour_ending=hour_ending)

    arr = np.asarray(load_24h, dtype=float).reshape(-1)
    if hour_ending == "0based":
        arr = np.concatenate([arr[1:], arr[:1]])          # rotate to HE 1..24
    arr = arr / arr.max()                                 # peak-norm curve

    centres = np.arange(0, 24) + 0.5

    # Zone-mean bands: one rectangle per contiguous zone run; height = zone mean.
    # off = light blue-gray fill, semi = very-light fill with a defining edge,
    # peak = highlighted by diagonal hatching.
    for x0, x1, z in _ZONE_SPANS:
        height = feat[f"mu_{z}"]
        if z == "off":
            ax.bar(x0, height, width=x1 - x0, align="edge", color=F_OFF,
                   edgecolor=C_SLATE, linewidth=0.5, zorder=1)
        elif z == "semi":
            ax.bar(x0, height, width=x1 - x0, align="edge", color=F_SEMI,
                   edgecolor=C_SLATE, linewidth=0.5, zorder=1)
        else:  # peak -> highlight with diagonal hatching
            ax.bar(x0, height, width=x1 - x0, align="edge", color=F_SEMI,
                   edgecolor=C_NAVY, hatch="////", linewidth=0.6, zorder=1)

    # Representative load curve (bold, no markers).
    ax.plot(centres, arr, color=C_NAVY, lw=2.7, solid_capstyle="round", zorder=4)

    # Interpretive reference lines (recolored, generic).
    ax.axhline(feat["Lbar"], color=C_BLUEGRAY, ls="--", lw=1.9, zorder=3)
    ax.axhline(feat["sigma"], color=C_SLATE, ls="-.", lw=1.9, zorder=3)

    ax.set_xlim(0, 24)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(range(0, 25, 2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    # Axis NAMES only on the grid edge (tick labels stay on every panel):
    # y-name in the first column, x-name in the bottom row.
    if show_xname:
        ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
    if show_yname:
        ax.set_ylabel("peak normalized load", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.grid(True, axis="y", alpha=0.30)

    # DAM panel only: retail-equivalent price on an independent right axis.
    # Its top is shared by all DAM panels (retail_top); the bottom is the panel's own.
    if retail_1 is not None:
        ax_r = ax.twinx()
        ax_r.plot(centres, retail_1, color=C_BODY, lw=2.4,
                  solid_capstyle="round", zorder=5)
        pad = _retail_pad(retail_1)
        top = retail_1.max() + pad if retail_top is None else retail_top
        ax_r.set_ylim(retail_1.min() - pad, top)
        ax_r.yaxis.set_major_locator(MultipleLocator(1.0))          # whole UAH/kWh
        ax_r.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax_r.set_ylabel("₴/kWh", fontsize=FS_LABEL)
        ax_r.tick_params(axis="y", labelsize=FS_TICK)


# ── Assemble the 5x3 figure ────────────────────────────────────────────────
def main():
    load_df, hours, hh = load_source()
    fam = family_repdays(load_df, hours, hh)
    esh = esh_repdays(load_df, hours)
    dam = read_dam()

    # Panel order: 8 families, then 3 ESH, then DAM 2021-2024 (fills the 5x3 grid).
    panels = []
    for fid, _c, _r in FAM_NON_ESH:
        panels.append((fid, fam[fid], "0based", None))
    for name in ESH_ARLPS:
        panels.append((name, esh[name], "0based", None))
    for year in DAM_YEARS:
        dam_load_1, dam_retail_1 = dam_year(dam, year)
        panels.append((f"DAM {year}", dam_load_1, "1based", dam_retail_1))
    assert len(panels) == 15

    # Common top of the DAM right axes: the reference year's own upper limit.
    ref_retail = next(p[3] for p in panels if p[0] == f"DAM {DAM_RIGHT_AXIS_REF_YEAR}")
    retail_top = ref_retail.max() + _retail_pad(ref_retail)

    fig, axes = plt.subplots(5, 3, figsize=(22, 30))
    for i, (ax, (title, data, he, retail)) in enumerate(zip(axes.ravel(), panels)):
        row, col = divmod(i, 3)
        paint_panel(ax, data, title=title, hour_ending=he, retail_1=retail,
                    retail_top=retail_top if retail is not None else None,
                    show_xname=(row == 4), show_yname=(col == 0))

    # Single shared legend at the very bottom.
    legend_handles = [
        Line2D([0], [0], color=C_NAVY, lw=2.7, label="Representative load profile"),
        Line2D([0], [0], color=C_BLUEGRAY, lw=1.9, ls="--", label="Load factor"),
        Line2D([0], [0], color=C_SLATE, lw=1.9, ls="-.", label="Standard deviation"),
        Patch(facecolor=F_OFF, edgecolor=C_SLATE, linewidth=0.7, label="Off-peak zone mean"),
        Patch(facecolor=F_SEMI, edgecolor=C_SLATE, linewidth=0.7, label="Shoulder zone mean"),
        Patch(facecolor=F_SEMI, edgecolor=C_NAVY, linewidth=0.7, hatch="////", label="Peak zone mean"),
        Line2D([0], [0], color=C_BODY, lw=2.4,
               label="Retail-equivalent DAM price, nominal ₴/kWh excl. VAT (right axis, DAM panels)"),
    ]
    fig.tight_layout(rect=(0, 0.045, 1, 1.0))
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, 0.006))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
