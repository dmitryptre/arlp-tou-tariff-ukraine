# make_seasonal_overlay.py
# ---------------------------------------------------------------------------
# The article's seasonal representative-day overlay, in the house blue/grey
# palette.
#
# Two panels (winter | summer). Each overlays, on an hour-ending axis:
#   * 8 HH-weighted non-ESH family rep-days  (thin blue-gray solid)
#   * 3 ESH ARLP rep-days                    (thin slate dashed)
#   * DAM market load 2023                   (bold navy dashed)
#   * DAM market load 2024                   (bold navy solid)
# with the TOU peak zone shown by diagonal hatching and the off-peak zone by a
# light fill (blue/grey tones only). One shared legend at the bottom.
#
# Data pipeline is reproduced 1:1 from notebook Cell 2.5 (the overlay); only the
# design is set here.
#
# Run from code/ :  python make_seasonal_overlay.py
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
INPUTS = VAULT / "inputs"
CODE_DIR = VAULT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from _common import TOU_PERIODS_1BASED, WINTER_MONTHS  # noqa: E402

OUT_PATH = VAULT / "outputs" / "ARLP_Clustering" / "non_esh" / "notebook_figures" / "repday_overlay_2season_house.png"
DAM_PATH = VAULT / "inputs" / "DAM_2019-2025_v05_Full.xlsx"

# ── House palette ───────────────────────────────────────────────────────────
NAVY = "#1A2C3C"
SLATE = "#676C80"
BLUEGRAY = "#5C7999"
LIGHT = "#D2D7DE"
LAVENDER = "#F5F2F6"

FS_TITLE = 18
FS_LABEL = 15
FS_TICK = 11
FS_LEGEND = 14

FAM = [("FAM_04_12", "ARLP_04", "ARLP_12"), ("FAM_05_13", "ARLP_05", "ARLP_13"),
       ("FAM_06_14", "ARLP_06", "ARLP_14"), ("FAM_07_15", "ARLP_07", "ARLP_15"),
       ("FAM_08_16", "ARLP_08", "ARLP_16"), ("FAM_09_17", "ARLP_09", "ARLP_17"),
       ("FAM_10_18", "ARLP_10", "ARLP_18"), ("FAM_11_19", "ARLP_11", "ARLP_19")]
ESH = ["ARLP_01", "ARLP_02", "ARLP_03"]
SEASONS = ["winter", "summer"]
DAM_YEARS = [2023, 2024]

HE = np.arange(1, 25)


def to_hour_ending(v24):
    """dt.hour-indexed (0..23) -> hour-ending order (1..24); index 0 -> HE 24."""
    return np.concatenate([v24[1:], v24[:1]])


def load_sources():
    load_df = pd.read_csv(INPUTS / "ARLPs(Segments)_Hourly_Load.csv")
    load_df.columns = [c.replace("-", "_") for c in load_df.columns]
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])
    ts = load_df["timestamp"]
    hours = ts.dt.hour.values
    season = np.where(np.isin(ts.dt.month.values, list(WINTER_MONTHS)), "winter", "summer")
    hh = pd.read_csv(INPUTS / "arlp_household_counts.csv").set_index("ARLP")["H_kHH"].to_dict()
    return load_df, hours, season, hh


def seasonal_repday(series_8760, hours, season, sn):
    m = season == sn
    return (pd.Series(series_8760[m]).groupby(hours[m]).mean()
            .reindex(range(24)).to_numpy())


def build_curves(load_df, hours, season, hh):
    """Return {name: {'kind','peak', season: repday24}} for families + ESH."""
    curves = {}
    for fid, c, r in FAM:
        Hc, Hr = float(hh.get(c, 0.0)), float(hh.get(r, 0.0))
        if Hc + Hr <= 0.0:
            Hc, Hr = 1.0, 1.0
        s8760 = (Hc * load_df[c].to_numpy() + Hr * load_df[r].to_numpy()) / (Hc + Hr)
        curves[fid] = {"kind": "family", "peak": float(s8760.max()),
                       **{sn: seasonal_repday(s8760, hours, season, sn) for sn in SEASONS}}
    for name in ESH:
        s8760 = load_df[name].to_numpy()
        curves[name] = {"kind": "esh", "peak": float(s8760.max()),
                        **{sn: seasonal_repday(s8760, hours, season, sn) for sn in SEASONS}}
    return curves


def build_dam():
    dam = pd.read_excel(DAM_PATH, sheet_name="DAM", engine="openpyxl",
                        usecols=["Timestamp", "volume_buy_mwh"])
    dam["Timestamp"] = pd.to_datetime(dam["Timestamp"])
    dam["year"] = dam["Timestamp"].dt.year
    dam["hour"] = dam["Timestamp"].dt.hour
    dam["season"] = np.where(np.isin(dam["Timestamp"].dt.month.values, list(WINTER_MONTHS)),
                             "winter", "summer")
    out = {}
    for y in DAM_YEARS:
        ym = dam["year"] == y
        peak = float(dam.loc[ym, "volume_buy_mwh"].max())
        out[y] = {sn: (dam.loc[ym & (dam["season"] == sn)]
                       .groupby("hour")["volume_buy_mwh"].mean()
                       .reindex(range(24)).to_numpy() / peak)
                  for sn in SEASONS}
    return out


def shade_zones(ax):
    for h in TOU_PERIODS_1BASED["peak"]:
        ax.axvspan(h - 1, h, facecolor=LAVENDER, edgecolor=NAVY, hatch="////",
                   linewidth=0.0, alpha=0.55, zorder=0)
    for h in TOU_PERIODS_1BASED["off"]:
        ax.axvspan(h - 1, h, facecolor=LIGHT, edgecolor="none", alpha=0.45, zorder=0)


def main():
    load_df, hours, season, hh = load_sources()
    curves = build_curves(load_df, hours, season, hh)
    dam = build_dam()

    dam_style = {2023: dict(color=NAVY, lw=2.6, ls="--"),
                 2024: dict(color=NAVY, lw=3.2, ls="-")}

    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)
    for ax, sn in zip(axes, SEASONS):
        shade_zones(ax)
        for name, c in curves.items():
            v = to_hour_ending(c[sn] / c["peak"])
            if c["kind"] == "esh":
                ax.plot(HE, v, color=SLATE, lw=1.3, ls="--", alpha=0.75, zorder=2)
            else:
                ax.plot(HE, v, color=BLUEGRAY, lw=1.1, ls="-", alpha=0.55, zorder=2)
        for y in DAM_YEARS:
            ax.plot(HE, to_hour_ending(dam[y][sn]), zorder=5, **dam_style[y])
        ax.set_title(sn.capitalize(), fontsize=FS_TITLE, color=NAVY)
        ax.set_xlabel("hour-ending", fontsize=FS_LABEL)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("load / annual peak", fontsize=FS_LABEL)

    handles = [
        Line2D([0], [0], color=BLUEGRAY, lw=1.6, ls="-", label="Non-ESH families"),
        Line2D([0], [0], color=SLATE, lw=1.6, ls="--", label="Electric-heating ARLPs"),
        Line2D([0], [0], color=NAVY, lw=2.6, ls="--", label="DAM market load — 2023"),
        Line2D([0], [0], color=NAVY, lw=3.2, ls="-", label="DAM market load — 2024"),
        Patch(facecolor=LAVENDER, edgecolor=NAVY, hatch="////", label="TOU peak zone"),
        Patch(facecolor=LIGHT, edgecolor="none", label="TOU off-peak zone"),
    ]
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(VAULT)}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
