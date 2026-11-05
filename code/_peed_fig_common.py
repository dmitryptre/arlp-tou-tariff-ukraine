# _peed_fig_common.py
# ---------------------------------------------------------------------------
# Shared pieces of the source-specific PEED figure scripts: the house palette
# and fonts, the root and output folder, the readers of the sensitivity
# outputs (layer13_peed_sensitivity/), and the zone shading of the house hour axis.
# ---------------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peed_sensitivity as ps

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["hatch.linewidth"] = 1.1

NAVY = "#1A2C3C"
BLUEGRAY = "#5C7999"
SLATE = "#676C80"
LIGHT = "#D2D7DE"
LAVENDER = "#F5F2F6"
MID = "#A9B6C6"
MINUS = "−"

MATRICES = ("F11-SR", "F11-LR", "A19")       # fixed order, equal visual weight
CLASSES = ("A", "B", "C", "ESH-1", "ESH-2")
SEASONS = ("winter", "summer")
HOUR_ORDER = list(range(1, 24)) + [0]        # array index -> hour-ending 1..24
X_HE = np.arange(1, 25)
# hour-ending h drawn at the middle of its hour [h - 1, h], the zone edges on whole hours and the axis 0-24, as in the
# representative-load-shape figure (used by the source-specific load-response figure; the other figures use X_HE and
# the default zone edges)
X_MID = X_HE - 0.5
LAMBDA_LABEL = {1.0: "full transferred-elasticity response (λ = 1.00)", 0.25: "attenuated response (λ = 0.25)"}
N_RET_LABEL = "change in retailer contribution under the market-procurement counterfactual"
N_RE_LABEL = "retail-equivalent net position"
# the PSO-compensation and supplier-balance figure shows these two, relative to the modelling-window baseline
DV_PSO_LABEL = "change in regulated PSO compensation"
DPI_LABEL = "change in the supplier balance"


def find_vault() -> Path:
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "inputs" / "ARLPs(Segments)_Hourly_Load.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate the root folder (the one containing inputs/).")


def out_dir(override=None) -> Path:
    if override is not None:
        return Path(override)
    return find_vault() / "outputs" / "ARLP_Clustering" / "notebook_only" / "layer13_peed_sensitivity"


def scenario_id(matrix, lam, treatment, allocation):
    return ps.scenario_id(matrix, lam, treatment, allocation)


def combined(npz, sid, scheme, season):
    """Combined cohort-weighted aggregate load at p_t = 1 of a scheme (P3: F->P3 and E2->P3; E3: F->E3 and E2->E3)."""
    s0 = float(npz["S0"])
    tf, t2 = {"P3": ("F_to_P3", "E2_to_P3"), "E3": ("F_to_E3", "E2_to_E3")}[scheme]
    base = sum(npz[f"L0__{c}__{season}"] for c in CLASSES)
    d_f = sum(npz[f"{sid}__{tf}__{c}__{season}"] - npz[f"L0__{c}__{season}"] for c in CLASSES)
    d_2 = sum(npz[f"{sid}__{t2}__{c}__{season}"] - npz[f"L0__{c}__{season}"] for c in CLASSES)
    return base, base + (1 - s0) * d_f + s0 * d_2


def shade_zones(ax, period, on_hours=False):
    """Shade the peak and off-peak hours. Default: each hour-ending h around x = h (h - 0.5 to h + 0.5, with X_HE);
    on_hours=True: over its hour [h - 1, h] (with X_MID; zone edges on whole hours, as in the representative-load-shape figure)."""
    for c in range(24):
        zt = period[(c + 1) % 24]
        x0 = c if on_hours else c + 0.5
        if zt == "peak":
            ax.axvspan(x0, x0 + 1, facecolor=LAVENDER, edgecolor=NAVY, hatch="////", linewidth=0.0, alpha=0.6, zorder=0)
        elif zt == "off-peak":
            ax.axvspan(x0, x0 + 1, facecolor=LIGHT, edgecolor="none", alpha=0.5, zorder=0)


def peak_row(peak_csv, sid, series, season):
    p = peak_csv
    q = p[(p.scenario_id == sid) & (p.series == series) & (p.season == season) & (p.p_t == 1.0)]
    assert len(q) == 1, (sid, series, season)
    return q.iloc[0]


def fmt_bn(v):
    return f"{'+' if v >= 0 else MINUS}{abs(v):,.1f}"


def save(fig, path, vault=None, dpi=130):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:
        shown = path.relative_to(find_vault())
    except (ValueError, FileNotFoundError):
        shown = path
    print(f"wrote {shown}  ({path.stat().st_size / 1024:.0f} KB)")


def read(folder, name):
    return pd.read_csv(Path(folder) / name)
