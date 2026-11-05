"""
_common.py
==========
Shared constants and helpers used across the ARLP clustering pipeline.

Regulatory facts shared with the companion study's billing code:
- ``WINTER_MONTHS`` / ``SUMMER_MONTHS``: the billing calendar (winter = Oct-Apr).
- ``ESH_ARLPS``: the three electric-space-heating ARLPs the NEURC discount applies to.

Single source of truth for stratification: the pipeline runs the three-lens
clustering and TOU design independently inside ``non_esh`` (16 ARLPs, k=4) and
``esh`` (3 ARLPs, k=2).
"""
from __future__ import annotations

ARLP_NAMES_ALL = [f"ARLP_{i:02d}" for i in range(1, 20)]

ESH_ARLPS = frozenset({"ARLP_01", "ARLP_02", "ARLP_03"})
NON_ESH_ARLPS = frozenset(ARLP_NAMES_ALL) - ESH_ARLPS

WINTER_MONTHS = frozenset({1, 2, 3, 4, 10, 11, 12})   # Oct-Apr
SUMMER_MONTHS = frozenset({5, 6, 7, 8, 9})            # May-Sep

K_POLICY_BY_STRATUM = {"non_esh": 3, "esh": 2}    # non-ESH k = 3 on the merged-family cohort

# Formal k-selection (identical-pair dedup + gap statistic).
IDENTICAL_PAIR_EUCLID_EPS = 1e-6
LENS2_KSELECTION_SUBDIR = "lens2_kselection"

# City/rural ARLP family consolidation (pre-clustering merge).
FAMILY_ID_PREFIX = "FAM"
FAMILY_CONSTRUCTION_SUBDIR = "family_construction"
LENS2_FAMILIES_SUBDIR = "lens2_families"
LENS2_FAMILIES_KSELECTION_SUBDIR = "lens2_families_kselection"

STRATA = {
    "non_esh": sorted(NON_ESH_ARLPS),
    "esh":     sorted(ESH_ARLPS),
}


def stratum_of(arlp: str) -> str:
    return "esh" if arlp in ESH_ARLPS else "non_esh"


def linestyle_for(arlp: str) -> str:
    """Dashed line for ESH ARLPs, solid for non-ESH. Use in overlay charts."""
    return "--" if arlp in ESH_ARLPS else "-"


ESH_LEGEND_NOTE = "dashed = ESH (electric space heating)"


# ─────────────────────────────────────────────────────────────────────
# Curated metric set produced by the feature-screening pipeline.
# The 9-metric `reduced_9` set is exposed as RADAR_METRICS; downstream
# consumers (radar charts, metric tables, ESH equivalents) take this list as
# the source of truth so the notebook and the .py pipeline always render the
# same 9 axes / columns in the same order.
# ─────────────────────────────────────────────────────────────────────
RADAR_METRICS = [
    "load_factor", "base_load_share", "peak_width_80pct",
    "morning_evening_ratio", "modal_peak_hour",
    "energy_share_winter",
    "ramp_p95_norm", "p95_mean_ratio",
]
RADAR_LABELS = {
    "load_factor":           "LF",
    "base_load_share":       "Base",
    "peak_width_80pct":      "PkW",
    "morning_evening_ratio": "M/E",
    "modal_peak_hour":       "PkHr",
    "energy_share_winter":   "Wint",
    "ramp_p95_norm":         "Ramp",
    "p95_mean_ratio":        "P95",
}
METRIC_DESCRIPTIONS = {
    "load_factor":           "mean(load) / peak(load) — overall capacity utilisation (0..1)",
    "base_load_share":       "10th-percentile load / peak — baseload intensity (0..1)",
    "peak_width_80pct":      "mean hours per day at load >= 80% of that day's peak",
    "morning_evening_ratio": "mean(load 07-12h) / mean(load 17-22h) — intra-day orientation",
    "modal_peak_hour":       "modal hour of daily peak (period-ending, 1..24)",
    "energy_share_winter":   "share of annual energy in Oct-Apr (winter dominance, 0..1)",
    "ramp_p95_norm":         "95th-percentile |load(t+1) - load(t)| / peak load",
    "p95_mean_ratio":        "95th-percentile load / mean load — extreme-load intensity",
}

# ─────────────────────────────────────────────────────────────────────
# Unified 19-ARLP / 5-class label scheme. Non-ESH (k = 3) gets cluster IDs
# 0..2 (Classes A..C); ESH (k = 2) gets 3..4 (Classes E1, E2).
# ─────────────────────────────────────────────────────────────────────
# Non-ESH k = 4 partitions are still computable from the long-form
# `cluster_assignments.csv` as a sensitivity case.
CLUSTER_LABELS_ALL19 = {
    0: "Class A",
    1: "Class B",
    2: "Class C",
    3: "Class E1 (ESH)",
    4: "Class E2 (ESH)",
}
ESH_CLUSTERS = frozenset({3, 4})

# NEURC residential base regime numerals (USD/kWh).
NON_ESH_BASE_PRICE = 0.103
ESH_BASE_PRICE_LOW = NON_ESH_BASE_PRICE * 2.64 / 4.32    # winter ≤ 2000 kWh discount; NEURC 2.64 vs 4.32 UAH/kWh
ESH_BASE_PRICE_HIGH = 0.103    # winter > 2000 kWh and summer

# NEURC household fixed prices (UAH/kWh, excluding VAT) — the model currency.
# CMU Resolution No. 483 of 05.06.2019 (as amended), in force 01.06.2024-31.10.2026; mirrored by
# inputs/neurc_household_tariffs_uah.csv. 2.20/3.60 = 2.64/4.32, the ratio of the USD constants above.
NON_ESH_BASE_PRICE_UAH = 3.60
ESH_BASE_PRICE_LOW_UAH = 2.20     # Oct-Apr, month <= 2000 kWh, whole volume
ESH_BASE_PRICE_HIGH_UAH = 3.60    # Oct-Apr above 2000 kWh, and May-Sep
RATE_LABEL = "₴/kWh"
MONEY_LABEL = "₴ bn"

# Publication-facing tariff names. The code keeps the internal ids (flat, e2z, e3z, p3z).
TARIFF_ID = {"flat": "F", "e2z": "E2", "e3z": "E3", "p3z": "P3"}
TARIFF_NAME = {"flat": "flat tariff", "e2z": "existing 2z tariff",
               "e3z": "existing 3z tariff", "p3z": "proposed 3z tariff"}


def transition_label(src: str, dst: str, sep: str = " → ") -> str:
    """Figure label of a tariff transition: transition_label("e2z", "p3z") == "E2 → P3"."""
    return f"{TARIFF_ID[src]}{sep}{TARIFF_ID[dst]}"


def pretty(name: str) -> str:
    """snake_case -> 'Sentence case' (first letter upper, underscores -> spaces)."""
    s = str(name).replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def pretty_cols(df):
    """Return a copy of `df` with snake_case column headers rewritten via `pretty`."""
    return df.rename(columns={c: pretty(c) for c in df.columns})


# ─────────────────────────────────────────────────────────────────────
# TOU (time-of-use) schedule + feature helpers
# ─────────────────────────────────────────────────────────────────────
# The Ukrainian residential three-zone TOU schedule (zones: off / semi / peak)
# in BOTH the 0-based (h00..h23) and 1-based (h01..h24) hour-ending indexings.
# The 0-based form also appears in the notebook as PEAK_H / OFF_H;
# the 1-based form is what the per-ARLP TOU annotation chart uses.

TOU_PERIODS_0BASED = {
    "off":  [0, 1, 2, 3, 4, 5, 6, 7],                       # 23:00-07:00
    "semi": [8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23],    # 07-08, 11-20, 22-23
    "peak": [9, 10, 11, 21, 22],                             # 08-11, 20-22
}
TOU_PERIODS_1BASED = {
    "off":  [1, 2, 3, 4, 5, 6, 7, 24],
    "semi": [8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23],
    "peak": [9, 10, 11, 21, 22],
}
TOU_ZONES = ("off", "semi", "peak")
TOU_ZONE_SIZE = {z: len(TOU_PERIODS_1BASED[z]) for z in TOU_ZONES}


def _tou_consistency_check():
    """Verify the two indexings describe the same schedule and partition {1..24}."""
    def _h0_to_h1(h0): return 24 if h0 == 0 else h0
    for z in TOU_ZONES:
        converted = sorted(_h0_to_h1(h) for h in TOU_PERIODS_0BASED[z])
        if converted != sorted(TOU_PERIODS_1BASED[z]):
            raise AssertionError(
                f"TOU zone {z!r}: 0-based {TOU_PERIODS_0BASED[z]} disagrees with "
                f"1-based {TOU_PERIODS_1BASED[z]}")
    all_1b = sorted(TOU_PERIODS_1BASED["off"] + TOU_PERIODS_1BASED["semi"] +
                    TOU_PERIODS_1BASED["peak"])
    if all_1b != list(range(1, 25)):
        raise AssertionError(f"TOU zones do not partition {{1..24}}: union = {all_1b}")
    if TOU_ZONE_SIZE != {"off": 8, "semi": 11, "peak": 5}:
        raise AssertionError(f"TOU_ZONE_SIZE drift: {TOU_ZONE_SIZE}")


_tou_consistency_check()


def _to_1based(load_24h, hour_ending):
    import numpy as np
    arr = np.asarray(load_24h, dtype=float).reshape(-1)
    if arr.shape[0] != 24:
        raise ValueError(f"expected 24-hour vector, got shape {arr.shape}")
    if hour_ending == "1based":
        return arr
    if hour_ending == "0based":
        return np.concatenate([arr[1:], arr[:1]])
    raise ValueError(f"hour_ending must be '0based' or '1based', got {hour_ending!r}")


def _zone_mean(arr_1based, zone):
    import numpy as np
    idx = [h - 1 for h in TOU_PERIODS_1BASED[zone]]
    return float(np.mean(arr_1based[idx]))


def _peak_normalise(arr):
    """Divide a 24-h profile by its daily peak. Raises on non-positive peak."""
    import numpy as np
    denom = float(arr.max())
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError(f"non-positive / non-finite peak: {denom}")
    return arr / denom


def tou_zone_means(load_24h, *, hour_ending="1based"):
    """Three TOU-zone means (off, semi, peak) of a 24-h profile under PEAK normalisation.

    Each hour is divided by the daily peak before zone aggregation,
    so every output value lies in [0, 1]. The companion helper ``tou_chart_features``
    returns the daily load factor (mean/peak) and sigma as overlays.

    Mean normalisation is not supported here; if needed for a one-off
    sensitivity, pre-divide at the call site: ``tou_zone_means(v / v.mean())``.

    Parameters
    ----------
    load_24h : array-like, shape (24,)
    hour_ending : {"1based", "0based"}
        Input indexing convention. "1based": idx 0 = h=1 = end-of-01:00.
        "0based": idx 0 = h=0 = end-of-00:00 (23:00-00:00).
    """
    arr = _peak_normalise(_to_1based(load_24h, hour_ending))
    return _zone_mean(arr, "off"), _zone_mean(arr, "semi"), _zone_mean(arr, "peak")


def tou_chart_features(load_24h, *, hour_ending="1based"):
    """Five TOU-relevant chart features {mu_off, mu_semi, mu_peak, Lbar, sigma}.

    PEAK normalisation only. ``Lbar`` is the daily load factor
    (mean / peak) and ``sigma`` is the standard deviation of the peak-normalised
    profile. Both are interpretive overlays on the per-ARLP TOU annotation
    chart only — they are NOT clustering features and are NOT consumed by
    Lens 2 or Lens 2-GMM.
    """
    arr = _peak_normalise(_to_1based(load_24h, hour_ending))
    return {
        "mu_off":  _zone_mean(arr, "off"),
        "mu_semi": _zone_mean(arr, "semi"),
        "mu_peak": _zone_mean(arr, "peak"),
        "Lbar":    float(arr.mean()),     # = mean / peak = load factor
        "sigma":   float(arr.std(ddof=0)),
    }


# No workday/weekend differentiation. The representative-day matrix is
# built by season only (winter, summer); within each season, all calendar
# days are averaged together. The TOU schedule (off / semi / peak) is
# hour-of-day only (TOU_PERIODS_*BASED) and the tariff design applies the
# same hourly schedule to every day.
SEG_ORDER = ["winter", "summer"]


def build_repday_matrix(load_matrix, timestamps, arlp_names, seg_order=None):
    """Build the (n, 48) per-season representative-day matrix.

    Within each season, all calendar days are averaged hour-by-hour without
    any day-type bucketing (no workday/weekend split). Returns a DataFrame
    indexed by ARLP name with columns named ``f"{season}_h{h:02d}"`` for
    season in {winter, summer} and h in 0..23 (0-based hour-ending).
    """
    import numpy as np
    import pandas as pd
    if seg_order is None:
        seg_order = SEG_ORDER
    ts = pd.to_datetime(pd.Series(timestamps))
    hour  = ts.dt.hour.to_numpy()
    month = ts.dt.month.to_numpy()
    season = np.where(np.isin(month, list(WINTER_MONTHS)), "winter", "summer")
    rows = {}
    load_arr = np.asarray(load_matrix)
    for i, name in enumerate(arlp_names):
        vec = []
        for sn in seg_order:
            mask = season == sn
            block = (pd.Series(load_arr[i][mask])
                     .groupby(hour[mask]).mean().reindex(range(24)).to_numpy())
            vec.extend(block)
        rows[name] = vec
    cols = [f"{sn}_h{h:02d}" for sn in seg_order for h in range(24)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def build_tou_feature_matrix(repday_df, arlp_names=None, seg_order=None):
    """Build the (n, 6) season-level TOU feature matrix.

    For each row and each season in ``seg_order``, the 24-hour block is
    peak-normalised and the three TOU zone means
    ``(mu_off, mu_semi, mu_peak)`` are emitted. Output columns:
        winter_off, winter_semi, winter_peak,
        summer_off, summer_semi, summer_peak.

    No workday/weekend bucketing; the per-season profile in ``repday_df``
    already aggregates all days of that season.
    """
    import numpy as np
    import pandas as pd
    if seg_order is None:
        seg_order = SEG_ORDER
    if arlp_names is None:
        arlp_names = list(repday_df.index)
    rows = []
    for name in arlp_names:
        row = []
        for sn in seg_order:
            cols = [f"{sn}_h{h:02d}" for h in range(24)]
            v = repday_df.loc[name, cols].to_numpy(float)
            mu_off, mu_semi, mu_peak = tou_zone_means(v, hour_ending="0based")
            row.extend([mu_off, mu_semi, mu_peak])
        rows.append(row)
    X = np.asarray(rows, dtype=float)
    cols_x = [f"{sn}_{z}" for sn in seg_order for z in TOU_ZONES]
    return pd.DataFrame(X, index=arlp_names, columns=cols_x)


_ZONE_COLOR = {"off": "seagreen", "semi": "goldenrod", "peak": "crimson"}
_ZONE_HUMAN = {"off": "Off-peak", "semi": "Semi-peak", "peak": "Peak"}
_ZONE_OF_HOUR = {}
for _z in TOU_ZONES:
    for _h in TOU_PERIODS_1BASED[_z]:
        _ZONE_OF_HOUR[_h] = _z


def plot_tou_annotation_chart(load_24h, *, title, out_path, hour_ending="1based",
                                true_peak_h: int | None = None):
    """Render the peak-norm TOU annotation chart for a 24-h load profile.

    Layout:
      - Bar for hour-ending h spans the physical clock interval [h-1, h]; the
        y-axis sits at x=0 and x=24 marks the end of the day.
      - Each hour's bar is coloured by its TOU zone (off / semi / peak) and its
        HEIGHT equals the zone mean.
      - A green line + markers traces the peak-normalised representative load.
      - Two horizontal overlays show the load factor and the standard deviation.
      - Legend uses plain English labels (no LaTeX symbols). No footer text.

    The optional ``true_peak_h`` keyword draws a black dashed vertical line
    at the centre of the bar for that hour (``true_peak_h + 0.5`` on the
    x-axis). It defaults to ``None`` (no line).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pathlib import Path as _Path

    feat = tou_chart_features(load_24h, hour_ending=hour_ending)
    arr_raw = np.asarray(load_24h, dtype=float).reshape(-1)
    if hour_ending == "0based":
        arr_raw = np.concatenate([arr_raw[1:], arr_raw[:1]])
    arr = arr_raw / arr_raw.max()      # peak-norm

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Coloured zone columns: bar h spans [h-1, h]; height = zone mean.
    left_edges = np.arange(0, 24)
    hour_endings = np.arange(1, 25)
    heights = np.array([feat[f"mu_{_ZONE_OF_HOUR[h]}"] for h in hour_endings])
    colours = [_ZONE_COLOR[_ZONE_OF_HOUR[h]] for h in hour_endings]
    ax.bar(left_edges, heights, width=1.0, color=colours, alpha=0.45,
           edgecolor="none", align="edge")

    # Representative load profile — markers at bar centres
    centres = left_edges + 0.5
    ax.plot(centres, arr, color="forestgreen", marker="o", lw=1.8, ms=5,
            label="Representative load profile")

    # Interpretive overlays
    ax.axhline(feat["Lbar"],  color="crimson",   ls="--", lw=1.3,
               label=f"Load factor = {feat['Lbar']:.2f}")
    ax.axhline(feat["sigma"], color="steelblue", ls="-.", lw=1.3,
               label=f"Standard deviation = {feat['sigma']:.2f}")

    # Optional vertical at the true peak hour
    if true_peak_h is not None:
        ax.axvline(true_peak_h + 0.5, color="black", ls="--", lw=1.2,
                   label=f"True peak (h = {true_peak_h + 1})")

    zone_handles = [
        Patch(facecolor=_ZONE_COLOR[z], alpha=0.45, edgecolor="none",
              label=f"{_ZONE_HUMAN[z]} mean load = {feat[f'mu_{z}']:.2f}")
        for z in ("off", "semi", "peak")
    ]

    ax.set_xlim(0, 24)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(0, 25))
    ax.set_xticklabels([str(h) for h in range(0, 25)], fontsize=8)
    ax.set_xlabel("hour-ending")
    ax.set_ylabel("peak normalized load")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(line_handles + zone_handles,
              line_labels + [h.get_label() for h in zone_handles],
              loc="upper left", fontsize=8, framealpha=0.92, ncol=2)

    fig.tight_layout()
    out = _Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return feat
