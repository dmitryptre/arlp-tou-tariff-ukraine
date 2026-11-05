"""Within-class distribution of passive bill changes: pure functions.

Passive change of a class-EULP unit (c, j), F -> P3, unchanged hourly consumption, no response, ₴ excl. VAT:
    B0  = sum_m b_cjm E_jm,    BP3 = sum_m sum_z r(c, tau(m), z, g_cjm) E_jmz,    dB = 100 (BP3 / B0 - 1)
(positive = a bill increase; the article's bill-impact table uses the opposite sign). The P3 rates are the fixed
4-decimal rates of `proposed_rates_per_class_season_period.csv`. ESH base prices by case:
- T: the dwelling's own monthly consumption sets the month's status (Oct-Apr, <= 2,000 kWh -> ₴2.20, else ₴3.60);
- A: each source ARLP's per-household monthly consumption sets it; a unit's bill is the
     attribution-weighted mixture of its bills under its ARLPs' statuses;
- L: no threshold, ₴2.20 in every ESH winter month (the convention that ignores the 2,000 kWh threshold).
A month's status is held in theta (share of the month billed at the ₴3.60 regime): rate = (1 - theta) r_lo + theta r_hi.
theta is 0 for non-ESH units (whose rates do not depend on it) and 1 in the ESH summer months.

Weighted quantiles are the weighted inverted-CDF ("lower") quantile (numpy 1.24 has no weights).
Display rules: `format_pct`, `format_share`; the near-zero band |x| < 0.05 prints "0.0".
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from _common import (
    ESH_BASE_PRICE_HIGH_UAH,
    ESH_BASE_PRICE_LOW_UAH,
    NON_ESH_BASE_PRICE_UAH,
    TOU_PERIODS_0BASED,
    WINTER_MONTHS,
)
from eulp_hierarchy import CLASS_ORDER, ESH_CLASSES, MONTHS, SEASONS, ZONES

CASES = ("T", "A", "L")
PANELS = ("A", "B", "C", "ESH-1", "ESH-2", "All")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_METHOD = "weighted_inverted_cdf"
Q_TOL = 1e-12
NEAR_ZERO_BAND = 0.05                     # the display precision of one decimal; not a materiality threshold
MINUS = "−"
TARIFFS = ("F", "E2", "E3", "P3")
COMPARISONS = (("F", "P3"), ("F", "E3"), ("E2", "P3"), ("E2", "E3"), ("E3", "P3"))
BILL_TABLE_TARIFF = {"F": "flat", "E2": "e2z", "E3": "e3z", "P3": "p3z"}
YAXIS_CEILINGS = (5, 10, 20, 25, 50, 75, 100)
YAXIS_STEP = {5: 1, 10: 2, 20: 5, 25: 5, 50: 10, 75: 25, 100: 25}
SEASON_IDX = np.array([0 if m in WINTER_MONTHS else 1 for m in MONTHS])      # month index -> 0 winter, 1 summer
WINTER_MASK = SEASON_IDX == 0
E2_COEF = np.array([0.5, 1.0, 1.0])       # zone order (off, shoulder, peak): night = the off-peak hours
E3_COEF = np.array([0.4, 1.0, 1.5])
P3_COLUMNS = ["cluster", "label", "season", "period", "base_tag", "alpha", "base_UAH_per_kWh", "rate_UAH_per_kWh"]


# ── fixed inputs ───────────────────────────────────────────────────────────────────────────────────────────────
def check_p3_rates(df) -> float:
    """The schema of the proposed-rates file and |rate - round(alpha x base, 4)| <= 1e-9 on all 42 rows; returns the max deviation."""
    assert list(df.columns) == P3_COLUMNS, list(df.columns)
    assert len(df) == 42, len(df)
    assert set(df["label"]) == set(CLASS_ORDER) and set(df["season"]) == set(SEASONS)
    assert set(df["period"]) == set(ZONES)
    dev = max(abs(r.rate_UAH_per_kWh - round(r.alpha * r.base_UAH_per_kWh, 4)) for r in df.itertuples())
    assert dev <= 1e-9, dev
    return float(dev)


def load_p3_rates(path):
    """Proposed-rates file -> (frame, {(class, season, zone, base_tag): rate})."""
    df = pd.read_csv(path)
    check_p3_rates(df)
    return df, {(r.label, r.season, r.period, r.base_tag): float(r.rate_UAH_per_kWh) for r in df.itertuples()}


def load_statutory(path) -> dict:
    """Statutory tariffs (inputs/neurc_household_tariffs_uah.csv) -> prices, threshold, discount months, zone hours
    (0-based) and coefficients; asserts that they equal the `_common` constants."""
    t = pd.read_csv(path)
    flat = t[t.tariff == "flat"].set_index("regime")

    def hours(tariff, zone):
        r = t[(t.tariff == tariff) & (t.regime == "regular") & (t.zone == zone)].iloc[0]
        return sorted(0 if int(h) == 24 else int(h) for h in str(r.hours_hour_ending).split(";"))

    def coef(tariff, zone, regime="regular"):
        return float(t[(t.tariff == tariff) & (t.regime == regime) & (t.zone == zone)].iloc[0].coefficient)

    esh = t[t.regime == "esh_discounted"]
    out = {"price_regular": float(flat.at["regular", "rate_uah_per_kwh_excl_vat"]),
           "price_esh_discounted": float(flat.at["esh_discounted", "rate_uah_per_kwh_excl_vat"]),
           "threshold_kWh": float(esh.monthly_threshold_kwh.unique().item()),
           "discount_months": sorted(int(m) for m in esh.months.unique().item().split(";")),
           "zone_hours": {"off": hours("e3z", "off-peak"), "shoulder": hours("e3z", "shoulder"),
                          "peak": hours("e3z", "peak")},
           "e2_night_hours": hours("e2z", "night"),
           "e2_coef": {"night": coef("e2z", "night"), "day": coef("e2z", "day")},
           "e3_coef": {"off": coef("e3z", "off-peak"), "shoulder": coef("e3z", "shoulder"), "peak": coef("e3z", "peak")}}
    assert out["price_regular"] == NON_ESH_BASE_PRICE_UAH == ESH_BASE_PRICE_HIGH_UAH
    assert out["price_esh_discounted"] == ESH_BASE_PRICE_LOW_UAH
    assert out["threshold_kWh"] == 2000.0
    assert set(out["discount_months"]) == set(WINTER_MONTHS)
    assert out["zone_hours"] == {"off": TOU_PERIODS_0BASED["off"], "shoulder": TOU_PERIODS_0BASED["semi"],
                                 "peak": TOU_PERIODS_0BASED["peak"]}
    assert out["e2_night_hours"] == TOU_PERIODS_0BASED["off"]
    assert [out["e2_coef"]["night"], 1.0, out["e2_coef"]["day"]] == list(E2_COEF)
    assert [out["e3_coef"][z] for z in ZONES] == list(E3_COEF)
    for z in ("off-peak", "shoulder", "peak"):                              # the discounted rows: same coefficients
        assert coef("e3z", z, "esh_discounted") == coef("e3z", z)
    return out


# ── regimes and bills ──────────────────────────────────────────────────────────────────────────────────────────
def base_regime(case, monthly_kWh, is_esh, arlp_theta=None, threshold=2000.0) -> np.ndarray:
    """theta (n, 12): the share of each month billed at the ₴3.60 ESH regime.

    Non-ESH units: 0 (their rates are the flat-base P3 rates whatever theta is). ESH units: 1 in May-September;
    in October-April, case T: 1 if the unit's own monthly kWh > threshold (2,000.0 -> lo, 2,000.0001 -> hi);
    case A: `arlp_theta` (n, 12), the attribution-weighted statuses of the unit's ARLPs; case L: 0.
    """
    monthly_kWh = np.asarray(monthly_kWh, dtype=float)
    is_esh = np.asarray(is_esh, dtype=bool)
    n = monthly_kWh.shape[0]
    if case == "T":
        winter = (monthly_kWh > threshold).astype(float)
    elif case == "A":
        if arlp_theta is None:
            raise ValueError("case A needs the ARLP statuses")
        winter = np.asarray(arlp_theta, dtype=float)
    elif case == "L":
        winter = np.zeros((n, 12))
    else:
        raise ValueError(case)
    theta = np.where(WINTER_MASK[None, :], winter, 1.0)
    return np.where(is_esh[:, None], theta, 0.0)


def rate_arrays(classes, rates) -> tuple:
    """Per unit and month: P3 rates at the lo and hi regimes (n, 12, 3) and base prices (n, 12) at lo and hi."""
    classes = list(classes)
    seasons = [SEASONS[i] for i in SEASON_IDX]
    r_lo = np.empty((len(classes), 12, 3))
    r_hi = np.empty_like(r_lo)
    cache = {}
    for c in set(classes):
        lo_tag, hi_tag = ("lo", "hi") if c in ESH_CLASSES else ("flat", "flat")
        cache[c] = (np.array([[rates[(c, s, z, lo_tag)] for z in ZONES] for s in seasons]),
                    np.array([[rates[(c, s, z, hi_tag)] for z in ZONES] for s in seasons]))
    for i, c in enumerate(classes):
        r_lo[i], r_hi[i] = cache[c]
    esh = np.array([c in ESH_CLASSES for c in classes])
    b_lo = np.where(esh, ESH_BASE_PRICE_LOW_UAH, NON_ESH_BASE_PRICE_UAH)[:, None] * np.ones((1, 12))
    b_hi = np.full((len(classes), 12), NON_ESH_BASE_PRICE_UAH)
    return r_lo, r_hi, b_lo, b_hi


def unit_bills(E, classes, theta, rates) -> dict:
    """Bills (₴ per dwelling-year) and their decomposition for units with monthly zone energies E (n, 12, 3).

    Returns arrays: B0, BP3, BE2, BE3, dbill_UAH, dbill_pct, contrib (n, 2, 3) in % (season x zone, summing to
    dbill_pct), b (n, 12), rate (n, 12, 3), sigma (n, 2, 3) (the b-weighted shares of B0), breakeven_share (NaN
    where dB <= 0), annual_kWh.
    """
    E = np.asarray(E, dtype=float)
    theta = np.asarray(theta, dtype=float)
    r_lo, r_hi, b_lo, b_hi = rate_arrays(classes, rates)
    rate = (1.0 - theta)[..., None] * r_lo + theta[..., None] * r_hi
    b = (1.0 - theta) * b_lo + theta * b_hi
    Em = E.sum(axis=2)
    B0 = (b * Em).sum(axis=1)
    BP3 = (rate * E).sum(axis=(1, 2))
    BE2 = (b[..., None] * E2_COEF * E).sum(axis=(1, 2))
    BE3 = (b[..., None] * E3_COEF * E).sum(axis=(1, 2))
    diff = (rate - b[..., None]) * E
    bE = b[..., None] * E
    contrib = np.stack([diff[:, SEASON_IDX == k, :].sum(axis=1) for k in (0, 1)], axis=1) * 100.0 / B0[:, None, None]
    sigma = np.stack([bE[:, SEASON_IDX == k, :].sum(axis=1) for k in (0, 1)], axis=1) / B0[:, None, None]
    dB = BP3 - B0
    denom = ((rate[:, :, 2] - rate[:, :, 0]) * E[:, :, 2]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        be = np.where(dB > 0, dB / denom, np.nan)
    return {"B0": B0, "BP3": BP3, "BE2": BE2, "BE3": BE3, "dbill_UAH": dB, "dbill_pct": 100.0 * (BP3 / B0 - 1.0),
            "contrib": contrib, "b": b, "rate": rate, "sigma": sigma, "breakeven_share": be,
            "annual_kWh": E.sum(axis=(1, 2))}


def class_level_bills(E, W, classes, theta, rates) -> dict:
    """The class bills (M₴, H basis) from the summed class energies split by regime, per class and tariff.
    The energies at the lo and hi regimes are summed over the units first; then the rates are applied."""
    E = np.asarray(E, dtype=float)
    W = np.asarray(W, dtype=float)
    theta = np.asarray(theta, dtype=float)
    r_lo, r_hi, b_lo, b_hi = rate_arrays(classes, rates)
    classes = np.asarray(list(classes))
    out = {}
    for c in CLASS_ORDER:
        m = classes == c
        if not m.any():
            continue
        e_lo = ((W[m] * 1.0)[:, None, None] * (1.0 - theta[m])[..., None] * E[m]).sum(axis=0)
        e_hi = ((W[m] * 1.0)[:, None, None] * theta[m][..., None] * E[m]).sum(axis=0)
        i = np.flatnonzero(m)[0]
        out[c] = {"F": float((b_lo[i][:, None] * e_lo + b_hi[i][:, None] * e_hi).sum() / 1e3),
                  "E2": float(((b_lo[i][:, None] * e_lo + b_hi[i][:, None] * e_hi) * E2_COEF).sum() / 1e3),
                  "E3": float(((b_lo[i][:, None] * e_lo + b_hi[i][:, None] * e_hi) * E3_COEF).sum() / 1e3),
                  "P3": float((r_lo[i] * e_lo + r_hi[i] * e_hi).sum() / 1e3)}
    return out


# ── statistics ─────────────────────────────────────────────────────────────────────────────────────────────────
def weighted_quantile(x, w, qs, sort_keys=()) -> np.ndarray:
    """Weighted inverted-CDF ("lower") quantile: sort by (x, *sort_keys) ascending, F_k = cumulative weight
    share, Q(q) = x_(k*) with k* = min{k : F_k >= q - 1e-12}. Zero weights are excluded. With w = 1 it equals
    np.quantile(x, q, method="inverted_cdf"). Every value returned is an x of the input."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = w > 0
    if not keep.any():
        raise ValueError("no positive weight")
    keys = [np.asarray(k)[keep] for k in sort_keys]
    x, w = x[keep], w[keep]
    order = np.lexsort(tuple(reversed([x, *keys])))
    xs, ws = x[order], w[order]
    F = np.cumsum(ws) / ws.sum()
    return np.array([xs[int(np.argmax(F >= q - Q_TOL))] for q in np.atleast_1d(qs)])


def _share(w, mask, tot):
    return 100.0 * float(np.asarray(w)[mask].sum()) / tot


def top_weight_order(w, cls_idx, eid):
    """Unit order by weight, descending; ties by class order, then eulp_id (deterministic)."""
    return np.lexsort((np.asarray(eid), np.asarray(cls_idx), -np.asarray(w, dtype=float)))


def n_units_for_share(w, order, share):
    c = np.cumsum(np.asarray(w, dtype=float)[order]) / float(np.sum(w))
    return int(np.argmax(c >= share - Q_TOL)) + 1


def panel_stats(x, w, cls, eid, *, B0=None, BP3=None, q=None, dUAH=None, kwh=None, be=None) -> dict:
    """The statistics of one panel and weighting. Money columns only when B0 is given."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    cls = np.asarray(cls)
    cidx = np.array([CLASS_ORDER.index(c) for c in cls])
    eid = np.asarray(eid)
    keys = (cidx, eid)
    tot = float(w.sum())
    out = {"n_units": int((w > 0).sum()), "total_weight_kHH": tot}
    for qv, v in zip(QUANTILES, weighted_quantile(x, w, QUANTILES, keys)):
        out[f"p{round(qv * 100):02d}_pct"] = float(v)
    order = top_weight_order(w, cidx, eid)
    n99 = n_units_for_share(w, order, 0.99)
    top99 = np.zeros(len(w), dtype=bool)
    top99[order[:n99]] = True
    out.update({"min_pct": float(x.min()), "max_pct": float(x.max()),
                "min_top99w_pct": float(x[top99].min()), "max_top99w_pct": float(x[top99].max()),
                "share_ge0_pct": _share(w, x >= 0, tot), "share_lt0_pct": _share(w, x < 0, tot),
                "share_gt_p5_pct": _share(w, x > 5, tot), "share_gt_p10_pct": _share(w, x > 10, tot),
                "share_lt_m5_pct": _share(w, x < -5, tot), "share_lt_m10_pct": _share(w, x < -10, tot),
                "share_abs_lt_0p05_pct": _share(w, np.abs(x) < NEAR_ZERO_BAND, tot),
                "share_ge0_abs_lt_0p05_pct": _share(w, (x >= 0) & (x < NEAR_ZERO_BAND), tot),
                "share_lt0_abs_lt_0p05_pct": _share(w, (x < 0) & (x > -NEAR_ZERO_BAND), tot),
                "n_abs_lt_0p05": int((np.abs(x) < NEAR_ZERO_BAND).sum()), "n_exact_zero": int((x == 0.0).sum()),
                "mean_hh_pct": float((w * x).sum() / tot)})
    i_max = order[0]
    out.update({"largest_unit_weight_pct": 100.0 * float(w[i_max]) / tot, "largest_unit_eulp_id": int(eid[i_max]),
                "largest_unit_class": str(cls[i_max]), "kish_neff": tot ** 2 / float((w ** 2).sum()),
                "top5_weight_pct": 100.0 * float(w[order[:5]].sum()) / tot,
                "n_units_90w": n_units_for_share(w, order, 0.90), "n_units_99w": n99})
    if B0 is not None:
        B0, BP3, q = np.asarray(B0, float), np.asarray(BP3, float), np.asarray(q, float)
        d = BP3 - B0
        out["aggregate_pct"] = 100.0 * ((w * BP3).sum() / (w * B0).sum() - 1.0)
        for sfx, f in (("", np.ones_like(q)), ("_table_basis", q)):
            out[f"bill_F_MUAH{sfx}"] = float((w * f * B0).sum() / 1e3)
            out[f"bill_P3_MUAH{sfx}"] = float((w * f * BP3).sum() / 1e3)
            out[f"positive_part_MUAH{sfx}"] = float((w * f * np.maximum(d, 0.0)).sum() / 1e3)
            out[f"negative_part_MUAH{sfx}"] = float((w * f * np.maximum(-d, 0.0)).sum() / 1e3)
            out[f"net_change_MUAH{sfx}"] = float((w * f * d).sum() / 1e3)
        out["positive_part_pct_of_bill_F"] = 100.0 * out["positive_part_MUAH"] / out["bill_F_MUAH"]
        out["positive_part_pct_of_bill_F_table_basis"] = 100.0 * out["positive_part_MUAH_table_basis"] / out["bill_F_MUAH_table_basis"]
    if dUAH is not None:
        p = weighted_quantile(dUAH, w, (0.05, 0.50, 0.95), keys)
        out.update({"dbill_UAH_p05": float(p[0]), "dbill_UAH_p50": float(p[1]), "dbill_UAH_p95": float(p[2]),
                    "annual_kWh_p50": float(weighted_quantile(kwh, w, (0.5,), keys)[0])})
    if be is not None:
        be = np.asarray(be, dtype=float)
        pos = x > 0
        if pos.any():
            out["breakeven_median_share"] = float(weighted_quantile(be[pos], w[pos], (0.5,),
                                                                    (cidx[pos], eid[pos]))[0])
            out["share_breakeven_unattainable_pct"] = 100.0 * float(w[pos & (be > 1.0)].sum()) / float(w[pos].sum())
        else:
            out["breakeven_median_share"] = np.nan
            out["share_breakeven_unattainable_pct"] = np.nan
    return out


def top99_mask(w, cls, eid):
    w = np.asarray(w, dtype=float)
    cidx = np.array([CLASS_ORDER.index(c) for c in np.asarray(cls)])
    order = top_weight_order(w, cidx, eid)
    m = np.zeros(len(w), dtype=bool)
    m[order[:n_units_for_share(w, order, 0.99)]] = True
    return m


# ── histogram rule and y-axis rule ─────────────────────────────────────────────────────────────────────────────
def bin_rule(pooled_p005, pooled_p995, panel_p005, panel_p995, max_bins=80) -> dict:
    """Width 0.5 / 1 / 2 pp from R = pooled P99.5 - P0.5 (R <= 15, <= 40, > 40); the next width if more than
    `max_bins` interior bins; L = w floor(min P0.5 / w) - w, U = w ceil(max P99.5 / w) + w, L <= -w, U >= +w."""
    R = float(pooled_p995 - pooled_p005)
    widths = (0.5, 1.0, 2.0)
    start = 0 if R <= 15 else 1 if R <= 40 else 2
    for w in widths[start:]:
        L = min(w * math.floor(min(panel_p005) / w) - w, -w)
        U = max(w * math.ceil(max(panel_p995) / w) + w, w)
        n = round((U - L) / w)
        if n <= max_bins:
            return {"width_pp": w, "range_L_pct": L, "range_U_pct": U, "rule_R_pp": R, "n_bins": n}
    raise ValueError(f"bin rule: more than {max_bins} bins at 2 pp (R = {R}); the bin rule does not cover it")


def x_tick_step(width_pp) -> float:
    """X ticks every 1 pp at a bin width of 0.5 pp, every 5 pp at
    1 pp, every 10 pp at 2 pp."""
    return {0.5: 1.0, 1.0: 5.0, 2.0: 10.0}[float(width_pp)]


def bin_edges(rule) -> np.ndarray:
    return np.array([rule["range_L_pct"] + k * rule["width_pp"] for k in range(rule["n_bins"] + 1)])


def histogram(x, w, rule) -> pd.DataFrame:
    """Left-closed bins [a, b) on the common edges, plus the two open edge bins (values < L and >= U), drawn at
    [L - w, L) and [U, U + w). Shares in % of the panel's weight; all rows kept (empty ones too)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    edges = bin_edges(rule)
    wd = rule["width_pp"]
    k = np.searchsorted(edges, x, side="right") - 1          # -1: below L; n_bins: at or above U
    tot = float(w.sum())
    rows = [{"bin_index": 0, "left_pct": edges[0] - wd, "right_pct": edges[0], "is_edge_low": True,
             "is_edge_high": False, "share_pct": _share(w, k < 0, tot), "n_units": int((k < 0).sum())}]
    for i in range(rule["n_bins"]):
        rows.append({"bin_index": i + 1, "left_pct": edges[i], "right_pct": edges[i + 1], "is_edge_low": False,
                     "is_edge_high": False, "share_pct": _share(w, k == i, tot), "n_units": int((k == i).sum())})
    rows.append({"bin_index": rule["n_bins"] + 1, "left_pct": edges[-1], "right_pct": edges[-1] + wd,
                 "is_edge_low": False, "is_edge_high": True, "share_pct": _share(w, k >= rule["n_bins"], tot),
                 "n_units": int((k >= rule["n_bins"]).sum())})
    return pd.DataFrame(rows)


def yaxis_rule(h) -> dict:
    """The tick ceiling T = the smallest of 5, 10, 20, 25, 50, 75, 100 that is >= h (h clipped to 100),
    ticks 0..T with the fixed step, the axis top max(T, 1.08 h), or max(T, 1.15 h) when the bar carries a label
    (h > 50)."""
    h = float(h)
    if h > 100.0 + 1e-9:
        raise ValueError(f"tallest bar {h} > 100 %")
    h = min(h, 100.0)
    T = next(t for t in YAXIS_CEILINGS if t >= h)
    label = h > 50.0
    return {"tallest_bar_pct": h, "tick_ceiling_pct": float(T), "ticks": list(range(0, T + 1, YAXIS_STEP[T])),
            "y_top_pct": max(float(T), (1.15 if label else 1.08) * h), "bar_label": label}


# ── display ───────────────────────────────────────────────────────────────────────────────────────────────────
def _round1(x):
    return Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def format_pct(x, unit=True) -> str:
    """A bill change: one decimal, explicit sign, U+2212 for minus; |x| < 0.05 prints "0.0" without a sign."""
    x = float(x)
    if abs(x) < NEAR_ZERO_BAND:
        s = "0.0"
    else:
        d = _round1(x)
        s = f"+{d}" if d > 0 else f"{MINUS}{-d}"
    return s + ("%" if unit else "")


def format_share(x, unit=True) -> str:
    """A share: one decimal; > 0 rounding to 0.0 -> "<0.1"; < 100 rounding to 100.0 -> ">99.9"; exact 0 and 100."""
    x = float(x)
    if x == 0.0:
        s = "0.0"
    elif x == 100.0:
        s = "100.0"
    else:
        d = _round1(x)
        s = "<0.1" if d == 0 else ">99.9" if d == 100 else f"{d}"
    return s + ("%" if unit else "")


def format_count(n) -> str:
    return f"{int(n):,}"


def format_edge(v) -> str:
    """An axis value (an edge or a tick): sign, U+2212, no trailing .0."""
    v = float(v)
    s = f"{abs(v):g}"
    return "0" if v == 0 else (f"+{s}" if v > 0 else f"{MINUS}{s}")


def format_pct_fine(x, unit=True) -> str:
    """A bill change at three decimals for the note sentence on near-zero values: explicit sign,
    U+2212, half away from zero ("−0.047%")."""
    d = Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    s = "0.000" if d == 0 else (f"+{d}" if d > 0 else f"{MINUS}{-d}")
    return s + ("%" if unit else "")


# ── bill-impact table reproduction ─────────────────────────────────────────────────────────────────────────────
def change_pct(b_from, b_to):
    """This module's sign (positive = a bill increase): 100 (B_to / B_from - 1)."""
    return 100.0 * (b_to / b_from - 1.0)


def bill_table_reconciliation(bills_H, V_H, ref, *, reference, case, route, tol_pp, tol_rel) -> list:
    """Reconciliation rows against the article's bill-impact table for one reference, case and route.

    bills_H: {class: {tariff: M₴ on the H basis}}; V_H: {class: GWh}; ref: bill-impact table frame (a reference CSV).
    Per class: this module's average tariffs bill / V_H, the bills on the table's basis Q(c) x avg, the five passive
    changes (this module's sign) against the table's (sign converted). Total: the class bills on the table's basis
    summed, and its change from those sums (no weighted mean of percentages).
    """
    ref = ref.set_index("class")
    rows = []
    tot = {x: 0.0 for x in TARIFFS}
    for c in CLASS_ORDER:
        Q = float(ref.at[c, "cons_GWh"])
        for x in TARIFFS:
            tot[x] += Q * bills_H[c][x] / V_H[c]
    for c in (*CLASS_ORDER, "Total"):
        Q = float(ref.at[c, "cons_GWh"])
        if c == "Total":
            VH = sum(V_H[k] for k in CLASS_ORDER)
            t5b = dict(tot)
            avg = {x: t5b[x] / Q for x in TARIFFS}
            bh = {x: sum(bills_H[k][x] for k in CLASS_ORDER) for x in TARIFFS}
        else:
            VH = V_H[c]
            avg = {x: bills_H[c][x] / VH for x in TARIFFS}
            t5b = {x: Q * avg[x] for x in TARIFFS}
            bh = bills_H[c]
        for fr, to in COMPARISONS:
            ef, et = BILL_TABLE_TARIFF[fr], BILL_TABLE_TARIFF[to]
            agg = change_pct(t5b[fr], t5b[to])
            ref_pct = -float(ref.at[c, f"{ef}_{et}_passive_pct"])
            d_avg = max(abs(avg[fr] / float(ref.at[c, f"avg_{ef}"]) - 1), abs(avg[to] / float(ref.at[c, f"avg_{et}"]) - 1))
            d_bill = max(abs(t5b[fr] / float(ref.at[c, f"bill_{ef}_MUAH"]) - 1),
                         abs(t5b[to] / float(ref.at[c, f"bill_{et}_MUAH"]) - 1))
            ok = abs(agg - ref_pct) <= tol_pp and d_avg <= tol_rel and d_bill <= tol_rel
            note = ""
            if c == "Total":
                note = f"H-basis pooled change {change_pct(bh[fr], bh[to]):.9f} % (not compared)"
            rows.append({"reference": reference, "case": case, "class": c, "comparison": f"{fr}->{to}", "route": route,
                         "V_H_GWh": VH, "Q_GWh": Q, "q_factor": Q / VH,
                         "avg_from_units": avg[fr], "avg_to_units": avg[to],
                         "avg_from_table": float(ref.at[c, f"avg_{ef}"]), "avg_to_table": float(ref.at[c, f"avg_{et}"]),
                         "bill_from_table_basis_MUAH": t5b[fr], "bill_to_table_basis_MUAH": t5b[to],
                         "bill_from_table_MUAH": float(ref.at[c, f"bill_{ef}_MUAH"]),
                         "bill_to_table_MUAH": float(ref.at[c, f"bill_{et}_MUAH"]),
                         "aggregate_pct": agg, "table_pct_sign_converted": ref_pct, "diff_pp": agg - ref_pct,
                         "diff_avg_rel": d_avg, "diff_bill_rel": d_bill, "tolerance_pp": tol_pp,
                         "tolerance_rel": tol_rel, "status": "PASS" if ok else "FAIL", "note": note})
    return rows


def read_bill_table(path_or_bytes) -> pd.DataFrame:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        import io
        return pd.read_csv(io.BytesIO(path_or_bytes))
    return pd.read_csv(Path(path_or_bytes))
