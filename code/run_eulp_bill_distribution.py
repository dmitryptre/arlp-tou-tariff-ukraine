"""Within-class distribution of passive bill changes: the runner.

    python -X utf8 run_eulp_bill_distribution.py [--out DIR] [--no-figures]

Reads the inputs I-01 to I-13 (INPUTS below), computes the F->P3 passive bill change of every class-EULP pair in the
ESH billing cases T, A and L, runs the gates G-A to G-D and writes the files 01 to 11 into
outputs/ARLP_Clustering/eulp_bill_distribution/ (or --out). With a failed gate everything goes to
outputs/ARLP_Clustering/_unchecked/eulp_bill_distribution/ instead and the exit code is 1. Then the histogram figure
of the passive changes and the figure of the bill change against the peak share are drawn from the CSV files, unless
--no-figures. Nothing existing is written: the output folder is new.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
import time
from pathlib import Path

import eulp_bill_distribution as bd
import eulp_hierarchy as eh
import numpy as np
import pandas as pd
from _common import ARLP_NAMES_ALL, TOU_PERIODS_0BASED, WINTER_MONTHS

OUT_REL = Path("outputs/ARLP_Clustering/eulp_bill_distribution")
UNCHECKED_REL = Path("outputs/ARLP_Clustering/_unchecked/eulp_bill_distribution")
L13 = "outputs/ARLP_Clustering/notebook_only/layer13_peed_dr"
INPUTS = {   # paths below the root folder
    "I-01": "inputs/EULP_4435_hourly_load.csv",                       # EULP hourly loads (kWh), 8,760 x 4,435
    "I-02": "inputs/ARLPs_EULPs weight matrix.csv",                   # phi: EULP weights of each ARLP (columns sum to 1)
    "I-03": "inputs/ARLPs_EULPs absolute weight matrix.csv",          # ARLP-implied absolute weights
    "I-04": "inputs/ARLPs(Segments)_Hourly_Load.csv",                 # the 19 ARLP hourly loads
    "I-05": "inputs/arlp_household_counts.csv",                       # H_s, households per ARLP (kHH)
    "I-06": "inputs/neurc_household_tariffs_uah.csv",                 # statutory prices, ESH threshold, zone hours
    "I-07": f"{L13}/proposed_rates_per_class_season_period.csv",      # P3 rates (4 dp) and multipliers
    "I-08": f"{L13}/table5_bill_impacts.csv",                         # the notebook's class bill-impact table
    "I-09": f"{L13}/table4_proposed_tariff.csv",                      # the printed proposed tariff (3 dp)
    "I-10": "code/ARLP_clustering_analysis_autonomous.ipynb",         # zone hours of the notebook
    "I-11": f"{L13}/esh_threshold_monthly_per_arlp.csv",              # ESH ARLP monthly kWh and threshold status
    "I-12": f"{L13}/esh_effective_winter_base.csv",                   # zonal effective winter bases of the ESH classes
    "I-13": "inputs/reference/bill_impacts_no_esh_threshold.csv",     # the bill-impact table without the ESH threshold
}
# Tolerances fixed before any run. "rel" is relative to the reference value; for the reimbursement identity C11 to the
# larger of R+ and R- (the terms of the identity; the net can be close to 0).
TOL = {"B1": 1e-5, "B2": 1e-5, "B3": 5.0e-5 + 2e-6, "B4": 2e-8, "B8_kwh_i04": 1e-12, "B8_kwh_eulp": 1e-5,
       "C9": 1e-9, "C1": 1e-12, "C5": 1e-9, "C6": 1e-12, "C11": 1e-12, "C2d": 1e-12,
       "C2_arlp_pp": 1e-9, "C2_arlp_rel": 1e-10, "C2_eulp_pp": 1e-3, "C2_eulp_rel": 1e-5,
       "D1": 1e-9, "D2": 1e-6, "D3": 1e-9, "D3_display": 0.1, "D7": 1e-9, "D9": 1e-9, "A4": 0.00055,
       "I02_colsum": 2e-8, "I03_norm": 1e-9}
EXPECTED_UNITS = {"A": 536, "B": 721, "C": 1750, "ESH-1": 146, "ESH-2": 177}
EXPECTED_H = {"A": 9090, "B": 7047, "C": 258, "ESH-1": 765, "ESH-2": 99}
SEASON_ZONE = [(s, z) for s in eh.SEASONS for z in eh.ZONES]


class Gates:
    def __init__(self):
        self.rows = []

    def add(self, gate, cid, desc, value, tol, status, detail=""):
        self.rows.append({"gate": gate, "check_id": cid, "description": desc, "value": value, "tolerance": tol,
                          "status": status, "detail": detail})
        print(f"  [{status}] {cid}: {desc} | value {value} | tol {tol} {('| ' + detail) if detail else ''}")

    def check(self, gate, cid, desc, value, tol, ok, detail=""):
        self.add(gate, cid, desc, value, tol, "PASS" if ok else "FAIL", detail)

    @property
    def failed(self):
        return [r["check_id"] for r in self.rows if r["status"] == "FAIL"]


def write_csv(df, path):
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def notebook_cell(vault, cell_id):
    nb = json.loads((vault / INPUTS["I-10"]).read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if c.get("id") == cell_id)["source"])


# ── the scalar spot check of C9 ───────────────────────────────────────────────────────────────────────────────
def scalar_bills(load_col, ts, cls, rates, statuses_by_arlp, case, shares):
    """Month-by-month bills of one unit with plain loops (an independent path of C9)."""
    E = [[0.0, 0.0, 0.0] for _ in range(12)]
    zone_of = {h: i for i, z in enumerate(("off", "semi", "peak")) for h in TOU_PERIODS_0BASED[z]}
    for t, v in zip(ts, load_col):
        E[t.month - 1][zone_of[t.hour]] += float(v)
    esh = cls in eh.ESH_CLASSES

    def bills(status):          # status[m] in {0, 1}: 1 = the ₴3.60 regime
        b0 = bp = be2 = be3 = 0.0
        for m in range(12):
            season = "winter" if (m + 1) in WINTER_MONTHS else "summer"
            hi = (not esh) or season == "summer" or status[m] == 1
            base = 3.60 if hi else 2.20
            tag = "flat" if not esh else ("hi" if hi else "lo")
            for zi, z in enumerate(eh.ZONES):
                b0 += base * E[m][zi]
                bp += rates[(cls, season, z, tag)] * E[m][zi]
                be2 += base * (0.5 if z == "off" else 1.0) * E[m][zi]
                be3 += base * {"off": 0.4, "shoulder": 1.0, "peak": 1.5}[z] * E[m][zi]
        return np.array([b0, bp, be2, be3])

    if case == "T":
        return bills([1 if sum(E[m]) > 2000.0 else 0 for m in range(12)])
    if case == "L" or not esh:
        return bills([0] * 12)
    return sum(shares[a] * bills(statuses_by_arlp[a]) for a in shares)


FOLDER_README = """# Within-class distribution of passive bill changes

Written by `code/run_eulp_bill_distribution.py`; the figures by `make_bill_change_histograms.py`
(the histograms of the passive changes) and `make_bill_change_vs_peak_share.py` (the bill change against the annual
peak-zone share). The established outputs of the notebook are read as fixed inputs; nothing of them is changed.

## What is computed

For every class-EULP pair (c, j) with W_cj = sum_{s in c} H_s phi_sj > 0 (3,330 pairs from 3,145 EULPs; an EULP in
an ESH and a non-ESH class is two pairs), the annual bill under the flat tariff (F) and under the proposed 3z tariff
(P3) with the unchanged hourly consumption of the EULP dwelling: no behavioural response, no PEED, ₴ excl. VAT.
**The change is dbill_pct = 100 (bill_P3 / bill_F - 1): positive = a bill increase** (the bill-impact table of the
article uses the opposite sign). P3 rates: the fixed 4-decimal rates of
`notebook_only/layer13_peed_dr/proposed_rates_per_class_season_period.csv`.

**ESH billing cases** (October-April, a month of at most 2,000 kWh at ₴2.20/kWh on its whole volume, otherwise
₴3.60/kWh; May-September ₴3.60/kWh; the change of 1 February 2025 not modelled):
- **T** (primary; the histogram figure): the status of a month is set by the dwelling's own monthly consumption;
- **A**: by the per-household monthly consumption of each source ESH ARLP (the class-level convention of the
  notebook); a pair's bill is the attribution-weighted mixture of its bills under its ARLPs' statuses; it reproduces the
  bill-impact table (`table5_bill_impacts.csv`);
- **L**: no threshold, ₴2.20/kWh in every ESH winter month; it reproduces the bill-impact table computed without the
  threshold (`inputs/reference/bill_impacts_no_esh_threshold.csv`).
Non-ESH pairs are identical in all three cases.

**Weights.** `weighted` = W_cj (kHH), the household basis of the tariff design (H_s of `inputs/arlp_household_counts.csv`);
`unweighted` = 1 per pair (the eligible ResStock dwellings, not a household share); `arlp_implied` (diagnostic only) =
the ARLP-implied absolute weights / 1000. The weighted distribution describes the calibrated synthetic model, not
Ukrainian households. Weighted percentiles: the weighted inverted-CDF ("lower") quantile; every percentile is the
change of an actual pair.

**Volume bases.** Money on the H basis is sum_j W_cj x bill (M₴ = kHH x ₴ / 1000). `_table_basis` scales each class
by q_c = Q(c) / V_H(c), Q(c) = `cons_GWh` of the bill-impact table, V_H(c) = sum_j W_cj x annual kWh; this reproduces
the table's class bills. The H-basis volumes exceed Q(c) by 3.1-3.6 %.

**Modelled gross reimbursement requirement** (`positive_part_*`): the refunds of a full annual refund, for the first
year, of every passive increase over the shadow flat-tariff bill, in the model and before any response (annual
settlement, full take-up, no administrative cost, the volumes of U.S. dwellings). It is not an estimate or an upper
bound of the cost of an actual scheme; a monthly settlement would refund more (not computed).

**Near-zero changes** (|change| < 0.05 %, printed 0.0): potentially sensitive to the four-decimal rounding of the
tariff rates; their division between the >= 0 and < 0 groups should be read with caution. No sign is attributed to
the rounding.

## Files

| File | One row per | Content |
|---|---|---|
| `01_eulp_monthly_zone_kwh.csv` | EULP (3,145) | annual kWh and kWh per month (01-12) and zone (off, shoulder, peak) per dwelling; months and zones by the hour-ending stamp |
| `02_class_eulp_units.csv` | (case, class, EULP): 9,990 | weights, energy shares, deviations from the class aggregate, bills F/P3/E2/E3 (₴ per dwelling-year), the change and its six season-zone contributions, break-even share, validation comparisons |
| `03_class_eulp_arlp_attribution.csv` | (class, ARLP, EULP) with phi > 0: 6,337 | phi, H_s, W_s = H_s phi, share of the pair's weight (ARLP-16 has H = 0) |
| `04_distribution_summary.csv` | (case, panel, weighting, weight_basis): 54 | the six metrics of the histogram figure and the full statistics |
| `05_histogram_bins.csv` | (case, panel, weighting, bin) | the common bins and the y-axis rule results |
| `06_by_arlp_summary.csv` | (case, ARLP): 57 | the mean bill change of the ARLP's dwellings and within-ARLP percentiles (weights phi) |
| `07_aggregation_identity.csv` | (reference, case, class, comparison, route) | gate C: the reproduction of the bill-impact table (case A) and of the table without the threshold (case L), the Total from monetary bills, the identity of the summed bills |
| `08_threshold_diagnostics.csv` | (ESH class, case, quantity, key) | months above 2,000 kWh, case differences, the month-boundary diagnostic |
| `09_concentration_diagnostics.csv` | panel | Kish effective number, largest weights (both definitions), top-5 share, units for 90 % / 99 % of the weight |
| `10_figure_source.csv` | (panel, metric) | every data string printed in the histogram figure (`where = figure`), with its value, rule and the rendered text; the data strings of the class-B sentence of the figure note (`where = note`) |
| `11_qa_gates.csv` | check | every gate with PASS / FAIL / REPORT |
| `fig_bill_change_histograms.png` | — | the histograms of the passive changes |
| `fig_bill_change_vs_peak_share.png` | — | the bill change against the annual peak-zone share |

Conventions: UTF-8, LF, full float precision, kWh, kHH, ₴ excl. VAT, %.
"""


def main(out=None, figures=True):
    t0 = time.time()
    vault = eh.find_vault()
    g = Gates()
    print(f"within-class bill distribution; root {vault}")
    target = Path(out) if out is not None else vault / OUT_REL

    # Load ----------------------------------------------------------------------------------------------------------
    wdf = eh.load_weights(vault)
    wabs = eh.load_weights(vault, absolute=True)
    H = eh.load_household_counts(vault)
    arlp = eh.load_arlp_hourly(vault)
    ids = wdf.index.to_numpy()
    t_read = time.time()
    ts, L = eh.load_eulp_hourly(vault, ids)
    print(f"I-01 read: {L.shape} in {time.time() - t_read:.1f} s")
    p3df, rates = bd.load_p3_rates(vault / INPUTS["I-07"])
    stat = bd.load_statutory(vault / INPUTS["I-06"])
    t5post = pd.read_csv(vault / INPUTS["I-08"])
    t5pre = pd.read_csv(vault / INPUTS["I-13"])
    t4 = pd.read_csv(vault / INPUTS["I-09"])
    i11 = pd.read_csv(vault / INPUTS["I-11"])
    i12 = pd.read_csv(vault / INPUTS["I-12"])

    # I-01..I-03 validation at read
    phi = wdf[ARLP_NAMES_ALL].to_numpy()
    colsum_dev = float(np.abs(phi.sum(axis=0) - 1).max())
    g.check("G-B", "I-02 read", "phi: 3,145 rows; column sums 1; no negatives", f"rows {len(wdf)}, max|sum-1| "
            f"{colsum_dev:.2e}", TOL["I02_colsum"], len(wdf) == 3145 and colsum_dev <= TOL["I02_colsum"]
            and (phi >= 0).all())
    absn = wabs.loc[ids, ARLP_NAMES_ALL] / wabs[ARLP_NAMES_ALL].sum(axis=0)
    norm_dev = float(np.nanmax(np.abs(absn.to_numpy() - phi)))
    g.check("G-B", "I-03 read", "absolute weights normalized = phi", f"rows {len(wabs)}, {norm_dev:.2e}",
            TOL["I03_norm"], len(wabs) == 4435 and norm_dev <= TOL["I03_norm"])
    g.check("G-B", "I-01 read", "8,760 rows; no NaN; all >= 0", f"{L.shape}", "exact",
            L.shape == (8760, 3145) and not np.isnan(L).any() and (L >= 0).all())

    # B6, B5 --------------------------------------------------------------------------------------------------
    cal = eh.calendar(ts)
    cal4 = eh.calendar(arlp["timestamp"])
    sh_counts = cal.groupby(["season", "hour"]).size()
    month_hours = cal.groupby("month").size().tolist()
    b6 = (len(ts) == 8760 and (pd.Series(ts).reset_index(drop=True) == arlp["timestamp"].reset_index(drop=True)).all()
          and set(sh_counts.loc["winter"]) == {212} and set(sh_counts.loc["summer"]) == {153}
          and month_hours == [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])
    g.check("G-B", "B6", "calendar: 8,760 stamps, I-01 = I-04, 212/153 per (season, hour), month hours",
            f"{len(ts)}; {month_hours}", "exact", b6)
    c20 = notebook_cell(vault, "9dc9bc97")
    m = re.search(r"PEAK_H, OFF_H = (\[[^\]]*\]), (\[[^\]]*\])", c20)
    peak_nb, off_nb = (json.loads(m.group(1)), json.loads(m.group(2))) if m else (None, None)
    all_h = sorted(TOU_PERIODS_0BASED["off"] + TOU_PERIODS_0BASED["semi"] + TOU_PERIODS_0BASED["peak"])
    b5 = (peak_nb == TOU_PERIODS_0BASED["peak"] and off_nb == TOU_PERIODS_0BASED["off"] and all_h == list(range(24))
          and stat["zone_hours"]["peak"] == peak_nb)
    g.check("G-B", "B5", "zones: _common = I-06 (24 -> 0) = the notebook's PEAK_H/OFF_H; a partition of 0..23",
            f"peak {peak_nb} off {off_nb}", "exact", b5)
    g.add("G-B", "T11", "statutory prices, threshold, months, coefficients = _common (load_statutory)",
          f"{stat['price_regular']}/{stat['price_esh_discounted']}, {stat['threshold_kWh']}", "exact", "PASS")

    # Weights, units (B4) ----------------------------------------------------------------------------------------
    units, att = eh.class_eulp_weights(wdf, H)
    Wc = eh.class_weight_matrix(wdf, H)
    Hc = {c: sum(H[a] for a in ARLP_NAMES_ALL if eh.CLASS_MAP[a] == c) for c in eh.CLASS_ORDER}
    sumW = units.groupby("class")["W_kHH"].sum()
    dev_b4 = max(abs(sumW[c] / Hc[c] - 1) for c in eh.CLASS_ORDER)
    counts = units.groupby("class").size().to_dict()
    in_cls = units.groupby("eulp_id")["class"].apply(set)
    two = in_cls[in_cls.apply(len) == 2]
    two_esh_non = int(sum(1 for s in two if len(s & eh.ESH_CLASSES) == 1))
    both_esh = int(sum(1 for s in in_cls if {"ESH-1", "ESH-2"} <= s))
    b4 = (dev_b4 <= TOL["B4"] and counts == EXPECTED_UNITS and len(units) == 3330 and len(two) == 185
          and two_esh_non == 185 and both_esh == 0 and Hc == EXPECTED_H and int((in_cls.apply(len) > 2).sum()) == 0)
    g.check("G-B", "B4", "sum_j W_cj = H_c; units 536/721/1,750/146/177 = 3,330; 185 EULPs in an ESH and a "
            "non-ESH class; none in ESH-1 and ESH-2", f"max rel {dev_b4:.2e}; {counts}; two-class {len(two)}",
            TOL["B4"], b4)
    g.check("G-B", "file 03 rows", "attribution rows with phi > 0", len(att), "6,337", len(att) == 6337)

    # B1 (rebuild of the 19 ARLPs) -------------------------------------------------------------------------------
    rebuilt = eh.rebuild_arlp(phi, L)
    arlp_mat = arlp[ARLP_NAMES_ALL].to_numpy()
    b1 = {a: float(np.abs(rebuilt[:, i] - arlp_mat[:, i]).max() / arlp_mat[:, i].max())
          for i, a in enumerate(ARLP_NAMES_ALL)}
    g.check("G-B", "B1", "sum_j phi_sj l_j(t) rebuilds the 19 ARLP columns of I-04 (max |diff| / max)",
            f"{max(b1.values()):.3e} ({max(b1, key=b1.get)})", TOL["B1"], max(b1.values()) <= TOL["B1"])

    # Monthly zone energies ------------------------------------------------------------------------------------------
    E = eh.monthly_zone_energy(L, cal)                                    # (3145, 12, 3)
    E_arlp = eh.monthly_zone_energy(arlp_mat, cal4)                       # (19, 12, 3)
    eidx = {int(j): i for i, j in enumerate(ids)}
    Hs = np.array([H[a] for a in ARLP_NAMES_ALL])
    # B2: class season-zone energies, EULP route vs ARLP route
    Wmat = Wc.loc[ids, list(eh.CLASS_ORDER)].to_numpy()                  # (3145, 5)
    b2 = 0.0
    for ci, c in enumerate(eh.CLASS_ORDER):
        members = [i for i, a in enumerate(ARLP_NAMES_ALL) if eh.CLASS_MAP[a] == c]
        for si in (0, 1):
            msk = bd.SEASON_IDX == si
            e_eulp = (Wmat[:, ci][:, None] * E[:, msk, :].sum(axis=1)).sum(axis=0)
            e_arlp = (Hs[members][:, None] * E_arlp[members][:, msk, :].sum(axis=1)).sum(axis=0)
            b2 = max(b2, float(np.abs(e_eulp / e_arlp - 1).max()))
    g.check("G-B", "B2", "class season-zone energies: EULP route = ARLP route (relative)", f"{b2:.3e}",
            TOL["B2"], b2 <= TOL["B2"])
    # B3: alpha from the EULP-built class representative days vs I-07
    alpha_e = eh.recompute_alpha(eh.class_repday_from_eulps(L, Wc.loc[ids], ts))
    b3 = []
    for r in p3df[p3df.base_tag.isin(["flat", "lo"])].itertuples():
        b3.append((abs(alpha_e[(r.label, r.season)][eh.ZONES.index(r.period)] - r.alpha), r.label, r.season, r.period))
    b3max = max(b3)
    g.check("G-B", "B3", "alpha recomputed from the EULP-built class representative days = I-07 alpha",
            f"{b3max[0]:.3e} ({b3max[1]} {b3max[2]} {b3max[3]})", TOL["B3"], b3max[0] <= TOL["B3"])

    # A4: I-07 alpha (4 dp) within 0.00055 of the I-09 alpha (3 dp) -------------------------------------------------
    t4c = t4[t4.Class.isin(eh.CLASS_ORDER)]
    a4 = 0.0
    for r in t4c.itertuples():
        s = "winter" if r.Season == "w" else "summer"
        for col, z in (("a_off", "off"), ("a_sho", "shoulder"), ("a_peak", "peak")):
            a7 = float(p3df[(p3df.label == r.Class) & (p3df.season == s) & (p3df.period == z)].alpha.iloc[0])
            a4 = max(a4, abs(a7 - getattr(r, col)))
    g.check("G-A", "A4", "the 30 alpha of I-07 (4 dp) within 0.00055 of I-09 (3 dp)",
            f"{a4:.6f} over {3 * len(t4c)}", TOL["A4"], a4 <= TOL["A4"] and len(t4c) == 10)

    # Case A statuses (B8) and the zonal shares (C2d) -----------------------------------------------------------
    esh_arlps = [a for a in ARLP_NAMES_ALL if eh.CLASS_MAP[a] in eh.ESH_CLASSES]
    ai = {a: ARLP_NAMES_ALL.index(a) for a in ARLP_NAMES_ALL}
    em_i04 = {a: E_arlp[ai[a]].sum(axis=1) for a in esh_arlps}           # kWh per household per month
    em_eulp = {a: (phi[:, ai[a]][:, None] * E.sum(axis=2)).sum(axis=0) for a in esh_arlps}
    thr = stat["threshold_kWh"]
    status = {a: np.where(bd.WINTER_MASK, (em_i04[a] > thr).astype(float), 1.0) for a in esh_arlps}
    st_eulp = {a: np.where(bd.WINTER_MASK, (em_eulp[a] > thr).astype(float), 1.0) for a in esh_arlps}
    above = {a: sorted(int(mm) for mm in np.array(eh.MONTHS)[bd.WINTER_MASK & (em_i04[a] > thr)]) for a in esh_arlps}
    i11k = i11.set_index(["ARLP", "month"])
    kwh_dev = 0.0
    st_i11 = True
    for a in esh_arlps:
        for mm in sorted(WINTER_MONTHS):
            r = i11k.loc[(a, mm)]
            kwh_dev = max(kwh_dev, abs(em_i04[a][mm - 1] / r.kWh_per_household - 1),
                          *(abs(E_arlp[ai[a], mm - 1, eh.ZONES.index(z)] / getattr(r, col) - 1)
                            for z, col in (("peak", "kWh_peak"), ("shoulder", "kWh_shoulder"), ("off", "kWh_off_peak"))))
            st_i11 &= bool(r.above_threshold) == bool(em_i04[a][mm - 1] > thr)
    eulp_dev = max(float(np.abs(em_eulp[a][bd.WINTER_MASK] / em_i04[a][bd.WINTER_MASK] - 1).max()) for a in esh_arlps)
    min_dist = min(float(np.abs(em_i04[a][bd.WINTER_MASK] - thr).min()) for a in esh_arlps)
    b8 = (all((status[a] == st_eulp[a]).all() for a in esh_arlps) and above == {"ARLP_01": [], "ARLP_02": [],
          "ARLP_03": [1, 2, 3, 12]} and st_i11 and kwh_dev <= TOL["B8_kwh_i04"] and eulp_dev <= TOL["B8_kwh_eulp"])
    g.check("G-B", "B8", "case-A statuses: I-04 route = EULP route = I-11; only ARLP_03 above the threshold (Dec-Mar); "
            "I-04 kWh = I-11", f"above {above}; kWh rel {kwh_dev:.1e}; EULP route rel {eulp_dev:.1e}; nearest "
            f"{min_dist:.0f} kWh", f"{TOL['B8_kwh_i04']} / {TOL['B8_kwh_eulp']}", b8)
    c2d = 0.0
    i12k = i12.set_index(["class", "zone"])
    zname = {"off": "off-peak", "shoulder": "shoulder", "peak": "peak"}
    for c in ("ESH-1", "ESH-2"):
        members = [a for a in esh_arlps if eh.CLASS_MAP[a] == c]
        for zi, z in enumerate(eh.ZONES):
            e_all = sum(H[a] * E_arlp[ai[a], bd.WINTER_MASK, zi].sum() for a in members)
            e_hi = sum(H[a] * (E_arlp[ai[a], :, zi] * (bd.WINTER_MASK & (em_i04[a] > thr))).sum() for a in members)
            w_z = e_hi / e_all
            r = i12k.loc[(c, zname[z])]
            c2d = max(c2d, abs(w_z - r.w_share_at_3_60), abs(2.20 + 1.40 * w_z - r.effective_base_UAH_per_kWh))
    g.check("G-C", "C2d", "zonal shares w_z and effective winter bases recomputed from I-04 with H_s = I-12",
            f"{c2d:.2e}", TOL["C2d"], c2d <= TOL["C2d"])

    # Units: energies, weights and bills per case ----------------------------------------------------------------
    uix = units["eulp_id"].map(eidx).to_numpy()
    Eu = E[uix]
    ucls = units["class"].to_numpy()
    uW = units["W_kHH"].to_numpy()
    is_esh = np.isin(ucls, list(eh.ESH_CLASSES))
    att_esh = att[att["class"].isin(eh.ESH_CLASSES) & (att.W_s_kHH > 0)]
    arlp_theta = np.zeros((len(units), 12))
    shares_by_unit = {}
    ukey = {(c, int(j)): i for i, (c, j) in enumerate(zip(ucls, units["eulp_id"]))}
    for r in att_esh.itertuples():
        i = ukey[(r[1], r.eulp_id)]
        arlp_theta[i] += r.share_of_unit * status[r.arlp]
        shares_by_unit.setdefault(i, {})[r.arlp] = r.share_of_unit
    theta = {"T": bd.base_regime("T", Eu.sum(axis=2), is_esh, threshold=thr),
             "A": bd.base_regime("A", Eu.sum(axis=2), is_esh, arlp_theta=np.where(bd.WINTER_MASK, arlp_theta, 0.0)),
             "L": bd.base_regime("L", Eu.sum(axis=2), is_esh)}
    res = {k: bd.unit_bills(Eu, ucls, theta[k], rates) for k in bd.CASES}
    Vh = {c: float((uW[ucls == c] * res["L"]["annual_kWh"][ucls == c]).sum() / 1e3) for c in eh.CLASS_ORDER}
    Qc = t5post.set_index("class")["cons_GWh"].to_dict()
    q_unit = np.array([Qc[c] / Vh[c] for c in ucls])

    # C9: spot check of three named units, month by month with plain loops --------------------------------------
    bmask = ucls == "B"
    i_b = int(np.flatnonzero(bmask)[np.argmax(uW[bmask])])
    e2 = np.flatnonzero((ucls == "ESH-2") & (theta["T"][:, bd.WINTER_MASK].sum(axis=1) > 0))
    i_e2 = int(e2[np.argmax(uW[e2])]) if len(e2) else None
    e2ids = set(units.loc[ucls == "ESH-2", "eulp_id"])
    cm = np.flatnonzero((ucls == "C") & units["eulp_id"].isin(e2ids).to_numpy())
    i_c = int(cm[np.argmax(uW[cm])])
    spot, c9 = [], 0.0
    pyts = list(np.array(pd.Series(ts).dt.to_pydatetime()))
    st_int = {a: [int(v) for v in status[a]] for a in esh_arlps}
    for i in [i_b, i_e2, i_c]:
        if i is None:
            continue
        col = L[:, uix[i]]
        for k in bd.CASES:
            sb = scalar_bills(col, pyts, ucls[i], rates, st_int, k, shares_by_unit.get(i, {}))
            vb = np.array([res[k]["B0"][i], res[k]["BP3"][i], res[k]["BE2"][i], res[k]["BE3"][i]])
            c9 = max(c9, float(np.abs(sb - vb).max()))
        spot.append(f"{ucls[i]}:{units.eulp_id.iloc[i]}")
    g.check("G-C", "C9", "three named units billed month by month with plain loops (B largest; ESH-2 with "
            "months above 2,000 kWh; C also in ESH-2), cases T/A/L, F/P3/E2/E3", f"{c9:.2e} ₴ ({', '.join(spot)})",
            TOL["C9"], c9 <= TOL["C9"] and len(spot) == 3)

    # C1: the identity of the summed bills -----------------------------------------------------------------------
    c1 = 0.0
    for k in bd.CASES:
        cl = bd.class_level_bills(Eu, uW, ucls, theta[k], rates)
        for c in eh.CLASS_ORDER:
            m_ = ucls == c
            for x, key in (("F", "B0"), ("P3", "BP3"), ("E2", "BE2"), ("E3", "BE3")):
                c1 = max(c1, abs(float((uW[m_] * res[k][key][m_]).sum() / 1e3) / cl[c][x] - 1))
    g.check("G-C", "C1", "sum_j W_cj B^x_cj = the bill of the summed class energies (per class, case, tariff)",
            f"{c1:.2e}", TOL["C1"], c1 <= TOL["C1"])

    # C5: contributions; the deviation identity in case L ---------------------------------------------------------
    c5 = max(float(np.abs(res[k]["contrib"].sum(axis=(1, 2)) - res[k]["dbill_pct"]).max()) for k in bd.CASES)
    agg_sigma = {}
    dev_id = 0.0
    for k in bd.CASES:
        for c in eh.CLASS_ORDER:
            m_ = ucls == c
            num = (uW[m_][:, None, None] * res[k]["sigma"][m_] * res[k]["B0"][m_][:, None, None]).sum(axis=0)
            agg_sigma[(k, c)] = num / float((uW[m_] * res[k]["B0"][m_]).sum())
    for c in eh.CLASS_ORDER:
        m_ = ucls == c
        rL = res["L"]
        agg = 100 * ((uW[m_] * rL["BP3"][m_]).sum() / (uW[m_] * rL["B0"][m_]).sum() - 1)
        i0 = np.flatnonzero(m_)[0]
        ahat = np.stack([(rL["rate"][i0][bd.SEASON_IDX == si] / rL["b"][i0][bd.SEASON_IDX == si][:, None])[0]
                         for si in (0, 1)])                                # (2, 3): r / b per season and zone
        rhs = 100 * ((ahat - 1)[None] * (rL["sigma"][m_] - agg_sigma[("L", c)][None])).sum(axis=(1, 2))
        dev_id = max(dev_id, float(np.abs(rL["dbill_pct"][m_] - agg - rhs).max()))
    g.check("G-C", "C5", "contributions sum to the change (all cases); the deviation identity (case L)",
            f"{c5:.2e} / {dev_id:.2e} pp", TOL["C5"], c5 <= TOL["C5"] and dev_id <= TOL["C5"])

    # C6 ----------------------------------------------------------------------------------------------------------
    ne = ~is_esh
    bitwise = all(np.array_equal(res["T"][key][ne], res[k][key][ne]) for k in ("A", "L")
                  for key in ("B0", "BP3", "BE2", "BE3", "dbill_pct"))
    c6 = 0.0
    for k in ("T", "A"):
        lo_only = is_esh & (theta[k][:, bd.WINTER_MASK].sum(axis=1) == 0)
        c6 = max(c6, float(np.abs(res[k]["dbill_pct"][lo_only] - res["L"]["dbill_pct"][lo_only]).max(initial=0.0)))
    g.check("G-C", "C6", "non-ESH rows identical in T, A, L (bitwise); ESH units with no month at ₴3.60 have "
            "the change of case L", f"bitwise {bitwise}; {c6:.2e} pp", TOL["C6"], bitwise and c6 <= TOL["C6"])

    # Reproduction of the bill-impact tables, ARLP and EULP routes (C2c, C2a, C2b, C4) ------------------------------------------
    arlp_cls = [eh.CLASS_MAP[a] for a in ARLP_NAMES_ALL]
    arlp_esh = np.isin(arlp_cls, list(eh.ESH_CLASSES))
    th_arlp = {"A": np.array([status[a] if a in status else np.zeros(12) for a in ARLP_NAMES_ALL]),
               "L": bd.base_regime("L", E_arlp.sum(axis=2), arlp_esh)}
    arlp_res = {k: bd.unit_bills(E_arlp, arlp_cls, th_arlp[k], rates) for k in ("A", "L")}
    key_of = {"F": "B0", "P3": "BP3", "E2": "BE2", "E3": "BE3"}

    def class_bills(route, k):
        bills, vol = {}, {}
        for c in eh.CLASS_ORDER:
            if route == "ARLP":
                m_ = np.array([cc == c for cc in arlp_cls])
                bills[c] = {x: float((Hs[m_] * arlp_res[k][key_of[x]][m_]).sum() / 1e3) for x in bd.TARIFFS}
                vol[c] = float((Hs[m_] * arlp_res[k]["annual_kWh"][m_]).sum() / 1e3)
            else:
                m_ = ucls == c
                bills[c] = {x: float((uW[m_] * res[k][key_of[x]][m_]).sum() / 1e3) for x in bd.TARIFFS}
                vol[c] = float((uW[m_] * res[k]["annual_kWh"][m_]).sum() / 1e3)
        return bills, vol

    o7 = []
    for ref_name, k, ref in (("bill_impacts", "A", t5post), ("bill_impacts_no_threshold", "L", t5pre)):
        for route in ("ARLP", "EULP"):
            bills, vol = class_bills(route, k)
            cl = (bd.class_level_bills(Eu, uW, ucls, theta[k], rates) if route == "EULP"
                  else bd.class_level_bills(E_arlp, Hs, arlp_cls, th_arlp[k], rates))
            tp, tr = ((TOL["C2_arlp_pp"], TOL["C2_arlp_rel"]) if route == "ARLP"
                      else (TOL["C2_eulp_pp"], TOL["C2_eulp_rel"]))
            rows = bd.bill_table_reconciliation(bills, vol, ref, reference=ref_name, case=k, route=route, tol_pp=tp,
                                            tol_rel=tr)
            o7 += rows
            cid_cls = {"bill_impacts": "C2c", "bill_impacts_no_threshold": "C2a" if route == "ARLP" else "C2b"}[ref_name]
            cls_rows = [r for r in rows if r["class"] != "Total"]
            tot_rows = [r for r in rows if r["class"] == "Total"]
            for cid, rr, what in ((cid_cls, cls_rows, "per class"), ("C4", tot_rows, "Total from monetary bills")):
                worst = max(rr, key=lambda r: abs(r["diff_pp"]))
                g.add("G-C", f"{cid} {route}", f"{ref_name} case {k}, {what}: 5 comparisons, average tariffs, bills",
                      f"max |diff| {max(abs(r['diff_pp']) for r in rr):.2e} pp; avg rel "
                      f"{max(r['diff_avg_rel'] for r in rr):.1e}; bill rel {max(r['diff_bill_rel'] for r in rr):.1e}",
                      f"{tp} pp / rel {tr}", "PASS" if all(r["status"] == "PASS" for r in rr) else "FAIL",
                      f"worst {worst['class']} {worst['comparison']}")
            for r in rows:
                if r["class"] != "Total":
                    c_, (fr, to) = r["class"], r["comparison"].split("->")
                    r["sum_units_from_MUAH"], r["sum_units_to_MUAH"] = bills[c_][fr], bills[c_][to]
                    r["class_level_from_MUAH"], r["class_level_to_MUAH"] = cl[c_][fr], cl[c_][to]
                    r["identity_residual_rel"] = max(abs(bills[c_][fr] / cl[c_][fr] - 1),
                                                     abs(bills[c_][to] / cl[c_][to] - 1))
    # case T (no reference): the class aggregates and the threshold residual against case L
    billsT, volT = class_bills("EULP", "T")
    billsL, _ = class_bills("EULP", "L")
    for c in (*eh.CLASS_ORDER, "Total"):
        for fr, to in bd.COMPARISONS:
            if c == "Total":
                QT = sum(Qc[k_] for k_ in eh.CLASS_ORDER)
                bf = sum(Qc[k_] / volT[k_] * billsT[k_][fr] for k_ in eh.CLASS_ORDER)
                bt = sum(Qc[k_] / volT[k_] * billsT[k_][to] for k_ in eh.CLASS_ORDER)
                bfL = sum(Qc[k_] / volT[k_] * billsL[k_][fr] for k_ in eh.CLASS_ORDER)
                btL = sum(Qc[k_] / volT[k_] * billsL[k_][to] for k_ in eh.CLASS_ORDER)
                vh = sum(volT.values())
            else:
                QT, vh = Qc[c], volT[c]
                bf, bt = QT / vh * billsT[c][fr], QT / vh * billsT[c][to]
                bfL, btL = QT / vh * billsL[c][fr], QT / vh * billsL[c][to]
            agg, aggL = bd.change_pct(bf, bt), bd.change_pct(bfL, btL)
            o7.append({"reference": "none", "case": "T", "class": c, "comparison": f"{fr}->{to}", "route": "EULP",
                       "V_H_GWh": vh, "Q_GWh": QT, "q_factor": QT / vh, "avg_from_units": bf / QT, "avg_to_units": bt / QT,
                       "bill_from_table_basis_MUAH": bf, "bill_to_table_basis_MUAH": bt, "aggregate_pct": agg,
                       "status": "REPORT", "note": f"threshold residual vs case L {agg - aggL:+.9f} pp"})
    o7 = pd.DataFrame(o7)

    # Statistics (file 04), concentration (file 09) -----------------------------------------------------------------------
    W_impl = np.zeros(len(units))
    for i, (c, j) in enumerate(zip(ucls, units.eulp_id)):
        members = [a for a in ARLP_NAMES_ALL if eh.CLASS_MAP[a] == c]
        W_impl[i] = float(wabs.loc[j, members].sum()) / 1e3
    eid = units["eulp_id"].to_numpy()
    rows4, pan_mask = [], {p: (ucls == p) if p != "All" else np.ones(len(units), bool) for p in bd.PANELS}
    for k in bd.CASES:
        r_ = res[k]
        for p in bd.PANELS:
            m_ = pan_mask[p]
            for weighting, basis, w in (("weighted", "H_normalized", uW), ("unweighted", "none", np.ones(len(uW))),
                                        ("weighted", "arlp_implied", W_impl)):
                money = basis == "H_normalized"
                st = bd.panel_stats(r_["dbill_pct"][m_], w[m_], ucls[m_], eid[m_],
                                    B0=r_["B0"][m_] if money else None, BP3=r_["BP3"][m_] if money else None,
                                    q=q_unit[m_] if money else None, dUAH=r_["dbill_UAH"][m_], kwh=r_["annual_kWh"][m_],
                                    be=r_["breakeven_share"][m_])
                if not money:
                    st["aggregate_pct"] = 100.0 * ((w[m_] * r_["BP3"][m_]).sum() / (w[m_] * r_["B0"][m_]).sum() - 1)
                p_ext = bd.weighted_quantile(r_["dbill_pct"][m_], w[m_], (0.005, 0.995),
                                             (np.array([eh.CLASS_ORDER.index(c) for c in ucls[m_]]), eid[m_]))
                rows4.append({"case": k, "panel": p, "weighting": weighting, "weight_basis": basis,
                              "quantile_method": bd.QUANTILE_METHOD, **st, "p00_5_pct": float(p_ext[0]),
                              "p99_5_pct": float(p_ext[1])})
    o4 = pd.DataFrame(rows4)
    # C11 (reimbursement identity), C10, C3, C8 ----------------------------------------------------------
    w4 = o4[o4.weight_basis == "H_normalized"]
    c11 = 0.0
    for r in w4.itertuples():
        for sfx in ("", "_table_basis"):
            pp, nn, net = (getattr(r, f"positive_part_MUAH{sfx}"), getattr(r, f"negative_part_MUAH{sfx}"),
                           getattr(r, f"net_change_MUAH{sfx}"))
            c11 = max(c11, abs(pp - nn - net) / max(abs(pp), abs(nn)))
    for k in bd.CASES:
        for c in eh.CLASS_ORDER:
            m_ = ucls == c
            r = w4[(w4.case == k) & (w4.panel == c)].iloc[0]
            for x in ("positive_part_MUAH", "negative_part_MUAH", "net_change_MUAH", "bill_F_MUAH"):
                c11 = max(c11, abs(getattr(r, f"{x}_table_basis") / (Qc[c] / Vh[c] * getattr(r, x)) - 1))
    g.check("G-C", "C11", "R+ - R- = the net passive change (both bases); table basis = q_c x H basis",
            f"{c11:.2e}", f"{TOL['C11']} (rel. to max(R+, R-))", c11 <= TOL["C11"])
    for k in bd.CASES:
        d = w4[w4.case == k].set_index("panel")
        g.add("G-C", f"C10 {k}", "mean_hh_pct against aggregate_pct per panel (reported)",
              "; ".join(f"{p} {d.at[p, 'mean_hh_pct']:+.4f}/{d.at[p, 'aggregate_pct']:+.6f}" for p in bd.PANELS),
              "—", "REPORT")
    for k in ("L", "A"):
        d = w4[w4.case == k].set_index("panel")
        vals = {c: float(d.at[c, "aggregate_pct"]) for c in eh.CLASS_ORDER}
        g.add("G-C", f"C3 {k}", f"F->P3 class aggregate_pct, case {k} (rounding residual; ESH-2 case A zonal bases)",
              "; ".join(f"{c} {vals[c]:+.6f}" for c in eh.CLASS_ORDER), "reported", "REPORT")
    agg = {k: w4[w4.case == k].set_index("panel")["aggregate_pct"] for k in bd.CASES}
    g.add("G-C", "C8", "threshold residual: aggregate_pct(T) and (A) minus (L), ESH classes and All",
          "; ".join(f"{p}: T{agg['T'][p] - agg['L'][p]:+.6f} A{agg['A'][p] - agg['L'][p]:+.6f}"
                    for p in ("ESH-1", "ESH-2", "All")), "reported", "REPORT")

    o9 = []
    wsum_e = units.assign(W=uW).groupby("eulp_id")["W"].sum()
    for p in bd.PANELS:
        r = w4[(w4.case == "T") & (w4.panel == p)].iloc[0]
        row = {"panel": p, "n_units": r.n_units, "total_weight_kHH": r.total_weight_kHH, "kish_neff": r.kish_neff,
               "largest_unit_weight_pct": r.largest_unit_weight_pct, "largest_unit_eulp_id": r.largest_unit_eulp_id,
               "largest_unit_class": r.largest_unit_class, "top5_weight_pct": r.top5_weight_pct,
               "n_units_90w": r.n_units_90w, "n_units_99w": r.n_units_99w}
        if p == "All":
            row.update({"largest_eulp_pooled_pct": 100 * float(wsum_e.max()) / float(uW.sum()),
                        "largest_eulp_pooled_id": int(wsum_e.idxmax())})
        o9.append(row)
    o9 = pd.DataFrame(o9)

    # file 02 -----------------------------------------------------------------------------------------------------------
    o2 = []
    for k in bd.CASES:
        r_ = res[k]
        E_ann = Eu.sum(axis=(1, 2))
        shz = np.stack([Eu[:, bd.SEASON_IDX == si, :].sum(axis=1) for si in (0, 1)], axis=1) / E_ann[:, None, None]
        sh_pk = shz[:, :, 2].sum(axis=1)
        df = units[["class", "stratum", "eulp_id", "W_kHH"]].copy().assign(case=k)
        df = df[["case", "class", "stratum", "eulp_id", "W_kHH"]]
        df["w_share_class"] = uW / np.array([Hc[c] for c in ucls])
        df["n_arlps"] = units["n_arlps"].to_numpy()
        df["member_arlps"] = units["member_arlps"].to_numpy()
        df["annual_kWh"] = E_ann
        for si, s in enumerate(eh.SEASONS):
            for zi, z in enumerate(eh.ZONES):
                df[f"sh_{s}_{z}"] = shz[:, si, zi]
        df["sh_annual_peak"] = sh_pk
        cls_pk = {c: float((uW[ucls == c] * Eu[ucls == c][:, :, 2].sum(axis=1)).sum()
                           / (uW[ucls == c] * E_ann[ucls == c]).sum()) for c in eh.CLASS_ORDER}
        df["dev_annual_peak_pp"] = 100 * (sh_pk - np.array([cls_pk[c] for c in ucls]))
        for si, s in enumerate(eh.SEASONS):
            for zi, z in enumerate(eh.ZONES):
                df[f"dev_{s}_{z}"] = r_["sigma"][:, si, zi] - np.array([agg_sigma[(k, c)][si, zi] for c in ucls])
        df["winter_months_above_2000"] = np.where(is_esh, theta[k][:, bd.WINTER_MASK].sum(axis=1), np.nan)
        df["bill_F_UAH"], df["bill_P3_UAH"] = r_["B0"], r_["BP3"]
        df["dbill_UAH"], df["dbill_pct"] = r_["dbill_UAH"], r_["dbill_pct"]
        for si, s in enumerate(eh.SEASONS):
            for zi, z in enumerate(eh.ZONES):
                df[f"contrib_{s}_{z}_pct"] = r_["contrib"][:, si, zi]
        df["breakeven_share"] = r_["breakeven_share"]
        df["breakeven_attainable"] = pd.array([None if not d > 0 else bool(b <= 1.0)
                                               for d, b in zip(r_["dbill_pct"], r_["breakeven_share"])], dtype="boolean")
        df["bill_E2_UAH"], df["bill_E3_UAH"] = r_["BE2"], r_["BE3"]
        df["d_F_E3_pct"] = bd.change_pct(r_["B0"], r_["BE3"])
        df["d_E2_P3_pct"] = bd.change_pct(r_["BE2"], r_["BP3"])
        df["d_E2_E3_pct"] = bd.change_pct(r_["BE2"], r_["BE3"])
        df["d_E3_P3_pct"] = bd.change_pct(r_["BE3"], r_["BP3"])
        in99 = np.zeros(len(units), dtype=bool)
        for c in eh.CLASS_ORDER:
            m_ = ucls == c
            in99[m_] = bd.top99_mask(uW[m_], ucls[m_], eid[m_])
        df["in_top99w"] = in99
        df["W_impl_kHH"] = W_impl
        o2.append(df)
    o2 = pd.concat(o2, ignore_index=True)

    # file 06 (per ARLP) --------------------------------------------------------------------------------------------
    o6 = []
    for k in bd.CASES:
        for a in ARLP_NAMES_ALL:
            c = eh.CLASS_MAP[a]
            sub = att[(att.arlp == a)]
            js = sub.eulp_id.to_numpy()
            ph = sub.phi.to_numpy()
            ii = np.array([ukey[(c, int(j))] for j in js])
            if k == "A" and c in eh.ESH_CLASSES:          # the dwellings billed with their ARLP's own statuses
                rr = bd.unit_bills(Eu[ii], [c] * len(ii), np.tile(status[a], (len(ii), 1)), rates)
            else:
                rr = {key: res[k][key][ii] for key in ("B0", "BP3", "dbill_pct")}
            mean_ch = 100 * ((ph * rr["BP3"]).sum() / (ph * rr["B0"]).sum() - 1)
            q3 = bd.weighted_quantile(rr["dbill_pct"], ph, (0.05, 0.50, 0.95), (js,))
            th_a = (status[a] if c in eh.ESH_CLASSES else np.zeros(12)) if k == "A" else None
            if k == "T" and c in eh.ESH_CLASSES:
                th_a = np.where(bd.WINTER_MASK, (E_arlp[ai[a]].sum(axis=1) > thr).astype(float), 1.0)
            if th_a is None:
                th_a = th_arlp["L"][ai[a]]
            prof = bd.unit_bills(E_arlp[ai[a]][None], [c], th_a[None], rates)["dbill_pct"][0]
            o6.append({"case": k, "arlp": a, "class": c, "H_kHH": H[a], "n_eulps": len(js), "h_zero_flag": H[a] == 0,
                       "mean_bill_change_pct": mean_ch,
                       "mean_definition": ("mean of the dwellings' bills (threshold per dwelling; not the bill of the "
                                           "average profile)") if (k == "T" and c in eh.ESH_CLASSES)
                       else "aggregate of the dwellings' bills = the ARLP average profile's change (linearity)",
                       "avg_profile_change_pct": prof, "p05_pct": q3[0], "p50_pct": q3[1], "p95_pct": q3[2],
                       "share_ge0_pct": 100 * float(ph[rr["dbill_pct"] >= 0].sum()) / float(ph.sum()),
                       "kish_neff": eh.kish_neff(ph), "largest_phi": float(ph.max()),
                       "largest_phi_eulp_id": int(js[np.argmax(ph)])})
    o6 = pd.DataFrame(o6)
    lin = o6[~((o6.case == "T") & o6["class"].isin(eh.ESH_CLASSES))]
    g.add("G-C", "file 06 linearity", "per-ARLP mean change of the dwellings = the ARLP average profile's change "
          "(cases L and A, non-ESH in T; reported)", f"{(lin.mean_bill_change_pct - lin.avg_profile_change_pct).abs().max():.2e} pp",
          "reported (EULP rebuild, B1)", "REPORT")

    # file 08 (threshold diagnostics; C7) ------------------------------------------------------------------------
    o8 = []

    def add8(c, k, qn, key, v):
        o8.append({"class": c, "case": k, "quantity": qn, "key": key, "value": v})

    m_alt = (pd.Series(ts) - pd.Timedelta(hours=1)).dt.month.to_numpy()
    cal_alt = cal.assign(month=m_alt)
    E_alt = eh.monthly_zone_energy(L, cal_alt)
    E4_alt = eh.monthly_zone_energy(arlp_mat, eh.calendar(arlp["timestamp"]).assign(
        month=(arlp["timestamp"] - pd.Timedelta(hours=1)).dt.month.to_numpy()))
    for c in ("ESH-1", "ESH-2"):
        m_ = ucls == c
        Hcl = Hc[c]
        members = [a for a in esh_arlps if eh.CLASS_MAP[a] == c]
        kT = theta["T"][m_][:, bd.WINTER_MASK].sum(axis=1)
        for kk in range(8):
            add8(c, "T", "n_units_k_winter_months_above", kk, int((kT == kk).sum()))
            add8(c, "T", "weight_share_k_winter_months_above_pct", kk, 100 * float(uW[m_][kT == kk].sum()) / Hcl)
            ka = {a: int(status[a][bd.WINTER_MASK].sum()) for a in members}
            add8(c, "A", "weight_share_k_winter_months_above_pct", kk, 100 * sum(H[a] for a in members if ka[a] == kk) / Hcl)
            add8(c, "A", "n_units_k_winter_months_above", kk,
                 int(sum(1 for i in np.flatnonzero(m_) if any(ka[a] == kk for a in shares_by_unit.get(i, {})))))
        add8(c, "T", "weight_share_ge1_winter_month_above_pct", "", 100 * float(uW[m_][kT >= 1].sum()) / Hcl)
        add8(c, "A", "weight_share_ge1_winter_month_above_pct", "",
             100 * sum(H[a] for a in members if status[a][bd.WINTER_MASK].sum() >= 1) / Hcl)
        for mm in (10, 11, 12, 1, 2, 3, 4):
            add8(c, "T", "weight_share_above_by_month_pct", mm, 100 * float(uW[m_][theta["T"][m_][:, mm - 1] > 0].sum()) / Hcl)
            add8(c, "A", "weight_share_above_by_month_pct", mm, 100 * sum(H[a] for a in members if status[a][mm - 1] > 0) / Hcl)
        for k in ("T", "A"):
            d = res[k]["dbill_pct"][m_] - res["L"]["dbill_pct"][m_]
            ch = d != 0
            add8(c, k, "dB_minus_L_weighted_median_pp", "",
                 float(bd.weighted_quantile(d, uW[m_], (0.5,), (eid[m_],))[0]))
            add8(c, k, "dB_minus_L_max_abs_pp", "", float(np.abs(d).max()))
            add8(c, k, "dB_minus_L_n_units_changed", "", int(ch.sum()))
            add8(c, k, "dB_minus_L_weight_share_changed_pct", "", 100 * float(uW[m_][ch].sum()) / Hcl)
        for k in bd.CASES:
            a_ = float(agg[k][c])
            add8(c, k, "aggregate_pct", "", a_)
            if k != "L":
                add8(c, k, "aggregate_minus_L_pp", "", a_ - float(agg["L"][c]))
            den = float((uW[m_] * res[k]["B0"][m_]).sum())
            b_m = res[k]["b"][m_]
            for mm in eh.MONTHS:
                num = float((uW[m_] * ((res[k]["rate"][m_][:, mm - 1, :] - b_m[:, mm - 1][:, None])
                                       * Eu[m_][:, mm - 1, :]).sum(axis=1)).sum())
                add8(c, k, "aggregate_contribution_by_month_pp", mm, 100 * num / den)
        # C7: the hour 23:00-24:00 of a month's last day moved to that month
        Ealt_u = E_alt[uix[m_]].sum(axis=2)
        thT_alt = bd.base_regime("T", Ealt_u, np.ones(int(m_.sum()), bool), threshold=thr)
        chg = (thT_alt != theta["T"][m_]) & bd.WINTER_MASK[None, :]
        add8(c, "T", "month_boundary_units_regime_changed", "", int(chg.any(axis=1).sum()))
        add8(c, "T", "month_boundary_unit_months_changed", "", int(chg.sum()))
        n_am = sum(int(((E4_alt[ai[a]].sum(axis=1) > thr) != (em_i04[a] > thr))[bd.WINTER_MASK].sum()) for a in members)
        add8(c, "A", "month_boundary_arlp_months_changed", "", n_am)
    o8 = pd.DataFrame(o8)
    mb = o8[o8.quantity.str.startswith("month_boundary")]
    g.add("G-C", "C7", "month-boundary diagnostic: stamp 00:00 of the 1st moved to the previous month",
          "; ".join(f"{r['class']} {r.quantity.replace('month_boundary_', '')} {int(r.value)}" for _, r in mb.iterrows()),
          "—", "REPORT")

    # Bins (file 05) and G-D ------------------------------------------------------------------------------------------------
    t_w = o4[(o4.case == "T") & (o4.weight_basis == "H_normalized")].set_index("panel")
    rule = bd.bin_rule(t_w.at["All", "p00_5_pct"], t_w.at["All", "p99_5_pct"],
                       [t_w.at[p, "p00_5_pct"] for p in bd.PANELS], [t_w.at[p, "p99_5_pct"] for p in bd.PANELS])
    o5, d1, d7 = [], 0.0, 0.0
    for k in bd.CASES:
        for p in bd.PANELS:
            m_ = pan_mask[p]
            for weighting, w in (("weighted", uW), ("unweighted", np.ones(len(uW)))):
                hdf = bd.histogram(res[k]["dbill_pct"][m_], w[m_], rule)
                st = o4[(o4.case == k) & (o4.panel == p) & (o4.weighting == weighting)
                        & (o4.weight_basis != "arlp_implied")].iloc[0]
                d1 = max(d1, abs(hdf.share_pct.sum() - 100))
                d7 = max(d7, abs(hdf[hdf.right_pct <= 0].share_pct.sum() - st.share_lt0_pct),
                         abs(hdf[hdf.left_pct >= 0].share_pct.sum() - st.share_ge0_pct))
                hdf.insert(0, "weighting", weighting)
                hdf.insert(0, "panel", p)
                hdf.insert(0, "case", k)
                for key in ("width_pp", "rule_R_pp", "range_L_pct", "range_U_pct"):
                    hdf[key] = rule[key]
                if k == "T" and weighting == "weighted":
                    ya = bd.yaxis_rule(hdf.share_pct.max())
                    hdf["tallest_bar_pct"], hdf["tick_ceiling_pct"] = ya["tallest_bar_pct"], ya["tick_ceiling_pct"]
                    hdf["y_top_pct"], hdf["bar_label"] = ya["y_top_pct"], ya["bar_label"]
                o5.append(hdf)
    o5 = pd.concat(o5, ignore_index=True)
    g.check("G-D", "D1", "bin shares (edge bins included) sum to 100 % per panel and row", f"{d1:.1e}", TOL["D1"],
            d1 <= TOL["D1"])
    g.check("G-D", "D7", "bins left of 0 sum to S_<0; from 0 up to S_>=0", f"{d7:.1e}", TOL["D7"], d7 <= TOL["D7"])
    tw = w4[w4.case == "T"].set_index("panel")
    d2 = abs(tw.at["All", "total_weight_kHH"] / sum(EXPECTED_H.values()) - 1)
    d2b = abs(tw.at["All", "total_weight_kHH"] / sum(tw.at[c, "total_weight_kHH"] for c in eh.CLASS_ORDER) - 1)
    g.check("G-D", "D2", "pooled weight 17,259 kHH = the sum of the class panels; n = 3,330",
            f"{tw.at['All', 'total_weight_kHH']:.6f}; n {tw.at['All', 'n_units']}", TOL["D2"],
            max(d2, d2b) <= TOL["D2"] and tw.at["All", "n_units"] == 3330)
    d3 = float((o4.share_ge0_pct + o4.share_lt0_pct - 100).abs().max())
    d3d = max(abs(float(bd.format_share(a, False).strip("<>")) + float(bd.format_share(b, False).strip("<>")) - 100)
              for a, b in zip(o4.share_ge0_pct, o4.share_lt0_pct))
    g.check("G-D", "D3", "S_>=0 + S_<0 = 100; displayed one-decimal values sum to 100 +/- 0.1",
            f"{d3:.1e}; display {d3d:.2f}", f"{TOL['D3']} / {TOL['D3_display']}",
            d3 <= TOL["D3"] and d3d <= TOL["D3_display"] + 1e-9)
    d4 = True
    qcols = [f"p{round(q * 100):02d}_pct" for q in bd.QUANTILES]
    for r in o4.itertuples():
        vals = [getattr(r, c) for c in qcols]
        m_ = pan_mask[r.panel]
        d4 &= all(a <= b for a, b in itertools.pairwise(vals)) and all(np.any(res[r.case]["dbill_pct"][m_] == v) for v in vals)
    g.check("G-D", "D4", "percentiles monotone in q; each is the change of an existing unit", "", "exact", d4)
    d5 = True
    for r in o4[o4.weighting == "weighted"].itertuples():
        m_ = pan_mask[r.panel]
        w = (uW if r.weight_basis == "H_normalized" else W_impl)[m_]
        d5 &= r.largest_unit_weight_pct == 100 * w.max() / w.sum()
    d5 &= o9.set_index("panel").at["All", "largest_eulp_pooled_pct"] == 100 * float(wsum_e.max()) / float(uW.sum())
    g.check("G-D", "D5", "largest weights = max W / sum W (both definitions)", "", "exact", bool(d5))
    rule2 = bd.bin_rule(t_w.at["All", "p00_5_pct"], t_w.at["All", "p99_5_pct"],
                        list(t_w.loc[list(bd.PANELS), "p00_5_pct"]), list(t_w.loc[list(bd.PANELS), "p99_5_pct"]))
    g.check("G-D", "D8", "the bin rule result (w, L, U) = the bin rule applied to file 04",
            f"w {rule['width_pp']}, L {rule['range_L_pct']}, U {rule['range_U_pct']}, R {rule['rule_R_pp']:.3f}",
            "exact", rule2 == rule)
    d9 = float((o4.share_ge0_abs_lt_0p05_pct + o4.share_lt0_abs_lt_0p05_pct - o4.share_abs_lt_0p05_pct).abs().max())
    d9b = bool(((o4.share_ge0_abs_lt_0p05_pct <= o4.share_ge0_pct + 1e-12)
                & (o4.share_lt0_abs_lt_0p05_pct <= o4.share_lt0_pct + 1e-12)).all())
    fmt_ok = all((bd.format_pct(v, False) == "0.0") == (abs(v) < bd.NEAR_ZERO_BAND)
                 for k in bd.CASES for v in res[k]["dbill_pct"])
    g.check("G-D", "D9", "near-zero parts sum to the band share; each part <= its side; '0.0' exactly in the band",
            f"{d9:.1e}; parts <= sides {d9b}; formatter {fmt_ok}", TOL["D9"], d9 <= TOL["D9"] and d9b and fmt_ok)

    # Write ------------------------------------------------------------------------------------------------------------
    failed_pre = list(g.failed)
    if failed_pre:
        target = (Path(out) / "_unchecked") if out is not None else vault / UNCHECKED_REL
    target.mkdir(parents=True, exist_ok=True)
    print(f"writing to {target}")
    write_csv(eh.monthly_zone_table(ids, E), target / "01_eulp_monthly_zone_kwh.csv")
    write_csv(o2, target / "02_class_eulp_units.csv")
    abs_w = {(r.arlp, r.eulp_id): float(wabs.at[r.eulp_id, r.arlp]) for r in att.itertuples()}
    write_csv(att.assign(abs_weight_HH=[abs_w[(a, j)] for a, j in zip(att.arlp, att.eulp_id)]),
              target / "03_class_eulp_arlp_attribution.csv")
    write_csv(o4, target / "04_distribution_summary.csv")
    write_csv(o5, target / "05_histogram_bins.csv")
    write_csv(o6, target / "06_by_arlp_summary.csv")
    write_csv(o7, target / "07_aggregation_identity.csv")
    write_csv(o8, target / "08_threshold_diagnostics.csv")
    write_csv(o9, target / "09_concentration_diagnostics.csv")
    (target / "README.md").write_text(FOLDER_README, encoding="utf-8", newline="\n")

    fig_info = {}
    if figures and not failed_pre:
        import make_bill_change_histograms as s4
        import make_bill_change_vs_peak_share as s5
        try:
            fig_info = s4.main(target)
            g.check("G-D", "D6", "every data string printed in the histogram figure = display in file 10 (rendered text; the note strings are not drawn)",
                    f"{fig_info['n_strings']} strings", "exact", fig_info["d6_ok"])
            g.check("G-D", "D10", "y-axes by the y-axis rule; ticks <= 100; the tallest bar and its label inside "
                    "the axes; no text overlap; text >= 7 pt", fig_info["d10_detail"], "exact / geometric",
                    fig_info["d10_ok"])
        except AssertionError as ex:
            g.add("G-D", "D6/D10", "histogram figure assertions", str(ex)[:300], "exact", "FAIL")
        try:
            s1 = s5.main(target)
            g.check("G-D", "peak-share figure", "plotted points = file 02 (case T, five class panels)", f"{s1['n_points']} points",
                    "exact", s1["ok"])
        except AssertionError as ex:
            g.add("G-D", "peak-share figure", "peak-share figure assertions", str(ex)[:300], "exact", "FAIL")
    else:
        g.add("G-D", "D6/D10", "histogram figure not drawn in this run", "--no-figures" if not figures else "gate failed",
              "—", "REPORT")
    if g.failed and not failed_pre and out is None:
        dest = vault / UNCHECKED_REL
        dest.mkdir(parents=True, exist_ok=True)
        for f in target.iterdir():
            shutil.move(str(f), dest / f.name)
        target.rmdir()
        target = dest
    elif g.failed and not failed_pre:
        dest = Path(out) / "_unchecked"
        dest.mkdir(parents=True, exist_ok=True)
        for f in list(target.iterdir()):
            if f.is_file():
                shutil.move(str(f), dest / f.name)
        target = dest
    pd.DataFrame(g.rows).to_csv(target / "11_qa_gates.csv", index=False, lineterminator="\n", encoding="utf-8")


    print(f"done in {time.time() - t0:.0f} s; {len(g.rows)} checks; FAIL: {g.failed or 'none'}; -> {target}")
    return 1 if g.failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output folder (default: outputs/ARLP_Clustering/eulp_bill_distribution)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.out, figures=not a.no_figures))
