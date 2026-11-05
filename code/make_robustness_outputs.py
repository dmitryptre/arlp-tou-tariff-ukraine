"""Publication outputs of the robustness analysis: table_multiplier_deviations, table_multiplier_ranges,
table_clustering_stability, fig_coefficient_deviations, fig_ab_regimes, fig_reassignment (if its condition holds),
fig_ls1_timing, fig_ls2_one_class and reported_numbers.json, all from the CSV files of one run folder. No number is
typed by hand.

python -X utf8 make_robustness_outputs.py RUN_DIR                      # write the outputs into RUN_DIR
python -X utf8 make_robustness_outputs.py RUN_DIR --check-figures      # the figures against the run's CSV files
                                                                       # and the JSON; writes figure_checks.csv
Class keys are the class names of the proposed-tariff table. Printed scenario labels come from PUB (EW1 ... RC2-MN);
the JSON keys, the registry and the run's CSV files keep the internal scenario IDs (A1-U20 ... D2-MN). Monte Carlo
rule: a P5, P95 or P95|dev| whose split-half difference exceeds 0.001 is printed at 2 dp and marked "‡".
fig_ab_regimes and fig_reassignment (post-run visualisation): drawn from the LS3 (C3-RND) replicates of 03 only; the
experienced-multiplier decomposition is fixed classes + reassignment = total, nothing is re-run.
fig_ls1_timing and fig_ls2_one_class (post-run visualisation): the deterministic LS1 runs and the five single-class
LS2 runs (peak -> off-peak; the registry has no single-class run in the other direction), Delta alpha from 04,
classes fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CL = ("A", "B", "C", "ESH-1", "ESH-2")
SK = ("w", "s")
SEASON = {"w": "winter", "s": "summer"}
ZK = ("off", "sh", "peak")
ZNAME = {"off": "α_off", "sh": "α_sh", "peak": "α_peak", "ratio": "α_peak/α_off"}
STOCH = ("A1-U20", "B1-RND20", "C3-RND")
# Publication labels: the single source of truth for every printed scenario label, so that the
# experiment groups cannot be confused with the tariff classes A, B, C. The internal scenario IDs stay in the registry,
# the run's CSV files, the JSON keys and the provenance records.
PUB = {"A1-U20": "EW1", "A2-MIX25": "EW2", "B1-RND20": "HC1", "B2-STAT": "HC3",
       **{f"B3-CLS-{c}-{f}": "HC2" for c in CL for f in ("0.8", "1.2")},
       "C1-TIME-M1": "LS1", "C1-TIME-P1": "LS1", "C2-ZONE-CM-P5": "LS2", "C2-ZONE-CM-M5": "LS2",
       **{f"C2-ZONE-DM-{c}": "LS2" for c in CL}, "C3-RND": "LS3", "D2-NC": "RC2-NC", "D2-MN": "RC2-MN"}
RC1 = "RC1"                     # the re-clustering of every EW, HC and LS run on its own inputs (class mode "rec")
PUB_GROUP = {"EW": "EULP composition", "HC": "household counts and consumption", "LS": "load shape",
             "RC": "re-clustering"}
S4_FLAGS = f"{PUB['C3-RND']} + {RC1} runs: class flagged / membership changed, %"
PUB_MEANING = {"EW1": "EULP weights, random perturbation", "EW2": "EULP weights, concentration stress",
               "HC1": "household counts, random perturbation", "HC2": "household counts, class-level scaling",
               "HC3": "household consumption, statistical targets", "LS1": "load shape, timing shifts",
               "LS2": "load shape, TOU-zone redistribution", "LS3": "load shape, combined random perturbations",
               "RC1": "re-clustering on the perturbed inputs of every EW, HC and LS run",
               "RC2-NC": "alternative clustering: without city–rural consolidation",
               "RC2-MN": "alternative clustering: daily-mean instead of daily-peak normalisation"}
MINUS = "−"
MC_TOL = 0.001
ARLPS = [f"ARLP_{i:02d}" for i in range(1, 20)]
P0 = {**{a: "A" for a in ("ARLP_04", "ARLP_05", "ARLP_06", "ARLP_12", "ARLP_13", "ARLP_14")},
      **{a: "B" for a in ("ARLP_07", "ARLP_15")},
      **{a: "C" for a in ("ARLP_08", "ARLP_09", "ARLP_10", "ARLP_11", "ARLP_16", "ARLP_17", "ARLP_18", "ARLP_19")},
      "ARLP_01": "ESH-1", "ARLP_02": "ESH-2", "ARLP_03": "ESH-2"}
# clustering units of the baseline method: the 8 city-rural families and the 3 ESH ARLPs, in the class order of the
# proposed-tariff table
FAMILIES = [f"FAM_{i:02d}_{i + 8:02d}" for i in range(4, 12)]
UNIT_MEMBERS = {**{f: [f"ARLP_{f[4:6]}", f"ARLP_{f[7:9]}"] for f in FAMILIES},
                **{a: [a] for a in ("ARLP_01", "ARLP_02", "ARLP_03")}}
UNITS = [u for c in CL for u in UNIT_MEMBERS if P0[UNIT_MEMBERS[u][0]] == c]
# C3 regimes of the non-ESH partition after re-clustering (fig_ab_regimes): exhaustive and disjoint
REGIMES = ("unchanged", "05_13_to_B", "other")
FS41 = {"title": 8.5, "label": 8.0, "tick": 7.5, "legend": 7.5, "note": 7.0}    # type sizes of fig_coefficient_deviations


def q(x, p):
    return float(np.quantile(np.asarray(x, dtype=float), p))


def sgn(x, nd=3):
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return ("+" if x > 0 else MINUS) + s


def num(x, nd=3, mc_fail=False):
    return f"{x:.2f}‡" if mc_fail and nd > 2 else f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{x:.{nd}f}"


class Numbers:
    """Every printed number with its raw value and source; becomes reported_numbers.json."""

    def __init__(self):
        self.d = {}

    def put(self, key, text, raw=None, source=""):
        self.d[key] = {"text": text, "raw": None if raw is None else float(raw), "source": source}
        return text


# ── loading ───────────────────────────────────────────────────────────────────────────────────────────────────
def load(run):
    run = Path(run)
    return {"cs": pd.read_csv(run / "04_coefficient_summary.csv"), "c5": pd.read_csv(run / "05_clustering_summary.csv"),
            "n6": pd.read_csv(run / "06_neutrality_summary.csv"), "d7": pd.read_csv(run / "07_diagnostics.csv"),
            "a2": pd.read_csv(run / "07a_A2_concentration.csv"), "det": pd.read_csv(run / "02_deterministic_runs.csv"),
            "reps": pd.read_csv(run / "03_replicates.csv.gz"), "gates": pd.read_csv(run / "08_gates.csv"),
            "run": run}


def table4(vault):
    t = pd.read_csv(vault / "outputs/ARLP_Clustering/notebook_only/layer13_peed_dr/table4_proposed_tariff.csv")
    t = t[t.Class.isin(CL)].set_index(["Class", "Season"])
    return {(c, sk, z): float(t.loc[(c, sk), col]) for c in CL for sk in SK
            for z, col in (("off", "a_off"), ("sh", "a_sho"), ("peak", "a_peak"))} | \
           {(c, sk, "ratio"): float(t.loc[(c, sk), "ratio"]) for c in CL for sk in SK}


def cs_row(cs, sid, mode, c, sk, z):
    r = cs[(cs.scenario_id == sid) & (cs["mode"] == mode) & (cs["class"] == c) & (cs.season == sk) &
           (cs.quantity == z)]
    return r.iloc[0] if len(r) else None


def d_values(reps):
    """The RC1 column of table_multiplier_deviations: per scenario, the P95 of D_r(c, tau, z) and its split-half
    difference."""
    out = {}
    for sid in STOCH:
        g = reps[reps.scenario_id == sid]
        h = len(g) // 2
        for c in CL:
            for sk in SK:
                for z in ZK:
                    v = g[f"D_{c}_{sk}_{z}"].to_numpy(float)
                    out[(sid, c, sk, z)] = (q(v, .95), abs(q(v[:h], .95) - q(v[h:], .95)))
    return out


# ── table_multiplier_deviations ───────────────────────────────────────────────────────────────────────────────
def make_table9(D, t4, N):
    """Columns carry the publication labels (PUB); the JSON keys keep the internal short codes of the run."""
    cs, rows = D["cs"], []
    dv = d_values(D["reps"])
    for c in CL:
        for sk in SK:
            for z in ZK:
                r = {"Class": c, "Season": SEASON[sk], "Multiplier": ZNAME[z],
                     "Baseline": N.put(f"T9|{c}|{sk}|{z}|baseline", f"{t4[(c, sk, z)]:.3f}", t4[(c, sk, z)], "table4_proposed_tariff.csv")}
                for sid, col in (("A1-U20", "A1"), ("B1-RND20", "B1"), ("C3-RND", "C3")):
                    x = cs_row(cs, sid, "fix", c, sk, z)
                    fail = x.split_p95_absdev > MC_TOL
                    r[PUB[sid]] = N.put(f"T9|{c}|{sk}|{z}|{col}", num(x.p95_absdev, 3, fail), x.p95_absdev,
                                        f"04 {sid} fix p95_absdev")
                for sid, col in (("A2-MIX25", "A2"), ("B2-STAT", "B2")):
                    x = cs_row(cs, sid, "fix", c, sk, z)
                    r[PUB[sid]] = N.put(f"T9|{c}|{sk}|{z}|{col}", sgn(x.dev), x.dev, f"04 {sid} fix dev")
                best = max(STOCH, key=lambda s: dv[(s, c, sk, z)][0])
                val, split = dv[(best, c, sk, z)]
                txt = num(val, 3, split > MC_TOL) + (f" ({PUB[best]})" if val > 0 else "")
                r[RC1] = N.put(f"T9|{c}|{sk}|{z}|D", txt, val, f"03 D_{c}_{sk}_{z} P95, max over A1, B1, C3")
                for s in STOCH:
                    N.put(f"D|{s}|{c}|{sk}|{z}", num(dv[(s, c, sk, z)][0], 3, dv[(s, c, sk, z)][1] > MC_TOL),
                          dv[(s, c, sk, z)][0], f"03 D_{c}_{sk}_{z} P95 {s}")
                rows.append(r)
    t9 = pd.DataFrame(rows)[["Class", "Season", "Multiplier", "Baseline", PUB["A1-U20"], PUB["A2-MIX25"],
                             PUB["B1-RND20"], PUB["B2-STAT"], PUB["C3-RND"], RC1]]
    d7, c5 = D["d7"].set_index("scenario_id"), D["c5"].set_index("scenario_id")
    summ = []
    for lab, fn in (("Runs with an active shoulder clip, %", lambda s: d7.at[s, "clip_any_pct"]),
                    ("Runs with the α_peak order A > B > ESH-1 > C > ESH-2 in both seasons, %",
                     lambda s: _both_orders(D, s)),
                    ("Re-clustered runs with an unchanged partition, non-ESH / ESH, %",
                     lambda s: (c5.at[s, "same_nonesh_pct"], c5.at[s, "same_esh_pct"])),
                    ("Re-clustered runs with a class flagged substantially changed or ambiguous, %",
                     lambda s: _flagged(D, s))):
        r = {"Class": lab, "Season": "", "Multiplier": "", "Baseline": ""}
        for sid, col in (("A1-U20", "A1"), ("A2-MIX25", "A2"), ("B1-RND20", "B1"), ("B2-STAT", "B2"),
                         ("C3-RND", "C3")):
            v = fn(sid)
            if isinstance(v, tuple):
                r[PUB[sid]] = N.put(f"T9sum|{lab[:12]}|{col}", f"{pct(v[0])} / {pct(v[1])}", None, "05")
            else:
                r[PUB[sid]] = N.put(f"T9sum|{lab[:12]}|{col}", pct(v), v, "07/05")
        r[RC1] = ""
        summ.append(r)
    return pd.concat([t9, pd.DataFrame(summ)[list(t9.columns)]], ignore_index=True)


def _both_orders(D, sid):
    g = D["reps"][D["reps"].scenario_id == sid] if sid in STOCH else D["det"][D["det"].scenario_id == sid]
    return 100 * float((g.fix_order_full_w.astype(bool) & g.fix_order_full_s.astype(bool)).mean())


def _flagged(D, sid):
    g = D["reps"][D["reps"].scenario_id == sid] if sid in STOCH else D["det"][D["det"].scenario_id == sid]
    f = np.zeros(len(g), dtype=bool)
    for c in CL:
        f |= g[f"flag_sub_{c}"].astype(bool).to_numpy() | g[f"flag_amb_{c}"].astype(bool).to_numpy()
    return 100 * float(f.mean())


# ── table_multiplier_ranges ───────────────────────────────────────────────────────────────────────────────────
def make_tableS4(D, t4, N):
    cs, rows = D["cs"], []
    reps = D["reps"][D["reps"].scenario_id == "C3-RND"]
    for c in CL:
        flagged = 100 * float((reps[f"flag_sub_{c}"].astype(bool) | reps[f"flag_amb_{c}"].astype(bool)).mean())
        moved = 100 * float(reps[f"flag_mem_{c}"].astype(bool).mean())
        for sk in SK:
            for z in ZK + ("ratio",):
                nd = 2 if z == "ratio" else 3
                r = {"Class": c, "Season": SEASON[sk], "Quantity": ZNAME[z],
                     "Baseline": N.put(f"S4|{c}|{sk}|{z}|baseline", f"{t4[(c, sk, z)]:.{nd}f}", t4[(c, sk, z)], "table4_proposed_tariff.csv")}
                for sid, col, mode in (("A1-U20", "A1", "fix"), ("B1-RND20", "B1", "fix"), ("C3-RND", "C3", "fix"),
                                       ("C3-RND", "C3 re-clustered", "rec")):
                    head = PUB[sid] + (f" + {RC1}" if mode == "rec" else "")       # printed; col stays the key
                    x = cs_row(cs, sid, mode, c, sk, z)
                    # alpha (3 dp): a split-half difference > 0.001 prints at 2 dp with "‡"; ratios are printed at
                    # 2 dp (as in the proposed-tariff table) and marked "‡" when the split-half difference exceeds 0.01
                    f5, f95 = x.split_p05 > MC_TOL, x.split_p95 > MC_TOL
                    if z == "ratio":
                        f5, f95 = x.split_p05 > 0.01, x.split_p95 > 0.01

                    def fq(v, fail):
                        return num(v, 3, fail) if nd == 3 else f"{v:.2f}" + ("‡" if fail else "")
                    txt = f"{num(x['median'], nd)} [{fq(x.p05, f5)}, {fq(x.p95, f95)}]"
                    r[head] = N.put(f"S4|{c}|{sk}|{z}|{col}", txt, None, f"04 {sid} {mode}")
                    N.put(f"S4|{c}|{sk}|{z}|{col}|p05", num(x.p05, nd), x.p05, "04")
                    N.put(f"S4|{c}|{sk}|{z}|{col}|p95", num(x.p95, nd), x.p95, "04")
                for sid, col in (("A2-MIX25", "A2"), ("B2-STAT", "B2")):
                    x = cs_row(cs, sid, "fix", c, sk, z)
                    r[PUB[sid]] = N.put(f"S4|{c}|{sk}|{z}|{col}", f"{x.value:.{nd}f}", x.value, f"04 {sid}")
                r[S4_FLAGS] = f"{pct(flagged)} / {pct(moved)}" if (sk, z) == ("w", "off") else ""
                rows.append(r)
        N.put(f"S4|{c}|C3flagged", pct(flagged), flagged, "03")
        N.put(f"S4|{c}|C3moved", pct(moved), moved, "03")
    cols = ["Class", "Season", "Quantity", "Baseline", PUB["A1-U20"], PUB["A2-MIX25"], PUB["B1-RND20"],
            PUB["B2-STAT"], PUB["C3-RND"], f"{PUB['C3-RND']} + {RC1}", S4_FLAGS]
    return pd.DataFrame(rows)[cols]


# ── table_clustering_stability ────────────────────────────────────────────────────────────────────────────────
SCEN_S5 = [(sid, f"{PUB[sid]} {desc}") for sid, desc in (
    ("A1-U20", "EULP composition, ×U(0.8, 1.2)"), ("A2-MIX25", "concentration stress, 25 %"),
    ("B1-RND20", "household counts, ×U(0.8, 1.2)"), ("B2-STAT", "statistical energy targets"),
    ("C1-TIME-M1", "timing −1 h"), ("C1-TIME-P1", "timing +1 h"),
    ("C2-ZONE-CM-P5", "peak → off 5 %, all classes"), ("C2-ZONE-CM-M5", "off → peak 5 %, all classes"),
    ("C3-RND", "combined shape"), ("D2-NC", "no city–rural consolidation"), ("D2-MN", "daily-mean normalisation"))]


def make_tableS5(D, N):
    c5 = D["c5"].set_index("scenario_id")
    rows_a = []
    for sid, lab in SCEN_S5:
        x = c5.loc[sid]
        moved = sorted(((-x[f"moved_{a}_pct"], a) for a in ARLPS))            # by share, then by ARLP number
        moved = [(-v, a) for v, a in moved if v < 0]
        if sid in STOCH:
            moved = moved[:3]                                                  # deterministic runs: every moved ARLP
        flags = []
        for c in CL:
            parts = [f"{nm} {x[f'flag_{k}_{c}_pct']:.1f}" for k, nm in (("mem", "m"), ("sub", "s"), ("amb", "a"))
                     if x[f"flag_{k}_{c}_pct"] > 0]
            if parts:
                flags.append(f"{c}: " + ", ".join(parts))
        r = {"Scenario": lab,
             "Unchanged partition, non-ESH / ESH, %": N.put(f"S5a|{sid}|same", f"{pct(x.same_nonesh_pct)} / "
                                                                              f"{pct(x.same_esh_pct)}", None, "05"),
             "ARI (19 ARLPs), median [P5]": N.put(f"S5a|{sid}|ari", f"{x.ari_arlp_median:.2f} [{x.ari_arlp_p05:.2f}]",
                                                  None, "05"),
             "Households reassigned, P95 [max], %": N.put(f"S5a|{sid}|reassign", f"{pct(x.reassign_p95_pct)} "
                                                          f"[{pct(x.reassign_max_pct)}]", None, "05"),
             "ARLPs changing class most often, % of runs": "; ".join(f"{a.replace('ARLP_', '')} {v:.1f}"
                                                                     for v, a in moved) or "none",
             "Class flags, % of runs (m membership changed, s substantially changed, a ambiguous)":
                 "; ".join(flags) or "none",
             "Largest change of an experienced multiplier": N.put(f"S5a|{sid}|dre", f"{x.dre_absmax:.3f}",
                                                                   x.dre_absmax, "05")}
        N.put(f"S5a|{sid}|reassign_p95", pct(x.reassign_p95_pct), x.reassign_p95_pct, "05")
        N.put(f"S5a|{sid}|reassign_max", pct(x.reassign_max_pct), x.reassign_max_pct, "05")
        N.put(f"S5a|{sid}|same_nonesh", pct(x.same_nonesh_pct), x.same_nonesh_pct, "05")
        N.put(f"S5a|{sid}|same_esh", pct(x.same_esh_pct), x.same_esh_pct, "05")
        N.put(f"S5a|{sid}|ari_median", f"{x.ari_arlp_median:.2f}", x.ari_arlp_median, "05")
        N.put(f"S5a|{sid}|ari_p05", f"{x.ari_arlp_p05:.2f}", x.ari_arlp_p05, "05")
        for v, a in moved:
            N.put(f"S5a|{sid}|moved|{a}", f"{v:.1f}", v, "05")
        rows_a.append(r)
    # (c) the scenario-specific P95 behind the RC1 column of table_multiplier_deviations, largest over seasons and
    # zones, per class
    dv = d_values(D["reps"])
    rows_c = []
    for c in CL:
        r = {"Class": c}
        for s in STOCH:
            v = max(dv[(s, c, sk, z)][0] for sk in SK for z in ZK)
            r[PUB[s]] = N.put(f"S5c|{c}|{s}", f"{v:.3f}", v, "03")
        rows_c.append(r)
    # (b) frozen-coefficient neutrality, case T
    n6 = D["n6"]
    rows_b = []
    for sid, lab in [x for x in SCEN_S5 if not x[0].startswith("D2")] + [
            ("B1-RND20|energy", f"{PUB['B1-RND20']}, read as per-household energy (ESH threshold)")]:
        key = sid.split("|")[0]
        r = {"Scenario": lab}
        for c in CL:
            x = n6[(n6.scenario_id == key) & (n6["class"] == c) & (n6.case == "T")].iloc[0]
            col = "frz_dev_energy" if sid.endswith("energy") else "frz_dev"
            if key in STOCH:
                txt = f"{sgn(x[col + '_p05'])} to {sgn(x[col + '_p95'])}"
                N.put(f"S5b|{sid}|{c}|p05", sgn(x[col + "_p05"]), x[col + "_p05"], "06")
                N.put(f"S5b|{sid}|{c}|p95", sgn(x[col + "_p95"]), x[col + "_p95"], "06")
            else:
                txt = sgn(x[col + "_median"])
                N.put(f"S5b|{sid}|{c}|value", txt, x[col + "_median"], "06")
            r[c] = txt
        rows_b.append(r)
    # recalibrated: rounding and threshold parts, clip
    rr = {"Scenario": "Recalibrated coefficients (all EW, HC and LS runs): largest |rounding part| / "
                      "|ESH-threshold part| / clipping part, pp"}
    tt = n6[n6.case == "T"]
    for c in CL:
        x = tt[tt["class"] == c]
        rr[c] = N.put(f"S5b|recal|{c}", f"{x.rnd_part_absmax.max():.3f} / {x.thr_part_absmax.max():.3f} / "
                                         f"{x.clip_part_absmax.max():.3f}", None, "06")
        N.put(f"S5b|recal|{c}|rnd", f"{x.rnd_part_absmax.max():.3f}", x.rnd_part_absmax.max(), "06")
        N.put(f"S5b|recal|{c}|thr", f"{x.thr_part_absmax.max():.3f}", x.thr_part_absmax.max(), "06")
    rows_b.append(rr)
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b), pd.DataFrame(rows_c)


# ── headline numbers used in the text ────────────────────────────────────────────────────────────────────────
def headline(D, t4, N):
    cs = D["cs"]
    fix = cs[(cs["mode"] == "fix") & cs.quantity.isin(ZK)]
    for sid in STOCH:
        s = fix[fix.scenario_id == sid]
        i = s.p95_absdev.idxmax()
        N.put(f"H|{sid}|max_p95absdev", f"{s.p95_absdev.max():.3f}", s.p95_absdev.max(), "04")
        N.put(f"H|{sid}|max_p95absdev|where", f"{s.loc[i, 'class']} {SEASON[s.loc[i, 'season']]} "
                                              f"{ZNAME[s.loc[i, 'quantity']]}", None, "04")
        for cg, cls in (("AB", ("A", "B")), ("small", ("C", "ESH-1", "ESH-2"))):
            for zg, zz in (("pkoff", ("peak", "off")), ("sh", ("sh",))):
                ss = s[s["class"].isin(cls) & s.quantity.isin(zz)]
                N.put(f"H|{sid}|{cg}|{zg}|min", f"{ss.p95_absdev.min():.3f}", ss.p95_absdev.min(), "04")
                N.put(f"H|{sid}|{cg}|{zg}|max", f"{ss.p95_absdev.max():.3f}", ss.p95_absdev.max(), "04")
        for c in CL:
            ss = s[s["class"] == c]
            N.put(f"H|{sid}|{c}|max", f"{ss.p95_absdev.max():.3f}", ss.p95_absdev.max(), "04")
        rat = cs[(cs["mode"] == "fix") & (cs.quantity == "ratio") & (cs.scenario_id == sid)]
        for c in CL:
            for sk in SK:
                x = rat[(rat["class"] == c) & (rat.season == sk)].iloc[0]
                N.put(f"H|{sid}|ratio|{c}|{sk}|p05", f"{x.p05:.2f}", x.p05, "04")
                N.put(f"H|{sid}|ratio|{c}|{sk}|p95", f"{x.p95:.2f}", x.p95, "04")
    det = cs[(cs["mode"] == "fix") & cs.dev.notna() & cs.quantity.isin(ZK)]
    for sid in det.scenario_id.unique():
        s = det[det.scenario_id == sid]
        N.put(f"H|{sid}|max_absdev", f"{s.dev.abs().max():.3f}", s.dev.abs().max(), "04")
        for c in CL:
            ss = s[s["class"] == c]
            N.put(f"H|{sid}|{c}|max_absdev", f"{ss.dev.abs().max():.3f}", ss.dev.abs().max(), "04")
            for sk in SK:
                for z in ZK:
                    v = float(ss[(ss.season == sk) & (ss.quantity == z)].dev.iloc[0])
                    N.put(f"H|{sid}|{c}|{sk}|{z}", sgn(v), v, "04")
    # A2 concentration
    a2 = D["a2"]
    for c in CL:
        for stg in ("base", "mixed", "tilted"):
            x = a2[(a2.level == "class") & (a2.unit == c) & (a2.stage == stg)].iloc[0]
            N.put(f"A2conc|{c}|{stg}|max", f"{x['max']:.3f}", x["max"], "07a")
            N.put(f"A2conc|{c}|{stg}|kish", f"{x.kish:.1f}", x.kish, "07a")
    d7 = D["d7"].set_index("scenario_id")
    for sid in d7.index:
        N.put(f"D7|{sid}|clip", pct(d7.at[sid, "clip_any_pct"]), d7.at[sid, "clip_any_pct"], "07")
        for k in ("order_full_w_pct", "order_full_s_pct", "esh1_gt_c_s_pct", "order_nonesh_pct", "K_02_pct",
                  "K_03_pct", "K_04_pct", "flat_ratio_max_p95", "mass_ratio_min_p05", "mass_ratio_max_p95",
                  "flat_ratio_min_p05"):
            v = d7.at[sid, k]
            N.put(f"D7|{sid}|{k}", f"{v:.2f}" if "ratio" in k else pct(v), v, "07")
    N.put("H|n_runs", f"{len(D['reps']) + len(D['det']):,}", len(D["reps"]) + len(D["det"]), "02/03")
    N.put("H|n_reps", f"{len(D['reps']):,}", len(D["reps"]), "03")
    N.put("H|clip_total", str(int(D["reps"].clip_any.astype(bool).sum() + D["det"].clip_any.astype(bool).sum())),
          D["reps"].clip_any.astype(bool).sum() + D["det"].clip_any.astype(bool).sum(), "02/03")
    g = D["gates"]
    mc = g[g.check_id == "G13"].iloc[0]
    N.put("H|mc", str(mc.value), None, "08")
    n6 = D["n6"][D["n6"].case == "T"]
    for sid in ("A1-U20", "B1-RND20", "C3-RND"):
        x = n6[n6.scenario_id == sid]
        N.put(f"H|neu|{sid}|absmax", f"{x.frz_dev_absmax.max():.3f}", x.frz_dev_absmax.max(), "06")
    N.put("H|neu|rnd_max", f"{n6.rnd_part_absmax.max():.4f}", n6.rnd_part_absmax.max(), "06")
    N.put("H|neu|rnd_max3", f"{n6.rnd_part_absmax.max():.3f}", n6.rnd_part_absmax.max(), "06")
    x = n6[n6["class"] == "ESH-2"]
    N.put("H|neu|thr_esh2_max", f"{x.thr_part_absmax.max():.2f}", x.thr_part_absmax.max(), "06")
    b3 = n6[n6.scenario_id.str.startswith("B3") & n6["class"].isin(["A", "B", "C"])]
    N.put("H|neu|B3_nonesh_absmax", f"{b3.frz_dev_absmax.max():.1e}", b3.frz_dev_absmax.max(), "06")
    for sid in ("A1-U20", "B1-RND20", "C3-RND", "A2-MIX25", "B2-STAT", "C1-TIME-M1", "C1-TIME-P1", "C2-ZONE-CM-P5",
                "C2-ZONE-CM-M5"):
        x = n6[n6.scenario_id == sid]
        N.put(f"H|neu|{sid}|absmax2", f"{x.frz_dev_absmax.max():.2f}", x.frz_dev_absmax.max(), "06")
    # common- against differential-mode zone transfers, peak and off-peak multipliers
    dev = cs[(cs["mode"] == "fix") & cs.dev.notna() & cs.quantity.isin(("peak", "off"))]
    cm = dev[dev.scenario_id.isin(["C2-ZONE-CM-P5", "C2-ZONE-CM-M5"])].dev.abs().max()
    N.put("H|C2|CM|pkoff_max", f"{cm:.4f}", cm, "04")
    dm = max(dev[(dev.scenario_id == f"C2-ZONE-DM-{c}") & (dev["class"] == c)].dev.abs().max() for c in CL)
    N.put("H|C2|DM|own_pkoff_max", f"{dm:.4f}", dm, "04")
    for c in CL:
        v = dev[(dev.scenario_id == f"C2-ZONE-DM-{c}") & (dev["class"] == c)].dev.abs().max()
        N.put(f"H|C2|DM|{c}|own_pkoff_max", f"{v:.4f}", v, "04")
    sh = cs[(cs["mode"] == "fix") & cs.dev.notna() & (cs.quantity == "sh") &
            cs.scenario_id.isin(["C2-ZONE-CM-P5", "C2-ZONE-CM-M5"])].dev.abs().max()
    N.put("H|C2|CM|sh_max", f"{sh:.3f}", sh, "04")
    # pre-tilt diagnostic: the multipliers before the annual-energy tilt
    pre = cs[(cs["mode"] == "pre") & cs.quantity.isin(ZK)]
    if len(pre):
        a1 = pre[pre.scenario_id == "A1-U20"]
        N.put("H|pre|A1|max_p95absdev", f"{a1.p95_absdev.max():.3f}", a1.p95_absdev.max(), "04 pre")
        for c in CL:
            v = a1[a1["class"] == c].p95_absdev.max()
            N.put(f"H|pre|A1|{c}|max", f"{v:.3f}", v, "04 pre")
        a2 = pre[pre.scenario_id == "A2-MIX25"]
        for c in CL:
            v = a2[a2["class"] == c].dev.abs().max()
            N.put(f"H|pre|A2|{c}|max_absdev", f"{v:.3f}", v, "04 pre")
    # run counts
    N.put("H|n_det", str(len(D["det"]) - 1), len(D["det"]) - 1, "02 (without the baseline row)")
    import eulp_hierarchy as eh
    Hh = eh.load_household_counts(eh.find_vault())
    fam = Hh["ARLP_05"] + Hh["ARLP_13"]
    N.put("H|FAM_05_13_kHH", f"{fam:,.0f}", fam, "arlp_household_counts.csv")
    v = 100 - _both_orders(D, "B1-RND20")
    N.put("H|B1-RND20|order_fail", pct(v), v, "03")
    reg = pd.read_csv(D["run"] / "00_registry.csv")
    for r in reg[reg.n_replicates > 1].itertuples():
        N.put(f"H|seed|{r.scenario_id}", str(int(r.seed)), int(r.seed), "00_registry.csv")
    c5 = D["c5"].set_index("scenario_id")
    for sid in STOCH:
        N.put(f"H|{sid}|changed_nonesh", pct(100 - c5.at[sid, "same_nonesh_pct"]), 100 - c5.at[sid, "same_nonesh_pct"], "05")
        N.put(f"H|{sid}|changed_esh", pct(100 - c5.at[sid, "same_esh_pct"]), 100 - c5.at[sid, "same_esh_pct"], "05")
    rec = D["reps"]
    for sid in STOCH:
        g = rec[rec.scenario_id == sid]
        N.put(f"H|{sid}|both_orders", pct(_both_orders(D, sid)), _both_orders(D, sid), "03")
        N.put(f"H|{sid}|AB_swap", pct(100 * float((~(g.fix_order_nonesh_w.astype(bool) &
                                                    g.fix_order_nonesh_s.astype(bool))).mean())), None, "03")
        N.put(f"H|{sid}|rec_order", pct(100 * float((g.rec_order_full_w.astype(bool) &
                                                     g.rec_order_full_s.astype(bool)).mean())), None, "03")
        for a in ("ARLP_05", "ARLP_02", "ARLP_03"):
            v = 100 * float((g[f"cls_{a}"] != P0[a]).mean())
            N.put(f"H|{sid}|moved|{a}", pct(v, 2), v, "03")
            N.put(f"H|{sid}|moved1|{a}", pct(v, 1), v, "03")
        v = 100 * float((g.cls_ARLP_05 == "B").mean())
        N.put(f"H|{sid}|ARLP_05_to_B", pct(v, 2), v, "03")
        m = g[g.cls_ARLP_05 == "B"]
        if len(m):
            for c in ("A", "B"):
                N.put(f"H|{sid}|05toB|rec_{c}_w_peak", f"{m[f'a_rec_{c}_w_peak'].median():.2f}",
                      m[f"a_rec_{c}_w_peak"].median(), "03 median over the runs with ARLP_05 in B")


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────
def fig16(D, out):
    """fig_coefficient_deviations: EW1, HC1, LS3 as P5-P95 bars with medians; EW2 as a marker; HC2 (B3) as two
    markers per row, the row's own class scaled by 0.8 (open) and 1.2 (filled). The legend names the tests only."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    import _peed_fig_common as fc                                      # house palette and Arial (rcParams)
    cs = D["cs"]
    FS = {"title": 8.5, "label": 8.0, "tick": 7.5, "legend": 7.5}
    col = {"A1-U20": fc.NAVY, "B1-RND20": fc.BLUEGRAY, "C3-RND": fc.MID}
    off = {"A1-U20": -0.24, "B1-RND20": 0.0, "C3-RND": 0.24}
    hc_scale = {"0.8": dict(marker="s", ms=3.4, color=fc.NAVY, mfc="white", mew=0.8),
           "1.2": dict(marker="s", ms=3.4, color=fc.NAVY, mfc=fc.NAVY, mew=0.8)}
    rows = [(c, sk) for c in CL for sk in SK]
    fig = Figure(figsize=(190 / 25.4, 5.3), dpi=400)
    FigureCanvasAgg(fig)
    axes = [fig.add_axes([0.13 + i * 0.29, 0.16, 0.25, 0.77]) for i in range(3)]
    src = []
    for ax, z in zip(axes, ZK):
        lim = 0.0
        for ri, (c, sk) in enumerate(rows):
            if ri % 4 < 2:
                ax.axhspan(ri - 0.5, ri + 0.5, color=fc.LAVENDER, zorder=0, lw=0)
            for sid in STOCH:
                x = cs_row(cs, sid, "fix", c, sk, z)
                y = ri + off[sid]
                ax.plot([x.p05_dev, x.p95_dev], [y, y], color=col[sid], lw=3.2, solid_capstyle="butt", zorder=3)
                ax.plot([x["median"] - x.baseline_eulp_route], [y], marker="o", ms=3.0, color="white",
                        mec=col[sid] if sid != "C3-RND" else fc.NAVY, mew=0.8, zorder=4)
                lim = max(lim, abs(x.p05_dev), abs(x.p95_dev))
                src.append({"panel": z, "class": c, "season": sk, "scenario": sid, "label": PUB[sid],
                            "p05_dev": x.p05_dev,
                            "median_dev": x["median"] - x.baseline_eulp_route, "p95_dev": x.p95_dev})
            x = cs_row(cs, "A2-MIX25", "fix", c, sk, z)
            ax.plot([x.dev], [ri + off["A1-U20"]], marker="D", ms=3.6, color=fc.NAVY, mfc="none", mew=0.8, zorder=5)
            lim = max(lim, abs(x.dev))
            src.append({"panel": z, "class": c, "season": sk, "scenario": "A2-MIX25", "label": PUB["A2-MIX25"],
                        "dev": x.dev})
            for f, sty in hc_scale.items():                            # HC2 (B3): the row's own class scaled by f
                sid = f"B3-CLS-{c}-{f}"
                x = cs_row(cs, sid, "fix", c, sk, z)
                ln, = ax.plot([x.dev], [ri + off["B1-RND20"]], ls="", zorder=5, **sty)
                assert ln.get_xdata()[0] == x.dev
                lim = max(lim, abs(x.dev))
                src.append({"panel": z, "class": c, "season": sk, "scenario": sid, "label": PUB[sid],
                            "scaled_class": c, "factor": float(f), "dev": x.dev})
        lim *= 1.12
        ax.set_xlim(-lim, lim)
        ax.axvline(0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{c}, {SEASON[sk]}" for c, sk in rows] if z == "off" else [])
        ax.tick_params(axis="both", labelsize=FS["tick"], length=2.5, width=0.5, pad=2, colors=fc.NAVY)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color(fc.SLATE)
        ax.grid(axis="x", color=fc.LIGHT, lw=0.4, zorder=1)
        title = {"off": r"Off-peak multiplier $\alpha_{\mathrm{off}}$", "sh": r"Shoulder multiplier $\alpha_{\mathrm{sh}}$",
                 "peak": r"Peak multiplier $\alpha_{\mathrm{peak}}$"}[z]
        ax.set_title(title, fontsize=FS["title"], color=fc.NAVY, pad=4)
        ax.xaxis.set_major_formatter(_signed_formatter())
    axes[1].set_xlabel(r"Deviation from the baseline multiplier, $\Delta\alpha$", fontsize=FS["label"],
                       color=fc.NAVY, labelpad=3)
    # legend: test abbreviations only; the caption explains bars, circles and markers
    handles = [Line2D([], [], color=col[s], lw=3.2) for s in STOCH] + \
              [Line2D([], [], marker="D", ls="", ms=3.6, color=fc.NAVY, mfc="none", mew=0.8),
               tuple(Line2D([], [], ls="", **sty) for sty in hc_scale.values())]
    labels = [PUB[s] for s in STOCH] + [PUB["A2-MIX25"], PUB["B3-CLS-A-0.8"]]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=len(labels), fontsize=FS["legend"],
               frameon=False, bbox_to_anchor=(0.53, 0.0), handlelength=1.8, columnspacing=1.8,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
    png = out / "fig_coefficient_deviations.png"
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    to_tif(png)
    pd.DataFrame(src).to_csv(out / "fig_coefficient_deviations_source.csv", index=False, float_format="%.12g", lineterminator="\n")
    return png


def _signed_formatter():
    from matplotlib.ticker import FuncFormatter

    def f(x, _):
        if abs(x) < 1e-12:
            return "0"
        s = f"{abs(x):.3f}".rstrip("0").rstrip(".")
        return ("+" if x > 0 else MINUS) + s
    return FuncFormatter(f)


def figS2_needed(D):
    """Condition for drawing fig_reassignment: reassignment in >= 1 % of the replicates of any stochastic scenario,
    or in either D2 alternative."""
    c5 = D["c5"].set_index("scenario_id")
    stoch = any(100 - min(c5.at[s, "same_nonesh_pct"], c5.at[s, "same_esh_pct"]) >= 1.0 for s in STOCH)
    d2 = any(c5.at[s, "same_nonesh_pct"] < 100 or c5.at[s, "same_esh_pct"] < 100 for s in ("D2-NC", "D2-MN"))
    return stoch or d2


def c3_regimes(D):
    """The C3 replicates in replicate order and the regime of each under re-clustering (fig_ab_regimes): "unchanged" (non-ESH
    partition as P0), "05_13_to_B" (FAM_05_13 named B, with or without further moves) or "other" (FAM_05_13 stays in A,
    another family changes class). Only the non-ESH partition enters: ESH moves leave the A and B multipliers unchanged."""
    g = D["reps"][D["reps"].scenario_id == "C3-RND"].sort_values("replicate").reset_index(drop=True)
    same = g.same_nonesh.astype(bool).to_numpy()
    to_b = (g.cls_ARLP_05 == "B").to_numpy()
    reg = np.where(same, "unchanged", np.where(to_b, "05_13_to_B", "other"))
    assert not (same & to_b).any() and (g.cls_ARLP_05 == g.cls_ARLP_13).all()
    assert set(g.loc[reg == "other", "cls_ARLP_05"]) <= {"A"}, "FAM_05_13 left class A for a class other than B"
    return g, reg


def split_half(v, first, p, mask=None):
    """The Monte Carlo split-half difference of a figure quantile: |quantile over replicates 1-1,000 - quantile over
    1,001-2,000|, within mask."""
    v = np.asarray(v, dtype=float)
    m = np.ones(len(v), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return abs(q(v[m & first], p) - q(v[m & ~first], p))


def nonesh_pattern(g):
    """Matched class of each family in FAMILIES order, e.g. 'A A A B C C C C' for P0."""
    return g[[f"cls_{UNIT_MEMBERS[f][0]}" for f in FAMILIES]].astype(str).agg(" ".join, axis=1)


def baseline_row(D):
    return D["det"][D["det"].scenario_id == "BASELINE"].iloc[0]


def _style_axes(ax, fc):
    """Tick, spine and grid conventions of fig_coefficient_deviations."""
    ax.tick_params(axis="both", labelsize=FS41["tick"], length=2.5, width=0.5, pad=2, colors=fc.NAVY)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(fc.SLATE)


def fig17(D, out, N):
    """fig_ab_regimes: winter peak multipliers of classes A and B in the 2,000 C3 replicates, (a) with the classes of
    the proposed-tariff table fixed and (b) re-estimated (names by matching to P0); one point per replicate in both panels, coloured by the regime
    of (b). Every plotted coordinate is asserted against the source table."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    import _peed_fig_common as fc
    g, reg = c3_regimes(D)
    b = baseline_row(D)
    x0, y0 = float(b.a_fix_A_w_peak), float(b.a_fix_B_w_peak)
    assert (x0, y0) == (float(b.a_rec_A_w_peak), float(b.a_rec_B_w_peak))
    pts = {"fix": (g.a_fix_A_w_peak.to_numpy(float), g.a_fix_B_w_peak.to_numpy(float)),
           "rec": (g.a_rec_A_w_peak.to_numpy(float), g.a_rec_B_w_peak.to_numpy(float))}
    src = pd.DataFrame({"replicate": g.replicate.astype(int), "regime": reg, "cls_FAM_05_13": g.cls_ARLP_05,
                        "nonesh_classes": nonesh_pattern(g), "a_alpha_peak_A_w": pts["fix"][0],
                        "a_alpha_peak_B_w": pts["fix"][1], "b_alpha_peak_A_w": pts["rec"][0],
                        "b_alpha_peak_B_w": pts["rec"][1]})
    src = pd.concat([pd.DataFrame([{"replicate": 0, "regime": "baseline", "cls_FAM_05_13": "A",
                                    "nonesh_classes": "A A A B C C C C", "a_alpha_peak_A_w": x0, "a_alpha_peak_B_w": y0,
                                    "b_alpha_peak_A_w": x0, "b_alpha_peak_B_w": y0}]), src], ignore_index=True)
    lab_a, lab_b = PUB["C3-RND"], f"{PUB['C3-RND']} + {RC1}"
    src["scenario_id"] = np.where(src.replicate == 0, "BASELINE", "C3-RND")
    src["label_a"] = np.where(src.replicate == 0, "baseline", lab_a)
    src["label_b"] = np.where(src.replicate == 0, "baseline", lab_b)
    # identical limits in both panels, equal spans (so y = x runs at 45°), every point inside
    step = 0.05
    xs = np.concatenate([pts["fix"][0], pts["rec"][0], [x0]])
    ys = np.concatenate([pts["fix"][1], pts["rec"][1], [y0]])
    lo = (np.floor(xs.min() / step + 1e-9) * step, np.floor(ys.min() / step + 1e-9) * step)
    span = max(np.ceil(xs.max() / step - 1e-9) * step - lo[0], np.ceil(ys.max() / step - 1e-9) * step - lo[1])
    xlim, ylim = (lo[0], lo[0] + span), (lo[1], lo[1] + span)
    sty = {"unchanged": dict(marker="o", s=5.0, color=fc.MID, alpha=0.45, zorder=3),
           "other": dict(marker="s", s=6.5, color=fc.BLUEGRAY, alpha=0.75, zorder=4),
           "05_13_to_B": dict(marker="^", s=8.0, color=fc.NAVY, alpha=0.55, zorder=5)}
    label = {"unchanged": "Non-ESH classes unchanged", "05_13_to_B": "FAM_05_13 reassigned: A → B",
             "other": "Other non-ESH family reassignments"}
    # layout in inches: two square panels, the legend below
    fig_w, left, gap, ax_w, bottom, top = 190 / 25.4, 0.62, 0.22, 3.28, 0.95, 0.30
    fig_h = bottom + ax_w + top
    fig = Figure(figsize=(fig_w, fig_h), dpi=400)
    FigureCanvasAgg(fig)
    axes = [fig.add_axes([(left + i * (ax_w + gap)) / fig_w, bottom / fig_h, ax_w / fig_w, ax_w / fig_h])
            for i in range(2)]
    n = {r: int((reg == r).sum()) for r in REGIMES}
    first = g.replicate.to_numpy() <= len(g) // 2                          # replicates 1-1,000 (Monte Carlo rule)
    for ax, mode, title in zip(axes, ("fix", "rec"), (f"(a) {lab_a}, classes fixed",
                                                      f"(b) {lab_b}, classes re-estimated")):
        d0, d1 = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
        ax.plot([d0, d1], [d0, d1], color=fc.SLATE, lw=0.6, ls=":", zorder=1)
        ax.axvline(x0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
        ax.axhline(y0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
        x, y = pts[mode]
        for r in ("unchanged", "other", "05_13_to_B"):
            m = reg == r
            sc = ax.scatter(x[m], y[m], lw=0, **sty[r])
            assert np.array_equal(np.asarray(sc.get_offsets()), np.column_stack([x[m], y[m]]))
        ax.plot([x0], [y0], marker="*", ms=8.5, mfc="white", mec=fc.NAVY, mew=0.8, ls="", zorder=7)
        ax.annotate(f"Baseline (Table 4)\n{x0:.3f}, {y0:.3f}", xy=(x0, y0), xytext=(xlim[0] + 0.04, ylim[0] + 0.14),
                    fontsize=FS41["note"], color=fc.NAVY, ha="left", va="center", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=fc.SLATE, lw=0.5, shrinkA=2, shrinkB=4))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(0.1))
            axis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(True, color=fc.LIGHT, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        _style_axes(ax, fc)
        ax.set_title(title, fontsize=FS41["title"], color=fc.NAVY, pad=4, loc="left")
        ax.set_xlabel(r"$\alpha_{\mathrm{peak}}$ of class A, winter", fontsize=FS41["label"], color=fc.NAVY,
                      labelpad=3)
    axes[0].set_ylabel(r"$\alpha_{\mathrm{peak}}$ of class B, winter", fontsize=FS41["label"], color=fc.NAVY,
                       labelpad=3)
    axes[1].tick_params(labelleft=False)
    # (a): the fixed-class cloud magnified in the empty upper right (the panel limits stay those of (b))
    fx, fy = pts["fix"]
    half = 0.5 * np.ceil(1.15 * max(np.ptp(fx), np.ptp(fy)) / 0.01) * 0.01
    cx, cy = 0.5 * (fx.min() + fx.max()), 0.5 * (fy.min() + fy.max())
    ins = axes[0].inset_axes([0.585, 0.575, 0.385, 0.385])
    ins.axvline(x0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
    ins.axhline(y0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
    ins.plot([cx - half, cx + half], [cx - half, cx + half], color=fc.SLATE, lw=0.6, ls=":", zorder=1)
    for r in ("unchanged", "other", "05_13_to_B"):
        m = reg == r
        sc = ins.scatter(fx[m], fy[m], lw=0, **sty[r])
        assert np.array_equal(np.asarray(sc.get_offsets()), np.column_stack([fx[m], fy[m]]))
    ins.plot([x0], [y0], marker="*", ms=8.5, mfc="white", mec=fc.NAVY, mew=0.8, ls="", zorder=7)
    ins.set_xlim(cx - half, cx + half)
    ins.set_ylim(cy - half, cy + half)
    ins.set_aspect("equal", adjustable="box")
    for axis in (ins.xaxis, ins.yaxis):
        axis.set_major_locator(MultipleLocator(0.05))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))
    _style_axes(ins, fc)
    ins.tick_params(labelsize=FS41["note"])
    ins.set_facecolor("white")
    axes[0].indicate_inset_zoom(ins, edgecolor=fc.SLATE, lw=0.5, alpha=1.0)
    assert cx - half <= fx.min() and fx.max() <= cx + half and cy - half <= fy.min() and fy.max() <= cy + half
    # (b): the A -> B transition of FAM_05_13 and the number of replicates per regime
    ax = axes[1]
    m = reg == "05_13_to_B"
    mx, my = float(np.median(pts["rec"][0][m])), float(np.median(pts["rec"][1][m]))
    ax.annotate("", xy=(mx, my), xytext=(x0, y0), zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=fc.NAVY, lw=0.8, mutation_scale=7, shrinkA=6, shrinkB=9,
                                connectionstyle="arc3,rad=-0.25"))
    fam_h = N.d["H|FAM_05_13_kHH"]["text"]
    ax.text(x0 - 0.06, y0 + 0.14, f"FAM_05_13 ({fam_h} thousand HHs)\nmoves from class A to class B",
            fontsize=FS41["note"], color=fc.NAVY, ha="left", va="bottom", zorder=8)
    tx, ty, dy = xlim[0] + 0.60 * span, ylim[1] - 0.045 * span, 0.052 * span
    ax.text(tx - 0.02 * span, ty, f"Replicates of {len(g):,}", fontsize=FS41["note"], color=fc.SLATE, ha="left",
            va="center")
    for i, r in enumerate(("unchanged", "05_13_to_B", "other")):
        yy = ty - (i + 1) * dy
        share = 100 * n[r] / len(g)
        ax.scatter([tx], [yy], s=sty[r]["s"] * 2.2, marker=sty[r]["marker"], color=sty[r]["color"], lw=0, zorder=8)
        ax.text(tx + 0.03 * span, yy, f"{n[r]:,} ({share:.1f} %)" if r == "unchanged" else
                f"{n[r]:,} ({share:.2f} %)", fontsize=FS41["note"], color=fc.NAVY, ha="left", va="center", zorder=8)
    handles = [Line2D([], [], ls="", marker=sty[r]["marker"], ms=4.2, color=sty[r]["color"], label=label[r])
               for r in ("unchanged", "05_13_to_B", "other")] + \
              [Line2D([], [], ls="", marker="*", ms=7, mfc="white", mec=fc.NAVY, mew=0.8,
                      label="Baseline multipliers (Table 4), dashed lines"),
               Line2D([], [], color=fc.SLATE, lw=0.6, ls=":", label="Equal winter peak multipliers of A and B")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=FS41["legend"], frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.6)
    png = out / "fig_ab_regimes.png"
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    to_tif(png)
    src.to_csv(out / "fig_ab_regimes_source.csv", index=False, float_format="%.12g", lineterminator="\n")
    # numbers quoted with the figure
    N.put("F17|n_points", f"{len(g):,}", len(g), "03 C3-RND rows")
    N.put("F17|baseline|A", f"{x0:.3f}", x0, "02 BASELINE a_fix_A_w_peak")
    N.put("F17|baseline|B", f"{y0:.3f}", y0, "02 BASELINE a_fix_B_w_peak")
    for r in REGIMES:
        share = 100 * n[r] / len(g)
        N.put(f"F17|n|{r}", f"{n[r]:,}", n[r], "03 C3-RND, regime of the non-ESH partition")
        N.put(f"F17|pct1|{r}", pct(share, 1), share, "03 C3-RND")
        N.put(f"F17|pct2|{r}", pct(share, 2), share, "03 C3-RND")
        mm = reg == r
        for mode in ("fix", "rec"):
            for k, c in enumerate(("A", "B")):
                v = float(np.median(pts[mode][k][mm]))
                N.put(f"F17|med|{r}|{mode}|{c}", f"{v:.3f}", v, f"03 C3-RND a_{mode}_{c}_w_peak median, regime {r}")
                N.put(f"F17|med2|{r}|{mode}|{c}", f"{v:.2f}", v, f"03 C3-RND a_{mode}_{c}_w_peak median, regime {r}")
                for p, nm in ((0.05, "p05"), (0.95, "p95")):
                    v, sp = q(pts[mode][k][mm], p), split_half(pts[mode][k], first, p, mm)
                    N.put(f"F17|{nm}|{r}|{mode}|{c}", num(v, 3, sp > MC_TOL), v,
                          f"03 C3-RND a_{mode}_{c}_w_peak, regime {r}; split-half {sp:.4f}")
            v = 100 * float((pts[mode][1][mm] >= pts[mode][0][mm]).mean())
            N.put(f"F17|B_ge_A|{r}|{mode}", pct(v, 1), v, "03 C3-RND, share of the regime's replicates with "
                                                          "alpha_peak(B, w) >= alpha_peak(A, w)")
    for mode in ("fix", "rec"):
        for k, c in enumerate(("A", "B")):
            for p, nm in ((0.05, "p05"), (0.95, "p95")):
                v, sp = q(pts[mode][k], p), split_half(pts[mode][k], first, p)
                N.put(f"F17|{nm}|all|{mode}|{c}", num(v, 3, sp > MC_TOL), v,
                      f"03 C3-RND a_{mode}_{c}_w_peak, all replicates; split-half {sp:.4f}")
    pat = nonesh_pattern(g)
    v = int(((reg == "05_13_to_B") & (pat == "A B A B C C C C")).sum())
    N.put("F17|n|05_13_to_B_only", f"{v:,}", v, "03 C3-RND, FAM_05_13 the only family that changes class")
    N.put("F17|n|05_13_to_B_more", f"{n['05_13_to_B'] - v:,}", n["05_13_to_B"] - v, "03 C3-RND")
    for f in FAMILIES:
        a = UNIT_MEMBERS[f][0]
        v = int(((reg == "other") & (g[f"cls_{a}"] != P0[a])).sum())
        if v:
            N.put(f"F17|other|moved|{f}", f"{v:,}", v, "03 C3-RND, regime other: replicates in which the family moves")
    N.put("F17|xlim", f"{xlim[0]:.2f}–{xlim[1]:.2f}", None, "figure")
    N.put("F17|ylim", f"{ylim[0]:.2f}–{ylim[1]:.2f}", None, "figure")
    return png


def s2_decomposition(D, g):
    """The experienced-multiplier decomposition per clustering unit, season and zone over the replicates g: the change of the multiplier faced by the
    unit's HHs with the classes fixed (a_fix of its baseline class minus the baseline), the reassignment part (dre,
    = a_rec of its matched class minus that a_fix) and the total (their sum). The members of a unit share both classes,
    so the first member stands for the unit (asserted)."""
    b = baseline_row(D)
    out = {}
    for u in UNITS:
        a, c0 = UNIT_MEMBERS[u][0], P0[UNIT_MEMBERS[u][0]]
        for sk in SK:
            for z in ZK:
                fixed = g[f"a_fix_{c0}_{sk}_{z}"].to_numpy(float) - float(b[f"a_fix_{c0}_{sk}_{z}"])
                re = g[f"dre_{a}_{sk}_{z}"].to_numpy(float)
                for m in UNIT_MEMBERS[u][1:]:
                    assert np.array_equal(g[f"dre_{m}_{sk}_{z}"].to_numpy(float), re)
                out[(u, sk, z)] = {"fixed": fixed, "reassignment": re, "total": fixed + re}
    return out


def figS2(D, out, N):
    """fig_reassignment: (a) per clustering unit, the share of the C3 replicates in which it changes class and the share
    in which reassignment changes the winter peak multiplier its HHs face; (b) that change split into classes fixed,
    reassignment and total (P5-P95, median, full range over the 2,000 replicates)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    import _peed_fig_common as fc
    import eulp_hierarchy as eh
    Hh = eh.load_household_counts(eh.find_vault())
    g, _ = c3_regimes(D)
    c5 = D["c5"].set_index("scenario_id")
    dec = s2_decomposition(D, g)
    comp = ("fixed", "reassignment", "total")
    col = {"fixed": fc.MID, "reassignment": fc.NAVY, "total": fc.BLUEGRAY}
    ls3 = PUB["C3-RND"]
    # legend labels kept short to fit the layout of the figure; the source CSV carries the full ones
    name = {"fixed": "classes fixed", "reassignment": f"reassignment ({RC1})", "total": "total"}
    full = {"fixed": f"classes fixed ({ls3})", "reassignment": f"reassignment ({RC1})",
            "total": f"total ({ls3} + {RC1})"}
    yoff = {"fixed": -0.26, "reassignment": 0.0, "total": 0.26}
    rows, src = [], []
    first = g.replicate.to_numpy() <= len(g) // 2                          # replicates 1-1,000 (Monte Carlo rule)
    for u in UNITS:
        a, c0 = UNIT_MEMBERS[u][0], P0[UNIT_MEMBERS[u][0]]
        moved = g[f"cls_{a}"] != c0
        mv = 100 * float(moved.mean())
        for m in UNIT_MEMBERS[u]:
            assert abs(c5.at["C3-RND", f"moved_{m}_pct"] - mv) < 1e-9, (u, m)
        dest = g.loc[moved, f"cls_{a}"].value_counts()
        dest = ", ".join(dest.sort_index().sort_values(ascending=False, kind="stable").index)
        H = sum(Hh[m] for m in UNIT_MEMBERS[u])
        rows.append({"unit": u, "cls": c0, "H": H, "moved": mv, "dest": dest})
        for sk in SK:
            for z in ZK:
                d = dec[(u, sk, z)]
                nz = 100 * float((np.abs(d["reassignment"]) > 1e-12).mean())
                for k in comp:
                    v = d[k]
                    src.append({"scenario_id": "C3-RND", "label": ls3, "component_label": full[k],
                                "unit": u, "members": " ".join(UNIT_MEMBERS[u]), "baseline_class": c0,
                                "households_thousand": H, "moved_pct": mv, "moved_to": dest,
                                "reassignment_nonzero_pct": nz, "season": sk, "zone": z, "component": k,
                                "p05": q(v, .05), "median": q(v, .5), "p95": q(v, .95), "min": float(v.min()),
                                "max": float(v.max()), "p95_abs": q(np.abs(v), .95),
                                "split_p05": split_half(v, first, .05), "split_p95": split_half(v, first, .95),
                                "split_p95_abs": split_half(np.abs(v), first, .95),
                                "plotted": (sk, z) == ("w", "peak")})
        rows[-1]["nz"] = 100 * float((np.abs(dec[(u, "w", "peak")]["reassignment"]) > 1e-12).mean())
    src = pd.DataFrame(src)
    # layout in inches
    fig_w, left, w1, gap, w2, pitch = 190 / 25.4, 1.42, 1.95, 0.22, 3.65, 0.30
    bottom, top, nrow = 1.05, 0.30, len(UNITS)
    fig_h = bottom + nrow * pitch + top
    fig = Figure(figsize=(fig_w, fig_h), dpi=400)
    FigureCanvasAgg(fig)
    ax1 = fig.add_axes([left / fig_w, bottom / fig_h, w1 / fig_w, nrow * pitch / fig_h])
    ax2 = fig.add_axes([(left + w1 + gap) / fig_w, bottom / fig_h, w2 / fig_w, nrow * pitch / fig_h])
    blocks = [c for c in CL for _ in [u for u in UNITS if P0[UNIT_MEMBERS[u][0]] == c]]
    for i, r in enumerate(rows):
        if CL.index(blocks[i]) % 2 == 0:                    # class blocks shaded as in fig_coefficient_deviations
            for ax in (ax1, ax2):
                ax.axhspan(i - 0.5, i + 0.5, color=fc.LAVENDER, zorder=0, lw=0)
        b1 = ax1.barh(i, r["nz"], height=0.62, color=fc.LIGHT, edgecolor=fc.SLATE, linewidth=0.5, zorder=2)
        b2 = ax1.barh(i, r["moved"], height=0.26, color=fc.NAVY, linewidth=0, zorder=3)
        assert abs(b1[0].get_width() - r["nz"]) < 1e-12 and abs(b2[0].get_width() - r["moved"]) < 1e-12
        if r["dest"]:
            ax1.text(max(r["nz"], r["moved"]) + 0.6, i, f"→ {r['dest']}", fontsize=FS41["note"], color=fc.NAVY,
                     va="center", ha="left", zorder=4)
        x = src[(src.unit == r["unit"]) & src.plotted].set_index("component")
        for k in comp:
            y = i + yoff[k]
            ax2.plot([x.at[k, "min"], x.at[k, "max"]], [y, y], color=col[k], lw=0.6, solid_capstyle="butt", zorder=2)
            ax2.plot([x.at[k, "p05"], x.at[k, "p95"]], [y, y], color=col[k], lw=3.0, solid_capstyle="butt", zorder=3)
            ax2.plot([x.at[k, "median"]], [y], marker="o", ms=2.8, color="white",
                     mec=fc.NAVY if k == "fixed" else col[k], mew=0.7, zorder=4)
    labels = [f"{r['unit'].replace('ARLP_', 'ARLP-')} ({r['cls']}, {r['H']:,.0f})" for r in rows]
    shown = src[src.plotted]
    lim2 = (np.floor(shown["min"].min() / 0.1 - 1e-9) * 0.1, np.ceil(shown["max"].max() / 0.1 + 1e-9) * 0.1)
    for ax in (ax1, ax2):
        ax.set_ylim(nrow - 0.5, -0.5)
        ax.set_yticks(range(nrow))
        ax.grid(axis="x", color=fc.LIGHT, lw=0.4, zorder=1)
        _style_axes(ax, fc)
    ax1.set_yticklabels(labels)
    ax2.set_yticklabels([])
    ax1.set_xlim(0, 5 * np.ceil((max(r["nz"] for r in rows) + 6) / 5))
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax2.set_xlim(*lim2)
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    ax2.xaxis.set_major_formatter(_signed_formatter())
    ax2.axvline(0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
    ax1.set_title("(a) Replicates with a class change", fontsize=FS41["title"], color=fc.NAVY, pad=4, loc="left")
    ax2.set_title(f"(b) Change of the winter peak multiplier its HHs face ({ls3})", fontsize=FS41["title"],
                  color=fc.NAVY, pad=4, loc="left")
    ax1.set_xlabel(f"Share of the {len(g):,} {ls3} replicates, %", fontsize=FS41["label"], color=fc.NAVY,
                   labelpad=3)
    ax2.set_xlabel(r"Deviation from the baseline multiplier, $\Delta\alpha_{\mathrm{peak}}$ (winter)",
                   fontsize=FS41["label"], color=fc.NAVY, labelpad=3)
    # one legend under each panel (the panels encode different series)
    h1 = [Patch(facecolor=fc.LIGHT, edgecolor=fc.SLATE, linewidth=0.5, label="its HHs face a changed multiplier"),
          Patch(facecolor=fc.NAVY, label="the unit itself changes class")]
    h2 = [Line2D([], [], color=col[k], lw=3.0, label=f"{name[k]}, P5–P95") for k in comp] + \
         [Line2D([], [], marker="o", ls="", ms=2.8, color="white", mec=fc.NAVY, mew=0.7, label="median"),
          Line2D([], [], color=fc.NAVY, lw=0.6, label="full range")]
    fig.legend(handles=h1, loc="lower left", ncol=1, fontsize=FS41["legend"], frameon=False,
               bbox_to_anchor=((left - 0.1) / fig_w, 0.0), handlelength=1.6)
    fig.legend(handles=h2, loc="lower left", ncol=3, fontsize=FS41["legend"], frameon=False,
               bbox_to_anchor=((left + w1 + gap - 0.1) / fig_w, 0.0), handlelength=1.8, columnspacing=1.2)
    png = out / "fig_reassignment.png"
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    to_tif(png)
    src.to_csv(out / "fig_reassignment_source.csv", index=False, float_format="%.12g", lineterminator="\n")
    for r in rows:
        u = r["unit"]
        N.put(f"S2|{u}|H", f"{r['H']:,.0f}", r["H"], "arlp_household_counts.csv")
        N.put(f"S2|{u}|moved", pct(r["moved"], 2), r["moved"], "03 C3-RND = 05 moved_*_pct")
        N.put(f"S2|{u}|moved1", pct(r["moved"], 1), r["moved"], "03 C3-RND = 05 moved_*_pct")
        N.put(f"S2|{u}|nonzero", pct(r["nz"], 1), r["nz"], "03 C3-RND dre_*_w_peak != 0")
        x = src[(src.unit == u) & src.plotted].set_index("component")
        for k in comp:
            for stat in ("p05", "median", "p95", "min", "max"):
                # P5 and P95 follow the Monte Carlo rule (2 dp and "‡" above 0.001)
                fail = stat in ("p05", "p95") and x.at[k, f"split_{stat}"] > MC_TOL
                N.put(f"S2|{u}|w|peak|{k}|{stat}", sgn(x.at[k, stat], 2) + "‡" if fail else sgn(x.at[k, stat]),
                      x.at[k, stat], "fig_reassignment_source.csv")
            N.put(f"S2|{u}|w|peak|{k}|p95_abs", num(x.at[k, "p95_abs"], 3, x.at[k, "split_p95_abs"] > MC_TOL),
                  x.at[k, "p95_abs"], "fig_reassignment_source.csv")
    return png


# ── fig_ls1_timing and fig_ls2_one_class (post-run visualisation): the deterministic LS1 and single-class LS2 runs ─
LS1_IDS = ("C1-TIME-M1", "C1-TIME-P1")
LS2_ONE = tuple(f"C2-ZONE-DM-{c}" for c in CL)                     # 5 % peak -> off-peak in the units of one class
PANEL_TITLE = {"off": r"Off-peak multiplier $\alpha_{\mathrm{off}}$", "sh": r"Shoulder multiplier $\alpha_{\mathrm{sh}}$",
               "peak": r"Peak multiplier $\alpha_{\mathrm{peak}}$"}  # the panel titles of fig_coefficient_deviations
LS2_SHAPE = {"A": ("s", 3.3), "B": ("D", 3.1), "C": ("^", 4.0), "ESH-1": ("v", 4.0), "ESH-2": ("o", 3.5)}
LS2_YOFF = dict(zip(CL, (-0.32, -0.16, 0.0, 0.16, 0.32)))          # perturbed class, top to bottom within a row


def ls_limits(cs):
    """Horizontal limits shared by fig_ls1_timing and fig_ls2_one_class, per multiplier: 1.12 x the largest |dev| of
    the seven runs."""
    f = cs[(cs["mode"] == "fix") & cs.scenario_id.isin(LS1_IDS + LS2_ONE)]
    return {z: 1.12 * float(f[f.quantity == z].dev.abs().max()) for z in ZK}


def _ls_frame(pitch, nrow=10, bottom=0.85, top=0.37):
    """The frame of fig_coefficient_deviations (width, panel positions, bottom and top margins in inches) with a row pitch in inches."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig_w, fig_h = 190 / 25.4, bottom + nrow * pitch + top
    fig = Figure(figsize=(fig_w, fig_h), dpi=400)
    FigureCanvasAgg(fig)
    return fig, [fig.add_axes([0.13 + i * 0.29, bottom / fig_h, 0.25, nrow * pitch / fig_h]) for i in range(3)]


def _ls_style(axes, rows, lims, fc):
    """Class shading, zero line, row labels, ticks, grid, titles and x label as in fig_coefficient_deviations."""
    for ax, z in zip(axes, ZK):
        for ri in range(len(rows)):
            if ri % 4 < 2:
                ax.axhspan(ri - 0.5, ri + 0.5, color=fc.LAVENDER, zorder=0, lw=0)
        ax.set_xlim(-lims[z], lims[z])
        ax.axvline(0, color=fc.SLATE, lw=0.6, ls="--", zorder=2)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{c}, {SEASON[sk]}" for c, sk in rows] if z == "off" else [])
        _style_axes(ax, fc)
        ax.grid(axis="x", color=fc.LIGHT, lw=0.4, zorder=1)
        ax.set_title(PANEL_TITLE[z], fontsize=FS41["title"], color=fc.NAVY, pad=4)
        ax.xaxis.set_major_formatter(_signed_formatter())
    axes[1].set_xlabel(r"Deviation from the baseline multiplier, $\Delta\alpha$", fontsize=FS41["label"],
                       color=fc.NAVY, labelpad=3)


def _ls_inside_ls3(cs, sid):
    """Per multiplier: True if the run's deviation lies within the LS3 P5-P95 deviation range of 04 (classes fixed)."""
    s = cs[(cs.scenario_id == sid) & (cs["mode"] == "fix") & cs.quantity.isin(ZK)].set_index(["class", "season", "quantity"])
    r = cs[(cs.scenario_id == "C3-RND") & (cs["mode"] == "fix") & cs.quantity.isin(ZK)].set_index(
        ["class", "season", "quantity"]).loc[s.index]
    return (s.dev >= r.p05_dev) & (s.dev <= r.p95_dev)


def figS3(D, out, N):
    """fig_ls1_timing: the two LS1 runs (all profiles shifted by -1 h and +1 h), Delta alpha with the classes of the
    proposed-tariff table fixed, in the frame of fig_coefficient_deviations; open triangle -1 h, filled triangle +1 h
    (the open/filled convention of HC2 in fig_coefficient_deviations)."""
    from matplotlib.lines import Line2D
    import _peed_fig_common as fc
    cs = D["cs"]
    rows = [(c, sk) for c in CL for sk in SK]
    lims = ls_limits(cs)
    sty = {"C1-TIME-M1": dict(marker="^", ms=4.0, color=fc.NAVY, mfc="white", mew=0.8),
           "C1-TIME-P1": dict(marker="^", ms=4.0, color=fc.NAVY, mfc=fc.NAVY, mew=0.8)}
    yoff = {"C1-TIME-M1": -0.17, "C1-TIME-P1": 0.17}
    shift = {"C1-TIME-M1": -1, "C1-TIME-P1": 1}
    fig, axes = _ls_frame(pitch=0.30)
    src = []
    for ax, z in zip(axes, ZK):
        for ri, (c, sk) in enumerate(rows):
            for sid in LS1_IDS:
                x = cs_row(cs, sid, "fix", c, sk, z)
                ln, = ax.plot([x.dev], [ri + yoff[sid]], ls="", zorder=5, **sty[sid])
                assert ln.get_xdata()[0] == x.dev and abs(x.dev) <= lims[z]
                src.append({"panel": z, "class": c, "season": sk, "scenario": sid, "label": PUB[sid],
                            "shift_h": shift[sid], "baseline": x.baseline_eulp_route, "value": x.value, "dev": x.dev})
    _ls_style(axes, rows, lims, fc)
    handles = [Line2D([], [], ls="", **sty[s]) for s in LS1_IDS]
    labels = [f"{PUB['C1-TIME-M1']}, {MINUS}1 h", f"{PUB['C1-TIME-P1']}, +1 h"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=2, fontsize=FS41["legend"], frameon=False,
               bbox_to_anchor=(0.53, 0.0), handlelength=1.8, columnspacing=1.8)
    png = out / "fig_ls1_timing.png"
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    to_tif(png)
    src = pd.DataFrame(src)
    src.to_csv(out / "fig_ls1_timing_source.csv", index=False, float_format="%.12g", lineterminator="\n")
    # numbers quoted with the figure (the single deviations are in the JSON as H|<id>|<class>|<season>|<zone>)
    N.put("S3|n_points", str(len(src)), len(src), "fig_ls1_timing_source.csv")
    for z in ZK:
        N.put(f"S3S4|xlim|{z}", f"{MINUS}{lims[z]:.4f} to +{lims[z]:.4f}", lims[z], "figure (shared by fig_ls1_timing and fig_ls2_one_class)")
    n_in = sum(int(_ls_inside_ls3(cs, sid).sum()) for sid in LS1_IDS)
    N.put("S3|n_inside_LS3", str(n_in), n_in, "04: LS1 dev within the LS3 p05_dev-p95_dev range, classes fixed")
    pk = src[(src.panel == "peak") & (src["class"].isin(["A", "B"]))].pivot_table(
        index=["scenario", "season"], columns="class", values="dev")
    opp = bool((np.sign(pk["A"]) == -np.sign(pk["B"])).all())
    N.put("S3|AB_peak_opposite", "yes" if opp else "no", None, "04: alpha_peak of A and B move in opposite directions "
                                                              "in both LS1 runs and both seasons")
    return png


def figS4(D, out, N):
    """fig_ls2_one_class: the five single-class LS2 runs (5 % of the peak-zone energy moved to the off-peak zone in the
    units of one class), Delta alpha of all 30 multipliers with the classes of the proposed-tariff table fixed, in the
    frame of fig_coefficient_deviations. Within a row (measured class and season) the perturbed class runs from A (top) to ESH-2 (bottom) and has its own marker
    shape; filled = the row's own class perturbed, open = another class."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    import _peed_fig_common as fc
    cs = D["cs"]
    reg = pd.read_csv(D["run"] / "00_registry.csv").set_index("scenario_id")
    for k in CL:                                                    # the direction printed in the caption
        assert json.loads(reg.at[f"C2-ZONE-DM-{k}", "parameters"]) == {"delta": 0.05, "units": f"class {k}"}
    rows = [(c, sk) for c in CL for sk in SK]
    lims = ls_limits(cs)

    def sty(k, own):
        mk, ms = LS2_SHAPE[k]
        return dict(marker=mk, ms=ms, mew=0.8, color=fc.NAVY if own else fc.BLUEGRAY, mfc=fc.NAVY if own else "white")
    fig, axes = _ls_frame(pitch=0.62)
    src = []
    for ax, z in zip(axes, ZK):
        for ri, (c, sk) in enumerate(rows):
            for k in CL:
                sid = f"C2-ZONE-DM-{k}"
                x = cs_row(cs, sid, "fix", c, sk, z)
                ln, = ax.plot([x.dev], [ri + LS2_YOFF[k]], ls="", zorder=6 if k == c else 5, **sty(k, k == c))
                assert ln.get_xdata()[0] == x.dev and abs(x.dev) <= lims[z]
                src.append({"panel": z, "class": c, "season": sk, "scenario": sid, "label": PUB[sid],
                            "perturbed_class": k, "own_class": k == c, "baseline": x.baseline_eulp_route,
                            "value": x.value, "dev": x.dev})
    _ls_style(axes, rows, lims, fc)
    # legend: one entry per perturbed class, open (another class's row) and filled (own row), as HC2 in
    # fig_coefficient_deviations
    handles = [tuple(Line2D([], [], ls="", **sty(k, own)) for own in (False, True)) for k in CL]
    labels = [f"{PUB['C2-ZONE-DM-A']}, {k} only" for k in CL]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=len(CL), fontsize=FS41["legend"],
               frameon=False, bbox_to_anchor=(0.53, 0.0), handlelength=2.2, columnspacing=1.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
    png = out / "fig_ls2_one_class.png"
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    to_tif(png)
    src = pd.DataFrame(src)
    src.to_csv(out / "fig_ls2_one_class_source.csv", index=False, float_format="%.12g", lineterminator="\n")
    # numbers quoted with the figure (the single deviations are in the JSON as H|<id>|<class>|<season>|<zone>)
    N.put("S4|n_points", str(len(src)), len(src), "fig_ls2_one_class_source.csv")
    own, cross = src[src.own_class], src[~src.own_class]
    ok = bool((np.sign(own.dev) == own.panel.map({"off": -1, "sh": 1, "peak": -1})).all() and
              (np.sign(cross.dev) == cross.panel.map({"off": 1, "sh": -1, "peak": 1})).all())
    N.put("S4|sign_pattern", "yes" if ok else "no", None, "fig_ls2_one_class_source.csv: own class off-, sh+, peak-; every "
                                                         "other class off+, sh-, peak+")
    g = src.assign(a=src.dev.abs()).groupby(["class", "season", "panel"])
    ok = bool(all(gg[gg.own_class].a.iloc[0] > gg[~gg.own_class].a.max() for _, gg in g))
    N.put("S4|own_largest", "yes" if ok else "no", None, "fig_ls2_one_class_source.csv: in every row and panel the own-class "
                                                        "run moves the multiplier most")
    for k in CL:
        v = own[(own.perturbed_class == k) & (own.panel == "sh")].dev.max()
        N.put(f"S4|{k}|own_sh_max", f"{v:.3f}", v, "fig_ls2_one_class_source.csv")
    for grp, ks in (("AB", ("A", "B")), ("small", ("C", "ESH-1", "ESH-2"))):
        x = cross[cross.perturbed_class.isin(ks)]
        i = x.dev.abs().idxmax()
        N.put(f"S4|cross_max|{grp}", f"{abs(x.at[i, 'dev']):.3f}", abs(x.at[i, "dev"]), "fig_ls2_one_class_source.csv")
        N.put(f"S4|cross_max|{grp}|where", f"{x.at[i, 'class']} {SEASON[x.at[i, 'season']]} {ZNAME[x.at[i, 'panel']]}"
                                           f" ({x.at[i, 'perturbed_class']} perturbed)", None, "fig_ls2_one_class_source.csv")
    ins = pd.concat([_ls_inside_ls3(cs, f"C2-ZONE-DM-{k}").rename(k) for k in CL], axis=1)
    out_ = ~ins
    n_out = int(out_.to_numpy().sum())
    n_out_own = int(sum(out_.loc[ix, k] for k in CL for ix in out_.index if ix[0] == k))
    N.put("S4|n_outside_LS3", str(n_out), n_out, "04: single-class LS2 dev outside the LS3 p05_dev-p95_dev range")
    N.put("S4|n_outside_LS3_own", str(n_out_own), n_out_own, "04: of which in the perturbed class's own multipliers")
    return png


def label_map(D, out, N):
    """publication_labels.csv: every scenario of the registry (and the re-clustering mode) with its publication label,
    from PUB; the same pairs are appended to the JSON as LABEL|<internal id>. The registry itself is not changed."""
    reg = pd.read_csv(D["run"] / "00_registry.csv")
    assert set(reg.scenario_id) == set(PUB), sorted(set(reg.scenario_id) ^ set(PUB))
    rows = [{"internal_id": r.scenario_id, "publication_label": PUB[r.scenario_id],
             "group": PUB[r.scenario_id][:2], "group_name": PUB_GROUP[PUB[r.scenario_id][:2]],
             "meaning": PUB_MEANING[PUB[r.scenario_id]], "registry_mechanism": r.mechanism,
             "n_replicates": int(r.n_replicates), "seed": "" if pd.isna(r.seed) else int(r.seed)}
            for r in reg.itertuples()]
    rows.append({"internal_id": "class mode rec (a_rec_*, cls_*, dre_*, D_* of 02 and 03)", "publication_label": RC1,
                 "group": "RC", "group_name": PUB_GROUP["RC"], "meaning": PUB_MEANING[RC1],
                 "registry_mechanism": "re-clustering of the perturbed loads, applied to every A-C run", "n_replicates": "",
                 "seed": ""})
    pd.DataFrame(rows).to_csv(out / "publication_labels.csv", index=False, lineterminator="\n")
    for r in rows[:-1]:
        N.put(f"LABEL|{r['internal_id']}", r["publication_label"], None, "PUB in make_robustness_outputs.py")
    N.put("LABEL|rec", RC1, None, "PUB in make_robustness_outputs.py")


def to_tif(png):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    im.save(png.with_suffix(".tif"), compression="tiff_lzw", dpi=(300, 300))


# ── Markdown ──────────────────────────────────────────────────────────────────────────────────────────────────
def md(df):
    """Markdown pipe table; a literal '|' inside a cell is escaped so that the row keeps its columns."""
    def cell(v):
        return "" if (isinstance(v, float) and np.isnan(v)) else str(v).replace("|", r"\|")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cell(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)


def check_figures(run):
    """The figures against the run's 02, 03, 04, 05 and 00_registry.csv, their source CSVs, the JSON and the
    proposed-tariff table. Returns the table of checks (also written to figure_checks.csv)."""
    from PIL import Image
    import eulp_hierarchy as eh
    run = Path(run)
    D = load(run)
    nums = json.loads((run / "reported_numbers.json").read_text(encoding="utf-8"))
    t4 = table4(eh.find_vault())
    g, reg = c3_regimes(D)
    b = baseline_row(D)
    c5 = D["c5"].set_index("scenario_id")
    f17 = pd.read_csv(run / "fig_ab_regimes_source.csv")
    s2 = pd.read_csv(run / "fig_reassignment_source.csv")
    rows = []

    def put(cid, what, value, ok, tol=""):
        rows.append({"check_id": cid, "description": what, "value": value, "tolerance": tol,
                     "status": "PASS" if ok else "FAIL"})

    # fig_coefficient_deviations: every plotted value against 04 (EW1, HC1, LS3 bars and medians; EW2; B3 own class x 0.8 and x 1.2)
    f16 = pd.read_csv(run / "fig_coefficient_deviations_source.csv")
    cs, dev16, n16 = D["cs"], 0.0, 0
    for r in f16.to_dict("records"):
        x = cs_row(cs, r["scenario"], "fix", r["class"], r["season"], r["panel"])
        exp = ({"p05_dev": x.p05_dev, "median_dev": x["median"] - x.baseline_eulp_route, "p95_dev": x.p95_dev}
               if r["scenario"] in STOCH else {"dev": x.dev})
        for k, v in exp.items():
            dev16 = max(dev16, abs(r[k] - v))
        n16 += 1
    kinds = f16.groupby("label").size().to_dict()
    own = f16[f16.label == PUB["B3-CLS-A-0.8"]]
    ok = (dev16 <= 1e-9 and n16 == 30 * 6 and kinds == {PUB["A1-U20"]: 30, PUB["A2-MIX25"]: 30, PUB["B1-RND20"]: 30, PUB["B3-CLS-A-0.8"]: 60, PUB["C3-RND"]: 30}
          and (own.scaled_class == own["class"]).all())
    put("F16-01", f"fig_coefficient_deviations source = 04: EW1, HC1, LS3 (P5, median, P95), EW2 and {PUB['B3-CLS-A-0.8']} (own class "
                  f"x 0.8 and x 1.2) for all 30 multipliers; no {PUB['B2-STAT']}", f"{dev16:.1e}; rows {kinds}", ok, "<= 1e-9 (CSV %.12g)")
    # fig_ab_regimes
    pts = f17[f17.regime != "baseline"].reset_index(drop=True)
    ids = pts.replicate.to_numpy()
    put("F17-01", "one point per LS3 replicate (C3-RND): 2,000 rows, replicate ids 1-2,000 unique, no missing coordinate",
        f"{len(pts)} rows; ids {ids.min()}-{ids.max()}; unique {pd.Series(ids).is_unique}",
        len(pts) == 2000 and pd.Series(ids).is_unique and set(ids) == set(range(1, 2001)) and
        not pts.filter(like="alpha").isna().any().any(), "exact")
    dev = max(float(np.abs(pts[f"{p}_alpha_peak_{c}_w"].to_numpy() - g[f"a_{m}_{c}_w_peak"].to_numpy()).max())
              for p, m in (("a", "fix"), ("b", "rec")) for c in ("A", "B"))
    put("F17-02", "panel (a) = a_fix and panel (b) = a_rec of 03 for the same replicate id (A, B, winter, peak)",
        f"{dev:.1e}", dev <= 1e-9 and (pts.replicate.to_numpy() == g.replicate.to_numpy()).all(), "<= 1e-9 (CSV %.12g)")
    n = {r: int((pts.regime == r).sum()) for r in REGIMES}
    to_b = 100 * n["05_13_to_B"] / len(pts)
    ok = (abs(to_b - nums["H|C3-RND|ARLP_05_to_B"]["raw"]) < 1e-9 and
          abs(100 * n["unchanged"] / len(pts) - c5.at["C3-RND", "same_nonesh_pct"]) < 1e-9 and sum(n.values()) == 2000
          and (pts.regime.to_numpy() == reg).all())
    put("F17-03", "regimes: FAM_05_13 to B = reported share (JSON H|C3-RND|ARLP_05_to_B); unchanged = 05 same_nonesh_pct;"
                  " the three regimes partition the 2,000 replicates",
        f"{n['unchanged']} / {n['05_13_to_B']} ({to_b:.2f} %) / {n['other']}", ok, "exact")
    cls_cols = [c for c in g.columns if c.startswith("cls_")]
    names_ok = set(np.unique(g[cls_cols].to_numpy().astype(str))) <= set(CL)
    fam_ok = all((g[f"cls_{UNIT_MEMBERS[f][0]}"] == g[f"cls_{UNIT_MEMBERS[f][1]}"]).all() for f in FAMILIES)
    unch = g[reg == "unchanged"]
    p0_ok = all((unch[f"cls_{a}"] == P0[a]).all() for a in ARLPS if not P0[a].startswith("ESH"))
    same_ab = all(np.array_equal(unch[f"a_fix_{c}_w_peak"].to_numpy(), unch[f"a_rec_{c}_w_peak"].to_numpy())
                  for c in ("A", "B"))
    put("F17-04", "class identity: only Table 4 names; families move as one unit; regime 'unchanged' = P0 for every "
                  "non-ESH ARLP and (b) = (a) there exactly; regime 'other' keeps FAM_05_13 in A",
        f"names {names_ok}; families {fam_ok}; P0 {p0_ok}; (b)=(a) {same_ab}",
        names_ok and fam_ok and p0_ok and same_ab and set(g.loc[reg == "other", "cls_ARLP_05"]) <= {"A"}, "exact")
    base = f17[f17.regime == "baseline"].iloc[0]
    bx, by = float(b.a_fix_A_w_peak), float(b.a_fix_B_w_peak)
    ok = (abs(base.a_alpha_peak_A_w - bx) <= 1e-9 and abs(base.a_alpha_peak_B_w - by) <= 1e-9 and
          f"{bx:.3f}" == f"{t4[('A', 'w', 'peak')]:.3f}" and f"{by:.3f}" == f"{t4[('B', 'w', 'peak')]:.3f}")
    put("F17-05", "baseline point = 02 BASELINE (X0, P0) and equals Table 4 at 3 dp", f"{bx:.3f}, {by:.3f}", ok, "3 dp")
    m = pts[pts.regime == "05_13_to_B"]
    meds = (float(m.b_alpha_peak_A_w.median()), float(m.b_alpha_peak_B_w.median()))
    ok = (f"{meds[0]:.2f}" == nums["H|C3-RND|05toB|rec_A_w_peak"]["text"] and
          f"{meds[1]:.2f}" == nums["H|C3-RND|05toB|rec_B_w_peak"]["text"])
    put("F17-06", "panel (b) medians of the FAM_05_13 to B regime = the reported medians (JSON H|C3-RND|05toB|*)",
        f"{meds[0]:.3f}, {meds[1]:.3f}", ok, "2 dp")
    # fig_reassignment
    dec = s2_decomposition(D, g)
    sh = s2[s2.plotted.astype(bool)]
    mv_dev = max(abs(sh[sh.unit == u].moved_pct.iloc[0] - c5.at["C3-RND", f"moved_{a}_pct"])
                 for u in UNITS for a in UNIT_MEMBERS[u])
    ok = mv_dev < 1e-9 and all(f"{sh[sh.unit == unit_of(a)].moved_pct.iloc[0]:.2f}" ==
                               nums[f"H|C3-RND|moved|{a}"]["text"] for a in ("ARLP_02", "ARLP_03", "ARLP_05"))
    put("S2-01", "panel (a) shares of replicates with a class change = 05 moved_*_pct (C3-RND) for every member ARLP; "
                 "ARLP-02, ARLP-03, FAM_05_13 = JSON", f"{mv_dev:.1e}", ok, "<= 1e-9")
    # "zero": dre is formed in memory before the 12-digit CSV rounding; where the other stratum changed it carries
    # round-off of at most 2.2e-16, so the tolerance is that of G11 (1e-12), not exact 0
    ident, recon, zero = 0.0, 0.0, 0.0
    for u in UNITS:
        a, c0 = UNIT_MEMBERS[u][0], P0[UNIT_MEMBERS[u][0]]
        strat = "same_esh" if c0.startswith("ESH") else "same_nonesh"
        for sk in SK:
            for z in ZK:
                d = dec[(u, sk, z)]
                ident = max(ident, float(np.abs(d["total"] - d["fixed"] - d["reassignment"]).max()))
                a_rec = np.choose(pd.Categorical(g[f"cls_{a}"], categories=list(CL)).codes,
                                  [g[f"a_rec_{c}_{sk}_{z}"].to_numpy(float) for c in CL])
                recon = max(recon, float(np.abs(a_rec - g[f"a_fix_{c0}_{sk}_{z}"].to_numpy(float)
                                                - d["reassignment"]).max()))
                zero = max(zero, float(np.abs(d["reassignment"][g[strat].astype(bool).to_numpy()]).max()))
    put("S2-02", "classes fixed + reassignment = total; reassignment = a_rec(matched class) - a_fix(baseline "
                 "class); reassignment = 0 in every replicate whose stratum partition is unchanged",
        f"identity {ident:.1e}; reconstruction {recon:.1e}; zero {zero:.1e}",
        ident <= 1e-12 and recon <= 1e-9 and zero <= 1e-12, "<= 1e-12 / <= 1e-9 (CSV) / <= 1e-12 (G11)")
    Hh = eh.load_household_counts(eh.find_vault())
    ddev = 0.0
    for c in CL:
        us = [u for u in UNITS if P0[UNIT_MEMBERS[u][0]] == c]
        w = np.array([sum(Hh[m] for m in UNIT_MEMBERS[u]) for u in us])
        for sk in SK:
            for z in ZK:
                Dc = sum(wi * np.abs(dec[(u, sk, z)]["reassignment"]) for wi, u in zip(w, us)) / w.sum()
                ddev = max(ddev, float(np.abs(Dc - g[f"D_{c}_{sk}_{z}"].to_numpy(float)).max()))
    put("S2-03", "the unit reassignment parts, weighted by baseline HHs, reproduce 03 D_c (input of the RC1 column of table_multiplier_deviations)",
        f"{ddev:.1e}", ddev <= 1e-9, "<= 1e-9")
    sdev = 0.0
    first = g.replicate.to_numpy() <= len(g) // 2
    for r in s2.itertuples():
        v = dec[(r.unit, r.season, r.zone)][r.component]
        for stat, fn in (("p05", lambda x: q(x, .05)), ("median", lambda x: q(x, .5)), ("p95", lambda x: q(x, .95)),
                         ("min", np.min), ("max", np.max), ("p95_abs", lambda x: q(np.abs(x), .95)),
                         ("split_p05", lambda x: split_half(x, first, .05)),
                         ("split_p95", lambda x: split_half(x, first, .95)),
                         ("split_p95_abs", lambda x: split_half(np.abs(x), first, .95))):
            sdev = max(sdev, abs(getattr(r, stat) - float(fn(v))))
    put("S2-04", "every statistic of fig_reassignment_source.csv recomputed from 03", f"{sdev:.1e} over {len(s2)} rows",
        sdev <= 1e-11 and len(s2) == len(UNITS) * 6 * 3, "<= 1e-11 (CSV %.12g)")
    nz_dev = max(abs(sh[sh.unit == u].reassignment_nonzero_pct.iloc[0] -
                     100 * float((g[f"flag_mem_{P0[UNIT_MEMBERS[u][0]]}"].astype(bool)).mean())) for u in UNITS)
    put("S2-05", "panel (a) share with a changed multiplier = share of replicates in which the unit's baseline class "
                 "changes membership (03 flag_mem_*)", f"{nz_dev:.1e}", nz_dev < 1e-9, "<= 1e-9")
    # fig_ls1_timing and fig_ls2_one_class: every plotted value against 04, and the statements of the text
    fix = cs[(cs["mode"] == "fix") & cs.quantity.isin(ZK)].set_index(["scenario_id", "class", "season", "quantity"])
    for fig_id, stem, ids, n_exp in (("S3", "fig_ls1_timing_source.csv", LS1_IDS, 60),
                                     ("S4", "fig_ls2_one_class_source.csv", LS2_ONE, 150)):
        f = pd.read_csv(run / stem)
        dev = max(abs(r.dev - fix.at[(r.scenario, r["class"], r.season, r.panel), "dev"]) for _, r in f.iterrows())
        keys = set(zip(f.scenario, f["class"], f.season, f.panel))
        ok = (dev <= 1e-9 and len(f) == n_exp and len(keys) == n_exp and set(f.scenario) == set(ids) and
              set(f.label) == {PUB[ids[0]]})
        put(f"{fig_id}-01", f"{ {'S3': 'fig_ls1_timing', 'S4': 'fig_ls2_one_class'}[fig_id]} source = 04 (mode fix, column dev) for the {len(ids)} runs x 30 multipliers; "
                            f"one row each; label {PUB[ids[0]]}", f"{dev:.1e}; {len(f)} rows", ok, "<= 1e-9 (CSV %.12g)")
    s4 = pd.read_csv(run / "fig_ls2_one_class_source.csv")
    reg = pd.read_csv(run / "00_registry.csv").set_index("scenario_id")
    dm = reg[reg.index.str.startswith("C2-ZONE-DM")]
    ok = (sorted(dm.index) == sorted(LS2_ONE) and
          all(json.loads(p)["delta"] == 0.05 for p in dm.parameters) and
          (s4.perturbed_class == s4.scenario.str[len("C2-ZONE-DM-"):]).all() and
          (s4.own_class.astype(bool) == (s4.perturbed_class == s4["class"])).all())
    put("S4-02", "registry: exactly 5 single-class LS2 runs, all delta = +0.05 (peak -> off-peak), none in the other "
                 "direction; perturbed class and own-class flag of the source match the scenario id",
        f"{len(dm)} runs; deltas {sorted({json.loads(p)['delta'] for p in dm.parameters})}", ok, "exact")
    own, cross = s4[s4.own_class.astype(bool)], s4[~s4.own_class.astype(bool)]
    sgn_ok = ((np.sign(own.dev) == own.panel.map({"off": -1, "sh": 1, "peak": -1})).all() and
              (np.sign(cross.dev) == cross.panel.map({"off": 1, "sh": -1, "peak": 1})).all())
    big = all(abs(gg[gg.own_class.astype(bool)].dev.iloc[0]) > gg[~gg.own_class.astype(bool)].dev.abs().max()
              for _, gg in s4.groupby(["class", "season", "panel"]))
    xab = cross[cross.perturbed_class.isin(["A", "B"])].dev.abs().max()
    xsm = cross[~cross.perturbed_class.isin(["A", "B"])].dev.abs().max()
    ok = (sgn_ok and big and f"{xab:.3f}" == nums["S4|cross_max|AB"]["text"] and xsm < 0.003 and
          nums["S4|sign_pattern"]["text"] == "yes" and nums["S4|own_largest"]["text"] == "yes")
    put("S4-03", "text: own class off-, sh+, peak-, every other class the opposite; own-class run largest in every row and "
                 "panel; cross effects of A/B = JSON S4|cross_max|AB, of C/ESH-1/ESH-2 < 0.003",
        f"signs {sgn_ok}; own largest {big}; {xab:.4f}; {xsm:.4f}", ok, "exact")
    n_in3 = sum(int(_ls_inside_ls3(cs, s).sum()) for s in LS1_IDS)
    ins = pd.concat([_ls_inside_ls3(cs, s).rename(s[len("C2-ZONE-DM-"):]) for s in LS2_ONE], axis=1)
    outs = [(ix, k) for k in ins.columns for ix in ins.index if not ins.at[ix, k]]
    ok = (n_in3 == 60 and len(outs) == int(nums["S4|n_outside_LS3"]["text"]) and all(ix[0] == k for ix, k in outs)
          and nums["S3|n_inside_LS3"]["text"] == "60")
    put("S3S4-04", "text: all 60 LS1 deviations within the LS3 P5-P95 deviation range of 04; the single-class LS2 "
                   "deviations outside it are all own-class (count = JSON S4|n_outside_LS3)",
        f"LS1 inside {n_in3}/60; LS2 outside {len(outs)}, own {sum(ix[0] == k for ix, k in outs)}", ok, "exact")
    rec = cs[(cs["mode"] == "rec") & cs.quantity.isin(ZK)].set_index(["scenario_id", "class", "season", "quantity"])
    sids = list(LS1_IDS + LS2_ONE)
    same = all(c5.at[s, "same_nonesh_pct"] == 100 and c5.at[s, "same_esh_pct"] == 100 for s in sids)
    drec = max(abs(rec.at[ix, "value"] - fix.at[ix, "value"]) for ix in fix.index if ix[0] in sids)
    s3 = pd.read_csv(run / "fig_ls1_timing_source.csv")
    pk = s3[(s3.panel == "peak") & s3["class"].isin(["A", "B"])].pivot_table(index=["scenario", "season"],
                                                                            columns="class", values="dev")
    opp = bool((np.sign(pk["A"]) == -np.sign(pk["B"])).all())
    put("S3S4-05", "notes and text: RC1 leaves the partition unchanged in the 7 runs (05 = 100 / 100) and a_rec = a_fix "
                   "(04); LS1 moves alpha_peak of A and B in opposite directions in both runs and seasons",
        f"partition {same}; |rec - fix| {drec:.1e}; opposite {opp}", same and drec == 0 and opp, "exact")
    # images
    for stem in ("fig_coefficient_deviations", "fig_ab_regimes", "fig_reassignment",
                 "fig_ls1_timing", "fig_ls2_one_class"):
        p, t = Image.open(run / f"{stem}.png"), Image.open(run / f"{stem}.tif")
        ok = p.size == t.size and t.mode == "RGB" and t.info.get("compression") == "tiff_lzw"
        put("IMG", f"{stem}: PNG and TIF exist, same pixel size, TIF RGB LZW", f"{p.size[0]}x{p.size[1]} px", ok, "exact")
    out = pd.DataFrame(rows)
    out.to_csv(run / "figure_checks.csv", index=False, lineterminator="\n")
    return out


def unit_of(a):
    return next(u for u in UNITS if a in UNIT_MEMBERS[u])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--check-figures", action="store_true")
    args = ap.parse_args(argv)
    run = Path(args.run)
    if args.check_figures:
        res = check_figures(run)
        print(res.to_string(index=False))
        bad = int((res.status != "PASS").sum())
        print(f"figure checks: {len(res) - bad} PASS, {bad} FAIL (figure_checks.csv)")
        return 1 if bad else 0

    import eulp_hierarchy as eh
    vault = eh.find_vault()
    D = load(run)
    N = Numbers()
    t4 = table4(vault)
    t9 = make_table9(D, t4, N)
    s4 = make_tableS4(D, t4, N)
    s5a, s5b, s5c = make_tableS5(D, N)
    headline(D, t4, N)
    t9.to_csv(run / "table_multiplier_deviations.csv", index=False, lineterminator="\n")
    s4.to_csv(run / "table_multiplier_ranges.csv", index=False, lineterminator="\n")
    for nm, df in (("a", s5a), ("b", s5b), ("c", s5c)):
        df.to_csv(run / f"table_clustering_stability_{nm}.csv", index=False, lineterminator="\n")
    (run / "table_multiplier_deviations.md").write_text(md(t9) + "\n", encoding="utf-8", newline="\n")
    (run / "table_multiplier_ranges.md").write_text(md(s4) + "\n", encoding="utf-8", newline="\n")
    (run / "table_clustering_stability.md").write_text("Panel (a)\n\n" + md(s5a) + "\n\nPanel (b)\n\n" + md(s5b) + "\n\nPanel (c)\n\n" +
                                    md(s5c) + "\n", encoding="utf-8", newline="\n")
    fig16(D, run)
    made_s2 = figS2_needed(D)
    N.put("H|figS2_produced", "yes" if made_s2 else "no", None, "drawing condition")
    # post-run visualisation (fig_reassignment, fig_ab_regimes): their numbers follow those of the tables in the JSON
    if made_s2:
        figS2(D, run, N)
    fig17(D, run, N)
    label_map(D, run, N)                                                   # publication labels (PUB)
    # post-run visualisation (fig_ls1_timing, fig_ls2_one_class): their numbers follow all others in the JSON
    figS3(D, run, N)
    figS4(D, run, N)
    (run / "reported_numbers.json").write_text(json.dumps(N.d, indent=1, ensure_ascii=False), encoding="utf-8",
                                                  newline="\n")
    print(f"written: three tables, five figures{' (and the reassignment figure)' if made_s2 else ''}, "
          f"reported_numbers.json ({len(N.d)} numbers) in {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
