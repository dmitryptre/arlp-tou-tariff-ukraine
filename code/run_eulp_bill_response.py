"""Shift effect and combined effect within the classes: the runner.

    python -X utf8 run_eulp_bill_response.py [--out DIR]

Reads the outputs of run_eulp_bill_distribution.py (01 monthly zone energies, 02 units and weights, 04 statistics,
07 volume factors, the histogram figure), the P3 rates and four files of the PEED sensitivity layer of the notebook
(the scenario registry, the class bill impacts, the zone changes delta and the arrays of the class loads, S0 and the
zone of each hour). Writes into an empty folder (default outputs/ARLP_Clustering/eulp_bill_response/; with a failed
gate into outputs/ARLP_Clustering/_unchecked/...): 01_response_multipliers.csv, _response_units.csv,
03_response_summary.csv, 04_response_histogram_bins.csv, 05_response_class_identity.csv, 06_figure_source.csv,
07_qa_gates.csv, README.md and six figures (passive, shift, combined x cohorts both / F->P3). Every gate (R0b to R8)
is printed and written to 07_qa_gates.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import eulp_bill_distribution as bd
import eulp_bill_response as br
import eulp_hierarchy as eh
import make_bill_change_histograms as mh
import numpy as np
import pandas as pd
from _common import TOU_PERIODS_0BASED
from run_eulp_bill_distribution import Gates

OUT_REL = Path("outputs/ARLP_Clustering/eulp_bill_response")
UNCHECKED_REL = Path("outputs/ARLP_Clustering/_unchecked/eulp_bill_response")
DIST_REL = Path("outputs/ARLP_Clustering/eulp_bill_distribution")
SENS_REL = Path("outputs/ARLP_Clustering/notebook_only/layer13_peed_sensitivity")
RATES_REL = Path("outputs/ARLP_Clustering/notebook_only/layer13_peed_dr/proposed_rates_per_class_season_period.csv")
TOL = {"R1": 1e-9, "R2": 1e-6, "R3": 1e-9, "R4": 1e-9, "R5": 1e-9, "R6": 1e-9, "R8": 1e-12}
# (effect, cohorts): "both" = F->P3 and E2->P3 weighted 1 - S0 and S0; "F->P3" = the flat-tariff cohort only.
VARIANTS = (("passive", "both"), ("shift", "both"), ("combined", "both"),
            ("passive", "F->P3"), ("shift", "F->P3"), ("combined", "F->P3"))
FIG = {("passive", "both"): "fig_passive_both_cohorts_histograms.png",
       ("shift", "both"): "fig_shift_both_cohorts_histograms.png",
       ("combined", "both"): "fig_combined_both_cohorts_histograms.png",
       ("passive", "F->P3"): "fig_passive_F_to_P3_histograms.png",
       ("shift", "F->P3"): "fig_shift_effect_F_to_P3_histograms.png",
       ("combined", "F->P3"): "fig_combined_effect_F_to_P3_histograms.png"}
README = """# eulp_bill_response — shift effect and combined effect within the classes

Written by `code/run_eulp_bill_response.py` from the within-class outputs
(`../eulp_bill_distribution/`) and the PEED sensitivity outputs of the notebook
(`../notebook_only/layer13_peed_sensitivity/`).

**Method.** Each class-EULP pair (case T) responds to the proposed 3z tariff (P3) with its class's zone multipliers
max(1 + δ, 0) of the PEED sensitivity layer (`peed_validity_diagnostics.csv`), per pathway (F→P3, E2→P3), scenario and
season; under R each dwelling's seasonal energy is restored. Bills by the rules of the within-class runner (case T);
each ESH winter month keeps the dwelling's baseline 2,000-kWh status after the response, as the notebook's class bills
keep a fixed effective winter base (the variant in which the status follows the response energy is reported as a
sensitivity: `_dyn` columns, scope `pooled, ESH status follows the response (sensitivity)`). Changes in % of the
cohort's baseline bill (F or E2), **positive = a bill increase** (the bill-impact table of the article uses the
opposite sign): shift = P3 bill after − before the response; combined = P3 bill after the response − baseline bill
(passive + shift); passive = P3 bill − baseline bill, before any response. The shift and combined figures pool, per
panel, the 24 PEED scenarios (the valley-filling twins equal their proportional scenarios in bills). Two cohort
variants (`cohorts` column): **both** = F→P3 and E2→P3, each observation weighted W × (1 − S0 or S0) / 24 (the passive
change once, weighted W × (1 − S0 or S0)); **F→P3** = the flat-tariff cohort only (the passive variant reproduces the
histogram figure of `../eulp_bill_distribution/`). Figures: `fig_passive_both_cohorts_histograms.png`,
`fig_shift_both_cohorts_histograms.png` and `fig_combined_both_cohorts_histograms.png` (both cohorts; the figures of
the article), `fig_passive_F_to_P3_histograms.png`, `fig_shift_effect_F_to_P3_histograms.png`,
`fig_combined_effect_F_to_P3_histograms.png` (F→P3 only).

| File | Content |
|---|---|
| `01_response_multipliers.csv` | per matrix × λ × pathway × class × season × zone: δ, the floored multiplier a |
| `_response_units.csv` | per proportional scenario × pathway × unit: bills before/after, passive, shift, combined, η |
| `03_response_summary.csv` | effect × cohorts × scope (`pooled` or a scenario id) × panel: the distribution statistics (`n_units` = pairs, `n_obs`), and `mean_ge0_pct` / `mean_lt0_pct`, the weighted mean change of the observations with a change ≥0 / <0 (empty group: blank; the two mean rows of the figure tables) |
| `04_response_histogram_bins.csv` | effect × cohorts × panel: the pooled histograms (the bin rule and the y-axis rule of the within-class runner) |
| `05_response_class_identity.csv` | the W-weighted unit sums against the notebook's class bill impacts, per scenario × pathway × class |
| `06_figure_source.csv` | every data string of the six figures with its value, rule and rendered text (`effect`, `cohorts`) |
| `07_qa_gates.csv` | the gates R0b to R8 |
| the six `*.png` | see Method |
"""


def conditional_means(x, w):
    """The two mean rows of the figure tables: the weighted mean change of the observations with a change >= 0 and of
    those with a change < 0 (the weights of the shares; NaN for an empty group)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    out = {}
    for col, m in (("mean_ge0_pct", x >= 0), ("mean_lt0_pct", x < 0)):
        out[col] = float((w[m] * x[m]).sum() / w[m].sum()) if w[m].sum() > 0 else np.nan
    return out


def main(out=None):
    t0 = time.time()
    vault = eh.find_vault()
    g = Gates()
    dist, sens = vault / DIST_REL, vault / SENS_REL

    # ── units and energies ──────────────────────────────────────────────────────────────────────────────────────
    u = pd.read_csv(dist / "02_class_eulp_units.csv")
    u = u[u.case == "T"].reset_index(drop=True)
    e01 = pd.read_csv(dist / "01_eulp_monthly_zone_kwh.csv").set_index("eulp_id")
    cols = [f"m{m:02d}_{z}_kWh" for m in eh.MONTHS for z in eh.ZONES]
    E = e01.loc[u.eulp_id, cols].to_numpy(float).reshape(len(u), 12, 3)
    classes, eid, W = u["class"].to_numpy(), u.eulp_id.to_numpy(), u.W_kHH.to_numpy(float)
    esh = br.is_esh_array(classes)
    g.check("inputs", "R0b", "unit annual energies of file 02 = the sum of file 01's monthly zone energies",
            float(np.max(np.abs(E.sum(axis=(1, 2)) - u.annual_kWh.to_numpy()))), "1e-6 kWh",
            bool(np.allclose(E.sum(axis=(1, 2)), u.annual_kWh.to_numpy(), rtol=0, atol=1e-6)))
    _, rates = bd.load_p3_rates(vault / RATES_REL)
    reg = pd.read_csv(sens / "peed_scenario_registry.csv")
    reg = reg[reg.role == "scenario"].reset_index(drop=True)
    assert len(reg) == br.N_SCENARIOS
    diag = pd.read_csv(sens / "peed_validity_diagnostics.csv")
    mult = br.load_multipliers(diag)
    Z = np.load(sens / "_peed_sensitivity_arrays.npz")
    S0 = float(Z["S0"])
    period = [str(x) for x in Z["period"]]

    # 01: the multipliers
    rows1 = []
    for (mx, sc, pw, c, s), a in sorted(mult.items()):
        for z, av in zip(eh.ZONES, a):
            d = diag[(diag.peed_matrix == mx) & (diag.peed_scale == sc) & (diag.pathway == pw) & (diag["class"] == c)
                     & (diag.season == s)].iloc[0]
            rows1.append({"peed_matrix": mx, "peed_scale": sc, "pathway": pw, "class": c, "season": s, "zone": z,
                          "delta": float(d[f"delta_{br.ZONE_SENS[z]}"]), "a_floored_multiplier": float(av),
                          "floored": bool(av == 0.0)})
    o1 = pd.DataFrame(rows1)

    # R1: the zone of each hour and the multipliers reproduce the notebook's class loads (NR, proportional)
    zone_of_hour = {h: z for z, hs in (("off-peak", TOU_PERIODS_0BASED["off"]), ("shoulder", TOU_PERIODS_0BASED["semi"]),
                                       ("peak", TOU_PERIODS_0BASED["peak"])) for h in hs}
    g.check("R1", "R1a", "zone of each hour: `period` of the PEED sensitivity arrays = the zone hours of _common", "24 hours", "exact",
            period == [zone_of_hour[h] for h in range(24)])
    zi = np.array([eh.ZONES.index({"off-peak": "off", "shoulder": "shoulder", "peak": "peak"}[p]) for p in period])
    r1 = 0.0
    for sc in reg[(reg.energy_treatment == "NR") & (reg.allocation == "proportional")].itertuples():
        for pw in br.PATHWAYS:
            for c in eh.CLASS_ORDER:
                for s in eh.SEASONS:
                    l0 = Z[f"L0__{c}__{s}"]
                    l1 = Z[f"{sc.scenario_id}__{br.NPZ_PATHWAY[pw]}__{c}__{s}"]
                    exp = l0 * mult[(sc.peed_matrix, float(sc.peed_scale), pw, c, s)][zi]
                    r1 = max(r1, float(np.max(np.abs(l1 - exp) / np.maximum(np.abs(l0), 1e-300))))
    g.check("R1", "R1b", "notebook class loads (NR, proportional) = L0 x max(1 + delta, 0) by hour", r1,
            TOL["R1"], r1 <= TOL["R1"])
    # R4 (premise): valley filling keeps every zone energy of the class day
    r4 = 0.0
    for sc in reg[reg.allocation == "valley_fill"].itertuples():
        twin = sc.scenario_id[:-3] + "-P"
        for pw in br.PATHWAYS:
            for c in eh.CLASS_ORDER:
                for s in eh.SEASONS:
                    lv = Z[f"{sc.scenario_id}__{br.NPZ_PATHWAY[pw]}__{c}__{s}"]
                    lp = Z[f"{twin}__{br.NPZ_PATHWAY[pw]}__{c}__{s}"]
                    for k in range(3):
                        ev, ep = lv[zi == k].sum(), lp[zi == k].sum()
                        r4 = max(r4, abs(ev - ep) / max(abs(ep), 1e-300))
    g.check("R4", "R4", "valley-filling class days keep the zone energies of their proportional twin "
            "(so the bills of a twin are those of its proportional scenario)", r4, TOL["R4"], r4 <= TOL["R4"])

    # ── the responses (12 proportional scenarios x 2 pathways) ─────────────────────────────────────────────────
    b0 = br.bills(E, classes, rates, esh)
    res, r3, r5 = {}, 0.0, 0.0
    for sc in reg[reg.allocation == "proportional"].itertuples():
        for pw in br.PATHWAYS:
            A = br.unit_multipliers(classes, mult, sc.peed_matrix, float(sc.peed_scale), pw)
            E1, eta = br.respond(E, A, sc.energy_treatment)
            b1 = br.bills(E1, classes, rates, esh, theta=b0["theta"])      # primary: the baseline ESH status kept
            b1d = br.bills(E1, classes, rates, esh)                         # sensitivity: the status follows E1
            eff = br.effects(b0, b1, br.COHORT[pw])
            effd = br.effects(b0, b1d, br.COHORT[pw])
            res[(sc.scenario_id, pw)] = {"b1": b1, "b1d": b1d, "eta": eta, "E1kWh": E1.sum(axis=(1, 2)), **eff,
                                         "shift_dyn": effd["shift"], "combined_dyn": effd["combined"],
                                         "status_changed_months": (b1d["theta"] != b0["theta"]).sum(axis=1)}
            if sc.energy_treatment == "R":
                for k in (0, 1):
                    m = bd.SEASON_IDX == k
                    r3 = max(r3, float(np.max(np.abs(E1[:, m, :].sum(axis=(1, 2)) / E[:, m, :].sum(axis=(1, 2)) - 1))))
            if pw == "F->P3":
                r5 = max(r5, float(np.max(np.abs(eff["combined"] - eff["shift"] - u.dbill_pct.to_numpy()))))
    g.check("R3", "R3", "under R every unit's seasonal energy equals its baseline", r3, TOL["R3"], r3 <= TOL["R3"])
    g.check("R5", "R5", "F cohort: combined - shift = the passive change (file 02 dbill_pct)", r5,
            TOL["R5"], r5 <= TOL["R5"])

    # R2: the W-weighted unit sums against the notebook's class bill impacts (savings positive there)
    t4 = pd.read_csv(sens / "peed_sensitivity_table5_bill_impacts.csv")
    q = pd.read_csv(dist / "07_aggregation_identity.csv").groupby("class").q_factor.first().to_dict()
    rows5, r2_gate, r2_rep = [], 0.0, {"R, classes A-C": 0.0, "R, ESH": 0.0, "NR, ESH": 0.0,
                                       "NR, ESH, status follows the response (sensitivity)": 0.0}
    for (sid, pw), r in res.items():
        sc = reg.set_index("scenario_id").loc[sid]
        for c in eh.CLASS_ORDER:
            m = classes == c
            k = t4[(t4.scenario_id == sid) & (t4["class"] == c) & (t4.transition == pw)].iloc[0]
            base = q[c] * float((W[m] * r["base"][m]).sum()) / 1e3
            shift = q[c] * float((W[m] * (r["b1"]["P3"][m] - b0["P3"][m])).sum()) / 1e3
            passive = q[c] * float((W[m] * (b0["P3"][m] - r["base"][m])).sum()) / 1e3
            d_shift = abs(shift - (-k.response_MUAH)) / k.base_bill_MUAH
            d_pass = abs(passive - (-k.passive_MUAH)) / k.base_bill_MUAH
            d_base = abs(base - k.base_bill_MUAH) / k.base_bill_MUAH
            shift_d = q[c] * float((W[m] * (r["b1d"]["P3"][m] - b0["P3"][m])).sum()) / 1e3
            gated = sc.energy_treatment == "NR" and c not in eh.ESH_CLASSES
            if gated:
                r2_gate = max(r2_gate, d_shift, d_pass, d_base)
            else:
                key = f"{sc.energy_treatment}, " + ("ESH" if c in eh.ESH_CLASSES else "classes A-C")
                r2_rep[key] = max(r2_rep[key], d_shift)
            if sc.energy_treatment == "NR" and c in eh.ESH_CLASSES:
                k2 = "NR, ESH, status follows the response (sensitivity)"
                r2_rep[k2] = max(r2_rep[k2], abs(shift_d - (-k.response_MUAH)) / k.base_bill_MUAH)
            rows5.append({"scenario_id": sid, "pathway": pw, "class": c, "energy_treatment": sc.energy_treatment,
                          "gated": gated, "q_factor": q[c], "base_bill_units_table_basis_MUAH": base,
                          "base_bill_class_MUAH": k.base_bill_MUAH, "passive_units_MUAH": passive,
                          "passive_class_MUAH_sign_converted": -k.passive_MUAH, "shift_units_MUAH": shift,
                          "shift_class_MUAH_sign_converted": -k.response_MUAH,
                          "shift_diff_pct_of_base": 100 * (shift + k.response_MUAH) / k.base_bill_MUAH,
                          "shift_class_pct_sign_converted": -k.response_pct,
                          "shift_units_pct_of_base": 100 * shift / base,
                          "shift_units_status_follows_response_pct_of_base": 100 * shift_d / base})
    o5 = pd.DataFrame(rows5)
    g.check("R2", "R2", "classes A, B, C under NR: q_c x sum W (base, passive, shift) = the notebook's class bill impacts "
            "(relative to the base bill)", r2_gate, TOL["R2"], r2_gate <= TOL["R2"])
    for key, v in r2_rep.items():
        g.add("R2", f"R2r ({key})", "largest |q_c sum W shift - class-level shift| relative to the base bill "
              "(R: per-dwelling eta; ESH: monthly status vs the zonal effective winter base)", v, "report", "REPORT")

    # ── observations: 24 scenarios x 2 cohorts x the units ─────────────────────────────────────────────────────
    obs_keys = ("passive",) + br.EFFECTS + tuple(f"{e}_dyn" for e in br.EFFECTS)   # _dyn: ESH status follows E1
    obs = {e: [] for e in obs_keys}
    ow, ocls, oeid, osid, oco = [], [], [], [], []
    for sc in reg.itertuples():
        twin = sc.scenario_id if sc.allocation == "proportional" else sc.scenario_id[:-3] + "-P"
        for pw in br.PATHWAYS:
            r = res[(twin, pw)]
            for e in obs_keys:
                obs[e].append(r[e])
            ow.append(W * ((1.0 - S0) if pw == "F->P3" else S0) / br.N_SCENARIOS)
            ocls.append(classes)
            oeid.append(eid)
            osid.append(np.full(len(W), sc.scenario_id))
            oco.append(np.full(len(W), br.COHORT[pw]))
    obs = {e: np.concatenate(v) for e, v in obs.items()}
    ow, ocls, oeid, osid, oco = map(np.concatenate, (ow, ocls, oeid, osid, oco))
    pan = {p: (np.ones(len(ow), bool) if p == "All" else ocls == p) for p in bd.PANELS}
    upan = {p: (np.ones(len(W), bool) if p == "All" else classes == p) for p in bd.PANELS}
    # cohort variants: "both" (weights W (1 - S0 or S0) / 24) and "F->P3" (the F rows, weights W / 24)
    cmask = {"both": np.ones(len(ow), bool), "F->P3": oco == "F"}
    wv = {"both": ow, "F->P3": np.where(oco == "F", ow / (1.0 - S0), 0.0)}
    r6 = max(abs(wv[cv][pan[p] & cmask[cv]].sum() / W[upan[p]].sum() - 1) for p in bd.PANELS for cv in cmask)
    g.check("R6", "R6", "the observation weights of each panel and cohort variant sum to its unit weight", r6,
            TOL["R6"], r6 <= TOL["R6"])
    # R5b: the E2 cohort's passive change = the E2->P3 passive change of file 02 (d_E2_P3_pct)
    r5b = float(np.max(np.abs(res[(reg.scenario_id.iloc[0], "E2->P3")]["passive"] - u.d_E2_P3_pct.to_numpy())))
    g.check("R5", "R5b", "E2 cohort: passive = the E2->P3 passive change of file 02 (d_E2_P3_pct)", r5b,
            TOL["R5"], r5b <= TOL["R5"])

    # unit file
    rows2 = []
    for (sid, pw), r in res.items():
        rows2.append(pd.DataFrame({
            "scenario_id": sid, "vf_twin": sid[:-2] + "-VF", "pathway": pw, "cohort": br.COHORT[pw], "class": classes,
            "eulp_id": eid, "W_kHH": W, "cohort_share": (1.0 - S0) if pw == "F->P3" else S0,
            "base_bill_UAH": r["base"], "bill_P3_before_UAH": b0["P3"], "bill_P3_after_UAH": r["b1"]["P3"],
            "passive_pct": r["passive"], "shift_pct": r["shift"], "combined_pct": r["combined"],
            "eta_winter": r["eta"][:, 0], "eta_summer": r["eta"][:, 1], "annual_kWh_before": b0["kWh"],
            "annual_kWh_after": r["E1kWh"], "esh_winter_months_above_2000_before": b0["winter_months_above_2000"],
            "esh_winter_months_above_2000_if_status_follows": r["b1d"]["winter_months_above_2000"],
            "esh_months_status_would_change": r["status_changed_months"],
            "shift_pct_status_follows_response": r["shift_dyn"],
            "combined_pct_status_follows_response": r["combined_dyn"]}))
    o2 = pd.concat(rows2, ignore_index=True)

    # ── statistics (pooled and per scenario) ───────────────────────────────────────────────────────────────────
    first = osid == reg.scenario_id.iloc[0]          # the passive change does not depend on the scenario: one copy

    def vmask(e, p, cv, scope="pooled"):
        m = pan[p] & cmask[cv]
        if e.startswith("passive"):
            return m & first
        return m if scope == "pooled" else m & (osid == scope)

    def stats(e, p, cv, scope="pooled"):
        mask = vmask(e, p, cv, scope)
        x, w = obs[e][mask], wv[cv][mask]
        st = bd.panel_stats(x, w, ocls[mask], oeid[mask])
        pq = bd.weighted_quantile(x, w, (0.005, 0.995), (np.array([eh.CLASS_ORDER.index(c) for c in ocls[mask]]),
                                                         oeid[mask]))
        wu = W[upan[p]]
        return {"effect": e, "cohorts": cv, "scope": scope, "case": "T", "weighting": "weighted",
                "weight_basis": "H_normalized", "panel": p, **st, **conditional_means(x, w),
                "n_obs": st["n_units"], "n_units": int(upan[p].sum()),
                "n_scenarios": 1 if (e.startswith("passive") or scope != "pooled") else br.N_SCENARIOS, "S0": S0,
                "largest_unit_weight_pct": 100.0 * float(wu.max()) / float(wu.sum()),
                "largest_obs_weight_pct": st["largest_unit_weight_pct"],
                "p00_5_pct": float(pq[0]), "p99_5_pct": float(pq[1])}

    o3 = []
    for e, cv in VARIANTS:
        for p in bd.PANELS:
            o3.append(stats(e, p, cv))
            if e != "passive":
                for sid in reg.scenario_id:
                    o3.append(stats(e, p, cv, sid))
    for e in br.EFFECTS:                                                # the sensitivity, both cohorts, pooled only
        for p in bd.PANELS:
            o3.append({**stats(f"{e}_dyn", p, "both"), "effect": e,
                       "scope": "pooled, ESH status follows the response (sensitivity)"})
    o3 = pd.DataFrame(o3)
    # R8: the F->P3 passive variant reproduces the statistics of file 04 (case T, weighted)
    o4_dist = pd.read_csv(dist / "04_distribution_summary.csv")
    o4_dist = o4_dist[(o4_dist.case == "T") & (o4_dist.weighting == "weighted")
                      & (o4_dist.weight_basis == "H_normalized")]
    o4_dist = o4_dist.set_index("panel")
    fp = o3[(o3.effect == "passive") & (o3.cohorts == "F->P3") & (o3.scope == "pooled")].set_index("panel")
    r8 = max(abs(fp.at[p, c] - o4_dist.at[p, c]) for p in bd.PANELS
             for c in ("p05_pct", "p50_pct", "p95_pct", "share_ge0_pct", "share_lt0_pct", "largest_unit_weight_pct"))
    g.check("R8", "R8a", "F->P3 passive variant = the statistics of file 04 (case T): P5, median, P95, "
            "shares, largest weight", r8, TOL["R8"], r8 <= TOL["R8"])
    # R7b: the two mean rows add up, with the shares, to the overall weighted mean; signs; NaN only when
    # the group's share is 0
    ge, lt = o3.mean_ge0_pct, o3.mean_lt0_pct
    r7b = float((o3.share_ge0_pct * ge.fillna(0) + o3.share_lt0_pct * lt.fillna(0) - 100 * o3.mean_hh_pct).abs().max()
                / 100)
    signs = bool((ge.dropna() >= 0).all() and (lt.dropna() < 0).all()
                 and (ge.isna() == (o3.share_ge0_pct == 0)).all() and (lt.isna() == (o3.share_lt0_pct == 0)).all())
    g.check("R7", "R7b", "share_ge0 x mean_ge0 + share_lt0 x mean_lt0 = 100 x mean_hh_pct; mean_ge0 >= 0, "
            "mean_lt0 < 0; NaN only for an empty group", f"{r7b:.1e}; signs {signs}", "1e-9", r7b <= 1e-9 and signs)

    # ── histograms, figures ────────────────────────────────────────────────────────────────────────────────────
    o4, o6, figs, d17, rules = [], [], {}, 0.0, {}
    for e, cv in VARIANTS:
        s_e = o3[(o3.effect == e) & (o3.cohorts == cv) & (o3.scope == "pooled")].set_index("panel")
        rule = rules[f"{e}|{cv}"] = bd.bin_rule(s_e.at["All", "p00_5_pct"], s_e.at["All", "p99_5_pct"],
                                                 [s_e.at[p, "p00_5_pct"] for p in bd.PANELS],
                                                 [s_e.at[p, "p99_5_pct"] for p in bd.PANELS])
        for p in bd.PANELS:
            m = vmask(e, p, cv)
            h = bd.histogram(obs[e][m], wv[cv][m], rule)
            d17 = max(d17, abs(h.share_pct.sum() - 100), abs(h[h.right_pct <= 0].share_pct.sum() - s_e.at[p, "share_lt0_pct"]),
                      abs(h[h.left_pct >= 0].share_pct.sum() - s_e.at[p, "share_ge0_pct"]))
            h.insert(0, "weighting", "weighted")
            h.insert(0, "panel", p)
            h.insert(0, "case", "T")
            h.insert(0, "cohorts", cv)
            h.insert(0, "effect", e)
            for key in ("width_pp", "rule_R_pp", "range_L_pct", "range_U_pct"):
                h[key] = rule[key]
            ya = bd.yaxis_rule(h.share_pct.max())
            h["tallest_bar_pct"], h["tick_ceiling_pct"] = ya["tallest_bar_pct"], ya["tick_ceiling_pct"]
            h["y_top_pct"], h["bar_label"] = ya["y_top_pct"], ya["bar_label"]
            o4.append(h)
    o4 = pd.concat(o4, ignore_index=True)
    g.check("R7", "R7a", "histogram shares sum to 100 % and split at 0 as the summary's shares", d17, "1e-9",
            d17 <= 1e-9)

    def title_x24(p, row):
        return f": {bd.format_count(int(row['n_units']))} " + ("pairs × 24" if p == "All" else "EULPs × 24")

    for e, cv in VARIANTS:
        sel = (o3.effect == e) & (o3.cohorts == cv) & (o3.scope == "pooled")
        s_e = o3[sel].drop(columns=["effect", "cohorts"])
        b_e = o4[(o4.effect == e) & (o4.cohorts == cv)].drop(columns=["effect", "cohorts"]).copy()
        kw = {"x_tick_step": br.tick_step(rules[f"{e}|{cv}"]), "title_fmt": None if e == "passive" else title_x24,
              "note": False}
        # the table layout of the figures; R8b compares the layout of the within-class runner
        fig, disp, info = mh.build_figure(s_e, b_e, layout="table", **kw)
        if (e, cv) == ("passive", "F->P3"):
            fig_dist = mh.build_figure(s_e, b_e, **kw)[0]
        disp, d6, d10, detail = mh.check_rendered(fig, disp)
        for p, d in info["panels"].items():
            if d.get("raised"):
                o4.loc[(o4.effect == e) & (o4.cohorts == cv) & (o4.panel == p), "y_top_pct"] = d["y_top_final_pct"]
        figs[(e, cv)] = fig
        o6.append(disp.assign(effect=e, cohorts=cv))
        g.check("R7", f"R7-{e}-{cv}", f"{FIG[(e, cv)]}: rendered strings = source (D6); geometry (D10)", detail,
                "exact / geometric", bool(d6 and d10))
    o6 = pd.concat(o6, ignore_index=True)
    import io

    from PIL import Image
    buf = io.BytesIO()
    fig_dist.savefig(buf, format="png", dpi=mh.DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    pa = np.asarray(Image.open(buf).convert("RGB"))
    pb = np.asarray(Image.open(dist / "fig_bill_change_histograms.png").convert("RGB"))
    g.add("R8", "R8b", "the F->P3 passive figure, drawn in the layout of the within-class runner, is pixel-identical to its histogram figure",
          bool(pa.shape == pb.shape and np.array_equal(pa, pb)), "report", "REPORT")

    # ── write ──────────────────────────────────────────────────────────────────────────────────────────────────
    target = Path(out) if out else vault / (UNCHECKED_REL if g.failed else OUT_REL)
    if g.failed and out:
        target = Path(out) / "_unchecked"
    target.mkdir(parents=True, exist_ok=True)
    assert not any(target.iterdir()), f"{target} is not empty: the run writes into an empty folder"
    for name, df in (("01_response_multipliers.csv", o1), ("_response_units.csv", o2), ("03_response_summary.csv", o3),
                     ("04_response_histogram_bins.csv", o4), ("05_response_class_identity.csv", o5),
                     ("06_figure_source.csv", o6), ("07_qa_gates.csv", pd.DataFrame(g.rows))):
        df.to_csv(target / name, index=False, lineterminator="\n", encoding="utf-8")
    for e, fig in figs.items():
        fig.savefig(target / FIG[e], dpi=mh.DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    (target / "README.md").write_text(README, encoding="utf-8", newline="\n")

    print(f"done in {time.time() - t0:.0f} s; {len(g.rows)} checks; FAIL: {', '.join(g.failed) or 'none'}; -> {target}")
    return 1 if g.failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output folder (default: outputs/ARLP_Clustering/eulp_bill_response)")
    sys.exit(main(ap.parse_args().out))
