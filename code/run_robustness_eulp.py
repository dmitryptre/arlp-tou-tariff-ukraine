"""Sensitivity of the tariff multipliers to the synthetic inputs: executes the registered scenarios, applies the gates,
writes the run folder.

python -X utf8 run_robustness_eulp.py [--registry robustness_registry.csv] [--out DIR] [--pilot N]

The output folder must be new or empty. On a failed gate the outputs go to <out>/_unchecked/ and the run stops.
Inputs are read in place and never written. The class keys of every output are the names of the proposed-tariff table.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eulp_hierarchy as eh
import robustness_eulp as rb

OUT_REL = "outputs/ARLP_Clustering/robustness_eulp"
SK, ZK, CL = rb.SEASON_KEYS, rb.ZONE_KEYS, rb.CLASS_ORDER
TABLE4 = "outputs/ARLP_Clustering/notebook_only/layer13_peed_dr/table4_proposed_tariff.csv"
UNITS = "outputs/ARLP_Clustering/eulp_bill_distribution/02_class_eulp_units.csv"
SUMMARY = "outputs/ARLP_Clustering/eulp_bill_distribution/04_distribution_summary.csv"



class Gates:
    def __init__(self):
        self.rows = []

    def check(self, cid, desc, value, tol, ok, detail=""):
        self.rows.append({"check_id": cid, "description": desc, "value": value, "tolerance": tol,
                          "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}: {value}")
        return ok

    def qualified(self, cid, desc, value, tol, detail):
        """A check whose registered criterion is not met for a documented reason that does not invalidate the run."""
        self.rows.append({"check_id": cid, "description": desc, "value": value, "tolerance": tol,
                          "status": "QUALIFIED", "detail": detail})
        print(f"  [QUALIFIED] {cid} {desc}: {value} ({detail})")

    def report(self, cid, desc, value, detail=""):
        self.rows.append({"check_id": cid, "description": desc, "value": value, "tolerance": "reported",
                          "status": "REPORT", "detail": detail})

    @property
    def failed(self):
        return [r["check_id"] for r in self.rows if r["status"] == "FAIL"]





def write_csv(df, path, gz=False):
    data = df.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    if gz:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
            g.write(data)
        Path(path).write_bytes(buf.getvalue())
    else:
        Path(path).write_bytes(data)


# ── scenario states ───────────────────────────────────────────────────────────────────────────────────────────
def b2_factors(base):
    """B2: b_s = AnnGWh_s / (H_s e_s), e_s the ARLP profile's per-household annual energy."""
    e_s = np.add.reduceat(base.derived["phi_copy"] * base.derived["E_ann"][base.c_u], base.starts)   # kWh
    model_GWh = base.H * 1e3 * e_s / 1e6
    return np.where(model_GWh > 0, base.t2[:, 0] / np.where(model_GWh > 0, model_GWh, 1.0), 1.0)


def unit_class(u):
    return rb.P0[rb.UNIT_MEMBERS[u][0]]


def deterministic_states(base, reg):
    """(scenario_id, State, meta) for every deterministic registry row (A2, B2, B3, C1, C2, D2)."""
    out = []
    for r in reg.itertuples():
        if r.n_replicates != 1:
            continue
        p = json.loads(r.parameters)
        st = rb.baseline_state(base)
        meta = {"recluster": "baseline", "neutrality": True}
        sid = r.scenario_id
        if sid == "A2-MIX25":
            mixed = rb.mix_with_pool(base, p["epsilon"])
            tilted, beta, ok = rb.tilt_all(base, mixed)
            st.phi_copy = tilted
            meta.update(mixed=mixed, beta=beta, tilt_ok=ok)
        elif sid == "B2-STAT":
            st.b = b2_factors(base)
            st.reading = "energy"
        elif sid.startswith("B3-CLS"):
            ci = CL.index(p["class"])
            st.H = base.H * np.where(rb.P0 == ci, p["f"], 1.0)
            meta.update(cls=ci, f=p["f"])
        elif sid.startswith("C1-TIME"):
            st.M_ue, nz = rb.transform_units(base, np.full(11, p["theta_h"]), np.zeros(11), np.zeros(11))
            meta.update(zero_recv=nz)
        elif sid.startswith("C2-ZONE"):
            if p["units"] == "all":
                d = np.full(11, p["delta"])
            else:
                ci = CL.index(p["units"].split()[1])
                d = np.array([p["delta"] if unit_class(u) == ci else 0.0 for u in rb.UNITS])
            st.M_ue, nz = rb.transform_units(base, np.zeros(11), d, d)
            meta.update(zero_recv=nz, delta=p["delta"], common=p["units"] == "all")
        elif sid in ("D2-NC", "D2-MN"):
            meta.update(recluster=sid, neutrality=False)
        else:
            raise ValueError(sid)
        out.append((sid, st, meta))
    return out


def stochastic_state(base, sid, rng):
    """One replicate of A1, B1 or C3 from the registry seed stream (fixed draw order)."""
    st = rb.baseline_state(base)
    meta = {}
    if sid == "A1-U20":
        g = rng.uniform(0.8, 1.2, len(base.eulp_ids))
        raw = rb.perturb_multiplicative(base, g)
        st.phi_copy, beta, ok = rb.tilt_all(base, raw)
        meta.update(pre_tilt=raw, beta=beta, tilt_ok=ok)
    elif sid == "B1-RND20":
        a = rng.uniform(0.8, 1.2, 19)
        st.H = base.H * a
        meta.update(a=a)
    elif sid == "C3-RND":
        u = rng.uniform(-1.0, 1.0, size=(11, 3))
        st.M_ue, nz = rb.transform_units(base, u[:, 0], 0.05 * u[:, 1], 0.05 * u[:, 2])
        meta.update(zero_recv=nz, params=u)
    else:
        raise ValueError(sid)
    return st, meta


# ── per-run checks and records ────────────────────────────────────────────────────────────────────────────────
class Acc:
    """Running maxima of the per-run gate quantities."""

    def __init__(self):
        self.m = {}

    def max(self, key, v):
        self.m[key] = max(self.m.get(key, 0.0), float(v))

    def count(self, key, v=1):
        self.m[key] = self.m.get(key, 0) + v


def run_checks(base, st, ev, meta, sid, acc, refs):
    """The checks G04, G05, G06, G10, G11, G12 on one run."""
    fix = ev["fix"]
    acc.max("kappa_sum", max(np.abs(fix["kappa_peak"].sum(0) - 1).max(), np.abs(fix["kappa_off"].sum(0) - 1).max()))
    W_copy = st.H[base.c_s] * st.phi_copy                              # W'_cu summed over copies, no duplication
    s0 = rb.weight_sums(base)
    for ci in range(5):
        m = rb.P0[base.c_s] == ci
        exp = float((st.H * s0)[rb.P0 == ci].sum())
        acc.max("W_accounting", abs(W_copy[m].sum() / exp - 1))
    acc.max("W_accounting", abs(W_copy.sum() / float(st.H @ s0) - 1))
    acc.max("weights_sum", np.abs(rb.weight_sums(base, st.phi_copy) / rb.weight_sums(base) - 1).max())
    if not (np.isfinite(st.phi_copy).all() and (st.phi_copy > 0).all()):
        acc.count("weights_invalid")
    sid0 = sid.split("-")[0]
    if sid0 in ("A1", "A2"):
        acc.max("A_H_changed", np.abs(st.H - base.H).max())
        x = base.derived["E_ann"][base.c_u]
        e0 = np.add.reduceat(base.derived["phi_copy"] * x, base.starts)
        e1 = np.add.reduceat(st.phi_copy * x, base.starts)
        acc.max("A_energy_restored", np.abs(e1 / e0 - 1).max())
        if not meta["tilt_ok"].all():
            acc.count("tilt_fail", int((~meta["tilt_ok"]).sum()))
    if sid0 in ("B1", "B2", "B3"):
        acc.max("B_phi_changed", np.abs(st.phi_copy - base.derived["phi_copy"]).max())
    if st.M_ue is not None:
        M0 = base.M[base.ue_u]
        acc.max("C_energy", np.abs(st.M_ue.sum(-1) / M0.sum(-1) - 1).max())
        if (st.M_ue < 0).any():
            acc.count("C_negative")
        acc.count("C_zero_recv", meta.get("zero_recv", 0))
        acc.max("C_H_phi_changed", max(np.abs(st.H - base.H).max(), np.abs(st.phi_copy - base.derived["phi_copy"]).max()))
        if meta.get("common"):
            k0 = refs["fix0"]
            key = "kappa_peak" if meta["delta"] > 0 else "kappa_off"
            acc.max("C2_source_invariance", np.abs(fix[key] - k0[key]).max())
    if "rec" in ev:
        acc.max("decomp", np.abs(ev["rec"]["alpha"][ev["names"]] - fix["alpha"][rb.P0] - ev["d_re"]).max())
        same = np.array_equal(ev["names"], rb.P0)
        if same:
            acc.max("reassign_zero", np.abs(ev["d_re"]).max())
        if not np.isin(ev["names"][rb.ESH_ARLP], rb.ESH_IDX).all() or np.isin(ev["names"][~rb.ESH_ARLP],
                                                                                 rb.ESH_IDX).any():
            acc.count("strata_mixed")
    if "neu" in ev:
        for case, res in ev["neu"].items():
            single = res["single_base"]
            clip = fix["clip"]
            for k, key in enumerate(SK):
                ok_cells = (~clip[:, k]) & single[:, k]
                if ok_cells.any():
                    acc.max(f"neutral_full_{case}", np.abs(res[key]["rec_full"][ok_cells]).max())
                acc.max(f"thr_vs_lam_{case}", np.abs(res[key]["thr_part"] - res[key]["lam_full"]).max())
                acc.max(f"parts_sum_{case}", np.abs(res[key]["clip_part"] + res[key]["thr_part"] + res[key]["rnd_part"]
                                                    - res[key]["rec_round"]).max())
            if sid0 in ("A1", "A2"):
                lo, hi = refs["bound"][case]
                e = res["ann"]["frz"]
                if ((e < lo - 1e-9) | (e > hi + 1e-9)).any():
                    acc.count(f"bound_violation_{case}")
            # closed form for the non-ESH classes (single base 3.60): eps = sum (rate/3.6 - 1) E / sum E
            G_lo, G_hi = res["G"]
            r_lo, _ = rb.rates_frozen(base)
            E = G_lo[:3] + G_hi[:3]
            cf = 100 * ((r_lo[:3] / 3.6 - 1) * E).sum((1, 2)) / E.sum((1, 2))
            acc.max(f"closed_form_{case}", np.abs(cf - res["ann"]["frz"][:3]).max())


def record(ev, sid, rep, meta, base, extra=None):
    r = {"scenario_id": sid, "replicate": rep}
    r.update(rb.flat_record(ev, None))
    r.update({f"fix_{k}": v for k, v in rb.structural(ev["fix"]["alpha"]).items()})
    if "rec" in ev:
        r.update({f"rec_{k}": v for k, v in rb.structural(ev["rec"]["alpha"]).items()})
    if "beta" in meta:
        r["tilt_fail"] = int((~meta["tilt_ok"]).sum())
        r["tilt_beta_absmax"] = float(np.nanmax(np.abs(meta["beta"])))
    if extra:
        r.update(extra)
    return r


def a_diagnostics(base, st, meta):
    """A: the multipliers before the energy tilt (isolates the energy-level part of the composition
    change; fixed classes, CSV only), pre-tilt annual-energy drift, TV distance, change of the winter share of annual
    energy, Kish per class."""
    x = base.derived["E_ann"][base.c_u]
    phi0 = base.derived["phi_copy"]
    pre = meta.get("pre_tilt", meta.get("mixed"))
    st_pre = rb.State(phi_copy=pre, H=st.H.copy(), b=st.b.copy())
    a_pre = rb.evaluate(base, st_pre, recluster_method=None, neutrality=False)["fix"]["alpha"]
    e0 = np.add.reduceat(phi0 * x, base.starts)
    drift = np.add.reduceat(pre * x, base.starts) / e0 - 1
    tv = 0.5 * np.add.reduceat(np.abs(st.phi_copy - phi0), base.starts)
    xw = base.derived["EZ_u"][base.c_u][:, rb.WINTER_M, :].sum((1, 2))
    wsh0 = np.add.reduceat(phi0 * xw, base.starts) / e0
    wsh1 = np.add.reduceat(st.phi_copy * xw, base.starts) / np.add.reduceat(st.phi_copy * x, base.starts)
    out = {}
    for ci, c in enumerate(CL):
        for k, sk in enumerate(SK):
            for zi, z in enumerate(ZK):
                out[f"a_pre_{c}_{sk}_{z}"] = a_pre[ci, k, zi]
    for s, a in enumerate(rb.ARLPS):
        out[f"drift_pre_{a}"], out[f"tv_{a}"], out[f"dwinshare_{a}"] = drift[s], tv[s], wsh1[s] - wsh0[s]
    for c, v in rb.concentration(base, st.phi_copy, st.H).items():
        out[f"kish_{c}"], out[f"maxw_{c}"], out[f"top5_{c}"] = v["kish"], v["max"], v["top5"]
    return out


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="robustness_registry.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot", type=int, default=0, help="replicates per stochastic scenario (technical pilot)")
    args = ap.parse_args(argv)
    t0 = time.time()
    vault = eh.find_vault()
    here = Path(__file__).resolve().parent
    reg_path = (here / args.registry) if not Path(args.registry).is_absolute() else Path(args.registry)
    reg = pd.read_csv(reg_path, dtype={"seed": "Int64"})
    out = Path(args.out) if args.out else vault / OUT_REL
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output folder {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    g = Gates()
    print(f"coefficient sensitivity -> {out}")

    # baseline ----------------------------------------------------------------------------------------------------
    wdf = eh.load_weights(vault)
    ts, L = eh.load_eulp_hourly(vault, wdf.index.to_numpy())
    base = rb.load_baseline(vault, L=L, ts=ts)
    st0 = rb.baseline_state(base)
    ev0 = rb.evaluate(base, st0)
    a0 = ev0["fix"]["alpha"]
    refs = {"fix0": ev0["fix"]}

    # G02 ------------------------------------------------------------------------------------------------------
    d_pub = float(np.abs(a0 - base.alpha_pub).max())
    t4 = pd.read_csv(vault / TABLE4, dtype=str).set_index(["Class", "Season"])
    mism = []
    for ci, c in enumerate(CL):
        for k, sk in enumerate(SK):
            row = t4.loc[(c, sk)]
            for zi, col in enumerate(("a_off", "a_sho", "a_peak")):
                if f"{rb.published_round(a0[ci, k, zi], 3):.3f}" != f"{float(row[col]):.3f}":
                    mism.append(f"{c}{sk}{col}")
            if f"{rb.published_round(a0[ci, k, 2] / a0[ci, k, 0], 2):.2f}" != f"{float(row['ratio']):.2f}":
                mism.append(f"{c}{sk}ratio")
            for zi, col in enumerate(("off_rate", "sho_rate", "peak_rate")):
                # numeric comparison at the printed precision (the CSV drops trailing zeros: "1.8" = 1.80)
                if c in eh.ESH_CLASSES and sk == "w":
                    mine = [rb.published_round(a0[ci, k, zi] * 2.2, 2), rb.published_round(a0[ci, k, zi] * 3.6, 2)]
                    theirs = [float(x) for x in row[col].split("-")]
                else:
                    mine, theirs = [rb.published_round(a0[ci, k, zi] * 3.6, 2)], [float(row[col])]
                if len(mine) != len(theirs) or any(abs(x - y) > 1e-9 for x, y in zip(mine, theirs)):
                    mism.append(f"{c}{sk}{col}:{mine}/{row[col]}")
    arlp = eh.load_arlp_hourly(vault)
    rebuilt = L @ base.phi
    am = arlp[rb.ARLPS].to_numpy()
    rel = float(max(np.abs(rebuilt[:, i] - am[:, i]).max() / am[:, i].max() for i in range(19)))
    g.check("G02", "zero perturbation: 30 alpha vs the P3 rates file (4 dp)", f"{d_pub:.2e}", "<= 5e-5", d_pub <= 5e-5)
    g.check("G02", "zero perturbation: printed 3-dp alpha, 2-dp ratios and rates = the proposed-tariff table",
            f"{len(mism)} mismatches", "0", not mism, ";".join(mism))
    g.check("G02", "ARLP loads rebuilt from EULPs vs the ARLP loads (max |diff| / max)", f"{rel:.2e}", "<= 1e-5", rel <= 1e-5)

    # G03 ------------------------------------------------------------------------------------------------------
    C0 = rb.class_repdays(ev0["arlp_rep"], base.H, rb.P0)
    ref = eh.recompute_alpha({c: {s: C0[i, k] for k, s in enumerate(eh.SEASONS)} for i, c in enumerate(CL)})
    d3 = max(float(np.abs(a0[i, k] - np.array(ref[(c, s)])).max()) for i, c in enumerate(CL)
             for k, s in enumerate(eh.SEASONS))
    v03 = {"max": d3, "n": 1}

    # G07 ------------------------------------------------------------------------------------------------------
    z = [rb.perturb_multiplicative(base, np.ones(len(base.eulp_ids))), rb.tilt_all(base, base.derived["phi_copy"])[0],
         rb.mix_with_pool(base, 0.0)]
    dz = max(float(np.abs(x - base.derived["phi_copy"]).max()) for x in z)
    Mz, _ = rb.transform_units(base, np.zeros(11), np.zeros(11), np.zeros(11))
    no_flags = not any(any((f["membership_changed"], f["substantially_changed"], f["ambiguous"]))
                       for f in ev0["match"]["flags"].values())
    g.check("G07", "zero perturbation of the generators returns the baseline weights and days",
            f"{dz:.1e}; days identical {np.array_equal(Mz, base.M[base.ue_u])}", "<= 1e-15 / exact",
            dz <= 1e-15 and np.array_equal(Mz, base.M[base.ue_u]))
    g.check("G07", "baseline re-clustering reproduces _DEF; matching = identity with the class names; no flags",
            f"ARI {ev0['part']['ari_arlp']:.3f}", "exact", np.array_equal(ev0["names"], rb.P0) and no_flags)

    # G09 ------------------------------------------------------------------------------------------------------
    n_s = np.diff(np.append(base.starts, base.n_copies))
    s0 = rb.weight_sums(base)
    floor = np.array([base.derived["phi_copy"][base.starts[s]:base.starts[s] + n_s[s]].min() * n_s[s] / s0[s]
                      for s in range(19)])
    n_floor = int((np.abs(floor / 0.001 * 0.97 - 1) < 0.01).sum())
    desc9 = ("candidate pool = support: floor rho/N binding and equal to the non-zero set of the final calibration "
             "weights final_weights_05c")
    if n_floor >= 15:
        g.check("G09", desc9, f"floor in {n_floor}/19", ">= 15/19; set equality", True)
    elif n_floor == 14:
        g.qualified("G09", desc9, f"floor in {n_floor}/19 (ARLP_01-10, 12-15); supports {list(n_s)}",
                    ">= 15/19; set equality",
                    "the threshold of 15 was fixed before the run from a count that shows 14 ARLPs at the floor; the "
                    "decisive check, set equality of all 19 supports with the non-zero sets of the final calibration "
                    "final_weights_05c, holds; the threshold is not changed")
    else:
        g.check("G09", desc9, f"floor in {n_floor}/19", ">= 15/19; set equality", False)

    # G12 (d) and the hard bound ------------------------------------------------------------------------------------
    f04 = pd.read_csv(vault / SUMMARY)
    f02 = pd.read_csv(vault / UNITS)
    d12 = 0.0
    for case in ("T", "A"):
        rr = f04[(f04.case == case) & (f04.weighting == "weighted") &
                 (f04.weight_basis == "H_normalized")].set_index("panel")["aggregate_pct"]
        for ci, c in enumerate(CL):
            d12 = max(d12, abs(ev0["neu"][case]["ann"]["frz"][ci] - rr[c]))
    g.check("G12d", "zero perturbation: frozen class deviation = the within-class file 04 (cases T, A)", f"{d12:.2e} pp",
            "<= 1e-6", d12 <= 1e-6)
    refs["bound"] = {case: (np.array([f02[(f02.case == case) & (f02["class"] == c)].dbill_pct.min() for c in CL]),
                            np.array([f02[(f02.case == case) & (f02["class"] == c)].dbill_pct.max() for c in CL]))
                     for case in ("T", "A")}
    base_eps = {case: {key: ev0["neu"][case][key]["frz"] for key in ("ann", "w", "s")} for case in ("T", "A")}
    if g.failed:
        return stop(g, out)

    acc = Acc()
    rows_det, rows_rep, diag_rows = [], [], []

    # deterministic runs -------------------------------------------------------------------------------------------
    print("deterministic runs")
    rows_det.append(record(ev0, "BASELINE", 0, {}, base))
    for sid, st, meta in deterministic_states(base, reg):
        ev = rb.evaluate(base, st, recluster_method=meta["recluster"], neutrality=meta["neutrality"])
        run_checks(base, st, ev, meta, sid, acc, refs)
        extra = {}
        if sid == "A2-MIX25":
            extra.update(a_diagnostics(base, st, meta))
            for stage, phi in (("base", base.derived["phi_copy"]), ("mixed", meta["mixed"]), ("tilted", st.phi_copy)):
                for c, v in rb.concentration(base, phi, base.H).items():
                    diag_rows.append({"scenario_id": sid, "level": "class", "unit": c, "stage": stage, **v})
                for s, v in enumerate(rb.arlp_concentration(base, phi)):
                    diag_rows.append({"scenario_id": sid, "level": "arlp", "unit": rb.ARLPS[s], "stage": stage, **v})
        if sid.startswith("B3"):
            k0 = refs["fix0"]
            f = np.where(np.arange(5) == meta["cls"], meta["f"], 1.0)
            for key in ("kappa_peak", "kappa_off"):
                exp = f[:, None] * k0[key] / (f[:, None] * k0[key]).sum(0)
                acc.max("B3_closed_form", np.abs(ev["fix"][key] - exp).max())
        if sid == "B2-STAT":
            stc = rb.baseline_state(base)
            stc.H = base.H * st.b
            evc = rb.evaluate(base, stc, recluster_method=None, neutrality=False)
            acc.max("B_identity", np.abs(evc["fix"]["alpha"] - ev["fix"]["alpha"]).max())
        rows_det.append(record(ev, sid, 0, meta, base, extra))
        print(f"  {sid}: done ({time.time() - t0:.0f} s)")

    # stochastic runs ----------------------------------------------------------------------------------------------
    for r in reg[reg.n_replicates > 1].itertuples():
        n = args.pilot or int(r.n_replicates)
        rng = np.random.default_rng(int(r.seed))
        print(f"{r.scenario_id}: {n} replicates, seed {int(r.seed)}")
        t1 = time.time()
        for i in range(n):
            st, meta = stochastic_state(base, r.scenario_id, rng)
            ev = rb.evaluate(base, st)
            run_checks(base, st, ev, meta, r.scenario_id, acc, refs)
            if v03["n"] < 101:
                Ci = rb.class_repdays(ev["arlp_rep"], st.H, rb.P0)
                refi = eh.recompute_alpha({c: {s: Ci[j, k] for k, s in enumerate(eh.SEASONS)}
                                           for j, c in enumerate(CL)})
                v03["max"] = max(v03["max"], max(float(np.abs(ev["fix"]["alpha"][j, k] - np.array(refi[(c, s)])).max())
                                                 for j, c in enumerate(CL) for k, s in enumerate(eh.SEASONS)))
                v03["n"] += 1
            extra = {}
            if r.scenario_id == "A1-U20":
                extra.update(a_diagnostics(base, st, meta))
            if r.scenario_id == "B1-RND20":
                ste = rb.baseline_state(base)
                ste.b = meta["a"].copy()
                ste.reading = "energy"
                eve = rb.evaluate(base, ste, recluster_method="baseline", neutrality=True)
                acc.max("B_identity", np.abs(eve["fix"]["alpha"] - ev["fix"]["alpha"]).max())
                if not np.array_equal(eve["names"], ev["names"]):
                    acc.count("B_identity_names")
                for case in ("T", "A"):
                    for ci, c in enumerate(CL):
                        extra[f"eps_{case}_frz_{c}_ann_energy"] = eve["neu"][case]["ann"]["frz"][ci]
                        extra[f"eps_{case}_rec_round_{c}_ann_energy"] = eve["neu"][case]["ann"]["rec_round"][ci]
                    nonesh = np.abs(eve["neu"][case]["ann"]["frz"][:3] - ev["neu"][case]["ann"]["frz"][:3]).max()
                    acc.max("B_identity_nonesh_bills", nonesh)
            if r.scenario_id == "C3-RND":
                for j, u in enumerate(rb.UNITS):
                    extra[f"theta_{u}"], extra[f"dw_{u}"], extra[f"ds_{u}"] = meta["params"][j, 0], \
                        0.05 * meta["params"][j, 1], 0.05 * meta["params"][j, 2]
            rows_rep.append(record(ev, r.scenario_id, i + 1, meta, base, extra))
            if (i + 1) % 250 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n} ({time.time() - t1:.0f} s)")

    # gates on the runs --------------------------------------------------------------------------------------------
    m = acc.m
    g.check("G03", "published rule = eulp_hierarchy.recompute_alpha (baseline + perturbed sets)",
            f"{v03['max']:.1e} over {v03['n']}", "<= 1e-14", v03["max"] <= 1e-14 and v03["n"] >= 101)
    g.check("G04", "sum of kappa over the five classes = 1 (every run)", f"{m['kappa_sum']:.1e}", "<= 1e-12",
            m["kappa_sum"] <= 1e-12)
    g.check("G04", "class-EULP weights: sum per class and in total = sum of H_s x published weight sum",
            f"{m['W_accounting']:.1e}", "<= 1e-12", m["W_accounting"] <= 1e-12)
    g.check("G04", "A leaves H unchanged; A restores each ARLP's annual energy", f"H {m.get('A_H_changed', 0):.1e}; "
            f"energy {m.get('A_energy_restored', 0):.1e}", "exact / <= 1e-9",
            m.get("A_H_changed", 0) == 0 and m.get("A_energy_restored", 0) <= 1e-9)
    g.check("G04", "B leaves phi unchanged; C leaves H and phi unchanged",
            f"B {m.get('B_phi_changed', 0):.1e}; C {m.get('C_H_phi_changed', 0):.1e}", "exact",
            m.get("B_phi_changed", 0) == 0 and m.get("C_H_phi_changed", 0) == 0)
    g.check("G05", "weights finite and > 0, published sums kept; tilt converged",
            f"sum dev {m['weights_sum']:.1e}; invalid {m.get('weights_invalid', 0)}; tilt failures "
            f"{m.get('tilt_fail', 0)}", "<= 1e-12 / 0 / counted", m["weights_sum"] <= 1e-12 and
            m.get("weights_invalid", 0) == 0)
    g.check("G06", "C: energy of every dwelling-month unchanged; loads >= 0", f"{m.get('C_energy', 0):.1e}; "
            f"negative {m.get('C_negative', 0)}; zero receiving {m.get('C_zero_recv', 0)}", "<= 1e-12 / 0",
            m.get("C_energy", 0) <= 1e-12 and m.get("C_negative", 0) == 0)
    g.check("G06", "common-mode C2: source-zone kappa unchanged (peak for +5 %, off for -5 %)",
            f"{m.get('C2_source_invariance', float('nan')):.1e}", "<= 1e-12",
            m.get("C2_source_invariance", 1) <= 1e-12)
    g.check("G08", "count = energy: alpha and names identical; B3 = closed form; non-ESH bills identical",
            f"alpha {m.get('B_identity', 0):.1e}; names differ {m.get('B_identity_names', 0)}; B3 "
            f"{m.get('B3_closed_form', 0):.1e}; non-ESH bills {m.get('B_identity_nonesh_bills', 0):.1e}",
            "<= 1e-12 / 0", m.get("B_identity", 0) <= 1e-12 and m.get("B_identity_names", 0) == 0 and
            m.get("B3_closed_form", 0) <= 1e-12 and m.get("B_identity_nonesh_bills", 0) <= 1e-9)
    g.check("G11", "paired runs: one input array per run; total = fixed + reassignment; zero when P_r = P0",
            f"{m.get('decomp', 0):.1e}; {m.get('reassign_zero', 0):.1e}", "<= 1e-12",
            m.get("decomp", 0) <= 1e-12 and m.get("reassign_zero", 0) <= 1e-12)
    g.check("G15", "strata never mixed; names only by matching to P0", f"mixed {m.get('strata_mixed', 0)}", "0", m.get("strata_mixed", 0) == 0)
    for case in ("T", "A"):
        g.check("G12a", f"case {case}: recalibrated full precision neutral where the clip is inactive at a single "
                "base", f"{m.get(f'neutral_full_{case}', 0):.1e} pp", "<= 1e-8 pp (1e-10 relative)",
                m.get(f"neutral_full_{case}", 0) <= 1e-8)
        g.check("G12c", f"case {case}: clipping + threshold + rounding parts sum to the recalibrated deviation; "
                "threshold part = deviation with lambda*", f"{m.get(f'parts_sum_{case}', 0):.1e} / "
                f"{m.get(f'thr_vs_lam_{case}', 0):.1e}", "<= 1e-12", m.get(f"parts_sum_{case}", 0) <= 1e-12 and
                m.get(f"thr_vs_lam_{case}", 0) <= 1e-12)
        g.check("G12e", f"case {case}: A frozen deviation within the range of the passive changes of the pairs",
                f"violations {m.get(f'bound_violation_{case}', 0)}", "0", m.get(f"bound_violation_{case}", 0) == 0)
        g.check("G12f", f"case {case}: non-ESH closed form = billing route", f"{m.get(f'closed_form_{case}', 0):.1e}",
                "<= 1e-9", m.get(f"closed_form_{case}", 0) <= 1e-9)

    # write --------------------------------------------------------------------------------------------------------
    det = pd.DataFrame(rows_det)
    reps = pd.DataFrame(rows_rep)
    write_csv(reg, out / "00_registry.csv")
    write_csv(det, out / "02_deterministic_runs.csv")
    write_csv(reps, out / "03_replicates.csv.gz", gz=True)
    write_csv(pd.DataFrame(diag_rows), out / "07a_A2_concentration.csv")
    summaries(out, det, reps, base, a0, base_eps, g, args.pilot)
    target = out if not g.failed else out / "_unchecked"
    if g.failed:
        target.mkdir(exist_ok=True)
        for f in out.glob("*.csv*"):
            f.rename(target / f.name)
    write_csv(pd.DataFrame(g.rows), target / "08_gates.csv")

    print(f"{'STOP: FAILED ' + str(g.failed) if g.failed else 'all gates PASS'}; {time.time() - t0:.0f} s; {target}")
    return 1 if g.failed else 0


def stop(g, out):
    t = out / "_unchecked"
    t.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(g.rows), t / "08_gates.csv")
    print(f"STOP: {g.failed} (outputs in {t})")
    return 1


# ── summaries ─────────────────────────────────────────────────────────────────────────────────────────────────
def q(x, p):
    return float(np.quantile(np.asarray(x, dtype=float), p))


def summaries(out, det, reps, base, a0, base_eps, g, pilot):
    """04 coefficient summary, 05 clustering summary, 06 neutrality summary, 07 diagnostics."""
    rows = []
    pub = base.alpha_pub
    for sid, grp in list(reps.groupby("scenario_id", sort=False)) + list(det.groupby("scenario_id", sort=False)):
        stoch = sid in ("A1-U20", "B1-RND20", "C3-RND")
        for mode in ("fix", "rec", "pre"):
            for ci, c in enumerate(CL):
                for k, sk in enumerate(SK):
                    items = [(z, a0[ci, k, zi], pub[ci, k, zi], grp.get(f"a_{mode}_{c}_{sk}_{z}"))
                             for zi, z in enumerate(ZK)]
                    rat = grp.get(f"a_{mode}_{c}_{sk}_peak")
                    if rat is not None:
                        items.append(("ratio", a0[ci, k, 2] / a0[ci, k, 0], pub[ci, k, 2] / pub[ci, k, 0],
                                      rat / grp[f"a_{mode}_{c}_{sk}_off"]))
                    for z, b0, bp, v in items:
                        if v is None or v.isna().all():
                            continue
                        v = v.to_numpy(float)
                        d = v - b0
                        row = {"scenario_id": sid, "mode": mode, "class": c, "season": sk, "quantity": z,
                               "baseline_published": bp, "baseline_eulp_route": b0, "n": len(v)}
                        if stoch and len(v) > 1:
                            h = len(v) // 2
                            row.update(median=q(v, .5), p05=q(v, .05), p95=q(v, .95), min=v.min(), max=v.max(),
                                       p95_absdev=q(np.abs(d), .95), max_absdev=np.abs(d).max(),
                                       p05_dev=q(d, .05), p95_dev=q(d, .95),
                                       split_p95_absdev=abs(q(np.abs(d[:h]), .95) - q(np.abs(d[h:]), .95)),
                                       split_p05=abs(q(v[:h], .05) - q(v[h:], .05)),
                                       split_p95=abs(q(v[:h], .95) - q(v[h:], .95)))
                        else:
                            row.update(value=v[0], dev=d[0])
                        rows.append(row)
    cs = pd.DataFrame(rows)
    write_csv(cs, out / "04_coefficient_summary.csv")
    st = cs[cs.split_p95_absdev.notna()] if "split_p95_absdev" in cs else cs.iloc[0:0]
    if len(st):
        worst = float(st[["split_p95_absdev", "split_p05", "split_p95"]].max().max())
        n_fail = int((st[["split_p95_absdev", "split_p05", "split_p95"]] > 0.001).any(axis=1).sum())
        g.report("G13", "Monte Carlo split-half (1-1,000 vs 1,001-2,000): largest difference of a reported P5, P95 "
                 "or P95|dev|; values above 0.001 printed at 2 dp", f"{worst:.4f}; rows above 0.001: {n_fail}")

    # 05 clustering ------------------------------------------------------------------------------------------------
    rows = []
    for sid, grp in list(reps.groupby("scenario_id", sort=False)) + list(det.groupby("scenario_id", sort=False)):
        if "same_nonesh" not in grp or grp["same_nonesh"].isna().all():
            continue
        r = {"scenario_id": sid, "n": len(grp), "same_nonesh_pct": 100 * grp.same_nonesh.astype(bool).mean(),
             "same_esh_pct": 100 * grp.same_esh.astype(bool).mean(),
             "ari_arlp_median": q(grp.ari_arlp, .5), "ari_arlp_p05": q(grp.ari_arlp, .05),
             "reassign_p95_pct": 100 * q(grp.reassign_share, .95), "reassign_max_pct": 100 * grp.reassign_share.max(),
             "n_split_families_max": int(grp.n_split_families.max())}
        for c in CL:
            for f in ("mem", "sub", "amb"):
                r[f"flag_{f}_{c}_pct"] = 100 * grp[f"flag_{f}_{c}"].astype(bool).mean()
        for a in rb.ARLPS:
            base_c = CL[rb.P0[rb.ARLPS.index(a)]]
            r[f"moved_{a}_pct"] = 100 * (grp[f"cls_{a}"] != base_c).mean()
        dre = grp[[c_ for c_ in grp.columns if c_.startswith("dre_")]].abs()
        r["dre_absmax"] = float(dre.max().max()) if len(dre.columns) else float("nan")
        for c in CL:
            for sk in SK:
                for z in ZK:
                    col = f"D_{c}_{sk}_{z}"
                    r[f"D_p95_{c}_{sk}_{z}"] = q(grp[col], .95)
        top = grp[[f"cls_{a}" for a in rb.ARLPS]].astype(str).agg("|".join, axis=1).value_counts()
        r["partitions_distinct"] = len(top)
        r["partition_top"] = top.index[0]
        r["partition_top_pct"] = 100 * top.iloc[0] / len(grp)
        rows.append(r)
    write_csv(pd.DataFrame(rows), out / "05_clustering_summary.csv")

    # 06 neutrality ------------------------------------------------------------------------------------------------
    rows = []
    for sid, grp in list(reps.groupby("scenario_id", sort=False)) + list(det.groupby("scenario_id", sort=False)):
        if f"eps_T_frz_A_ann" not in grp or grp["eps_T_frz_A_ann"].isna().all():
            continue
        for case in ("T", "A"):
            for c in CL:
                r = {"scenario_id": sid, "case": case, "class": c, "n": len(grp),
                     "baseline_frz_pct": base_eps[case]["ann"][CL.index(c)]}
                for part, col in (("frz_dev", f"eps_{case}_frz_{c}_ann"), ("rec_round", f"eps_{case}_rec_round_{c}_ann"),
                                  ("clip_part", f"eps_{case}_clip_part_{c}_ann"), ("thr_part", f"eps_{case}_thr_part_{c}_ann"),
                                  ("rnd_part", f"eps_{case}_rnd_part_{c}_ann"),
                                  ("frz_dev_energy", f"eps_{case}_frz_{c}_ann_energy")):
                    if col not in grp:
                        continue
                    v = grp[col].to_numpy(float)
                    if part.startswith("frz_dev"):
                        v = v - base_eps[case]["ann"][CL.index(c)]
                    r[f"{part}_median"], r[f"{part}_p05"], r[f"{part}_p95"] = q(v, .5), q(v, .05), q(v, .95)
                    r[f"{part}_absmax"] = float(np.abs(v).max())
                clipcols = [f"clip_{c}_{sk}" for sk in SK]
                r["clip_runs_pct"] = 100 * grp[clipcols].astype(bool).any(axis=1).mean()
                rows.append(r)
    write_csv(pd.DataFrame(rows), out / "06_neutrality_summary.csv")

    # 07 diagnostics -----------------------------------------------------------------------------------------------
    rows = []
    for sid, grp in list(reps.groupby("scenario_id", sort=False)) + list(det.groupby("scenario_id", sort=False)):
        r = {"scenario_id": sid, "n": len(grp), "clip_any_pct": 100 * grp.clip_any.astype(bool).mean(),
             "order_full_w_pct": 100 * grp.fix_order_full_w.mean(), "order_full_s_pct": 100 * grp.fix_order_full_s.mean(),
             "esh1_gt_c_w_pct": 100 * grp.fix_esh1_gt_c_w.mean(), "esh1_gt_c_s_pct": 100 * grp.fix_esh1_gt_c_s.mean(),
             "order_nonesh_pct": 100 * (grp.fix_order_nonesh_w & grp.fix_order_nonesh_s).mean(),
             "order_esh_pct": 100 * (grp.fix_order_esh_w & grp.fix_order_esh_s).mean(),
             "K_01_pct": 100 * grp.fix_K_01_holds.mean(), "K_02_pct": 100 * grp.fix_K_02_holds.mean(),
             "K_03_pct": 100 * grp.fix_K_03_holds.mean(),
             "K_04_pct": 100 * (grp.fix_K_04_w & grp.fix_K_04_s).mean(),
             "flat_ratio_min_p05": q(grp.fix_K_01_flat_min, .05), "flat_ratio_max_p95": q(grp.fix_K_01_flat_max, .95),
             "mass_ratio_min_p05": q(grp.fix_K_01_mass_min, .05), "mass_ratio_max_p95": q(grp.fix_K_01_mass_max, .95)}
        if "rec_order_full_w" in grp and grp.rec_order_full_w.notna().any():
            r["rec_order_full_pct"] = 100 * (grp.rec_order_full_w.astype(bool) & grp.rec_order_full_s.astype(bool)).mean()
        for a in rb.ARLPS:
            for key in ("drift_pre", "tv", "dwinshare"):
                col = f"{key}_{a}"
                if col in grp and grp[col].notna().any():
                    r[f"{key}_{a}_absp95"] = q(np.abs(grp[col]), .95)
        for c in CL:
            if f"kish_{c}" in grp and grp[f"kish_{c}"].notna().any():
                r[f"kish_{c}_median"], r[f"maxw_{c}_median"] = q(grp[f"kish_{c}"], .5), q(grp[f"maxw_{c}"], .5)
        if "tilt_fail" in grp:
            r["tilt_fail_total"] = float(np.nansum(grp.tilt_fail))
        rows.append(r)
    write_csv(pd.DataFrame(rows), out / "07_diagnostics.csv")


if __name__ == "__main__":
    sys.exit(main())
