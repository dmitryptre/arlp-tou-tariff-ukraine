"""Sensitivity of the tariff multipliers to the synthetic inputs.

Hierarchy: 3,145 weighted EULPs -> 19 ARLPs (phi) -> 11 clustering units (8 city-rural families + 3 ESH ARLPs)
-> 5 classes. The baseline class map P0 is `_DEF` of notebook cell e3019676 (`eulp_hierarchy.CLASS_MAP_INT`).

Fixed rules:
- the multipliers follow the published rule of notebook cell 20 (kappa over all five classes, alpha_peak =
  1 + kappa_peak, alpha_off = 1 - kappa_off, alpha_shoulder = clip(lambda*, alpha_off, alpha_peak)). lambda* is kept
  only to flag an active clip; it is never used as a coefficient.
- class names are those of P0 (the article's class names). Experiments A-C use P0. Re-estimated clusters (D) are named only by
  Hungarian matching to P0 on the baseline-household-weighted membership overlap; household-count ranking is never
  applied.

Perturbations act on phi (A), on H or on the per-household level (B), or on the monthly mean days of the dwellings
(C); the class-EULP weights are always recomputed, never perturbed directly. The ESH bills reuse the regime rule of the
within-class bill distribution (`eulp_bill_distribution.base_regime`, cases T and A) at the level of (ARLP, EULP) copies.

Calendar convention: that of `eulp_hierarchy` (stamp month, stamp hour, 0 = hour-ending 24). Nothing here writes a
file.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import eulp_bill_distribution as bd
import eulp_hierarchy as eh
from _common import ARLP_NAMES_ALL, TOU_PERIODS_0BASED, build_tou_feature_matrix

CLASS_ORDER = tuple(eh.CLASS_ORDER)                      # ("A", "B", "C", "ESH-1", "ESH-2") - the article's class names
NON_ESH_IDX = (0, 1, 2)
ESH_IDX = (3, 4)
SEASON_KEYS = ("w", "s")                                  # winter (Oct-Apr), summer
ZONE_KEYS = ("off", "sh", "peak")                         # index order of every zone axis (= eulp_hierarchy.ZONES)
ARLPS = list(ARLP_NAMES_ALL)
FAMILIES = [("FAM_04_12", "ARLP_04", "ARLP_12"), ("FAM_05_13", "ARLP_05", "ARLP_13"),
            ("FAM_06_14", "ARLP_06", "ARLP_14"), ("FAM_07_15", "ARLP_07", "ARLP_15"),
            ("FAM_08_16", "ARLP_08", "ARLP_16"), ("FAM_09_17", "ARLP_09", "ARLP_17"),
            ("FAM_10_18", "ARLP_10", "ARLP_18"), ("FAM_11_19", "ARLP_11", "ARLP_19")]
UNITS = [f[0] for f in FAMILIES] + ["ARLP_01", "ARLP_02", "ARLP_03"]
UNIT_MEMBERS = {**{f[0]: [ARLPS.index(f[1]), ARLPS.index(f[2])] for f in FAMILIES},
                **{a: [ARLPS.index(a)] for a in ("ARLP_01", "ARLP_02", "ARLP_03")}}
UNIT_OF_ARLP = np.array([next(i for i, u in enumerate(UNITS) if s in UNIT_MEMBERS[u]) for s in range(19)])
P0 = np.array([eh.CLASS_MAP_INT[a] for a in ARLPS])     # baseline class index per ARLP
ESH_ARLP = np.array([P0[s] in ESH_IDX for s in range(19)])

PEAK_H = list(TOU_PERIODS_0BASED["peak"])
OFF_H = list(TOU_PERIODS_0BASED["off"])
SH_H = [h for h in range(24) if h not in PEAK_H and h not in OFF_H]
ZONE_HOURS = (OFF_H, SH_H, PEAK_H)                        # ZONE_KEYS order
TIME_ORDER = np.array(list(range(1, 24)) + [0])          # hour-ending 1..24 in stamp-hour indices (0 = hour-ending 24)
WINTER_M = bd.WINTER_MASK.copy()                          # month index 0..11 -> winter?
THRESHOLD_KWH = 2000.0
BASE_LO = np.array([3.6, 3.6, 3.6, 2.2, 2.2])            # base price at the low ESH regime (non-ESH: the flat rate)
BASE_HI = np.full(5, 3.6)
TILT_SCALE_KWH = 1000.0
STAT_TOL = 1e-9


# ── the published coefficient rule ──────────────────────────────────────────────────────────────────────────────
def alpha_published_rule(C) -> dict:
    """Notebook cell 20, vectorised. C: (5, 2, 24) class representative-day loads (any common scale) in the article's
    class order and the stamp-hour index. Returns alpha (5, 2, 3) in ZONE_KEYS order with the published clip,
    lam_star (5, 2) before the clip, clip (5, 2) True where the clip is active, kappa_peak, kappa_off (5, 2)."""
    C = np.asarray(C, dtype=float)
    ctot = C.sum(axis=0)
    e_p, e_o, e_s = C[..., PEAK_H].sum(-1), C[..., OFF_H].sum(-1), C[..., SH_H].sum(-1)
    cpeak = e_p / ctot[:, PEAK_H].sum(-1)
    coff = e_o / ctot[:, OFF_H].sum(-1)
    a_peak, a_off = 1.0 + cpeak, 1.0 - coff
    lam_star = 1.0 - (cpeak * e_p - coff * e_o) / e_s
    lam = np.clip(lam_star, a_off, a_peak)
    return {"alpha": np.stack([a_off, lam, a_peak], axis=-1), "lam_star": lam_star,
            "clip": (lam_star < a_off) | (lam_star > a_peak), "kappa_peak": cpeak, "kappa_off": coff}


def published_round(x, nd):
    """Python's round, element by element (cells 20, 50 and 52 use the built-in round)."""
    return np.vectorize(lambda v: round(float(v), nd))(np.asarray(x, dtype=float))


# ── calendar helpers ──────────────────────────────────────────────────────────────────────────────────────────
def monthly_mean_days(L, cal):
    """(8760, n) hourly loads -> monthly mean days (n, 12, 24) by stamp month and stamp hour, and the stamp count per
    (month, hour). Raises if the count is not constant within a month (then the mean day would not carry energy)."""
    L = np.asarray(L, dtype=float)
    idx = (cal["month"].to_numpy() - 1) * 24 + cal["hour"].to_numpy()
    counts = np.bincount(idx, minlength=288).astype(float)
    S = np.zeros((288, L.shape[1]))
    np.add.at(S, idx, L)
    M = (S / counts[:, None]).T.reshape(L.shape[1], 12, 24)
    cnt = counts.reshape(12, 24)
    if not np.all(cnt == cnt[:, :1]):
        raise ValueError("stamp counts differ within a month")
    return M, cnt[:, 0]


def zone_energy(M, days):
    """(..., 12, 24) mean days -> (..., 12, 3) monthly zone energies (days x zone sum)."""
    M = np.asarray(M, dtype=float)
    return np.stack([M[..., h].sum(-1) for h in ZONE_HOURS], axis=-1) * np.asarray(days)[:, None]


def repday(M, days):
    """(..., 12, 24) -> (..., 2, 24) seasonal representative days: the stamp-weighted mean over the season's months,
    as `_common.build_repday_matrix` (mean over all stamps of the season per stamp hour)."""
    M = np.asarray(M, dtype=float)
    out = []
    for mask in (WINTER_M, ~WINTER_M):
        w = np.asarray(days, dtype=float) * mask
        out.append((M * w[:, None]).sum(axis=-2) / w.sum())
    return np.stack(out, axis=-2)


# ── baseline object ───────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Baseline:
    vault: Path
    eulp_ids: np.ndarray            # (3145,)
    phi: np.ndarray                 # (3145, 19)
    H: np.ndarray                   # (19,) kHH
    M: np.ndarray                   # (3145, 12, 24) monthly mean days, kWh/h per dwelling
    days: np.ndarray                # (12,)
    c_s: np.ndarray                 # copy -> ARLP (sorted by ARLP)
    c_u: np.ndarray                 # copy -> EULP row
    c_ue: np.ndarray                # copy -> unit-EULP row
    ue_unit: np.ndarray             # unit-EULP -> unit
    ue_u: np.ndarray                # unit-EULP -> EULP row
    starts: np.ndarray              # first copy of each ARLP (reduceat)
    t2: np.ndarray                  # (19, 4) _T2: AnnGWh, HHths, implGWh, implHHths
    rates_pub: dict                 # published P3 rates {(class, season, zone, tag): rate}
    alpha_pub: np.ndarray           # (5, 2, 3) published alpha (4 dp)
    derived: dict = field(default_factory=dict)

    @property
    def n_copies(self):
        return len(self.c_s)


def build_copies(phi):
    """Copies (s, u) with phi_su > 0, sorted by ARLP then EULP row; unit-EULP rows (unit, u) shared by the members of
    a unit (city-rural members of a family have identical supports)."""
    s_idx, u_idx = [], []
    for s in range(phi.shape[1]):
        us = np.flatnonzero(phi[:, s] > 0)
        s_idx.append(np.full(len(us), s)), u_idx.append(us)
    c_s, c_u = np.concatenate(s_idx), np.concatenate(u_idx)
    key = UNIT_OF_ARLP[c_s] * 100000 + c_u
    uniq, c_ue = np.unique(key, return_inverse=True)
    ue_unit, ue_u = uniq // 100000, uniq % 100000
    starts = np.searchsorted(c_s, np.arange(phi.shape[1]))
    return c_s, c_u, c_ue, ue_unit, ue_u, starts


def load_baseline(vault=None, L=None, ts=None) -> Baseline:
    """Read the inputs (EULP hourly loads, ARLP-EULP weights phi, household counts, published P3 rates and the
    notebook's `_T2` table) and build the baseline object. L/ts may be passed to avoid re-reading the hourly loads."""
    import json
    import re
    vault = eh.find_vault() if vault is None else Path(vault)
    wdf = eh.load_weights(vault)
    phi = wdf[ARLPS].to_numpy(float)
    Hd = eh.load_household_counts(vault)
    H = np.array([Hd[a] for a in ARLPS], dtype=float)
    if L is None:
        ts, L = eh.load_eulp_hourly(vault, wdf.index.to_numpy())
    cal = eh.calendar(ts)
    M, days = monthly_mean_days(L, cal)
    c_s, c_u, c_ue, ue_unit, ue_u, starts = build_copies(phi)
    nb = json.loads((vault / "code/ARLP_clustering_analysis_autonomous.ipynb").read_text(encoding="utf-8"))
    src = {c.get("id"): "".join(c["source"]) for c in nb["cells"]}
    t2 = np.array(eval(re.search(r"_T2 = (\[.*?\])\n", src["cb1d4eb1"], re.S).group(1)), dtype=float)
    _, rates = bd.load_p3_rates(vault / "outputs/ARLP_Clustering/notebook_only/layer13_peed_dr/"
                                        "proposed_rates_per_class_season_period.csv")
    seasons = ("winter", "summer")
    zones = ("off", "shoulder", "peak")
    p3 = pd.read_csv(vault / "outputs/ARLP_Clustering/notebook_only/layer13_peed_dr/"
                             "proposed_rates_per_class_season_period.csv")
    alpha_pub = np.array([[[float(p3[(p3.label == c) & (p3.season == s) & (p3.period == z)].alpha.iloc[0])
                            for z in zones] for s in seasons] for c in CLASS_ORDER])
    b = Baseline(vault=vault, eulp_ids=wdf.index.to_numpy(), phi=phi, H=H, M=M, days=days, c_s=c_s, c_u=c_u,
                 c_ue=c_ue, ue_unit=ue_unit, ue_u=ue_u, starts=starts, t2=t2, rates_pub=rates, alpha_pub=alpha_pub)
    b.derived["EZ_u"] = zone_energy(M, days)                   # (3145, 12, 3)
    b.derived["R_u"] = repday(M, days)                         # (3145, 2, 24)
    b.derived["E_ann"] = b.derived["EZ_u"].sum(axis=(1, 2))    # (3145,)
    b.derived["phi_copy"] = phi[c_u, c_s]
    return b


# ── run state ─────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class State:
    """One perturbed input set. phi_copy: weight of each (ARLP, EULP) copy (sums to 1 per ARLP); H: households per
    ARLP; b: per-household energy factor per ARLP; M_ue: transformed monthly mean days per unit-EULP row, or None."""
    phi_copy: np.ndarray
    H: np.ndarray
    b: np.ndarray
    M_ue: np.ndarray | None = None
    reading: str = "count"                                     # B: "count" or "energy" (bills only)


def baseline_state(base: Baseline) -> State:
    return State(phi_copy=base.derived["phi_copy"].copy(), H=base.H.copy(), b=np.ones(19))


def copy_arrays(base: Baseline, st: State):
    """Per copy: representative days (n, 2, 24) and monthly zone energies (n, 12, 3), both x b_s."""
    if st.M_ue is None:
        R = base.derived["R_u"][base.c_u]
        EZ = base.derived["EZ_u"][base.c_u]
    else:
        R = repday(st.M_ue, base.days)[base.c_ue]
        EZ = zone_energy(st.M_ue, base.days)[base.c_ue]
    f = st.b[base.c_s][:, None, None]
    return R * f, EZ * f


def arlp_repdays(base: Baseline, st: State, R_copy=None):
    """(19, 2, 24) per-household ARLP representative days, sum_u phi_su R_u."""
    if R_copy is None:
        R_copy, _ = copy_arrays(base, st)
    x = (st.phi_copy[:, None, None] * R_copy).reshape(len(st.phi_copy), 48)
    return np.add.reduceat(x, base.starts, axis=0).reshape(19, 2, 24)


def class_repdays(arlp_rep, H, class_of):
    """(5, 2, 24) TCL: sum_{s in c} H_s l_s for the class map `class_of` (19,) (indices into CLASS_ORDER)."""
    out = np.zeros((5, 2, 24))
    np.add.at(out, np.asarray(class_of), np.asarray(H)[:, None, None] * arlp_rep)
    return out


# ── Experiment A: weights ─────────────────────────────────────────────────────────────────────────────────────
def weight_sums(base: Baseline, w_copy=None):
    """Per ARLP sum of the copy weights. The published phi sums to 1 within CSV rounding (about 3e-9), and every
    operator keeps each ARLP's published sum, so that zero perturbation reproduces the published route exactly."""
    return np.add.reduceat(base.derived["phi_copy"] if w_copy is None else w_copy, base.starts)


def renormalise(base: Baseline, w_copy):
    s0 = weight_sums(base)
    return w_copy * (s0 / weight_sums(base, w_copy))[base.c_s]


def tilt_to_energy(p, x, target):
    """Exponential tilting p_k exp(beta x_k) / Z with mean(x) = target (x in MWh). Returns (p_new, beta, ok)."""
    p = np.asarray(p, dtype=float)
    x = np.asarray(x, dtype=float)

    def tilted(beta):
        z = beta * x
        w = p * np.exp(z - z.max())
        return w / w.sum()

    def f(beta):
        return float(tilted(beta) @ x - target)

    if abs(f(0.0)) <= STAT_TOL * abs(target):
        return p / p.sum(), 0.0, True
    lo, hi = -1.0, 1.0
    for _ in range(60):
        if f(lo) < 0 < f(hi):
            break
        lo, hi = lo * 2, hi * 2
    else:
        return p / p.sum(), float("nan"), False
    beta = brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    q = tilted(beta)
    return q, float(beta), abs(q @ x - target) <= STAT_TOL * abs(target)


def tilt_all(base: Baseline, phi_copy):
    """Restore each ARLP's baseline per-household annual energy by exponential tilting within its support."""
    x_all = base.derived["E_ann"][base.c_u] / TILT_SCALE_KWH
    phi0 = base.derived["phi_copy"]
    s0 = weight_sums(base)
    out = phi_copy.copy()
    beta, ok = np.zeros(19), np.ones(19, dtype=bool)
    for s in range(19):
        sl = slice(base.starts[s], base.starts[s + 1] if s < 18 else len(phi_copy))
        target = float(phi0[sl] @ x_all[sl]) / s0[s]                  # per-household annual energy (MWh)
        if np.array_equal(phi_copy[sl], phi0[sl]):
            continue
        q, beta[s], ok[s] = tilt_to_energy(phi_copy[sl], x_all[sl], target)
        out[sl] = q * s0[s]
    return out, beta, ok


def perturb_multiplicative(base: Baseline, g):
    """A1: phi'_su ∝ phi_su g_u (g one draw per EULP row, shared by all ARLPs), renormalised per ARLP."""
    return renormalise(base, base.derived["phi_copy"] * np.asarray(g)[base.c_u])


def mix_with_pool(base: Baseline, eps):
    """A2: (1 - eps) phi_s + eps uniform over the support S_s of each ARLP."""
    n_s = np.diff(np.append(base.starts, base.n_copies))
    s0 = weight_sums(base)
    return (1.0 - eps) * base.derived["phi_copy"] + eps * (s0 / n_s)[base.c_s]


def concentration(base: Baseline, phi_copy, H):
    """Per class: Kish N_eff, largest weight share and five-largest share of the class-EULP weights W_cu."""
    out = {}
    W_copy = np.asarray(H)[base.c_s] * phi_copy
    for ci, c in enumerate(CLASS_ORDER):
        m = P0[base.c_s] == ci
        W = pd.Series(W_copy[m]).groupby(base.c_u[m]).sum().to_numpy()
        W = W[W > 0]
        sh = np.sort(W)[::-1] / W.sum()
        out[c] = {"kish": float(W.sum() ** 2 / (W ** 2).sum()), "max": float(sh[0]), "top5": float(sh[:5].sum())}
    return out


def arlp_concentration(base: Baseline, phi_copy):
    out = []
    for s in range(19):
        sl = slice(base.starts[s], base.starts[s + 1] if s < 18 else len(phi_copy))
        p = np.sort(phi_copy[sl])[::-1]
        out.append({"kish": float(1.0 / (p ** 2).sum()), "max": float(p[0]), "top5": float(p[:5].sum())})
    return out


# ── Experiment C: shape operators on monthly mean days ───────────────────────────────────────────────────────
def shift_time(M, theta):
    """Fractional circular timing shift in time order (hour-ending 1..24), |theta| <= 1 h; theta > 0 delays.
    M: (n, 12, 24) stamp-hour index; theta: (n,) per row. Energy of every row-month is conserved exactly."""
    M = np.asarray(M, dtype=float)
    th = np.asarray(theta, dtype=float).reshape(-1, 1, 1)
    if np.any(np.abs(th) > 1 + 1e-12):
        raise ValueError("|theta| > 1 h")
    X = M[..., TIME_ORDER]
    nb = np.where(th > 0, np.roll(X, 1, axis=-1), np.roll(X, -1, axis=-1))
    Y = (1.0 - np.abs(th)) * X + np.abs(th) * nb
    out = np.empty_like(M)
    out[..., TIME_ORDER] = Y
    return out


def move_zone_energy(M, delta_m):
    """Peak <-> off-peak transfer on each row-month: delta > 0 moves delta of the peak-zone energy to the off-peak
    hours in proportion to the row-month's own off-peak profile; delta < 0 moves |delta| of the off-peak energy to
    the peak hours. Energy of every row-month is conserved. delta_m: (n, 12). A row-month with zero energy in the
    receiving zone is left unchanged and counted. Returns (M', n_zero_receiving)."""
    M = np.asarray(M, dtype=float).copy()
    d = np.asarray(delta_m, dtype=float)
    ep, eo = M[..., PEAK_H].sum(-1), M[..., OFF_H].sum(-1)
    pos, neg = d > 0, d < 0
    zero_recv = (pos & (eo <= 0)) | (neg & (ep <= 0))
    pos &= ~zero_recv
    neg &= ~zero_recv
    with np.errstate(divide="ignore", invalid="ignore"):
        f_peak = np.where(pos, 1.0 - d, np.where(neg, 1.0 + (-d) * eo / ep, 1.0))
        f_off = np.where(pos, 1.0 + d * ep / eo, np.where(neg, 1.0 - (-d), 1.0))
    M[..., PEAK_H] *= f_peak[..., None]
    M[..., OFF_H] *= f_off[..., None]
    return M, int(zero_recv.sum())


def transform_units(base: Baseline, theta_unit, delta_w_unit, delta_s_unit):
    """C operators with unit parameters (11,) on the unit-EULP rows: theta first, then delta (season by month)."""
    th = np.asarray(theta_unit, dtype=float)[base.ue_unit]
    M = base.M[base.ue_u]
    if np.any(th != 0):
        M = shift_time(M, th)
    dm = np.where(WINTER_M[None, :], np.asarray(delta_w_unit)[base.ue_unit][:, None],
                  np.asarray(delta_s_unit)[base.ue_unit][:, None])
    nz = 0
    if np.any(dm != 0):
        M, nz = move_zone_energy(M, dm)
    return M, nz


# ── Experiment D: features, clustering, matching ──────────────────────────────────────────────────────────────
def _features_from_repdays(rep, normalisation):
    """rep (n, 2, 24) -> (n, 6) zone means per season (winter off, sh, peak, summer off, sh, peak)."""
    if normalisation == "peak":
        cols = [f"{sn}_h{h:02d}" for sn in ("winter", "summer") for h in range(24)]
        df = pd.DataFrame(np.asarray(rep).reshape(len(rep), 48), columns=cols, index=[f"i{i}" for i in range(len(rep))])
        return build_tou_feature_matrix(df).to_numpy(float)
    if normalisation == "mean":
        v = np.asarray(rep) / np.asarray(rep).mean(axis=-1, keepdims=True)
        return np.concatenate([np.stack([v[:, k, h].mean(-1) for h in ZONE_HOURS], axis=-1) for k in (0, 1)], axis=1)
    raise ValueError(normalisation)


def unit_repdays(arlp_rep, H):
    """(11, 2, 24): families merged with the current household counts, ESH ARLPs as they are."""
    out = []
    for u in UNITS:
        m = UNIT_MEMBERS[u]
        h = np.asarray(H, dtype=float)[m]
        out.append((h[:, None, None] * arlp_rep[m]).sum(0) / h.sum() if h.sum() > 0 else arlp_rep[m].mean(0))
    return np.stack(out)


def zscore(X):
    X = np.asarray(X, dtype=float)
    return (X - X.mean(0)) / X.std(0, ddof=0)


def pam(Xz, k):
    from sklearn_extra.cluster import KMedoids
    return KMedoids(n_clusters=k, metric="euclidean", method="pam", random_state=42).fit(Xz).labels_


def recluster(arlp_rep, H, method="baseline"):
    """Raw cluster label per ARLP (19,) with the cell-17 settings, within each stratum. method: "baseline" (families,
    peak normalisation), "D2-MN" (families, daily-mean normalisation), "D2-NC" (16 non-ESH ARLPs, no merge)."""
    raw = np.full(19, -1)
    norm = "mean" if method == "D2-MN" else "peak"
    urep = unit_repdays(arlp_rep, H)
    X = _features_from_repdays(urep, norm)
    esh_units = [UNITS.index(a) for a in ("ARLP_01", "ARLP_02", "ARLP_03")]
    lab_e = pam(zscore(X[esh_units]), 2)
    for j, ui in enumerate(esh_units):
        raw[UNIT_MEMBERS[UNITS[ui]][0]] = 100 + lab_e[j]
    if method == "D2-NC":
        items = [s for s in range(19) if not ESH_ARLP[s]]
        Xa = _features_from_repdays(arlp_rep[items], "peak")
        lab = pam(zscore(Xa), 3)
        for j, s in enumerate(items):
            raw[s] = lab[j]
    else:
        fam_units = list(range(8))
        lab = pam(zscore(X[fam_units]), 3)
        for j, ui in enumerate(fam_units):
            for s in UNIT_MEMBERS[UNITS[ui]]:
                raw[s] = lab[j]
    return raw


def assign(O):
    """Optimal one-to-one assignment of baseline classes (rows) to clusters (columns) maximising the total overlap:
    the Hungarian method, checked against the enumeration of all permutations (k <= 3), which also finds ties. A tie is
    broken by the lexicographic overlap vector in the baseline class order. Returns (perm, best, margin, tie)."""
    O = np.asarray(O, dtype=float)
    r_i, c_i = linear_sum_assignment(O, maximize=True)
    hung = float(O[r_i, c_i].sum())
    n = O.shape[0]
    totals = {p: float(sum(O[i, p[i]] for i in range(n))) for p in itertools.permutations(range(O.shape[1]))}
    best = max(totals.values())
    opt = [p for p, t in totals.items() if abs(t - best) <= 1e-9]
    if abs(hung - best) > 1e-9:
        raise AssertionError("Hungarian optimum differs from the enumerated optimum")
    perm = max(opt, key=lambda p: tuple(O[i, p[i]] for i in range(n)))
    others = sorted({t for t in totals.values() if t < best - 1e-9}, reverse=True)
    return perm, best, (best - others[0] if others else float("inf")), len(opt) > 1


def match_to_baseline(raw, H0=None):
    """Name the raw clusters by Hungarian matching to P0 on the baseline-household-weighted overlap.

    Returns (names (19,) class indices, info) where info holds per stratum the overlap matrix, the optimal total, the
    margin to the second-best total, tie flag, and per class the flags membership_changed, substantially_changed
    (household-weighted Jaccard < 0.5) and ambiguous (tie, or the cluster named c holds more baseline households of
    another class than of c). P0 is the reference in every call."""
    H0 = np.asarray(H0, dtype=float)
    raw = np.asarray(raw)
    names = np.full(19, -1)
    info = {"flags": {}, "strata": {}}
    for stratum, cls in (("non_esh", NON_ESH_IDX), ("esh", ESH_IDX)):
        arl = [s for s in range(19) if P0[s] in cls]
        labs = sorted(set(raw[arl]), key=lambda L: min(s for s in arl if raw[s] == L))   # label-free order
        if len(labs) != len(cls):
            raise ValueError(f"{stratum}: {len(labs)} clusters for {len(cls)} classes")
        O = np.array([[sum(H0[s] for s in arl if P0[s] == c and raw[s] == L) for L in labs] for c in cls])
        perm, best, margin, tie = assign(O)
        for i, c in enumerate(cls):
            for s in arl:
                if raw[s] == labs[perm[i]]:
                    names[s] = c
        info["strata"][stratum] = {"overlap": O, "best": best, "margin": margin, "tie": tie}
        for i, c in enumerate(cls):
            k = perm[i]
            m0 = {s for s in arl if P0[s] == c}
            mr = {s for s in arl if raw[s] == labs[k]}
            inter, union = sum(H0[s] for s in m0 & mr), sum(H0[s] for s in m0 | mr)
            jac = inter / union if union > 0 else 1.0
            contested = any(O[j, k] > O[i, k] + 1e-12 for j in range(len(cls)) if j != i)
            info["flags"][CLASS_ORDER[c]] = {"membership_changed": m0 != mr, "jaccard": jac,
                                             "substantially_changed": jac < 0.5, "ambiguous": tie or contested}
    return names, info


def partition_stats(names, method="baseline"):
    """ARI against P0: families (8, when no family is split), ESH ARLPs (3), all 19 ARLPs; split families."""
    fam_p0 = [P0[UNIT_MEMBERS[u][0]] for u in UNITS[:8]]
    split = [u for u in UNITS[:8] if len({names[s] for s in UNIT_MEMBERS[u]}) > 1]
    fam_r = [names[UNIT_MEMBERS[u][0]] for u in UNITS[:8]]
    esh = [s for s in range(19) if ESH_ARLP[s]]
    return {"ari_fam": adjusted_rand_score(fam_p0, fam_r) if not split else float("nan"),
            "ari_esh": adjusted_rand_score(P0[esh], names[esh]), "ari_arlp": adjusted_rand_score(P0, names),
            "n_split_families": len(split),
            "same_nonesh": bool(all(names[s] == P0[s] for s in range(19) if not ESH_ARLP[s])),
            "same_esh": bool(all(names[s] == P0[s] for s in esh))}


# ── rates and bills ───────────────────────────────────────────────────────────────────────────────────────────
def rates_from_alpha(alpha, mode):
    """(5, 2, 3) alpha -> (r_lo, r_hi) per class, month and zone (5, 12, 3). mode "published": round(round(alpha, 4)
    x base, 4) as cells 20/52; "full": alpha x base."""
    a = np.asarray(alpha, dtype=float)
    if mode == "published":
        a4 = published_round(a, 4)
        lo = published_round(a4 * BASE_LO[:, None, None], 4)
        hi = published_round(a4 * BASE_HI[:, None, None], 4)
    elif mode == "full":
        lo, hi = a * BASE_LO[:, None, None], a * BASE_HI[:, None, None]
    else:
        raise ValueError(mode)
    sidx = np.where(WINTER_M, 0, 1)
    return lo[:, sidx, :], hi[:, sidx, :]


def rates_frozen(base: Baseline):
    """The published 4-dp P3 rates (proposed_rates_per_class_season_period.csv) as (r_lo, r_hi) (5, 12, 3)."""
    seasons = ("winter", "summer")
    zones = ("off", "shoulder", "peak")
    lo = np.zeros((5, 2, 3))
    hi = np.zeros((5, 2, 3))
    for ci, c in enumerate(CLASS_ORDER):
        tl, th = ("lo", "hi") if c in eh.ESH_CLASSES else ("flat", "flat")
        for si, s in enumerate(seasons):
            for zi, z in enumerate(zones):
                lo[ci, si, zi], hi[ci, si, zi] = base.rates_pub[(c, s, z, tl)], base.rates_pub[(c, s, z, th)]
    sidx = np.where(WINTER_M, 0, 1)
    return lo[:, sidx, :], hi[:, sidx, :]


def regime_energies(base: Baseline, st: State, EZ_copy, case):
    """Class energies at the low and high ESH regimes, G_lo, G_hi (5, 12, 3) in kHH x kWh, and the per-class-month
    single-base flag. theta per copy by the regime rule of eulp_bill_distribution: case T from the copy's own monthly kWh (the dwelling, x b_s);
    case A from its ARLP's per-household monthly kWh."""
    kwh = EZ_copy.sum(axis=2)                                             # (n, 12)
    is_esh = ESH_ARLP[base.c_s]
    if case == "T":
        theta = bd.base_regime("T", kwh, is_esh, threshold=THRESHOLD_KWH)
    elif case == "A":
        arlp_kwh = np.add.reduceat(st.phi_copy[:, None] * kwh, base.starts, axis=0)            # (19, 12)
        status = (arlp_kwh > THRESHOLD_KWH).astype(float)
        theta = bd.base_regime("A", kwh, is_esh, arlp_theta=np.where(WINTER_M, status[base.c_s], 0.0))
    else:
        raise ValueError(case)
    w = st.H[base.c_s] * st.phi_copy
    G_lo, G_hi = np.zeros((5, 12, 3)), np.zeros((5, 12, 3))
    cls = P0[base.c_s]
    np.add.at(G_lo, cls, (w[:, None] * (1.0 - theta))[..., None] * EZ_copy)
    np.add.at(G_hi, cls, (w[:, None] * theta)[..., None] * EZ_copy)
    single = np.ones((5, 12), dtype=bool)
    for ci in ESH_IDX:
        m = (cls == ci) & (w > 0)
        t = theta[m]
        single[ci] = np.all(t == t[:1], axis=0)
    return G_lo, G_hi, single


def bills_from_energies(G_lo, G_hi, r_lo, r_hi):
    """Flat and P3 bills per class and month (5, 12) in kHH x ₴."""
    F = (BASE_LO[:, None, None] * G_lo + BASE_HI[:, None, None] * G_hi).sum(-1)
    P = (r_lo * G_lo + r_hi * G_hi).sum(-1)
    return F, P


def eps(F, P, months=None):
    """Bill-weighted class deviation, % (the change of the class total bill)."""
    m = slice(None) if months is None else months
    return 100.0 * (P[:, m].sum(1) / F[:, m].sum(1) - 1.0)


# ── one evaluation ────────────────────────────────────────────────────────────────────────────────────────────
def evaluate(base: Baseline, st: State, recluster_method="baseline", neutrality=True, cases=("T", "A")):
    """All quantities of one perturbed input set: fixed-class multipliers (P0), re-clustered multipliers (P_r named by
    matching to P0), experienced-multiplier decomposition, partition statistics and the neutrality diagnostic."""
    R_copy, EZ_copy = copy_arrays(base, st)
    rep = arlp_repdays(base, st, R_copy)
    fix = alpha_published_rule(class_repdays(rep, st.H, P0))
    out = {"fix": fix, "arlp_rep": rep}
    if recluster_method is not None:
        raw = recluster(rep, st.H, recluster_method)
        names, info = match_to_baseline(raw, base.H)
        rec = alpha_published_rule(class_repdays(rep, st.H, names))
        a_fix_exp = fix["alpha"][P0]                                 # (19, 2, 3) experienced, fixed classes
        a_rec_exp = rec["alpha"][names]
        d_re = a_rec_exp - a_fix_exp
        D = np.zeros((5, 2, 3))
        for ci in range(5):
            m = P0 == ci
            D[ci] = (base.H[m][:, None, None] * np.abs(d_re[m])).sum(0) / base.H[m].sum()
        part = partition_stats(names, recluster_method)
        part["reassign_share"] = reassign_share(names, base.H)
        out.update({"rec": rec, "names": names, "match": info, "d_re": d_re, "D": D, "part": part})
    if neutrality:
        neu = {}
        r_froz = rates_frozen(base)
        r_pub = rates_from_alpha(fix["alpha"], "published")
        r_full = rates_from_alpha(fix["alpha"], "full")
        a_lam = fix["alpha"].copy()
        a_lam[..., 1] = fix["lam_star"]
        r_lam = rates_from_alpha(a_lam, "full")
        for case in cases:
            G_lo, G_hi, single = regime_energies(base, st, EZ_copy, case)
            Ff, Pf = bills_from_energies(G_lo, G_hi, *r_froz)
            Fr, Pr = bills_from_energies(G_lo, G_hi, *r_pub)
            Fu, Pu = bills_from_energies(G_lo, G_hi, *r_full)
            Fl, Pl = bills_from_energies(G_lo, G_hi, *r_lam)
            # clipping part per class-season: (alpha_sh - lambda*) x shoulder energy at its base / flat bill
            sh_val = (BASE_LO[:, None] * G_lo[..., 1] + BASE_HI[:, None] * G_hi[..., 1])         # (5, 12)
            res = {}
            for key, mm in (("ann", None), ("w", WINTER_M), ("s", ~WINTER_M)):
                e_fz, e_rr, e_rf, e_lm = eps(Ff, Pf, mm), eps(Fr, Pr, mm), eps(Fu, Pu, mm), eps(Fl, Pl, mm)
                if key == "ann":
                    clip_part = sum(100.0 * (fix["alpha"][:, k, 1] - fix["lam_star"][:, k]) *
                                    sh_val[:, msk].sum(1) for k, msk in ((0, WINTER_M), (1, ~WINTER_M))) / Fu.sum(1)
                else:
                    k = 0 if key == "w" else 1
                    clip_part = 100.0 * (fix["alpha"][:, k, 1] - fix["lam_star"][:, k]) * sh_val[:, mm].sum(1) \
                        / Fu[:, mm].sum(1)
                res[key] = {"frz": e_fz, "rec_round": e_rr, "rec_full": e_rf, "lam_full": e_lm,
                            "clip_part": clip_part, "thr_part": e_rf - clip_part, "rnd_part": e_rr - e_rf}
            res["single_base"] = np.stack([single[:, WINTER_M].all(1), single[:, ~WINTER_M].all(1)], axis=1)
            res["G"] = (G_lo, G_hi)
            res["bills_frz"] = (Ff, Pf)
            neu[case] = res
        out["neu"] = neu
    return out


# ── record flattening ─────────────────────────────────────────────────────────────────────────────────────────
def flat_record(ev, alpha0):
    """One CSV row of a run: multipliers, clips, partition, flags, decomposition, neutrality. Class keys are the
    article's class names."""
    r = {}
    for ci, c in enumerate(CLASS_ORDER):
        for k, sk in enumerate(SEASON_KEYS):
            for zi, z in enumerate(ZONE_KEYS):
                r[f"a_fix_{c}_{sk}_{z}"] = ev["fix"]["alpha"][ci, k, zi]
                if "rec" in ev:
                    r[f"a_rec_{c}_{sk}_{z}"] = ev["rec"]["alpha"][ci, k, zi]
                    r[f"D_{c}_{sk}_{z}"] = ev["D"][ci, k, zi]
            r[f"lam_{c}_{sk}"] = ev["fix"]["lam_star"][ci, k]
            r[f"clip_{c}_{sk}"] = bool(ev["fix"]["clip"][ci, k])
    r["clip_any"] = bool(ev["fix"]["clip"].any())
    if "rec" in ev:
        r["clip_any_rec"] = bool(ev["rec"]["clip"].any())
        for s, a in enumerate(ARLPS):
            r[f"cls_{a}"] = CLASS_ORDER[ev["names"][s]]
            for k, sk in enumerate(SEASON_KEYS):
                for zi, z in enumerate(ZONE_KEYS):
                    r[f"dre_{a}_{sk}_{z}"] = ev["d_re"][s, k, zi]
        for c, f in ev["match"]["flags"].items():
            r[f"flag_mem_{c}"], r[f"flag_sub_{c}"], r[f"flag_amb_{c}"] = (f["membership_changed"],
                                                                         f["substantially_changed"], f["ambiguous"])
            r[f"jac_{c}"] = f["jaccard"]
        for st_, d in ev["match"]["strata"].items():
            r[f"margin_{st_}"], r[f"tie_{st_}"] = d["margin"], d["tie"]
        r.update(ev["part"])
    if "neu" in ev:
        for case, res in ev["neu"].items():
            for ci, c in enumerate(CLASS_ORDER):
                for key in ("ann", "w", "s"):
                    for part in ("frz", "rec_round", "rec_full", "clip_part", "thr_part", "rnd_part"):
                        r[f"eps_{case}_{part}_{c}_{key}"] = res[key][part][ci]
                for k, sk in enumerate(SEASON_KEYS):
                    r[f"single_{case}_{c}_{sk}"] = bool(res["single_base"][ci, k])
    return r


def reassign_share(names, H0):
    H0 = np.asarray(H0, dtype=float)
    return float(H0[np.asarray(names) != P0].sum() / H0.sum())


def structural(alpha):
    """Per run: cross-class alpha_peak order A > B > ESH-1 > C > ESH-2 per season, the ESH-1/C pair, and the numeric
    claims on the ratios (keys K_*; the article's class names)."""
    a = np.asarray(alpha)
    ratio = a[..., 2] / a[..., 0]                                      # (5, 2)
    iA, iB, iC, iE1, iE2 = range(5)
    out = {}
    for k, sk in enumerate(SEASON_KEYS):
        pk = a[:, k, 2]
        out[f"order_full_{sk}"] = bool(pk[iA] > pk[iB] > pk[iE1] > pk[iC] > pk[iE2])
        out[f"esh1_gt_c_{sk}"] = bool(pk[iE1] > pk[iC])
        out[f"order_nonesh_{sk}"] = bool(pk[iA] > pk[iB] > pk[iC])
        out[f"order_esh_{sk}"] = bool(pk[iE1] > pk[iE2])
        out[f"K_04_{sk}"] = bool(ratio[iA, k] > ratio[iB, k])
    flat = ratio[[iC, iE1, iE2]].ravel()
    mass = ratio[[iA, iB]].ravel()
    out["K_01_flat_min"], out["K_01_flat_max"] = float(flat.min()), float(flat.max())
    out["K_01_mass_min"], out["K_01_mass_max"] = float(mass.min()), float(mass.max())
    out["K_01_holds"] = bool(round(flat.min(), 2) == 1.04 and round(flat.max(), 2) == 1.33 and
                               round(mass.min(), 2) == 1.88 and round(mass.max(), 2) == 2.99)
    out["K_02_holds"] = bool((ratio[iA] > 2.03).all() and (flat < 1.38).all())
    out["K_03_holds"] = bool(ratio[iA, 1] > ratio[iA, 0] and ratio[iB, 1] > ratio[iB, 0] and
                               all(ratio[i, 1] < ratio[i, 0] for i in (iC, iE1, iE2)))
    return out
