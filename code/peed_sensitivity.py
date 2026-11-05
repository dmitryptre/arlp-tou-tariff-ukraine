"""Source-specific PEED scenarios: helpers for the source-specific PEED cells of the autonomous notebook.

eps(omega, lambda) = lambda * eps(omega). omega is one of three published matrices (F11-SR, F11-LR, A19);
lambda (`peed_scale`) is the PEED response intensity, a sensitivity parameter. It is not the participation
share p_t and not an estimated behavioural parameter.

The functions here are pure: they take arrays, frames and mappings and return new objects. The notebook
supplies the loads, price signals, rates and prices; an exact check ties these helpers to the notebook's
reference cells 52, 53, 61 and 63.

Hour index h = 0..23 is the house axis of the notebook: index 0 is hour-ending 24, index h is hour-ending h.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── design ────────────────────────────────────────────────────────────────────────────────────────────────────
PEED_MATRIX_IDS = ("F11-SR", "F11-LR", "A19")            # no central matrix; this order is fixed for tables
PEED_MATRIX_CODE = {"F11-SR": "F11SR", "F11-LR": "F11LR", "A19": "A19"}
PEED_FILES = {"F11-SR": "filippini2011_lsdvc_sr.csv", "F11-LR": "filippini2011_lsdvc_lr.csv",
              "A19": "auray2019_aids_table4.csv"}
PEED_SCALES = {"full": 1.00, "attenuated": 0.25}          # plus the no-response control, lambda = 0
PEED_SCALE_LABEL = {1.00: "full transferred-elasticity response (λ = 1.00)",
                    0.25: "attenuated response (λ = 0.25)", 0.0: "no response (λ = 0)"}
PEED_SWEEP = [round(i / 100, 2) for i in range(101)]      # 0.50 is a point, not a scenario
ALLOCATIONS = ("proportional", "valley_fill")
ALLOCATION_CODE = {"proportional": "P", "valley_fill": "VF"}
ENERGY_TREATMENTS = ("R", "NR")
ZONES_2 = ("peak", "off-peak")
ZONES_3 = ("peak", "shoulder", "off-peak")
PATHWAYS = ("F->P3", "E2->P3", "F->E3", "E2->E3", "E3->P3")
PRINCIPAL_PATHWAYS = PATHWAYS[:4]                         # E3->P3 is a bill comparison only
SCHEMES = {"P3": ("F->P3", "E2->P3"), "E3": ("F->E3", "E2->E3")}
PATHWAY_TARIFFS = {"F->P3": ("F", "P3"), "E2->P3": ("E2", "P3"), "F->E3": ("F", "E3"),
                   "E2->E3": ("E2", "E3"), "E3->P3": ("E3", "P3")}

# ── accounting ────────────────────────────────────────────────────────────────────────────────────────────────
# the gross price is net + dist + trans
COMPONENTS = ("net", "dist", "trans")
PRICE_COLUMNS = {"gross": "DAM_gross_price_uah_mwh", "net": "DAM_net_price_uah_mwh",
                 "dist": "distribution_uah_mwh", "trans": "transmission_uah_mwh"}
NET_LABELS = {"N_RE": "retail-equivalent net position",
              "N_ret": "change in retailer contribution under the market-procurement counterfactual",
              # changes relative to the modelling-window baseline
              "dV_PSO": "change in regulated PSO compensation",
              "dPi": "change in the supplier balance (reference value minus procurement cost)",
              "dG": "change in the cost-based financing gap (dV_PSO - dPi)"}

# ── classification ───────────────────────────────────────────────────────────────────────────────────────────
TAU_FLOOR = 0.10               # F2 if the relative floor influence is at least 10 %
A_MONEY_MUAH = 100.0           # materiality floor of money results, ₴ M (the ₴0.1 bn rule)
A_PP = 0.1                     # materiality floor of the bill-impact table's totals, percentage points
X_RED = -0.80                  # extreme reduction: final zone change of -80 % or less
X_RED_BINS = (-0.80, -0.90, -0.95, -1.00)
X_INC = 1.00                   # extreme increase: final zone change of +100 % or more
X_RES = (0.80, 1.25)           # extreme rescale: eta outside this interval (R only)
NF_ETA_STABLE = (0.5, 2.0)     # NF reference unstable if eta_NF lies outside this interval (floored cases)
VF_TOL_REL = 1e-12             # valley filling acts only if the off-peak increment exceeds 1e-12 x zone energy
PEAK_TIE_MW = 1e-6             # hours within 1e-6 MW of the maximum are tied
FLOOR_CLASSES = ("F0", "F?", "F2", "F1")
SUPPORT_LABELS = {1: "not assessable: no applicable cells", 2: "holds in every cell, with clean support",
                  3: "holds in every clean cell", 4: "conditional", 5: "rests only on flagged cells",
                  6: "not supported"}
DIMENSIONS = ("peed_matrix", "peed_scale", "energy_treatment", "allocation")


# ── scenario identifiers ───────────────────────────────────────────────────────────────────────────────────
def scenario_id(peed_matrix, peed_scale, energy_treatment=None, allocation=None, nf=False):
    """X-<matrix>-<100*lambda>-<R|NR>-<P|VF>[-NF]; the control is X-CTRL-000."""
    if peed_scale == 0:
        return "X-CTRL-000"
    sid = (f"X-{PEED_MATRIX_CODE[peed_matrix]}-{round(100 * peed_scale):03d}-{energy_treatment}-"
           f"{ALLOCATION_CODE[allocation]}")
    return sid + "-NF" if nf else sid


def sweep_id(peed_matrix, energy_treatment, allocation):
    return f"X-{PEED_MATRIX_CODE[peed_matrix]}-S-{energy_treatment}-{ALLOCATION_CODE[allocation]}"


def scenario_grid():
    """The 24 scenarios in the fixed order matrix -> lambda (1.00, 0.25) -> treatment -> allocation."""
    return [{"peed_matrix": m, "peed_scale": s, "energy_treatment": t, "allocation": a}
            for m in PEED_MATRIX_IDS for s in (PEED_SCALES["full"], PEED_SCALES["attenuated"])
            for t in ENERGY_TREATMENTS for a in ALLOCATIONS]


# ── matrices ─────────────────────────────────────────────────────────────────────────────────────────────────
def load_source_matrices(folder):
    """Read the three source-specific 2x2 matrices: index `zone_t` = consumption zone, columns = price zone."""
    out = {}
    for mid in PEED_MATRIX_IDS:
        df = pd.read_csv(Path(folder) / PEED_FILES[mid], index_col=0)
        assert df.index.name == "zone_t", (mid, df.index.name)
        assert tuple(df.index) == ZONES_2 and tuple(df.columns) == ZONES_2, (mid, df.index, df.columns)
        assert df.shape == (2, 2) and df.notna().all().all(), mid
        out[mid] = df.astype(float)
    return out


def shoulder_extension(m2):
    """3x3 matrix by the shoulder-inert rule; identical to the notebook's build_3x3 (cell 22)."""
    epp, eoo = m2.loc["peak", "peak"], m2.loc["off-peak", "off-peak"]
    epo, eop = m2.loc["peak", "off-peak"], m2.loc["off-peak", "peak"]
    m3 = pd.DataFrame(0.0, index=list(ZONES_3), columns=list(ZONES_3))
    m3.loc["peak", "peak"] = epp
    m3.loc["peak", "off-peak"] = epo
    m3.loc["off-peak", "peak"] = eop
    m3.loc["off-peak", "off-peak"] = eoo
    m3.loc["shoulder", "shoulder"] = 0.5 * (epp + eoo)
    return m3


def scale_peed(matrix, peed_scale):
    """lambda * matrix; lambda >= 0."""
    peed_scale = float(peed_scale)
    assert peed_scale >= 0.0, peed_scale
    return matrix * peed_scale


# ── response pipeline ──────────────────────────────────────────────────────────────────────────────────────
def zone_deltas(du, matrix):
    """delta_i = sum_z eps_iz * du_z, summed in the order of ZONES_3 as apply_peed_zonal does (bit-identical)."""
    return {z: sum(matrix.loc[z, zh] * du[zh] for zh in ZONES_3) for z in ZONES_3}


def linear_response(l0, du, matrix, period_of_hour):
    """Linear response before the floor, l0 * (1 + delta_i(h)); bit-identical to apply_peed_zonal before max(., 0).

    Returns (l_lin, delta_by_zone)."""
    dz = zone_deltas(du, matrix)
    dvec = np.array([dz[period_of_hour[h]] for h in range(24)], dtype=float)
    return np.asarray(l0, float) * (1.0 + dvec), dz


def floor_threshold(delta):
    """lambda* = -1/delta for delta < 0 (the zone reaches zero at lambda*), else inf; delta at lambda = 1."""
    return -1.0 / delta if delta < 0 else math.inf


def fractional_factor(peed_scale, gamma):
    """phi(lambda) = lambda (1 + gamma) / (1 + lambda gamma): the R response relative to the unclipped lambda = 1
    response, for lambda below the smallest floor threshold."""
    return peed_scale * (1.0 + gamma) / (1.0 + peed_scale * gamma)


def energy_factor(l0, l_plus, energy_treatment):
    """eta: sum(l0)/sum(l_plus) under R (sum(l_plus) > 0 asserted; no fallback), exactly 1.0 under NR."""
    if energy_treatment == "NR":
        return 1.0
    assert energy_treatment == "R", energy_treatment
    e0, e1 = float(np.sum(l0)), float(np.sum(l_plus))
    assert e1 > 0.0, f"sum of the floored load is {e1}: the R rescale is undefined"
    return e0 / e1


def fill_level(l0_off, delta_e):
    """theta with sum(max(l0, theta) - l0) = delta_e > 0 over the off-peak hours (closed form)."""
    v = np.sort(np.asarray(l0_off, float))
    assert delta_e > 0.0, delta_e
    csum = np.cumsum(v)
    m = 1
    for k in range(1, len(v) + 1):
        if k * v[k - 1] - csum[k - 1] <= delta_e:
            m = k
    return (delta_e + csum[m - 1]) / m


def valley_fill(l0, l1, off_hours, tol=None):
    """Water-fill the off-peak increment of l1 from the baseline shape l0 (a heuristic).

    Returns (l_vf, theta, n_filled, applied). If the off-peak increment is not above tol (default
    VF_TOL_REL x the baseline off-peak energy), l1 is returned unchanged with applied = False."""
    l0 = np.asarray(l0, float)
    l1 = np.asarray(l1, float)
    off = list(off_hours)
    if tol is None:
        tol = VF_TOL_REL * float(l0[off].sum())
    delta_e = float((l1[off] - l0[off]).sum())
    if not delta_e > tol:
        return l1.copy(), math.nan, 0, False
    theta = fill_level(l0[off], delta_e)
    out = l1.copy()
    out[off] = np.maximum(l0[off], theta)
    return out, float(theta), int((l0[off] < theta).sum()), True


def volume_timing_split(value0, e0, e1, value1):
    """Split the change value1 - value0 of a quantity linear in the load into volume and timing parts.

    nu = e1/e0; volume = (nu - 1) value0; timing = value1 - nu value0; volume + timing = change.
    Returns (change, volume, timing)."""
    nu = e1 / e0
    return value1 - value0, (nu - 1.0) * value0, value1 - nu * value0


def bill_volume_aware(q_annual, rates_by_season, l0_by_season, l_by_season, days):
    """Volume-aware bill: Bill = Q(c) sum_tau |D_tau| sum_h r l / sum_tau |D_tau| sum_h l0 (the baseline energy).

    Returns (total, {season: season term over the same annual denominator})."""
    den = sum(days[s] * float(np.sum(l0_by_season[s])) for s in l0_by_season)
    terms = {s: q_annual * days[s] * float(np.dot(np.asarray(rates_by_season[s], float),
                                                  np.asarray(l_by_season[s], float))) / den
             for s in l_by_season}
    return sum(terms.values()), terms


def price_components(prices_df, seasons, winter_months):
    """Seasonal hourly mean prices by component in ₴/kWh, grouped exactly as cell 53 groups the gross
    price: {component: DataFrame indexed by hour 0..23 with one column per season}."""
    d = prices_df.copy()
    d["season"] = np.where(d["timestamp"].dt.month.isin(set(winter_months)), "winter", "summer")
    d["hour"] = d["timestamp"].dt.hour
    out = {}
    for k, col in PRICE_COLUMNS.items():
        out[k] = (d.groupby(["season", "hour"])[col].mean().unstack(0).reindex(range(24)) / 1000)[list(seasons)]
    return out


# ── retailer side ────────────────────────────────────────────────────────────────────────────────────────────
def retailer_metrics(terms, xi):
    """Both retailer-side metrics of one scheme, from per-class terms that are already cohort-weighted.

    terms: DataFrame indexed by class with the retailer-sign columns R_pas, R_resp, R_resp_vol, R_resp_tim and,
           for k in COMPONENTS, the cost savings S_k, S_k_vol, S_k_tim (profile basis; S = -change in cost).
    xi:    Series by class, Q(c) / annual profile energy of the baseline (the bill-basis ratio).
    Returns a Series (₴ M): the components, S_en, S_ext, S_DAM (= S_ext), S_ext_Q (each with _vol and _tim),
    R_pas, R_resp(_vol, _tim), R_tot, N_RE and N_ret (each with _vol and _tim), bridge_basis (= N_RE - N_ret)."""
    xi = pd.Series(xi).reindex(terms.index)
    assert xi.notna().all(), "xi missing for a class"
    out = {}
    for part in ("", "_vol", "_tim"):
        for k in COMPONENTS:
            out[f"S_{k}{part}"] = float(terms[f"S_{k}{part}"].sum())
        ext_c = terms[f"S_net{part}"] + terms[f"S_dist{part}"] + terms[f"S_trans{part}"]
        out[f"S_en{part}"] = out[f"S_net{part}"]
        out[f"S_ext{part}"] = float(ext_c.sum())
        out[f"S_DAM{part}"] = out[f"S_ext{part}"]           # profile basis
        out[f"S_ext_Q{part}"] = float((xi * ext_c).sum())
    out["R_pas"] = float(terms["R_pas"].sum())
    for part in ("", "_vol", "_tim"):
        out[f"R_resp{part}"] = float(terms[f"R_resp{part}"].sum())
    out["R_tot"] = out["R_pas"] + out["R_resp"]
    out["N_RE"] = out["R_tot"] + out["S_DAM"]
    out["N_ret"] = out["R_tot"] + out["S_ext_Q"]
    for part in ("_vol", "_tim"):
        out[f"N_RE{part}"] = out[f"R_resp{part}"] + out[f"S_DAM{part}"]
        out[f"N_ret{part}"] = out[f"R_resp{part}"] + out[f"S_ext_Q{part}"]
    out["bridge_basis"] = out["S_ext"] - out["S_ext_Q"]
    return pd.Series(out)


# ── peaks ──────────────────────────────────────────────────────────────────────────────────────────────────────
def hour_ending(idx):
    """House index 0..23 -> hour-ending 24, 1..23."""
    return 24 if int(idx) == 0 else int(idx)


def peak_metrics(base, profile, peak_hours, period_of_hour, tol=PEAK_TIE_MW):
    """Maximum, hour-ending, zone, shave and flags of a 24-h profile against its baseline.

    Ties: hours within tol of the maximum; the first in clock order from hour-ending 24 (index 0) is reported,
    with the span. migrated: the zone of the maximum differs from the baseline's; boundary: the maximum sits next
    to an hour of another zone; offpeak_plateau: the tied hours cover the whole off-peak zone."""
    base = np.asarray(base, float)
    prof = np.asarray(profile, float)
    pk = list(peak_hours)
    off = [h for h in range(24) if period_of_hour[h] == "off-peak"]
    if not np.all(np.isfinite(prof)):
        keys = ("base_max_MW", "base_hour_ending", "base_zone", "max_MW", "hour_ending", "zone", "shave_MW", "n_tied",
                "tied_hour_endings", "migrated", "boundary", "offpeak_plateau", "peakzone_mean_change_pct",
                "offpeak_mean_change_pct")
        return {k: math.nan for k in keys}

    def _max(v):
        tied = [h for h in range(24) if v[h] >= v.max() - tol]
        return float(v.max()), tied[0], tied

    bmax, bidx, _ = _max(base)
    nmax, nidx, tied = _max(prof)
    nb = [(nidx - 1) % 24, (nidx + 1) % 24]
    return {"base_max_MW": bmax, "base_hour_ending": hour_ending(bidx), "base_zone": period_of_hour[bidx],
            "max_MW": nmax, "hour_ending": hour_ending(nidx), "zone": period_of_hour[nidx],
            "shave_MW": bmax - nmax, "n_tied": len(tied), "tied_hour_endings": " ".join(str(hour_ending(h)) for h in tied),
            "migrated": period_of_hour[nidx] != period_of_hour[bidx],
            "boundary": any(period_of_hour[h] != period_of_hour[nidx] for h in nb),
            "offpeak_plateau": len(tied) > 1 and set(off) <= set(tied),
            "peakzone_mean_change_pct": 100.0 * (prof[pk].mean() / base[pk].mean() - 1.0),
            "offpeak_mean_change_pct": 100.0 * (prof[off].mean() / base[off].mean() - 1.0)}


def crossing(xs, ys):
    """The x values where ys changes sign (strictly), refined by linear interpolation; exact zeros included."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    out = []
    for i in range(len(xs)):
        if ys[i] == 0.0:
            out.append(float(xs[i]))
        elif i + 1 < len(xs) and ys[i] * ys[i + 1] < 0.0:
            out.append(float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])))
    return out


# ── physical-validity diagnostics ────────────────────────────────────────────────────────────────────────────
def validity_diagnostics(l0, l_plus, delta_by_zone, du, zone_hours):
    """Physical-validity diagnostics for one class x season x pathway case at one matrix and lambda.

    l_plus is the floored linear load actually used (R rescales it, NR keeps it). Returns a flat dict."""
    l0 = np.asarray(l0, float)
    l_plus = np.asarray(l_plus, float)
    mult = {z: 1.0 + float(delta_by_zone[z]) for z in ZONES_3}
    e0, ep = float(l0.sum()), float(l_plus.sum())
    gamma_plus = ep / e0 - 1.0
    eta = e0 / ep if ep > 0 else math.nan
    fin = {"R": {z: eta * max(mult[z], 0.0) - 1.0 for z in ZONES_3},
           "NR": {z: max(mult[z], 0.0) - 1.0 for z in ZONES_3}}
    zero = [z for z in ZONES_3 if mult[z] <= 0.0]
    e_s0 = float(l0[zone_hours["shoulder"]].sum())
    out = {f"delta_{z}": float(delta_by_zone[z]) for z in ZONES_3}
    out.update({f"mult_{z}": mult[z] for z in ZONES_3})
    out.update({f"lambda_star_{z}": floor_threshold(float(delta_by_zone[z])) for z in ZONES_3})
    out["m_min"] = min(mult.values())
    out["negative_load"] = out["m_min"] < 0.0
    out["floored"] = out["m_min"] <= 0.0
    out["n_zones_zero"] = len(zero)
    out["zones_zero"] = " ".join(zero)
    out["gamma_plus"] = gamma_plus
    out["eta_R"] = eta
    for t in ENERGY_TREATMENTS:
        for z in ZONES_3:
            out[f"final_change_{t}_{z}"] = fin[t][z]
        out[f"max_final_change_{t}"] = max(fin[t].values())
        out[f"min_final_change_{t}"] = min(fin[t].values())
    out["max_raw_delta"] = max(float(v) for v in delta_by_zone.values())
    out["min_raw_delta"] = min(float(v) for v in delta_by_zone.values())
    out["shoulder_change_by_rescale_pct"] = (100.0 * (eta - 1.0) * float(l_plus[zone_hours["shoulder"]].sum()) / e_s0
                                             if e_s0 > 0 else math.nan)
    still = [z for z in ZONES_3 if du[z] == 0.0]
    out["zones_with_unchanged_price"] = " ".join(still)
    out["unchanged_price_change_R_pct"] = (100.0 * max(fin["R"][z] for z in still) if still else math.nan)
    return out


def x_flags(final_changes, etas, energy_treatment):
    """Extreme-response flags from final zone changes (fractions) and rescale factors."""
    fc = [float(v) for v in final_changes]
    return {"X_red": any(v <= X_RED for v in fc), "X_inc": any(v >= X_INC for v in fc),
            "X_res": energy_treatment == "R" and any(not (X_RES[0] <= float(e) <= X_RES[1]) for e in etas)}


# ── floor influence ────────────────────────────────────────────────────────────────────────────────────────
def floor_class(fed_by_floored, nf_status, delta_f=None, d_r=None, a_r=0.0, tau=TAU_FLOOR, hour_changed=None):
    """F0 -> F? -> F2 -> F1, the first that applies.

    fed_by_floored: any case feeding the result is floored; nf_status: "ok", "undefined" or "unstable" (the worst
    over the floored feeding cases); for the hour of a maximum pass hour_changed instead of delta_f."""
    if not fed_by_floored:
        return "F0"
    if nf_status in ("undefined", "unstable"):
        return "F?"
    if hour_changed is not None:
        return "F2" if hour_changed else "F1"
    return "F2" if relative_floor_influence(delta_f, d_r, a_r) >= tau else "F1"


def relative_floor_influence(delta_f, d_r, a_r=0.0):
    """rho_F = |Delta_F| / max(D_r, A_r)."""
    den = max(float(d_r), float(a_r))
    if den > 0.0:
        return abs(float(delta_f)) / den
    return 0.0 if float(delta_f) == 0.0 else math.inf


# ── robustness of a claim ──────────────────────────────────────────────────────────────────────────────────
def _partition(cells, holds, dims):
    key = cells[list(dims)].astype(str).agg(" | ".join, axis=1)
    groups = holds.groupby(key).agg(["min", "max"])
    if (groups["min"] != groups["max"]).any():
        return None
    held = sorted(groups.index[groups["max"].astype(bool)])
    return f"({', '.join(dims)}) in {{{'; '.join(held)}}}"


def classify_claim(cells, holds, levels):
    """Coverage axis, evidence axis and support label of one claim over distinct cells.

    cells:  DataFrame, one row per distinct cell, with the columns of DIMENSIONS that vary in the grid;
    holds:  Series aligned with cells: True / False, or NaN where the criterion cannot be evaluated (u0);
    levels: Series aligned with cells: the highest evidence level of the cell, "L2" (clean), "L1" (no F2 or F?
            result used by the claim, but an X flag) or "L0" (everything else).
    Returns a dict with n0, h0, f0, u0, n1, h1, n2, h2, the condition, the matrices and lambdas in L2, and the
    label (number and text). Subsets are sized before the claim is tested (an empty subset never supports)."""
    holds = pd.Series(holds, index=cells.index)
    levels = pd.Series(levels, index=cells.index)
    ev = holds.notna()
    h = holds[ev].astype(bool)
    lv = levels[ev]
    n0, h0 = int(ev.sum()), int(h.sum())
    in1 = lv.isin(["L1", "L2"])
    in2 = lv.eq("L2")
    n1, h1 = int(in1.sum()), int(h[in1].sum()) if in1.any() else 0
    n2, h2 = int(in2.sum()), int(h[in2].sum()) if in2.any() else 0
    if n0 == 0:
        label = 1
    elif h0 == n0 and n2 >= 1:
        label = 2
    elif n2 >= 1 and h2 == n2 and h0 < n0:
        label = 3
    elif n2 >= 1 and 1 <= h2 < n2:
        label = 4
    elif h0 >= 1 and h2 == 0:
        label = 5
    else:
        label = 6
    condition = None
    if 0 < h0 < n0:
        sub = cells[ev]
        dims = [d for d in DIMENSIONS if d in sub.columns and sub[d].nunique() > 1]
        for k in (1, 2):
            for combo in itertools.combinations(dims, k):
                condition = _partition(sub, h, combo)
                if condition is not None:
                    break
            if condition is not None:
                break
        if condition is None:
            condition = "not separable by one or two dimensions; holding cells: " + "; ".join(
                " | ".join(str(v) for v in row) for row in sub[h].itertuples(index=False))
    l2 = cells[ev][in2]
    return {"n0": n0, "h0": h0, "f0": n0 - h0, "u0": int((~ev).sum()), "n1": n1, "h1": h1, "n2": n2, "h2": h2,
            "condition": condition,
            "matrices_L2": " ".join(sorted(set(l2["peed_matrix"]))) if "peed_matrix" in l2 else "",
            "lambdas_L2": " ".join(f"{v:.2f}" for v in sorted(set(l2["peed_scale"]))) if "peed_scale" in l2 else "",
            "label": label, "label_text": SUPPORT_LABELS[label]}
