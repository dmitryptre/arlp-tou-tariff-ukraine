"""Within-class distribution of the shift effect and the combined effect.

The units of the passive bill distribution (class-EULP pairs, case T) respond to the proposed 3z tariff (P3) with their
class's zone multipliers of the source-specific PEED scenarios (`peed_validity_diagnostics.csv`, file 7):
    E1_umz = E_umz * a_z,tau(m),   a = max(1 + delta, 0)                       (NR)
    E1_umz <- eta_u,tau E1_umz,    eta_u,tau = sum_{m in tau, z} E / sum_{m in tau, z} E1   (R: each dwelling's
                                                                                          seasonal energy restored)
Bills by the passive-bill rules (eulp_bill_distribution.unit_bills, case T: the ESH winter base of a month set by the
dwelling's own baseline consumption; the response keeps that status, as the class bills of the source-specific PEED
scenarios keep a fixed effective winter base; the variant in which the status follows the response energy is a reported
sensitivity). Cohorts F->P3 and E2->P3; changes in % of the cohort's baseline bill B_s(E0),
positive = a bill increase (the article's bill tables use the opposite sign):
    shift    = 100 (B_P3(E1) - B_P3(E0)) / B_s(E0)
    combined = 100 (B_P3(E1) - B_s(E0)) / B_s(E0)      (= passive + shift)
Within-zone allocation (proportional / valley filling) does not change the zone energies, so not the bills.
"""
from __future__ import annotations

import eulp_bill_distribution as bd
import numpy as np
import pandas as pd
from eulp_hierarchy import ESH_CLASSES, SEASONS, ZONES

PATHWAYS = ("F->P3", "E2->P3")
COHORT = {"F->P3": "F", "E2->P3": "E2"}
NPZ_PATHWAY = {"F->P3": "F_to_P3", "E2->P3": "E2_to_P3"}
ZONE_SENS = {"off": "off-peak", "shoulder": "shoulder", "peak": "peak"}     # eulp_hierarchy zone -> peed_sensitivity zone name
EFFECTS = ("shift", "combined")
N_SCENARIOS = 24


def load_multipliers(diag: pd.DataFrame) -> dict:
    """File 7 -> {(peed_matrix, peed_scale, pathway, class, season): a (3,) in ZONES order}, a = max(1 + delta, 0).
    Also asserts that file 7's `mult_<zone>` equals 1 + delta (the multiplier before the floor)."""
    out = {}
    for r in diag[diag.pathway.isin(PATHWAYS)].to_dict("records"):
        d = np.array([r[f"delta_{ZONE_SENS[z]}"] for z in ZONES], dtype=float)
        m = np.array([r[f"mult_{ZONE_SENS[z]}"] for z in ZONES], dtype=float)
        assert np.allclose(m, 1.0 + d, rtol=0, atol=1e-12), (r["peed_matrix"], r["pathway"], r["class"], r["season"])
        out[(r["peed_matrix"], float(r["peed_scale"]), r["pathway"], r["class"], r["season"])] = np.maximum(1.0 + d, 0.0)
    return out


def unit_multipliers(classes, mult: dict, matrix: str, scale: float, pathway: str) -> np.ndarray:
    """(n, 12, 3): each unit's zone multipliers for each month, from its class and the month's season."""
    classes = list(classes)
    seasons = [SEASONS[i] for i in bd.SEASON_IDX]
    cache = {c: np.array([mult[(matrix, float(scale), pathway, c, s)] for s in seasons]) for c in set(classes)}
    return np.stack([cache[c] for c in classes])


def respond(E, A, energy_treatment: str):
    """E (n, 12, 3) monthly zone energies, A (n, 12, 3) multipliers -> (E1, eta (n, 2) by season; 1 under NR)."""
    E = np.asarray(E, dtype=float)
    E1 = E * np.asarray(A, dtype=float)
    eta = np.ones((E.shape[0], 2))
    if energy_treatment == "NR":
        return E1, eta
    assert energy_treatment == "R", energy_treatment
    for k in (0, 1):
        m = bd.SEASON_IDX == k
        e0, e1 = E[:, m, :].sum(axis=(1, 2)), E1[:, m, :].sum(axis=(1, 2))
        assert (e1 > 0).all(), "a unit's seasonal energy is zero after the floor: the R rescale is undefined"
        eta[:, k] = e0 / e1
        E1[:, m, :] *= eta[:, k][:, None, None]
    return E1, eta


def bills(E, classes, rates, is_esh, theta=None) -> dict:
    """The passive-rule bills of energy E. theta None: case T, the ESH status of each winter month from E's own monthly
    consumption; else the given statuses (the primary response bills keep each month's baseline status, as the class
    bills of the source-specific PEED scenarios keep a fixed effective winter base)."""
    if theta is None:
        theta = bd.base_regime("T", np.asarray(E).sum(axis=2), is_esh)
    ub = bd.unit_bills(E, classes, theta, rates)
    return {"F": ub["B0"], "E2": ub["BE2"], "P3": ub["BP3"], "kWh": ub["annual_kWh"], "theta": theta,
            "winter_months_above_2000": (theta[:, bd.WINTER_MASK] > 0).sum(axis=1) * np.asarray(is_esh, bool)}


def effects(b0: dict, b1: dict, cohort: str) -> dict:
    """Shift and combined changes in % of the cohort's baseline bill (positive = increase), and the passive part."""
    bs = b0[cohort]
    return {"base": bs, "passive": 100.0 * (b0["P3"] - bs) / bs, "shift": 100.0 * (b1["P3"] - b0["P3"]) / bs,
            "combined": 100.0 * (b1["P3"] - bs) / bs}


def is_esh_array(classes) -> np.ndarray:
    return np.array([c in ESH_CLASSES for c in classes])


def tick_step(rule, max_labels=8) -> float:
    """x-tick step of the supplementary shift- and combined-effect histograms: the passive-histogram step for the bin
    width (eulp_bill_distribution.x_tick_step), widened to 10, 20, 25 or 50 pp until the axis carries at most
    `max_labels` intervals (that step alone overlaps above about 100 pp)."""
    s = bd.x_tick_step(rule["width_pp"])
    span = rule["range_U_pct"] - rule["range_L_pct"] + 2 * rule["width_pp"]
    for c in (s, 10.0, 20.0, 25.0, 50.0):
        if c >= s and span / c <= max_labels:
            return c
    return 100.0
