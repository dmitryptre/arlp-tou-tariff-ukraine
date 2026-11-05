# peed_rollout_retailer.py
# ---------------------------------------------------------------------------
# The rollout of the proposed 3z tariff (P3) per designed scenario and year
# 2025-2040 (file 17): the data of the participation-trajectory figure and of
# the rollout paragraphs of the article.
#
# The figure shows, relative to the modelling-window baseline,
#   dV_t   = sum_m r_ref_m dQ_m(t) - dRev_t     change in regulated PSO compensation
#   dPi_t  = sum_m r_ref_m dQ_m(t) - dC_sup_t   change in the supplier balance
# with the load L_t = L0 + w_F(t) dL(F) + w_E2(t) dL(E2) and the baseload block of
# year t taken from the minimum of L_{t-1} (the first year: L0), so dV_t is
# additive in the cohort shares and dPi_t is not (peed_pso.rollout). The loads
# come from _peed_sensitivity_arrays.npz, the prices from inputs/ (as the notebook).
# At the cohort weights 1 - S0 and S0, dV, dPi and B reproduce file 5 (asserted).
#
# Auxiliary columns: per channel (F = newly metered F->P3, E2 = households on
# the existing 2z tariff, E2->P3) and class, for all households of the channel
# (unweighted),
#   delta_R   = -(passive + response bill change)   (retail revenue)
#   delta_C_Q = -S_ext_Q                             (supply cost, annual-consumption basis)
#   N_ret     = delta_R - delta_C_Q
# and per year N_t^ret = p_flat_to_3z(t) * N_ret(F) + p_2z_to_3z(t) * N_ret(E2).
#
# Reads layer13_peed_sensitivity/ files 15 (peed_volume_timing.csv), 5
# (peed_sensitivity_retailer.csv), 1b (peed_scenario_registry.csv), 7c
# (peed_validity_economic.csv) and _peed_sensitivity_arrays.npz,
# layer13_peed_dr/penetration_forecast.csv, inputs/hourly_dam_prices_uah.csv and
# inputs/base_m_prices_uah.csv. Writes peed_rollout_retailer.csv (file 17), read
# by make_participation_trajectory.py.
#
# Run from code/ :  python peed_rollout_retailer.py
# ---------------------------------------------------------------------------
from pathlib import Path

import _peed_fig_common as fc
import numpy as np
import pandas as pd
import peed_pso as pp
from _common import WINTER_MONTHS

OUT_NAME = "peed_rollout_retailer.csv"
FORECAST = ("outputs", "ARLP_Clustering", "notebook_only", "layer13_peed_dr", "penetration_forecast.csv")
NPZ = "_peed_sensitivity_arrays.npz"
PRICES = ("inputs", "hourly_dam_prices_uah.csv")
BASE_M = ("inputs", "base_m_prices_uah.csv")
CHANNEL_PATHWAY = {"F": "F_to_P3", "E2": "E2_to_P3"}
PSO_COLUMNS = {"dV": "dV_bn", "dV_F": "dV_F_bn", "dV_E2": "dV_E2_bn", "dV_vol": "dV_vol_bn", "dPi": "dPi_bn",
               "dC_sup": "dC_sup_bn", "B_t": "B_t_bn"}


def pso_inputs(folder, v):
    """The arrays, xi_c (file 15), the baseline aggregate load and the price tables of the notebook (peed_pso)."""
    z = np.load(Path(folder) / NPZ, allow_pickle=True)
    assert (v.groupby("class").xi_c.nunique() == 1).all()
    xi = v.groupby("class").xi_c.first().to_dict()
    l0 = {c: {s: z[f"L0__{c}__{s}"] for s in fc.SEASONS} for c in fc.CLASSES}
    vault = fc.find_vault()
    mp = pp.month_prices(pd.read_csv(vault.joinpath(*PRICES)), WINTER_MONTHS)
    pi_b = pp.base_m_by_month(pd.read_csv(vault.joinpath(*BASE_M), dtype={"supply_month": str}), mp)
    return z, xi, l0, pp.aggregate(l0, xi), mp, pi_b


def channel_change(z, sid, channel, xi, l0):
    """dL of one channel (all its households), aggregated on the Q(c) basis as the notebook does."""
    tr = CHANNEL_PATHWAY[channel]
    return pp.aggregate({c: {s: z[f"{sid}__{tr}__{c}__{s}"] - l0[c][s] for s in fc.SEASONS} for c in fc.CLASSES}, xi)


def compute(folder):
    v = fc.read(folder, "peed_volume_timing.csv")
    r = fc.read(folder, "peed_sensitivity_retailer.csv")
    registry = fc.read(folder, "peed_scenario_registry.csv").query("role == 'scenario'")
    econ = fc.read(folder, "peed_validity_economic.csv")
    pf = pd.read_csv(fc.find_vault().joinpath(*FORECAST))
    s0 = float(pf.S0.iloc[0])
    assert len(registry) == 24 and registry.scenario_id.is_unique
    assert np.allclose(pf.p_flat_to_3z, np.maximum(pf.smart_meter_share_S - s0, 0), rtol=0, atol=1e-14)
    assert np.allclose(pf.p_2z_to_3z, pf.beta * s0, rtol=0, atol=1e-14)
    assert np.allclose(pf.p_total, pf.p_2z_to_3z + pf.p_flat_to_3z, rtol=0, atol=1e-14)
    assert np.allclose(v.S_ext_Q_MUAH, v.xi_c * (v.S_net_MUAH + v.S_dist_MUAH + v.S_trans_MUAH), rtol=0, atol=1e-8)

    a = v[v.season == "annual"].copy()
    cols = ["S_ext_Q_MUAH", "bill_passive_MUAH", "bill_resp_vol_MUAH", "bill_resp_tim_MUAH"]
    keys = ["scenario_id", "scheme", "channel", "class"]
    seasonal = v[v.season != "annual"].groupby(keys)[cols].sum().sort_index()
    assert np.allclose(a.set_index(keys)[cols].sort_index(), seasonal, rtol=0, atol=1e-8)
    a["delta_R"] = -(a.bill_passive_MUAH + a.bill_resp_vol_MUAH + a.bill_resp_tim_MUAH)
    a["delta_C_Q"] = -a.S_ext_Q_MUAH
    a["N_ret"] = a.delta_R - a.delta_C_Q
    # the channel terms are unweighted: cohort_weight is metadata, only the w_* columns of file 15 are weighted
    for name in ("delta_R", "delta_C_Q", "N_ret"):
        a["weighted_" + name] = a.cohort_weight * a[name]
    endpoint = a.groupby(["scenario_id", "scheme"])[["weighted_delta_R", "weighted_delta_C_Q", "weighted_N_ret"]].sum()
    ref = r.set_index(["scenario_id", "scheme"])
    for name, col in (("delta_R", "total_retail_MUAH"), ("delta_C_Q", "S_ext_Q_MUAH"), ("N_ret", "net_ret_MUAH")):
        expected = ref.loc[endpoint.index, col] * (-1 if name == "delta_C_Q" else 1)
        err = float(np.max(np.abs(endpoint["weighted_" + name] - expected)))
        assert err < 1e-6, (name, err)

    p3 = a[(a.scheme == "P3") & a.scenario_id.isin(registry.scenario_id)]
    assert len(p3) == 24 * 2 * 5 and p3.groupby(["scenario_id", "channel"]).size().eq(5).all()
    channels = p3.groupby(["scenario_id", "channel"])[["delta_R", "delta_C_Q", "N_ret"]].sum()
    stress = econ[econ.quantity == "stress_scenario"].set_index("scenario_id").value.astype(bool)
    rows = []
    for p in pf.itertuples():
        for sid in registry.scenario_id:
            f, e2 = p.p_flat_to_3z * channels.loc[(sid, "F")], p.p_2z_to_3z * channels.loc[(sid, "E2")]
            terms = f + e2
            assert abs(terms.N_ret - (terms.delta_R - terms.delta_C_Q)) < 1e-8
            rows.append({"year": p.year, "scenario_id": sid, "p_total": p.p_total, "w_F": p.p_flat_to_3z,
                         "w_E2": p.p_2z_to_3z, "delta_R_bn": terms.delta_R / 1000, "delta_C_Q_bn": terms.delta_C_Q / 1000,
                         "N_ret_bn": terms.N_ret / 1000, "stress_scenario": bool(stress.loc[sid]),
                         "N_ret_F_bn": f.N_ret / 1000, "N_ret_E2_bn": e2.N_ret / 1000})
    roll = pd.DataFrame(rows).merge(registry, on="scenario_id", validate="many_to_one")
    extra = ["N_ret_F_bn", "N_ret_E2_bn"]
    roll = roll[[c for c in roll.columns if c not in extra] + extra]
    assert len(roll) == 24 * len(pf) and np.isfinite(roll.N_ret_bn).all()
    assert np.allclose(roll.N_ret_F_bn + roll.N_ret_E2_bn, roll.N_ret_bn, rtol=0, atol=1e-12)
    return roll.merge(pso_rollout(folder, v, ref, channels, registry, pf, s0), on=["year", "scenario_id"],
                      validate="one_to_one")


def pso_rollout(folder, v, ref, channels, registry, pf, s0):
    """dV_t and dPi_t per year and scenario, with the lagged baseload (peed_pso.rollout)."""
    z, xi, l0, l0h, mp, pi_b = pso_inputs(folder, v)
    assert abs(float(z["S0"]) - s0) < 1e-15
    weights = pf[["year", "p_flat_to_3z", "p_2z_to_3z"]].rename(columns={"p_flat_to_3z": "F", "p_2z_to_3z": "E2"})
    paths = []
    for sid in registry.scenario_id:
        d_l = {ch: channel_change(z, sid, ch, xi, l0) for ch in CHANNEL_PATHWAY}
        d_rev = {ch: float(channels.loc[(sid, ch), "delta_R"]) for ch in CHANNEL_PATHWAY}
        # full participation (cohort weights 1 - S0, S0) reproduces the notebook's file 5
        l1 = {s: l0h[s] + (1 - s0) * d_l["F"][s] + s0 * d_l["E2"][s] for s in fc.SEASONS}
        fp = pp.fp_metrics(l0h, l1, (1 - s0) * d_rev["F"] + s0 * d_rev["E2"], mp, pi_b)
        for k in ("dV_PSO", "dPi", "B", "dG"):
            err = abs(fp[k] - float(ref.loc[(sid, "P3"), f"{k}_MUAH"]))
            assert err < 1e-6, (sid, k, err)
        paths.append(pp.rollout(l0h, d_l, d_rev, weights, mp, pi_b).assign(scenario_id=sid))
    pso = pd.concat(paths, ignore_index=True)
    assert np.allclose(pso.dV, pso.dV_F + pso.dV_E2, rtol=0, atol=1e-6)            # dV is additive in the shares
    assert np.allclose(pso.dPi, pso.dV_vol - pso.dC_sup, rtol=0, atol=1e-6)
    assert np.allclose(pso[["w_F", "w_E2"]].to_numpy(),
                       pf.set_index("year").loc[pso.year, ["p_flat_to_3z", "p_2z_to_3z"]].to_numpy(), rtol=0, atol=0)
    out = pso[["year", "scenario_id"]].copy()
    for k, col in PSO_COLUMNS.items():
        out[col] = pso[k] / 1000
    for s in fc.SEASONS:
        out[f"b_t_{s}_MW"] = pso[f"b_{s}_MW"]
    out["surplus_hours"], out["surplus_MWh"] = pso.surplus_hours, pso.surplus_MWh
    assert len(out) == 24 * len(pf) and np.isfinite(out[list(PSO_COLUMNS.values())].to_numpy()).all()
    return out


def main(out=None):
    folder = fc.out_dir(out)
    roll = compute(folder)
    path = folder / OUT_NAME
    roll.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"wrote {path.name}  ({len(roll)} rows: {roll.year.nunique()} years x 24 scenarios)")


if __name__ == "__main__":
    main()
