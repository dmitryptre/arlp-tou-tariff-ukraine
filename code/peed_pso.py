"""PSO compensation, procurement cost and supplier balance.

V^PSO     = sum_m r_ref_m Q_m(L) - Rev                    regulated PSO compensation
C_sup(L;b) = sum_m |D_m| [24 pi_B_m b_m + sum_h pi_m(h) (L(h) - b_m) + sum_h n_m(h) L(h)]
Pi(L;b)   = sum_m r_ref_m Q_m(L) - C_sup(L;b)             supplier balance (independent of the tariff)
G         = C_sup(L;b) - Rev = V^PSO - Pi                 cost-based financing gap
with r_ref_m = pi_bar_m + n_bar_m (arithmetic monthly means of the DAM price and of the network charges),
pi_B_m the BASE_M price of the delivery month (DAM BASE index of M-3) and b_m = min_h L(tau(m), h).

The model has one representative day per season, so a month's load is the representative day of its season
repeated over the |D_m| days of the month, and b_m is equal in all months of a season. Prices are monthly
hour-of-day means on the model calendar of inputs/hourly_dam_prices_uah.csv.

Hour index h = 0..23 is the house axis of the notebook: index 0 is hour-ending 24, index h is hour-ending h.
Loads are in MW (aggregate, annual-consumption basis: sum_c xi_c l(c)), prices in UAH/MWh excluding VAT,
money in UAH M. The functions are pure: they take arrays, frames and mappings and return new objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONVENTIONS = ("stamp", "hour_start")     # month of the hour-ending stamp (the notebook's) or of the hour start
MUAH = 1e6
M_BASELOAD_RATIO = 0.5                # flag M: b1 < 0.5 b0 in at least one season


@dataclass(frozen=True)
class MonthPrices:
    """Monthly price table on the model calendar."""
    convention: str
    days: dict          # m -> |D_m|
    season: dict        # m -> season name
    pi_h: dict          # m -> np.ndarray(24): mean DAM net price by hour index, UAH/MWh
    n_h: dict           # m -> np.ndarray(24): mean network charge (distribution + transmission) by hour index
    ym: dict            # m -> 'YYYY-MM', the modal delivery month of the DAM prices of model month m

    @property
    def months(self):
        return sorted(self.days)

    def pi_bar(self, m):
        return float(self.pi_h[m].mean())

    def n_bar(self, m):
        return float(self.n_h[m].mean())

    def r_ref(self, m):
        """Reference price proxy: the arithmetic monthly mean of DAM + network charges."""
        return self.pi_bar(m) + self.n_bar(m)


def month_prices(prices_df: pd.DataFrame, winter_months, convention: str = "stamp") -> MonthPrices:
    """MonthPrices from inputs/hourly_dam_prices_uah.csv (columns timestamp, DAM_timestamp, DAM_net_price_uah_mwh,
    distribution_uah_mwh, transmission_uah_mwh). The hour index is the stamp hour in both conventions, as in
    peed_sensitivity.price_components; 'hour_start' assigns the month by the hour's start."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    d = prices_df
    ts = pd.to_datetime(d["timestamp"])
    month = ts.dt.month if convention == "stamp" else (ts - pd.Timedelta(hours=1)).dt.month
    hour = ts.dt.hour
    net = d["DAM_net_price_uah_mwh"].astype(float)
    net_ch = (d["distribution_uah_mwh"] + d["transmission_uah_mwh"]).astype(float)
    dam_month = (pd.to_datetime(d["DAM_timestamp"]) - pd.Timedelta(hours=1)).dt.strftime("%Y-%m")
    days, season, pi_h, n_h, ym = {}, {}, {}, {}, {}
    winter = set(winter_months)
    for m in range(1, 13):
        sel = month == m
        n_rows = int(sel.sum())
        if n_rows % 24:
            raise ValueError(f"month {m}: {n_rows} hours, not whole days")
        days[m] = n_rows // 24
        season[m] = "winter" if m in winter else "summer"
        g = pd.DataFrame({"h": hour[sel], "pi": net[sel], "n": net_ch[sel]}).groupby("h")
        counts = g.size().reindex(range(24))
        if not (counts == days[m]).all():
            raise ValueError(f"month {m}: hours of the day are not equally represented")
        pi_h[m] = g["pi"].mean().reindex(range(24)).to_numpy()
        n_h[m] = g["n"].mean().reindex(range(24)).to_numpy()
        ym[m] = dam_month[sel].mode().iloc[0]
    return MonthPrices(convention, days, season, pi_h, n_h, ym)


def base_m_by_month(base_m_df: pd.DataFrame, mp: MonthPrices) -> dict:
    """pi_B_m (UAH/MWh) of each model month: the BASE_M price of its delivery month (inputs/base_m_prices_uah.csv)."""
    price = base_m_df.set_index(base_m_df["supply_month"].astype(str))["base_m_price_uah_mwh"].astype(float)
    missing = [mp.ym[m] for m in mp.months if mp.ym[m] not in price.index]
    if missing:
        raise ValueError(f"no BASE_M price for delivery months {missing}")
    return {m: float(price[mp.ym[m]]) for m in mp.months}


def aggregate(profiles: dict, xi) -> dict:
    """Aggregate load on the annual-consumption basis: {season: sum_c xi_c l(c, season)} from {c: {season: l}}."""
    xi = dict(xi)
    seasons = next(iter(profiles.values())).keys()
    return {s: sum(float(xi[c]) * np.asarray(profiles[c][s], float) for c in profiles) for s in seasons}


def baseload(L: dict) -> dict:
    """b by season (MW): the minimum hourly load of the representative day."""
    return {s: float(np.min(L[s])) for s in L}


def min_hour(L: dict) -> dict:
    """Hour index of the minimum of each season's representative day (the first on a tie)."""
    return {s: int(np.argmin(L[s])) for s in L}


def monthly_energy(L: dict, mp: MonthPrices) -> dict:
    """Q_m (MWh) of each model month."""
    return {m: mp.days[m] * float(np.sum(L[mp.season[m]])) for m in mp.months}


def reference_value(L: dict, mp: MonthPrices) -> float:
    """sum_m r_ref_m Q_m(L), UAH M."""
    q = monthly_energy(L, mp)
    return sum(mp.r_ref(m) * q[m] for m in mp.months) / MUAH


def supply_cost_parts(L: dict, b: dict, mp: MonthPrices, pi_b: dict) -> dict:
    """C_sup(L;b) and its parts, UAH M: C_mkt = sum_m |D_m| sum_h (pi + n) L (all energy at hourly prices) and
    lag = sum_m 24 |D_m| (pi_B_m - pi_bar_m) b_m (the BASE_M block instead of DAM purchases), C_sup = C_mkt + lag.
    Load below b in an hour is credited at the hourly DAM price (the C_sup formula above)."""
    c_mkt = lag = 0.0
    for m in mp.months:
        s, n_days = mp.season[m], mp.days[m]
        load = np.asarray(L[s], float)
        c_mkt += n_days * float(np.dot(mp.pi_h[m] + mp.n_h[m], load))
        lag += n_days * b[s] * (24.0 * pi_b[m] - float(mp.pi_h[m].sum()))
    return {"C_mkt": c_mkt / MUAH, "lag": lag / MUAH, "C_sup": (c_mkt + lag) / MUAH}


def supply_cost(L: dict, b: dict, mp: MonthPrices, pi_b: dict) -> float:
    """C_sup(L;b), UAH M."""
    return supply_cost_parts(L, b, mp, pi_b)["C_sup"]


def surplus(L: dict, b: dict, mp: MonthPrices) -> tuple[int, float]:
    """Hours of the year with load below the baseload block, and the energy below it (MWh)."""
    hours, mwh = 0, 0.0
    for m in mp.months:
        s = mp.season[m]
        gap = np.maximum(b[s] - np.asarray(L[s], float), 0.0)
        hours += mp.days[m] * int((gap > 0).sum())
        mwh += mp.days[m] * float(gap.sum())
    return hours, mwh


def levels(L0: dict, rev0_muah: float, mp: MonthPrices, pi_b: dict) -> tuple[dict, pd.DataFrame]:
    """Window levels of the baseline: totals (UAH M) and the monthly table (the notebook's file 18)."""
    b0 = baseload(L0)
    q = monthly_energy(L0, mp)
    rows = []
    for m in mp.months:
        s = mp.season[m]
        one = MonthPrices(mp.convention, {m: mp.days[m]}, {m: s}, {m: mp.pi_h[m]}, {m: mp.n_h[m]}, {m: mp.ym[m]})
        parts = supply_cost_parts(L0, b0, one, {m: pi_b[m]})
        rows.append({"month": m, "delivery_month": mp.ym[m], "season": s, "days": mp.days[m],
                     "pi_bar_uah_mwh": mp.pi_bar(m), "n_bar_uah_mwh": mp.n_bar(m), "r_ref_uah_mwh": mp.r_ref(m),
                     "pi_B_uah_mwh": pi_b[m], "b_MW": b0[s], "Q_MWh": q[m],
                     "ref_value_MUAH": mp.r_ref(m) * q[m] / MUAH, "C_mkt_MUAH": parts["C_mkt"],
                     "lag_MUAH": parts["lag"], "C_sup_MUAH": parts["C_sup"]})
    t = pd.DataFrame(rows)
    ref, c_mkt, lag = t["ref_value_MUAH"].sum(), t["C_mkt_MUAH"].sum(), t["lag_MUAH"].sum()
    tot = {"ref_value": ref, "C_mkt": c_mkt, "lag": lag, "C_sup": c_mkt + lag, "Rev0": rev0_muah,
           "V0": ref - rev0_muah, "G0": c_mkt + lag - rev0_muah, "Pi0": ref - c_mkt - lag,
           "Pi0_profile": ref - c_mkt, "Pi0_lag": -lag, "Q_MWh": t["Q_MWh"].sum(),
           "baseload_MWh": sum(24 * mp.days[m] * b0[mp.season[m]] for m in mp.months)}
    return tot, t


def fp_metrics(L0: dict, L1: dict, d_rev_muah: float, mp: MonthPrices, pi_b: dict, period=None,
               b1: dict | None = None) -> dict:
    """Changes between two steady states (full participation): each configuration with its own baseload.

    d_rev_muah: the change in residential revenue (passive + shift), UAH M. b1 overrides the post-response
    baseload (checks only, e.g. b1 = b0). flag_M: the baseload block falls by more than half in at least one season
    (b1 < M_BASELOAD_RATIO b0). period: zone name of each hour index, for the diagnostic zone of the minimum."""
    b0 = baseload(L0)
    b1 = baseload(L1) if b1 is None else dict(b1)
    vol = reference_value(L1, mp) - reference_value(L0, mp)
    c0, c1 = supply_cost_parts(L0, b0, mp, pi_b), supply_cost_parts(L1, b1, mp, pi_b)
    d_mkt, b_term = c1["C_mkt"] - c0["C_mkt"], c1["lag"] - c0["lag"]
    d_sup = d_mkt + b_term
    out = {"dV_PSO": vol - d_rev_muah, "dV_vol": vol, "dRev": d_rev_muah, "dC_sup": d_sup, "dC_mkt": d_mkt,
           "B": b_term, "dPi": vol - d_sup, "dG": d_sup - d_rev_muah}
    for s in L0:
        out[f"b0_{s}_MW"], out[f"b1_{s}_MW"] = b0[s], b1[s]
    hrs = min_hour(L1)
    for s in L1:
        out[f"min_hour_{s}"] = hrs[s]
        out[f"min_zone_{s}"] = None if period is None else str(period[hrs[s]])
    out["flag_M"] = any(b1[s] < M_BASELOAD_RATIO * b0[s] for s in L0)
    return out


def rollout(L0: dict, dL: dict, d_rev: dict, weights: pd.DataFrame, mp: MonthPrices, pi_b: dict) -> pd.DataFrame:
    """Rollout path: L_t = L0 + sum_k w_k(t) dL_k, Rev_t = Rev0 + sum_k w_k(t) dRev_k, and the baseload of
    year t is the minimum of L_{t-1} (the first year: of L0).

    dL: {channel: {season: dL}}, d_rev: {channel: UAH M}, weights: DataFrame with 'year' and one column per
    channel, in year order. Returns one row per year: dV (additive in the shares), its channel parts, dPi
    (not additive), its baseload part B_t, b_t by season and the surplus diagnostic, relative to L0 and b0."""
    chans = list(dL)
    seasons = list(L0)
    w = weights.sort_values("year").reset_index(drop=True)
    ref0 = reference_value(L0, mp)
    b0 = baseload(L0)
    c0 = supply_cost_parts(L0, b0, mp, pi_b)
    rows = []
    prev = L0
    for r in w.itertuples(index=False):
        wt = {k: float(getattr(r, k)) for k in chans}
        L_t = {s: np.asarray(L0[s], float) + sum(wt[k] * np.asarray(dL[k][s], float) for k in chans)
               for s in seasons}
        b_t = baseload(prev)
        vol = reference_value(L_t, mp) - ref0
        c_t = supply_cost_parts(L_t, b_t, mp, pi_b)
        d_rev_t = sum(wt[k] * float(d_rev[k]) for k in chans)
        h, mwh = surplus(L_t, b_t, mp)
        row = {"year": int(r.year), **{f"w_{k}": wt[k] for k in chans},
               "dV": vol - d_rev_t, "dV_vol": vol, "dRev": d_rev_t,
               "dC_sup": c_t["C_sup"] - c0["C_sup"], "dC_mkt": c_t["C_mkt"] - c0["C_mkt"],
               "B_t": c_t["lag"] - c0["lag"], "dPi": vol - (c_t["C_sup"] - c0["C_sup"]),
               **{f"b_{s}_MW": b_t[s] for s in seasons}, "surplus_hours": h, "surplus_MWh": mwh}
        for k in chans:
            Lk = {s: np.asarray(L0[s], float) + wt[k] * np.asarray(dL[k][s], float) for s in seasons}
            row[f"dV_{k}"] = reference_value(Lk, mp) - ref0 - wt[k] * float(d_rev[k])
        rows.append(row)
        prev = L_t
    return pd.DataFrame(rows)
