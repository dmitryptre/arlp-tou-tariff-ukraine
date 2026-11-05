"""EULP hierarchy helpers for the within-class bill distribution.

3,145 weighted EULPs -> 19 ARLPs -> five tariff classes. The unit of the within-class analysis is the class-EULP pair
(c, j) with W_cj = sum_{s in c} H_s * phi_sj (kHH), where phi_sj is the normalized EULP weight of ARLP s.

Calendar convention: the hour-ending stamp of the input files; month = stamp month, hour = stamp hour
(0 = hour-ending 24), season by `_common.WINTER_MONTHS`, zone by `_common.TOU_PERIODS_0BASED` ("semi" = shoulder).
This is the convention of `build_repday_matrix` and notebook cell 20, which produced alpha.

All functions are pure apart from the loaders; nothing here writes a file. The inputs are read in place: the 350 MB
EULP file is read with `usecols` and never copied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _common import (
    ARLP_NAMES_ALL,
    TOU_PERIODS_0BASED,
    WINTER_MONTHS,
    build_repday_matrix,
)

CLASS_ORDER = ("A", "B", "C", "ESH-1", "ESH-2")
ESH_CLASSES = frozenset({"ESH-1", "ESH-2"})
# The canonical 5-class map: the literal `_DEF` of the notebook cell e3019676.
CLASS_MAP_INT = {**{n: 0 for n in ["ARLP_04", "ARLP_05", "ARLP_06", "ARLP_12", "ARLP_13", "ARLP_14"]},
                 **{n: 1 for n in ["ARLP_07", "ARLP_15"]},
                 **{n: 2 for n in ["ARLP_08", "ARLP_09", "ARLP_10", "ARLP_11", "ARLP_16", "ARLP_17", "ARLP_18",
                                   "ARLP_19"]},
                 "ARLP_01": 3, "ARLP_02": 4, "ARLP_03": 4}
CLASS_LABEL = {0: "A", 1: "B", 2: "C", 3: "ESH-1", 4: "ESH-2"}
CLASS_MAP = {a: CLASS_LABEL[k] for a, k in CLASS_MAP_INT.items()}
ZONES = ("off", "shoulder", "peak")              # index order of every zone axis
SEASONS = ("winter", "summer")
MONTHS = tuple(range(1, 13))
ZONE_OF_HOUR = {h: ("shoulder" if z == "semi" else z) for z, hs in TOU_PERIODS_0BASED.items() for h in hs}

F_EULP = "EULP_4435_hourly_load.csv"
F_WEIGHTS = "ARLPs_EULPs weight matrix.csv"
F_ABS_WEIGHTS = "ARLPs_EULPs absolute weight matrix.csv"
F_ARLP = "ARLPs(Segments)_Hourly_Load.csv"
F_HH = "arlp_household_counts.csv"


def find_vault(start=None) -> Path:
    """The root folder (the one holding inputs/ARLPs(Segments)_Hourly_Load.csv): walk up from `start`, else from
    this module's folder, then from the working directory."""
    starts = [Path(start)] if start is not None else [Path(__file__).resolve().parent, Path.cwd()]
    for s in starts:
        s = s.resolve()
        for c in [s, *s.parents]:
            if (c / "inputs" / F_ARLP).exists():
                return c
    raise FileNotFoundError("root folder not found (no inputs/ARLPs(Segments)_Hourly_Load.csv above the start folder)")





def _arlp_cols(df):
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def load_weights(vault, absolute=False) -> pd.DataFrame:
    """phi_sj (weight matrix) or the absolute weights in HH (absolute weight matrix): index `eulp_id` (int), columns ARLP_01..ARLP_19.
    Handles the BOM, the `ARLP-nn` names and the trailing empty column."""
    w = _arlp_cols(pd.read_csv(Path(vault) / "inputs" / (F_ABS_WEIGHTS if absolute else F_WEIGHTS), encoding="utf-8-sig"))
    w = w.rename(columns={w.columns[0]: "eulp_id"})
    w["eulp_id"] = w["eulp_id"].astype(np.int64)
    w = w.set_index("eulp_id")[ARLP_NAMES_ALL].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return w


def load_household_counts(vault) -> dict:
    """H_s in kHH (household counts per ARLP)."""
    return pd.read_csv(Path(vault) / "inputs" / F_HH).set_index("ARLP")["H_kHH"].astype(float).to_dict()


def load_arlp_hourly(vault) -> pd.DataFrame:
    """The ARLP hourly loads: `timestamp` (datetime) and ARLP_01..ARLP_19 in kWh/h per household."""
    df = _arlp_cols(pd.read_csv(Path(vault) / "inputs" / F_ARLP))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_eulp_hourly(vault, ids):
    """The EULP hourly loads for the given EULP ids, read in place with `usecols`: (timestamps, array of shape (8760, len(ids)))."""
    cols = [f"bldg_id_{int(i)}" for i in ids]
    df = pd.read_csv(Path(vault) / "inputs" / F_EULP, usecols=["timestamp", *cols],
                     dtype={c: np.float64 for c in cols})
    ts = pd.to_datetime(df["timestamp"])
    return ts, df[cols].to_numpy(dtype=np.float64)


def calendar(ts) -> pd.DataFrame:
    """Per stamp: month, hour (0 = hour-ending 24), season, zone (off / shoulder / peak), zone index."""
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    month, hour = ts.dt.month.to_numpy(), ts.dt.hour.to_numpy()
    zone = np.array([ZONE_OF_HOUR[int(h)] for h in hour])
    return pd.DataFrame({"month": month, "hour": hour,
                         "season": np.where(np.isin(month, sorted(WINTER_MONTHS)), "winter", "summer"),
                         "zone": zone, "zone_idx": np.array([ZONES.index(z) for z in zone])})


def monthly_zone_energy(load, cal) -> np.ndarray:
    """E[n, m-1, z] = sum of the hourly load (shape (8760, n)) over the stamps of month m and zone z."""
    load = np.asarray(load, dtype=np.float64)
    g = (cal["month"].to_numpy() - 1) * 3 + cal["zone_idx"].to_numpy()
    onehot = np.zeros((load.shape[0], 36))
    onehot[np.arange(load.shape[0]), g] = 1.0
    return (onehot.T @ load).T.reshape(load.shape[1], 12, 3)


def monthly_zone_table(eulp_ids, E) -> pd.DataFrame:
    """The layout of file 01 of the within-class runner: `eulp_id`, `annual_kWh`, then m01_off_kWh ... m12_peak_kWh."""
    data = {"eulp_id": np.asarray(eulp_ids, dtype=np.int64), "annual_kWh": E.sum(axis=(1, 2))}
    for m in MONTHS:
        for zi, z in enumerate(ZONES):
            data[f"m{m:02d}_{z}_kWh"] = E[:, m - 1, zi]
    return pd.DataFrame(data)


def class_eulp_weights(wdf, H, class_map=None):
    """Units and attribution (files 02 and 03 of the within-class runner).

    Returns (units, attribution):
    - units: one row per (class, eulp_id) with W_cj > 0, ordered by class (A, B, C, ESH-1, ESH-2) then eulp_id;
      columns class, stratum, eulp_id, W_kHH, n_arlps, member_arlps (the ARLPs with H_s * phi_sj > 0);
    - attribution: one row per (class, ARLP, EULP) with phi > 0 (ARLP-16, with H = 0, included at W_s = 0);
      columns class, arlp, eulp_id, phi, H_kHH, W_s_kHH, share_of_unit.
    """
    class_map = class_map or CLASS_MAP
    rows, att = [], []
    for c in CLASS_ORDER:
        members = [a for a in ARLP_NAMES_ALL if class_map[a] == c]
        sub = wdf[members]
        ws = sub.mul(pd.Series({a: H[a] for a in members}), axis=1)
        W = ws.sum(axis=1)
        for j in W.index[W > 0]:
            contrib = [a for a in members if ws.at[j, a] > 0]
            rows.append({"class": c, "stratum": "esh" if c in ESH_CLASSES else "non_esh", "eulp_id": int(j),
                         "W_kHH": float(W.at[j]), "n_arlps": len(contrib), "member_arlps": ";".join(contrib)})
        for a in members:
            for j in sub.index[sub[a] > 0]:
                att.append({"class": c, "arlp": a, "eulp_id": int(j), "phi": float(sub.at[j, a]), "H_kHH": float(H[a]),
                            "W_s_kHH": float(ws.at[j, a]),
                            "share_of_unit": float(ws.at[j, a] / W.at[j]) if W.at[j] > 0 else 0.0})
    units = pd.DataFrame(rows)
    attribution = pd.DataFrame(att)
    attribution["_o"] = attribution["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    attribution = attribution.sort_values(["_o", "arlp", "eulp_id"]).drop(columns="_o").reset_index(drop=True)
    return units, attribution


def class_weight_matrix(wdf, H, class_map=None) -> pd.DataFrame:
    """W_cj as a frame indexed by eulp_id with one column per class (zeros included)."""
    class_map = class_map or CLASS_MAP
    out = {}
    for c in CLASS_ORDER:
        members = [a for a in ARLP_NAMES_ALL if class_map[a] == c]
        out[c] = wdf[members].mul(pd.Series({a: H[a] for a in members}), axis=1).sum(axis=1)
    return pd.DataFrame(out)


def rebuild_arlp(phi, eulp_load) -> np.ndarray:
    """sum_j phi_sj l_j(t): phi of shape (n_eulp, 19), eulp_load (8760, n_eulp) -> (8760, 19) (check B1)."""
    return np.asarray(eulp_load) @ np.asarray(phi)


def class_repday_from_eulps(eulp_load, Wc, ts) -> dict:
    """The HH-weighted class representative days from the EULPs, {class: {season: 24 values}} (kHH x kWh/h),
    through `_common.build_repday_matrix` as notebook cell 20 builds C_season_all (check B3)."""
    cls_load = (np.asarray(eulp_load) @ Wc[list(CLASS_ORDER)].to_numpy()).T          # (5, 8760)
    rep = build_repday_matrix(cls_load, ts, list(CLASS_ORDER))
    return {c: {s: rep.loc[c, [f"{s}_h{h:02d}" for h in range(24)]].to_numpy(float) for s in SEASONS}
            for c in CLASS_ORDER}


def recompute_alpha(C_season) -> dict:
    """alpha per (class, season) by the cell-20 formula (with the clip), at full precision: {(c, s): (off, sh, pk)}."""
    peak_h = list(TOU_PERIODS_0BASED["peak"])
    off_h = list(TOU_PERIODS_0BASED["off"])
    sho_h = [h for h in range(24) if h not in peak_h and h not in off_h]
    out = {}
    for s in SEASONS:
        ctot = sum(C_season[c][s] for c in C_season)
        for c in C_season:
            ck = C_season[c][s]
            cpeak = ck[peak_h].sum() / ctot[peak_h].sum()
            coff = ck[off_h].sum() / ctot[off_h].sum()
            a_peak, a_off = 1 + cpeak, 1 - coff
            e_p, e_o, e_s = ck[peak_h].sum(), ck[off_h].sum(), ck[sho_h].sum()
            lam = float(np.clip(1 - (cpeak * e_p - coff * e_o) / e_s, a_off, a_peak))
            out[(c, s)] = (a_off, lam, a_peak)
    return out


def kish_neff(w) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())
