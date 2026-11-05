"""
arlp_families.py
================
City/rural ARLP family consolidation (pre-clustering).

Detect city/rural counterpart pairs in the non-ESH stratum from segment
metadata (`SEGMENT_META_ALL` in the notebook) and build
household-weighted ARLP families. Each family consolidates a (city, rural)
counterpart pair into a single residential segment whose per-household profile
is the household-weighted convex combination of the two members.

The merge criterion is **metadata-structural**, not numeric: a pair is merged
iff its two members agree on every household-defining attribute except
``Location``, where one is ``"city"`` and the other ``"rural"``. The structural
reason is that the NREL ResStock 2024 EULP basis used by the companion ARLP
construction pipeline (Monakov, in press) carries no Ukrainian-compatible
city/rural attribute, so the only thing distinguishing city/rural counterpart
members is the post-mapping household count — the per-household profile
differences are bounded by the QP-reweighting divergence between the two
household-count targets.

Specification + detection: ``detect_family_pairs``, ``family_id_of``,
``family_membership``. Household-weighted aggregation: ``build_families``,
``FamiliesBundle``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FAMILY_ID_PREFIX = "FAM"

# Household-defining attributes whose equality (in addition to Location ∈ {city,
# rural} differing) characterises a city/rural counterpart pair.
_SHARED_ATTRS = ("Consumption_band", "Dwelling", "Bedrooms", "Heating_fuel")


def _arlp_id_to_int(arlp: str) -> int:
    """ARLP_04 -> 4. Raises on malformed input."""
    if not arlp.startswith("ARLP_"):
        raise ValueError(f"unexpected ARLP id {arlp!r}; expected 'ARLP_NN'")
    try:
        return int(arlp.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"unexpected ARLP id {arlp!r}; expected 'ARLP_NN'") from exc


def detect_family_pairs(segment_meta: pd.DataFrame) -> list[tuple[str, str]]:
    """Detect city/rural ARLP counterpart pairs from segment metadata.

    A pair ``(s_city, s_rural)`` is returned iff:

    1. ``Location[s_city] == "city"`` and ``Location[s_rural] == "rural"``;
    2. for every attribute ``a in _SHARED_ATTRS``: ``a[s_city] == a[s_rural]``.

    Pairs against ``Location == "any"`` ESH ARLPs are not formed. The returned
    list is sorted by ``city_id`` ascending and contains exactly one tuple per
    pair.

    Parameters
    ----------
    segment_meta : pandas.DataFrame
        Must carry columns ``ARLP``, ``Location``, plus every attribute name
        in ``_SHARED_ATTRS``. Extra columns are ignored.

    Returns
    -------
    list[tuple[str, str]]
        ``(city_arlp_id, rural_arlp_id)`` tuples, ordered by ``city_arlp_id``.
    """
    missing = [c for c in ("ARLP", "Location", *_SHARED_ATTRS) if c not in segment_meta.columns]
    if missing:
        raise ValueError(
            f"segment_meta is missing required columns: {missing}; "
            f"got columns: {list(segment_meta.columns)}")

    city = segment_meta[segment_meta["Location"] == "city"].reset_index(drop=True)
    rural = segment_meta[segment_meta["Location"] == "rural"].reset_index(drop=True)

    pairs: list[tuple[str, str]] = []
    for _, c_row in city.iterrows():
        matches = rural
        for attr in _SHARED_ATTRS:
            matches = matches[matches[attr] == c_row[attr]]
        if len(matches) == 1:
            pairs.append((c_row["ARLP"], matches.iloc[0]["ARLP"]))
        elif len(matches) > 1:
            raise ValueError(
                f"ambiguous city/rural pairing for {c_row['ARLP']}: "
                f"matches {list(matches['ARLP'])}")
    pairs.sort(key=lambda t: _arlp_id_to_int(t[0]))
    return pairs


def family_id_of(arlp: str, pairs: list[tuple[str, str]]) -> str:
    """Return the family ID for ``arlp``.

    If ``arlp`` is one half of a city/rural pair, returns
    ``"FAM_<cc>_<rr>"`` (zero-padded two-digit IDs, city first). Otherwise
    returns the ARLP's own ID unchanged (singleton family).
    """
    for city, rural in pairs:
        if arlp == city or arlp == rural:
            return f"{FAMILY_ID_PREFIX}_{_arlp_id_to_int(city):02d}_{_arlp_id_to_int(rural):02d}"
    return arlp


def family_membership(arlp_names: Iterable[str],
                       pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Return ``{family_id: [arlp_id, ...]}`` keyed by family ID.

    Each paired family has exactly two members ordered ``[city, rural]``; each
    singleton family has exactly one member. The dict iteration order follows
    the input ``arlp_names``: paired families appear at their city member's
    position, singletons at their own position.
    """
    members: dict[str, list[str]] = {}
    seen_paired: set[str] = set()
    paired_lookup = {p[0]: p for p in pairs}  # city -> (city, rural)
    paired_lookup.update({p[1]: p for p in pairs})  # rural -> (city, rural)

    for arlp in arlp_names:
        if arlp in paired_lookup:
            city, rural = paired_lookup[arlp]
            fid = f"{FAMILY_ID_PREFIX}_{_arlp_id_to_int(city):02d}_{_arlp_id_to_int(rural):02d}"
            if fid in seen_paired:
                continue
            members[fid] = [city, rural]
            seen_paired.add(fid)
        else:
            members[arlp] = [arlp]
    return members


@dataclass(frozen=True)
class FamiliesBundle:
    """Household-weighted family construction result.

    Attributes
    ----------
    family_ids : list[str]
        Ordered list of family IDs. Length = ``n_families``.
    family_load_matrix : np.ndarray
        Per-family household-weighted 8 760-h profile, shape (n_families, n_hours).
    family_weight_matrix : np.ndarray
        Per-family household-weighted EULP weight vector, shape
        (n_families, n_eulps). Rows are probability distributions on the EULP
        simplex (``row.sum() == 1`` within floating-point tolerance).
    family_household_counts : np.ndarray
        Per-family aggregated household count (kHH), shape (n_families,).
    family_membership : dict[str, list[str]]
        ``{family_id: [arlp_id, ...]}`` with one or two members per family.
    arlp_to_family : dict[str, str]
        Inverse map ``{arlp_id: family_id}`` for every input ARLP.
    """
    family_ids: list[str]
    family_load_matrix: np.ndarray
    family_weight_matrix: np.ndarray
    family_household_counts: np.ndarray
    family_membership: dict[str, list[str]]
    arlp_to_family: dict[str, str]


def build_families(load_matrix: np.ndarray,
                    weight_matrix: np.ndarray,
                    segment_meta: pd.DataFrame,
                    arlp_names: list[str],
                    household_counts: dict[str, float]) -> FamiliesBundle:
    """Build household-weighted ARLP families from the per-ARLP inputs.

    For a family ``f`` with members ``{s_city, s_rural}`` and household counts
    ``H_city, H_rural``:

        H_f      = H_city + H_rural
        w_f(e)   = (H_city * w_city(e) + H_rural * w_rural(e)) / H_f
        l_f(h)   = (H_city * l_city(h) + H_rural * l_rural(h)) / H_f

    A singleton family (no counterpart) inherits its member's profile, weight
    vector, and household count unchanged. The per-household profile equality
    "household-weighted weight merge → reconstructed profile = household-weighted
    profile merge" follows from linearity of the EULP→profile map; we compute
    the direct profile path for production use and the weight path for the
    Lens 1 traceability artefact ``family_weight_matrix.csv``.

    Parameters
    ----------
    load_matrix : np.ndarray
        Shape ``(len(arlp_names), n_hours)`` per-ARLP kW/household profiles.
    weight_matrix : np.ndarray
        Shape ``(len(arlp_names), n_eulps)`` per-ARLP EULP weight vectors;
        each row is expected to sum to 1.
    segment_meta : pandas.DataFrame
        Used only to derive the city/rural pair list (see
        ``detect_family_pairs``). Must include rows for every ``arlp_names``.
    arlp_names : list[str]
        Input cohort order; the function preserves this order when emitting
        the family-level outputs (paired families inherit their city member's
        position, singletons their own).
    household_counts : dict[str, float]
        ``{arlp_id: H_kHH}`` mapping. Raises ``KeyError`` if any input ARLP is
        missing.

    Returns
    -------
    FamiliesBundle

    Raises
    ------
    ValueError
        If any input ARLP fails to be paired AND is not classified as a
        singleton (the detection is total over the input).
    """
    if load_matrix.shape[0] != len(arlp_names):
        raise ValueError(
            f"load_matrix has {load_matrix.shape[0]} rows but "
            f"len(arlp_names) == {len(arlp_names)}")
    if weight_matrix.shape[0] != len(arlp_names):
        raise ValueError(
            f"weight_matrix has {weight_matrix.shape[0]} rows but "
            f"len(arlp_names) == {len(arlp_names)}")

    arlp_idx = {name: i for i, name in enumerate(arlp_names)}
    pairs = detect_family_pairs(
        segment_meta[segment_meta["ARLP"].isin(arlp_names)])
    membership = family_membership(arlp_names, pairs)

    # Totality: every input ARLP must appear in exactly one family.
    covered = {a for ms in membership.values() for a in ms}
    missing = set(arlp_names) - covered
    if missing:
        raise ValueError(
            f"build_families: ARLPs not covered by the family membership "
            f"(neither paired nor singleton): {sorted(missing)}")

    family_ids: list[str] = []
    fam_load_rows: list[np.ndarray] = []
    fam_weight_rows: list[np.ndarray] = []
    fam_hh: list[float] = []
    arlp_to_family: dict[str, str] = {}

    for fid, members in membership.items():
        H_vec = np.array([float(household_counts[m]) for m in members], dtype=float)
        H_total = float(H_vec.sum())
        if H_total <= 0:
            # Equal-weight fallback so the family carries a meaningful profile
            # even when all members have a zero household count (e.g. ARLP-16
            # rural side with H = 0 in the current data). Documented edge case.
            H_vec = np.ones(len(members), dtype=float)
            H_total = float(H_vec.sum())
        load_rows = np.stack([load_matrix[arlp_idx[m]] for m in members], axis=0)
        weight_rows = np.stack([weight_matrix[arlp_idx[m]] for m in members], axis=0)
        fam_load_rows.append((H_vec @ load_rows) / H_total)
        fam_weight_rows.append((H_vec @ weight_rows) / H_total)
        fam_hh.append(sum(float(household_counts[m]) for m in members))
        family_ids.append(fid)
        for m in members:
            arlp_to_family[m] = fid

    return FamiliesBundle(
        family_ids=family_ids,
        family_load_matrix=np.stack(fam_load_rows, axis=0),
        family_weight_matrix=np.stack(fam_weight_rows, axis=0),
        family_household_counts=np.array(fam_hh, dtype=float),
        family_membership=membership,
        arlp_to_family=arlp_to_family,
    )


def _numeric_proximity_class(d_z: float) -> str:
    """Descriptive class for the 6-D (per-season) z-scored TOU distance
    ``d_z_tou6``."""
    if d_z <= 1e-6:
        return "exact"
    if d_z <= 0.3:
        return "near-duplicate"
    if d_z <= 1.5:
        return "similar"
    return "divergent"


def compute_family_merge_diagnostics(
    bundle: FamiliesBundle,
    *,
    load_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    arlp_names: list[str],
    household_counts: dict[str, float],
    jsd_matrix: pd.DataFrame,
    cosine_matrix: pd.DataFrame,
    tou_feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Per-pair empirical similarity diagnostics.

    For each city/rural counterpart pair the table reports:

    - ``jsd_eulp``: Jensen-Shannon distance on the EULP weight vectors (read
      directly from the Lens 1 ``jsd_distance_matrix.csv``, which already
      stores the scipy square-root form bounded in
      ``[0, sqrt(ln 2)] ~ [0, 0.832]``).
    - ``cosine_eulp``: cosine distance on the EULP weight vectors (Lens 1).
    - ``d_z_tou6``: Euclidean distance on the 6-D peak-norm TOU features after
      stratum-wise column z-scoring (the same metric the k-selection's
      identical-pair dedup uses); the features are per-season (winter, summer).
    - ``pearson_corr_8760h``: Pearson correlation between the two 8 760-h
      per-household profiles.
    - ``max_abs_diff_annual_peak_norm``: per-pair ``max_h |l_city(h)/peak_city
      - l_rural(h)/peak_rural|`` on the annual-peak-normalised profiles.
    - ``numeric_proximity_class``: descriptive class derived from ``d_z_tou6``.
    - ``merge_recommended``: always ``True`` (the merge criterion is structural).

    Parameters
    ----------
    bundle : FamiliesBundle
        The family construction result; carries the ``(city, rural)``
        membership lists.
    load_matrix, weight_matrix : np.ndarray
        Per-ARLP inputs in the same row order as ``arlp_names``.
    arlp_names : list[str]
        Cohort order used to index ``load_matrix`` / ``weight_matrix``.
    household_counts : dict[str, float]
        ``{arlp_id: H_kHH}`` for the cohort.
    jsd_matrix, cosine_matrix : pandas.DataFrame
        Square ARLP × ARLP distance tables from
        ``lens1_compositional/{jsd,cosine}_distance_matrix.csv``.
    tou_feature_matrix : pandas.DataFrame
        ``feature_matrix.csv`` from ``lens2_tou_features/`` (rows indexed by
        ARLP id, 6 columns).

    Returns
    -------
    pandas.DataFrame
        One row per city/rural pair, with the columns listed above.
    """
    X_tou = tou_feature_matrix.to_numpy(float)
    mu = X_tou.mean(axis=0, keepdims=True)
    sd = X_tou.std(axis=0, ddof=0, keepdims=True)
    Xz = (X_tou - mu) / np.where(sd > 0, sd, 1.0)
    Xz_df = pd.DataFrame(Xz, index=tou_feature_matrix.index,
                          columns=tou_feature_matrix.columns)
    arlp_idx = {n: i for i, n in enumerate(arlp_names)}

    rows = []
    for fid, members in bundle.family_membership.items():
        if len(members) != 2:
            continue
        city, rural = members[0], members[1]
        jsd_dist = float(jsd_matrix.loc[city, rural])
        cos_d = float(cosine_matrix.loc[city, rural])
        d_z = float(np.linalg.norm(Xz_df.loc[city].to_numpy() - Xz_df.loc[rural].to_numpy()))
        prof_city = load_matrix[arlp_idx[city]]
        prof_rural = load_matrix[arlp_idx[rural]]
        rho = float(np.corrcoef(prof_city, prof_rural)[0, 1])
        peak_city = float(prof_city.max())
        peak_rural = float(prof_rural.max())
        if peak_city > 0 and peak_rural > 0:
            max_abs_diff = float(np.max(np.abs(prof_city / peak_city - prof_rural / peak_rural)))
        else:
            max_abs_diff = float("nan")
        rows.append({
            "family_id": fid,
            "city_arlp": city,
            "rural_arlp": rural,
            "H_city": float(household_counts[city]),
            "H_rural": float(household_counts[rural]),
            "jsd_eulp": round(jsd_dist, 6),
            "cosine_eulp": round(cos_d, 6),
            "d_z_tou6": round(d_z, 6),
            "pearson_corr_8760h": round(rho, 6),
            "max_abs_diff_annual_peak_norm": round(max_abs_diff, 6),
            "numeric_proximity_class": _numeric_proximity_class(d_z),
            "merge_recommended": True,
        })
    return pd.DataFrame(rows)


def write_family_construction_artifacts(
    bundle: FamiliesBundle,
    *,
    out_dir: Path | str,
    n_hours: int = 8760,
    eulp_columns: list[str] | None = None,
) -> dict[str, Path]:
    """Persist the four family-construction CSVs.

    Returns ``{name: path}`` for the four files written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) family_membership.csv
    mem_rows = [
        {"family_id": fid, "member_arlps": ",".join(members)}
        for fid, members in bundle.family_membership.items()
    ]
    p_mem = out / "family_membership.csv"
    pd.DataFrame(mem_rows).to_csv(p_mem, index=False)

    # 2) family_household_counts.csv
    p_hh = out / "family_household_counts.csv"
    pd.DataFrame({
        "family_id": bundle.family_ids,
        "H_kHH": bundle.family_household_counts,
    }).to_csv(p_hh, index=False)

    # 3) family_load_matrix.csv (one row per family, 8 760 hour columns)
    hour_cols = [f"h{h:04d}" for h in range(n_hours)]
    p_load = out / "family_load_matrix.csv"
    pd.DataFrame(
        bundle.family_load_matrix, index=bundle.family_ids, columns=hour_cols
    ).rename_axis("family_id").to_csv(p_load)

    # 4) family_weight_matrix.csv (one row per family, n_eulps columns)
    n_e = bundle.family_weight_matrix.shape[1]
    cols = list(eulp_columns) if eulp_columns is not None else [f"e{i:04d}" for i in range(n_e)]
    p_w = out / "family_weight_matrix.csv"
    pd.DataFrame(
        bundle.family_weight_matrix, index=bundle.family_ids, columns=cols
    ).rename_axis("family_id").to_csv(p_w)

    return {
        "family_membership": p_mem,
        "family_household_counts": p_hh,
        "family_load_matrix": p_load,
        "family_weight_matrix": p_w,
    }
