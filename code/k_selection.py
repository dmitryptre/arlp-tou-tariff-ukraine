"""
k_selection.py
==============
Formal k-selection helpers for the Lens 2 (TOU-feature) clustering pipeline.

Public functions
----------------
- ``detect_identical_pairs(X, names, eps)`` — identical-pair connected components
  on a feature matrix, returned as one DataFrame row per multi-member component.
- ``deduplicate(X, names, eps)`` — collapse identical-pair components by keeping
  the lex-smallest ARLP name as the representative; returns the deduplicated
  feature matrix, the survivor names, and the membership map.
- ``gap_statistic(X, k_range, B, ref, random_state)`` — Tibshirani, Walther &
  Hastie (2001) gap statistic on the input feature matrix, using PAM
  (k-medoids) at each k. Supports two reference distributions:
  ``"uniform_aabb"`` (axis-aligned uniform — the headline; more conservative
  for small n) and ``"uniform_pca"`` (PCA-aligned uniform — sensitivity for
  anisotropic data).
- ``gap_decision(gap_df, rule)`` — apply the 1-SE rule to the gap-statistic
  table and return (k_star, decision_rule). Falls back to argmax with the
  ``"argmax_fallback"`` flag if no k in the candidate range satisfies the
  1-SE rule.

Used by the notebook, which applies the deduplication, the gap statistic and
the 1-SE decision to the Lens 2 feature matrix.

Algorithm notes:
- ``eps = 1e-6`` is the runtime default mirrored from
  ``_common.IDENTICAL_PAIR_EUCLID_EPS``.
- The PCA-aligned uniform null computes uniform draws in the bounding box of
  the PCA-rotated feature matrix and inverse-transforms them back to the
  original coordinates (Tibshirani et al. 2001 §3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


# ─────────────────────────────────────────────────────────────────────────
# Identical-pair detection and deduplication
# ─────────────────────────────────────────────────────────────────────────

def detect_identical_pairs(
    X: np.ndarray,
    names: list[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Return one row per multi-member identical-pair component.

    Two rows i, j of X are flagged as identical if their Euclidean distance
    is ``<= eps``. Components are computed as connected components on the
    induced adjacency graph (Union–Find), so a triple of mutually identical
    rows produces one component with three members.

    Returned columns:
        pair_index : int             — component index (0-based)
        members    : str             — comma-separated ARLP names
        euclidean_distance : float   — max pairwise distance within the component
                                       (== eps tolerance bound when eps is small)
        location_only_diff_flag : bool — placeholder column kept for parity with
                                         the notebook; downstream callers may
                                         populate it from metadata.

    Notes
    -----
    The ``location_only_diff_flag`` column is left as False here because this
    helper has no access to ARLP metadata. The caller joins the metadata in
    after the call.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    assert n == len(names), f"len(names)={len(names)} != X.shape[0]={n}"

    D = squareform(pdist(X, metric="euclidean"))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    rows = []
    for pix, grp in enumerate(
        sorted([g for g in components.values() if len(g) > 1],
               key=lambda g: min(names[k] for k in g))
    ):
        members = sorted([names[k] for k in grp])
        # Max pairwise distance inside the component (tight upper bound on eps)
        d_max = max(D[a, b] for a in grp for b in grp if a < b)
        rows.append({
            "pair_index": pix,
            "members": ", ".join(members),
            "euclidean_distance": float(d_max),
            "location_only_diff_flag": False,
        })
    return pd.DataFrame(
        rows,
        columns=["pair_index", "members", "euclidean_distance",
                 "location_only_diff_flag"])


def deduplicate(
    X: np.ndarray,
    names: list[str],
    eps: float = 1e-6,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Collapse identical-pair components in X by keeping the lex-smallest name.

    Returns
    -------
    X_dedup : np.ndarray, shape (n_components, d)
    survivor_names : list[str]    — lex-smallest representative per component
    membership : dict[str, list[str]]
        Maps each survivor name to the sorted list of original names in its
        component (always contains the survivor itself).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    assert n == len(names), f"len(names)={len(names)} != X.shape[0]={n}"

    D = squareform(pdist(X, metric="euclidean"))
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    keep_idx: list[int] = []
    membership: dict[str, list[str]] = {}
    for grp in components.values():
        grp_lex = sorted(grp, key=lambda k: names[k])
        rep = grp_lex[0]
        keep_idx.append(rep)
        membership[names[rep]] = sorted(names[k] for k in grp)
    keep_idx.sort()
    survivor_names = [names[k] for k in keep_idx]
    X_dedup = X[keep_idx]
    return X_dedup, survivor_names, membership


# ─────────────────────────────────────────────────────────────────────────
# Gap statistic (Tibshirani, Walther & Hastie 2001)
# ─────────────────────────────────────────────────────────────────────────

def _Wk(X: np.ndarray, labels: np.ndarray) -> float:
    """Pooled within-cluster sum of squared distances."""
    W = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) < 2:
            continue
        W += float(pdist(pts, metric="sqeuclidean").sum() / (2.0 * len(pts)))
    return W


def _fit_W(X: np.ndarray, k: int, random_state: int) -> float:
    """Fit PAM at k and return the within-cluster dispersion."""
    if k == 1:
        return _Wk(X, np.zeros(X.shape[0], dtype=int))
    # Import here so the module loads without sklearn_extra in non-clustering
    # contexts (e.g. running unit tests on the dedup helpers alone).
    from sklearn_extra.cluster import KMedoids
    pam = KMedoids(n_clusters=k, metric="euclidean", method="pam",
                   random_state=random_state).fit(X)
    return _Wk(X, pam.labels_)


def gap_statistic(
    X: np.ndarray,
    k_range: list[int],
    B: int = 100,
    ref: str = "uniform_aabb",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute the Tibshirani gap statistic on X across a range of k.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Z-scored feature matrix. The reference distribution is drawn in this
        coordinate frame (or its PCA rotation, depending on ``ref``).
    k_range : iterable of int
        Candidate k values. Should include k=1 as the reference point.
    B : int
        Number of null samples drawn.
    ref : {"uniform_aabb", "uniform_pca"}
        - ``"uniform_aabb"``: uniform on the axis-aligned bounding box of X
          (more conservative; the headline reference).
        - ``"uniform_pca"``: uniform in PCA-rotated bounding box,
          inverse-transformed back to X-space (Tibshirani 2001 §3).
    random_state : int
        Seed for both the null-sample draws and PAM fits.

    Returns
    -------
    DataFrame indexed by k, with columns:
        W_k, mean_log_W_ref, sd_log_W_ref, gap_k, s_k, gap_k_minus_s_kplus1
    where ``s_k = sd_log_W_ref * sqrt(1 + 1/B)``.
    """
    X = np.asarray(X, dtype=float)
    k_range = list(k_range)
    rng = np.random.default_rng(random_state)

    Wk_obs = np.array([_fit_W(X, k, random_state) for k in k_range])
    log_W = np.log(Wk_obs + 1e-30)

    log_ref = np.zeros((B, len(k_range)))
    if ref == "uniform_aabb":
        lo = X.min(axis=0); hi = X.max(axis=0)
        for b in range(B):
            Xr = rng.uniform(lo, hi, size=X.shape)
            for j, k in enumerate(k_range):
                log_ref[b, j] = np.log(_fit_W(Xr, k, random_state) + 1e-30)
    elif ref == "uniform_pca":
        from sklearn.decomposition import PCA
        # PCA is rank-bounded by min(n_samples, n_features); when n < d the
        # bounding box of the rotated coords lives in n-1 components (the
        # rest carry zero variance), and uniform draws there are degenerate
        # but well-defined.
        n_comp = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp).fit(X)
        Xp = pca.transform(X)
        lo = Xp.min(axis=0); hi = Xp.max(axis=0)
        for b in range(B):
            Xrp = rng.uniform(lo, hi, size=Xp.shape)
            Xr = pca.inverse_transform(Xrp)
            for j, k in enumerate(k_range):
                log_ref[b, j] = np.log(_fit_W(Xr, k, random_state) + 1e-30)
    else:
        raise ValueError(f"unknown ref distribution: {ref!r}")

    mean_log_ref = log_ref.mean(axis=0)
    sd_log_ref = log_ref.std(axis=0, ddof=0)
    gap = mean_log_ref - log_W
    s_k = sd_log_ref * np.sqrt(1.0 + 1.0 / B)

    rows = []
    for j, k in enumerate(k_range):
        g_next = gap[j + 1] if j + 1 < len(k_range) else float("nan")
        s_next = s_k[j + 1] if j + 1 < len(k_range) else float("nan")
        rows.append({
            "k": int(k),
            "W_k": float(Wk_obs[j]),
            "mean_log_W_ref": float(mean_log_ref[j]),
            "sd_log_W_ref": float(sd_log_ref[j]),
            "gap_k": float(gap[j]),
            "s_k": float(s_k[j]),
            "gap_k_minus_s_kplus1": (float(g_next - s_next)
                                      if np.isfinite(g_next) and np.isfinite(s_next)
                                      else float("nan")),
        })
    return pd.DataFrame(rows).set_index("k")


def gap_decision(
    gap_df: pd.DataFrame,
    rule: str = "1se",
    candidate_min_k: int = 2,
) -> tuple[int, str]:
    """Apply the 1-SE rule to a gap-statistic DataFrame.

    Smallest k >= ``candidate_min_k`` such that
        gap(k) >= gap(k+1) - s_{k+1}
    If no candidate k satisfies the rule, falls back to the argmax over the
    full ``gap_df`` index and flags the decision as ``"argmax_fallback"``.

    Returns
    -------
    (k_star, decision_rule) where ``decision_rule`` is one of
    {"1se", "argmax_fallback"}.
    """
    if rule != "1se":
        raise ValueError(f"unknown rule: {rule!r}; only '1se' is implemented")
    df = gap_df.sort_index()
    ks = list(df.index)
    gaps = df["gap_k"].to_numpy()
    s_k = df["s_k"].to_numpy()
    for j, k in enumerate(ks):
        if k < candidate_min_k:
            continue
        if j + 1 >= len(ks):
            continue
        if gaps[j] >= gaps[j + 1] - s_k[j + 1]:
            return int(k), "1se"
    return int(ks[int(np.argmax(gaps))]), "argmax_fallback"
