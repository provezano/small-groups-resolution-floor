"""
Baselines for tiny-bench with one uniform signature: fit(X, param) -> labels (n,), -1 = noise. Small groups are written
over the shared HDBSCAN base (base.hdb), so every method keeps the same big structure and scores are comparable.

Vendored from clustering.eval.nature.tiny_baselines / tiny_methods (same project):
- gapclust_k30 / gapclust_k200 / gapclust_euclid_k30 / _k200 (param = skewness threshold): NOT REDISTRIBUTED. The upstream
  GapClust R package is GPL-3, so the port was removed from this MIT package; these entries raise unless GAPCLUST_PY names your
  own port (see the stub below). The v1.0 leaderboard rows computed with the removed port are kept in results/.
- mccatch (param = share of top-scored microclusters kept, or "b0.05"/"b0.2" = plateau slope b at full share): OUR
  REIMPLEMENTATION of McCatch (Sanchez Vinces, Cordeiro, Faloutsos, ICDE 2024) from the paper; the official code
  could not be fetched. Not validated against the official code.
- lof_group / iforest_group / glosh_group (param = share of top outlier scores): top-q points grouped by mutual
  3-nearest neighbours (components of 2-15 points).
- hdbscan_mcs (param = min_cluster_size = min_samples).
- hdbscan_base (param ignored): the base alone.
Optional (not vendored, need their own code):
- sccad (param = score threshold): the authors' scCAD.py (Xu et al., Nat Commun 2024), path in env SCCAD_PY.
- cts / cts_bh (param = alpha): the collapse-then-scan method, imported from the method package if installed.
"""
import math
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .base import groups_over_base, hdb, hdbscan_fit, _mcs


# ---------------- GapClust: NOT REDISTRIBUTED (licensing) ----------------
# The upstream GapClust R package (github.com/fabotao/GapClust) is declared GPL-3 and has no LICENSE file. A port of its
# core cannot be distributed under this package's MIT licence and was removed. The names stay so the baseline registry
# still imports; they raise unless GAPCLUST_PY names your own Python file defining
#   gapclust_raw(Z, k=200, squared=True) -> (skew, top_ave, v1k, idx)
#   gapclust_groups(skew, top_ave, v1k, idx, thr=2.0) -> list of index arrays
# Leaderboard rows computed earlier with the removed port are kept in results/; they cannot be regenerated from this
# package alone.
import importlib.util


class GapClustUnavailable(RuntimeError):
    pass


_GAP_MSG = ("GapClust is not available in this package. The official GapClust package (github.com/fabotao/GapClust) is "
            "GPL-3, so a port of it cannot be redistributed under this MIT-licensed package and was removed. Obtain the "
            "official R package from github.com/fabotao/GapClust, port or call it yourself, and set GAPCLUST_PY to a "
            "Python file defining gapclust_raw(Z, k, squared) and gapclust_groups(skew, top_ave, v1k, idx, thr).")
_GAP_USER = {}


def _gapclust_user():
    path = os.environ.get("GAPCLUST_PY")
    if not path:
        raise GapClustUnavailable(_GAP_MSG)
    if path not in _GAP_USER:
        spec = importlib.util.spec_from_file_location("_gapclust_user", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GAP_USER[path] = mod
    return _GAP_USER[path]


def gapclust_available():
    return bool(os.environ.get("GAPCLUST_PY")) and os.path.isfile(os.environ["GAPCLUST_PY"])


def gapclust_raw(Z, k=200, squared=True):
    """Stub: delegates to the user's port named by GAPCLUST_PY, else raises GapClustUnavailable."""
    return _gapclust_user().gapclust_raw(Z, k, squared)


def gapclust_groups(skew, top_ave, v1k, idx, thr=2.0):
    """Stub: delegates to the user's port named by GAPCLUST_PY, else raises GapClustUnavailable."""
    return _gapclust_user().gapclust_groups(skew, top_ave, v1k, idx, thr)


# ---------------- McCatch (reimplementation) ----------------
def _ucode(z):
    """Universal code length <z> (Rissanen's log*), z >= 1."""
    z = max(float(z), 1.0)
    out, v = math.log2(2.865064), z
    while True:
        v = math.log2(v) if v > 1 else 0.0
        if v <= 0:
            break
        out += v
    return out


def _cost(V):
    V = np.asarray(V, float)
    if len(V) == 0:
        return 0.0
    m = V.mean()
    return _ucode(len(V)) + _ucode(1 + math.ceil(m)) + sum(_ucode(1 + math.ceil(abs(v - m))) for v in V)


def mccatch(Z, a=15, b=0.1, c_frac=0.1, return_all=False):
    n = len(Z)
    c = int(math.ceil(c_frac * n))
    rng = np.random.default_rng(0)
    s = Z[rng.choice(n, min(n, 2000), replace=False)]
    far = np.linalg.norm(s[:, None] - s[None], axis=2).max()
    l = float(far)
    R = np.array([l / 2 ** (a - e) for e in range(1, a + 1)])
    kk = min(n, c + 1)
    dist, idx = NearestNeighbors(n_neighbors=kk).fit(Z).kneighbors(Z)      # includes self at distance 0
    q = np.stack([(dist <= r).sum(1) for r in R], 1).astype(float)          # capped at c + 1 (> c = excused)
    q[:, -1] = n                                                           # radius l: everyone
    lr = np.log(R)
    x = np.zeros(n); y = np.zeros(n)
    for i in range(n):
        lq = np.log(q[i])
        slope = (lq[1:] - lq[:-1]) / (lr[1:] - lr[:-1])
        flat = slope <= b
        # plateaus = maximal runs [e, e'] with all slopes in between <= b
        runs, e = [], 0
        while e < a:
            e2 = e
            while e2 < a - 1 and flat[e2]:
                e2 += 1
            runs.append((e, e2))
            e = e2 + 1
        for (e1, e2) in runs:
            h = q[i, e1]
            if h > c:
                continue
            length = R[e2] - R[e1]
            if h == 1 and e1 == 0:
                x[i] = length if length > 0 else R[0]
            elif h > 1 and e2 != a - 1 and length > y[i]:
                y[i] = length
        if q[i, 0] > 1:
            x[i] = 0.0
    # MDL cutoff on the histogram of x over the radii bins
    xb = np.where(x > 0, np.clip(np.round(np.log2(np.maximum(x, 1e-300) / R[0])), 0, a - 1), -1).astype(int)
    h = np.array([(xb == e).sum() for e in range(a)])
    ep = int(np.argmax(h))
    best, d_idx = np.inf, a - 1
    for e in range(ep + 1, a):
        cst = _cost(h[ep:e]) + _cost(h[e:])
        if cst < best:
            best, d_idx = cst, e
    d = R[d_idx]
    A = (x >= d) | (y >= d)
    Mset = np.flatnonzero(y >= d)
    groups = []
    if len(Mset):
        xu = x[Mset].max()
        above = R[R > xu]
        rl = above[0] if len(above) else R[-1]
        G = NearestNeighbors(radius=rl).fit(Z[Mset]).radius_neighbors_graph(Z[Mset])
        comp = connected_components(G, directed=False)[1]
        groups = [Mset[comp == k] for k in np.unique(comp)]
    singles = [np.array([i]) for i in np.flatnonzero(A & ~np.isin(np.arange(n), Mset))]
    # scores (paper Def. 7): higher = more anomalous
    inl = np.flatnonzero(~A)
    t = Z.shape[1]
    if len(inl):
        g = NearestNeighbors(n_neighbors=1).fit(Z[inl]).kneighbors(Z)[0][:, 0]
    else:
        g = np.full(n, l)
    scores = []
    for M in groups:
        m = len(M)
        s1 = _ucode(m) + _ucode(n) + t * _ucode(max(g[M].min() / R[0], 1)) + (m - 1) * t * _ucode(1 + math.ceil(x[M].mean() / R[0]))
        scores.append(s1 / m)
    groups = [Mg for Mg in groups if 2 <= len(Mg) <= c]
    scores = [s for Mg, s in zip([M for M in groups], scores)] if groups else []
    if return_all:
        return groups, np.array(scores[: len(groups)]), {"d": d, "n_outliers": int(A.sum())}
    return groups


def _mccatch_fit(Z, f):
    base = hdb(Z)
    if isinstance(f, str) and f.startswith("b"):
        return groups_over_base(Z, mccatch(Z, b=float(f[1:])), base, max_size=15)
    groups, sc, _ = mccatch(Z, return_all=True)
    order = np.argsort(-sc) if len(sc) else np.array([], int)
    k = int(math.ceil(float(f) * len(order)))
    return groups_over_base(Z, [groups[i] for i in order[:k]], base, max_size=15)


# ---------------- outlier scorers + mutual-kNN grouping ----------------
def _group_outliers(Z, out_idx, base, k=3, max_size=15):
    lab = base.copy()
    k = min(k, len(out_idx) - 1)                 # guard: fewer than k + 1 outliers (small data sets)
    if k < 1:
        return lab
    _, nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[out_idx]).kneighbors(Z[out_idx])
    near = [set(r[1:]) for r in nn]
    ii, jj = [], []
    for a in range(len(out_idx)):
        for b in nn[a][1:]:
            if a in near[b]:
                ii.append(a); jj.append(b)
    if not ii:
        return lab
    comp = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(out_idx),) * 2), directed=False)[1]
    size = np.bincount(comp)
    nxt = lab.max() + 1
    for c in np.flatnonzero((size >= 2) & (size <= max_size)):
        lab[out_idx[comp == c]] = nxt; nxt += 1
    return lab


def _scored_fit(kind):
    def fit(Z, q):
        base = hdb(Z)
        if kind == "iforest":
            from sklearn.ensemble import IsolationForest
            s = -IsolationForest(random_state=0).fit(Z).score_samples(Z)
        elif kind == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            s = -LocalOutlierFactor(n_neighbors=20).fit(Z).negative_outlier_factor_
        else:
            s = np.nan_to_num(hdbscan_fit(Z, _mcs(len(Z)), max(3, _mcs(len(Z)) // 2)).outlier_scores_)
        return _group_outliers(Z, np.flatnonzero(s >= np.quantile(s, 1 - q)), base)
    return fit


def _gap_fit(k, squared=True):
    def fit(Z, thr):
        return groups_over_base(Z, gapclust_groups(*gapclust_raw(Z, k, squared), thr=thr))
    return fit


def _hdb_mcs(Z, c):
    c = int(c)
    return hdbscan_fit(Z, c, c).labels_


def _hdb_base(Z, _=None):
    return hdb(Z)


def _sccad_fit(floor=10):
    def fit(Z, thr):
        path = os.environ.get("SCCAD_PY")
        if not path:
            raise RuntimeError("scCAD is not vendored; set SCCAD_PY to the authors' scCAD.py (github.com/xuyp-csu/scCAD)")
        try:
            from clustering.eval.nature.tiny_baselines import sccad_raw
        except ImportError as e:
            raise RuntimeError("the scCAD wrapper needs the project's clustering package") from e
        sub, sc = sccad_raw(Z, floor)
        return groups_over_base(Z, [np.flatnonzero(sub == i) for i in np.flatnonzero(sc >= thr)])
    return fit


def _cts_fit(**kw):
    def fit(Z, alpha):
        try:
            from clustering.eval.nature.tiny_methods import cts3
        except ImportError as e:
            raise RuntimeError("CTS is a contestant, not part of the benchmark; install its package to run it") from e
        return cts3(Z, alpha=alpha, dim_est="local", scale_min=True, border=True, **kw)
    return fit


Q6 = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
#: name -> (fit(X, param) -> labels, default parameter grid for the frontier, needs optional code?)
BASELINES = {
    "hdbscan_base": (_hdb_base, (0,), False),
    "hdbscan_mcs": (_hdb_mcs, (2, 3, 4, 5, 8), False),
    "lof_group": (_scored_fit("lof"), Q6, False),
    "iforest_group": (_scored_fit("iforest"), Q6, False),
    "glosh_group": (_scored_fit("glosh"), Q6, False),
    "gapclust_k30": (_gap_fit(30), (1.0, 1.5, 2.0, 3.0, 5.0, 8.0), False),
    "gapclust_k200": (_gap_fit(200), (1.0, 1.5, 2.0, 3.0, 5.0, 8.0), False),
    "gapclust_euclid_k30": (_gap_fit(30, False), (1.0, 1.5, 2.0, 3.0, 5.0, 8.0), False),
    "gapclust_euclid_k200": (_gap_fit(200, False), (1.0, 1.5, 2.0, 3.0, 5.0, 8.0), False),
    "mccatch_reimpl": (_mccatch_fit, (1.0, 0.75, 0.5, 0.25, "b0.05", "b0.2"), False),
    "sccad_official": (_sccad_fit(10), (0.5, 0.6, 0.7, 0.8, 0.9), True),
    "cts_v5": (_cts_fit(), (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3), True),
    "cts_v5_bh": (_cts_fit(respect_base=True, select="bh"), (1e-6, 1e-4, 1e-2, 0.05, 0.1, 0.3), True),
}


def get(name):
    """Returns (fit, grid) for a baseline name."""
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    f, g, _ = BASELINES[name]
    return f, g
