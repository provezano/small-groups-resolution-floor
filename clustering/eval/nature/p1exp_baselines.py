"""
More baselines for the tiny-group benchmark, each with a parameter grid so the matched false-rate frontier
(`tinybench_cli.frontier`) can pick its best point per tier.

Every method is a function f(X, g) -> labels, where g is one grid point (a tuple). Work shared between grid points of
the same scene (a fit, a reachability plot, a persistence diagram) is cached on the scene's bytes, so a grid costs
about one fit. Methods:
- OPTICS (sklearn): one fit per min_samples, then xi extraction (xi swept) and DBSCAN-style extraction (eps swept).
- FINCH (Sarfraz et al., CVPR 2019; pip finch-clust 0.2.3): every partition level of the hierarchy, two metrics.
- ToMATo (Chazal et al., J. ACM 2013; gudhi 3.13): kNN graph + logDTM density, merge threshold swept.
- AuToMATo (pip automato 0.1.1): ToMATo with a bottleneck-bootstrap choice of the number of clusters, alpha swept.
- Density peaks (Rodriguez and Laio, Science 2014): Gaussian-kernel rho with cutoff dc (a percentile of pairwise
  distances), delta to the nearest denser point, centres = points with delta >= c x dc and rho above a floor; halo
  removal as in the paper (optional).
- Isolation forest + mutual-kNN grouping (the McCatch-style grouping already used for LOF), top-q share and k swept.
- DBSCAN: eps = a quantile of the distance to the (min_samples - 1)th neighbour, both swept.
- HDBSCAN: min_cluster_size 2-8, min_samples 1-3 (below the group size), eom and leaf.
Plus the references, rerun with the same scoring: CTS v5 (Bonferroni), CTS v5 + BH, GapClust k = 30 (needs your own port, GAPCLUST_PY; not shipped).
"""
import hashlib

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import methods as M
from .tiny_baselines import gapclust_groups, gapclust_raw, groups_over_base
from .tiny_methods import _group_outliers, cts3, hdb

_CACHE = {}


def _key(X, tag):
    return (tag, hashlib.blake2b(np.ascontiguousarray(X).view(np.uint8), digest_size=12).hexdigest())


def _cached(X, tag, fn):
    k = _key(X, tag)
    if k not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[k] = fn()
    return _CACHE[k]


def _small_noise(lab):
    """Singletons are not clusters: label them noise (FINCH and ToMATo assign every point)."""
    lab = np.asarray(lab).copy()
    u, c = np.unique(lab, return_counts=True)
    lab[np.isin(lab, u[c == 1])] = -1
    return lab


# ---------------- references ----------------
def cts_v5(X, a):
    return cts3(X, alpha=a, dim_est="local", scale_min=True, border=True)


def cts_v5_bh(X, a):
    return cts3(X, alpha=a, dim_est="local", scale_min=True, border=True, respect_base=True, select="bh")


def gapclust_k30(X, thr):
    raw = _cached(X, "gap30", lambda: (gapclust_raw(X, 30), hdb(X)))
    return groups_over_base(X, gapclust_groups(*raw[0], thr=thr), raw[1])


# ---------------- DBSCAN ----------------
def dbscan(X, g):
    from sklearn.cluster import DBSCAN
    ms, q = g
    kd = _cached(X, f"kd{ms}", lambda: NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)[0][:, -1])
    eps = max(float(np.quantile(kd, q)), 1e-12)
    return DBSCAN(eps=eps, min_samples=ms).fit(X).labels_


# ---------------- OPTICS ----------------
def _optics_fit(X, ms):
    from sklearn.cluster import OPTICS
    o = OPTICS(min_samples=ms, max_eps=np.inf, cluster_method="dbscan", eps=np.inf).fit(X)
    return o.reachability_, o.core_distances_, o.ordering_, o.predecessor_


def optics(X, g):
    from sklearn.cluster import cluster_optics_dbscan, cluster_optics_xi
    kind, ms, v = g
    reach, core, order, pred = _cached(X, f"optics{ms}", lambda: _optics_fit(X, ms))
    if kind == "xi":
        return cluster_optics_xi(reachability=reach, predecessor=pred, ordering=order, min_samples=ms,
                                 min_cluster_size=2, xi=v)[0]
    eps = float(np.quantile(core, v))
    return cluster_optics_dbscan(reachability=reach, core_distances=core, ordering=order, eps=eps)


# ---------------- FINCH ----------------
def finch(X, g):
    from finch import FINCH
    metric, level = g
    c = _cached(X, f"finch{metric}", lambda: FINCH(X, distance=metric, verbose=False, random_state=0)[0])
    return _small_noise(c[:, min(level, c.shape[1] - 1)])


# ---------------- ToMATo / AuToMATo ----------------
def tomato(X, g):
    from gudhi.clustering.tomato import Tomato
    k, tau = g
    t = _cached(X, f"tomato{k}", lambda: Tomato(density_type="logDTM", k=k, k_DTM=k).fit(X))
    t.merge_threshold_ = tau
    return _small_noise(t.labels_)


def automato(X, g):
    from automato import Automato
    k, a = g
    A = _cached(X, f"automato{k}", lambda: Automato(n_bootstrap=200, parallelize=0, random_state=0,
                                                     tomato_params=dict(k=k)).fit(X))
    A.alpha = a
    return _small_noise(A.labels_)


# ---------------- density peaks ----------------
def _dp_fit(X, pct):
    from scipy.spatial.distance import pdist, squareform
    dv = pdist(X)
    dc = float(np.percentile(dv, pct))
    D = squareform(dv)
    del dv
    rho = np.exp(-(D / dc) ** 2).sum(1) - 1
    order = np.argsort(-rho, kind="stable")
    delta = np.empty(len(X))
    nh = np.full(len(X), -1)
    delta[order[0]] = D[order[0]].max()
    for r in range(1, len(X)):
        i = order[r]
        hi = order[:r]
        j = hi[np.argmin(D[i, hi])]
        delta[i], nh[i] = D[i, j], j
    return D, dc, rho, delta, nh, order


def density_peaks(X, g):
    pct, cmult, halo = g
    D, dc, rho, delta, nh, order = _cached(X, f"dp{pct}", lambda: _dp_fit(X, pct))
    centres = (delta >= cmult * dc) & (rho >= 0.5)       # rho floor: at least one other point within ~dc
    centres[order[0]] = True
    lab = -np.ones(len(X), int)
    lab[centres] = np.arange(centres.sum())
    for i in order:
        if lab[i] < 0:
            lab[i] = lab[nh[i]]
    if halo:
        # Rodriguez-Laio halo: rho_b(c) = max over border pairs (different clusters, within dc) of their mean rho;
        # members of c with rho < rho_b(c) are halo (noise)
        ii, jj = np.nonzero(np.triu(D < dc, 1))
        diff = lab[ii] != lab[jj]
        rb = np.full(lab.max() + 1, -np.inf)
        m = (rho[ii[diff]] + rho[jj[diff]]) / 2
        np.maximum.at(rb, lab[ii[diff]], m)
        np.maximum.at(rb, lab[jj[diff]], m)
        lab = np.where(rho < rb[lab], -1, lab)
    return _small_noise(lab)


# ---------------- isolation forest + mutual kNN ----------------
def iforest_group(X, g):
    from sklearn.ensemble import IsolationForest
    q, k = g
    s, base = _cached(X, "iforest", lambda: (-IsolationForest(random_state=0).fit(X).score_samples(X), hdb(X)))
    out = np.flatnonzero(s >= np.quantile(s, 1 - q))
    return _group_outliers(X, out, base, k=min(k, len(out) - 1))


# ---------------- HDBSCAN, min_samples below the group size ----------------
def hdbscan_small(X, g):
    mcs, ms, meth = g
    return M.hdbscan_labels(X, mcs, ms, method=meth)


ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3, 0.6, 0.9)        # stage-8 grid + 0.6, 0.9 (0.3 was the edge)
ALPHAS_BH = (1e-6, 1e-4, 1e-2, 0.05, 0.1, 0.3, 0.5)
GRIDS = {
    "cts_v5": (cts_v5, list(ALPHAS)),
    "cts_v5_bh": (cts_v5_bh, list(ALPHAS_BH)),
    # alpha > 1 is no longer a family-wise level, only a looser threshold -log(alpha / n_tests): a free knob, the
    # way the baselines' grids are, to see how far the frontier goes when validity is given up
    "cts_v5_ext": (cts_v5, [0.9, 3.0, 10.0, 30.0, 100.0, 300.0]),
    "gapclust_k30": (gapclust_k30, [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]),
    "dbscan": (dbscan, [(ms, q) for ms in (2, 3, 4, 5, 8)
                          for q in (0.05, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)]),
    "optics": (optics, [("xi", ms, xi) for ms in (2, 3, 5, 8) for xi in (0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6)]
               + [("dbscan", ms, q) for ms in (2, 3, 5, 8) for q in (0.2, 0.5, 0.8, 0.9, 0.95, 0.98)]),
    "finch": (finch, [(m, lv) for m in ("euclidean", "cosine") for lv in (0, 1, 2, 3)]),
    "tomato": (tomato, [(k, t) for k in (3, 5, 10, 20) for t in (0.5, 5.0, 25.0, 50.0, 100.0, np.inf)]),
    "automato": (automato, [(k, a) for k in (3, 5, 10) for a in (0.05, 0.35, 0.6, 0.9, 0.95, 0.99)]),
    "density_peaks": (density_peaks, [(p, c, h) for p in (0.5, 1.0, 2.0, 5.0, 10.0) for c in (0.5, 1.0, 2.0, 4.0, 8.0)
                                      for h in (False, True)]),
    "iforest_group": (iforest_group, [(q, k) for q in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)
                                      for k in (3, 5, 8)]),
    "hdbscan_small": (hdbscan_small, [(mcs, ms, me) for mcs in (2, 3, 4, 5, 8) for ms in (1, 2, 3) if ms <= mcs
                                      for me in ("eom", "leaf")]),
}


# ---------------- CTS with a swapped base clustering (sensitivity) ----------------
def cts_v5_base(X, a, base="hdbscan", k=8, **kw):
    """CTS v5 groups written over another base: 'hdbscan' (default), 'none' (all noise) or 'kmeans' (k clusters).
    v5 writes its groups whatever the base holds (respect_base off), so swapping the base after the fact is exact."""
    lab = cts3(X, alpha=a, dim_est="local", scale_min=True, border=True, **kw)
    if base == "hdbscan":
        return lab
    h = hdb(X).max()
    if base == "none":
        b = -np.ones(len(X), int)
    else:
        from sklearn.cluster import KMeans
        b = _cached(X, f"km{k}", lambda: KMeans(k, n_init=4, random_state=0).fit_predict(X))
    out = b.copy()
    nxt = out.max() + 1
    for c in np.unique(lab[lab > h]):
        out[lab == c] = nxt; nxt += 1
    return out


def cts_v5_dbarm(X, a, ms=2, q=0.98):
    """Candidate fix for the isolated-tier loss (not calibrated, an option only): CTS v5, plus the small clusters
    (2-15 points) of one DBSCAN run at a global scale (eps = the q-quantile of the distance to the (ms - 1)th
    neighbour), added where they touch no CTS group. The Beta scan compares a set with its own ring only; a set that
    is dense against the whole data set but sits in an empty region at d = 2 has too little ring contrast to pass
    Bonferroni (error analysis in the stage-9 notes)."""
    from sklearn.cluster import DBSCAN
    lab = cts_v5(X, a)
    h = hdb(X).max()
    kd = _cached(X, f"kd{ms}", lambda: NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)[0][:, -1])
    db = _cached(X, f"db{ms}_{q}", lambda: DBSCAN(eps=float(np.quantile(kd, q)), min_samples=ms).fit(X).labels_)
    nxt = lab.max() + 1
    for c in np.unique(db[db >= 0]):
        pts = np.flatnonzero(db == c)
        if 2 <= len(pts) <= 15 and (lab[pts] <= h).all():
            lab[pts] = nxt; nxt += 1
    return lab


GRIDS["cts_v5_dbarm"] = (cts_v5_dbarm, list(ALPHAS))


# ---------------- collapse scale that does not assume the mass is half the data ----------------
def _otsu(v, bins=128):
    """Otsu's threshold on v (maximises the between-class variance of a two-class split)."""
    h, e = np.histogram(v, bins)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h); w2 = w[-1] - w
    m1 = np.cumsum(h * c) / np.maximum(w, 1)
    m2 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w2, 1)
    return float(c[np.argmax(w * w2 * (m1 - m2) ** 2)])


def cts_v5_eps(Z, alpha=0.05, eps_rule="otsu", big=20, max_m=15, k_bg=10, min_core=10, tight=0.25, attach=0.5,
               select="bonferroni", respect_base=False):
    """CTS v5 with the collapse scale eps chosen by `eps_rule`. 'median' reproduces tiny_methods.cts3 (eps = median
    r_10 over all points), which sits at the mass scale only while the near-duplicate mass is about half of the data;
    'otsu' puts eps at Otsu's split of log r_10 (the valley between the tight near-copies and everything else), and
    falls back to the median when the split finds no tight class. The rest is cts3's code (border join on)."""
    from scipy.sparse.csgraph import connected_components
    from .tiny_methods import _scan_select
    n = len(Z)
    r = NearestNeighbors(n_neighbors=min_core + 1).fit(Z).kneighbors(Z)[0][:, -1]
    eps = float(np.median(r))
    if eps_rule.startswith("otsu"):
        lr = np.log(np.maximum(r, 1e-12))
        t = float(np.exp(_otsu(lr)))
        lo, hi = r <= t, r > t
        # accept the split only if the low class is a tight family scale (>= 4x tighter than the rest, the same
        # 'tight' rule the families must pass later) and holds enough points to form a family
        if lo.sum() > big and hi.any() and np.median(r[lo]) <= tight * np.median(r[hi]):
            # the valley is wide (often 10x); its middle swallows groups that sit a few copy spreads from a family.
            # eps is put at the top of the tight class instead (its 90th percentile), as the median rule does when
            # the mass is half of the data
            eps = float(np.quantile(r[lo], int(eps_rule[6:]) / 100)) if eps_rule.startswith("otsu_q") else t
    core = r <= eps
    comp = -np.ones(n, int)
    pairs = NearestNeighbors(radius=eps).fit(Z[core]).radius_neighbors_graph(Z[core], mode="connectivity")
    comp[core] = connected_components(pairs, directed=False)[1]
    ref = np.median(r[~core]) if (~core).any() else np.inf
    if (~core).any():
        nc = np.flatnonzero(~core)
        dd, jj = NearestNeighbors(n_neighbors=1).fit(Z[core]).kneighbors(Z[nc])
        dn = NearestNeighbors(n_neighbors=2).fit(Z[nc]).kneighbors(Z[nc])[0][:, 1] if len(nc) > 1 else np.full(len(nc), np.inf)
        join = (dd[:, 0] <= eps) & (dd[:, 0] <= dn)
        comp[nc[join]] = comp[np.flatnonzero(core)[jj[join, 0]]]
    collapsed = np.zeros(n, bool)
    for c in np.unique(comp[core]):
        mem = np.flatnonzero(comp == c)
        if len(mem) > big and np.median(r[mem]) <= tight * ref:
            collapsed[mem] = True
    if collapsed.any():
        cidx = np.flatnonzero(collapsed)
        free = np.flatnonzero(~collapsed)
        dd, jj = NearestNeighbors(n_neighbors=1).fit(Z[cidx]).kneighbors(Z[free])
        near = dd[:, 0] <= attach * eps
        comp[free[near]] = comp[cidx[jj[near, 0]]]
        collapsed[free[near]] = True
    keep = np.flatnonzero(~collapsed)
    reps = []
    for c in np.unique(comp[collapsed]):
        mem = np.flatnonzero(collapsed & (comp == c))
        reps.append(mem[np.argmin(np.linalg.norm(Z[mem] - Z[mem].mean(0), axis=1))])
    R = np.r_[keep, np.array(reps, int)]
    is_rep = np.r_[np.zeros(len(keep), bool), np.ones(len(reps), bool)]
    return _scan_select(Z, R, is_rep, max_m, k_bg, alpha, dim_est="local", scale_min=True, select=select,
                        respect_base=respect_base)


def cts_v5_otsu(X, a):
    """Frozen on seeds 600-602: eps = 90th percentile of the tight class of Otsu's split of log r_10, and big = 10
    (families of 11+ near-copies collapse; groups of <= 10 points have no core points of their own)."""
    return cts_v5_eps(X, a, eps_rule="otsu_q90", big=10)


GRIDS["cts_v5_otsu"] = (cts_v5_otsu, list(ALPHAS))
