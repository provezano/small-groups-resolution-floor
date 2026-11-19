"""
Official / published baselines for the tiny-group benchmark, as f(Z) -> labels with small groups written over
HDBSCAN's default labels (same convention as tiny_methods.scored, so scores are comparable).

- scCAD (Xu et al., Nat Commun 2024): the authors' scCAD.py from github.com/xuyp-csu/scCAD, imported by path (code read
  before use; no network or shell calls). Run with normalization off on features shifted to be non-negative. Its rare
  sub-clusters are the groups. One guard patch: its PCA step asks for 40 components even on < 40 features; capped at
  the feature count. NOTE: the official code scores only sub-clusters of >= 10 cells (`len(id) < 10` ->
  score 0), so it cannot report groups of 2-9 by design. `sccad_floor2` is our patched copy with that floor at 2
  (labelled as modified). The overlap threshold is swept post hoc on the returned per-sub-cluster scores.
- McCatch (Sanchez Vinces, Cordeiro, Faloutsos, ICDE 2024): the official code (Java/C++, figshare private link) could
  not be fetched (figshare returns 403 to scripted access) and there is no Java runtime here, so this is OUR
  REIMPLEMENTATION from the paper's definitions (arXiv 2403.08027v1): radii l/2^(a-e), neighbour counts per radius,
  plateaus by log-log slope <= b, first plateau (x = 1NN distance) and longest middle plateau (y = group 1NN distance),
  MDL cutoff d on the histogram of x, outliers x >= d or y >= d, microclusters = components of the points with
  y >= d joined at the first radius above their largest x, scores by the paper's compression cost. Defaults a = 15,
  b = 0.1, c = ceil(0.1 n). Not validated against the official code.
- GapClust (Fa et al., Nat Commun 2021): NOT REDISTRIBUTED. The upstream R package is GPL-3, so the Python port that was here
  was removed from this MIT-licensed deposit; the functions are stubs that load your own port from GAPCLUST_PY (below).
"""
import contextlib
import importlib.util
import io
import math
import os
import types

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .tiny_methods import hdb

try:
    from . import _tgpaths as _TGP
except ImportError:
    import _tgpaths as _TGP
SCCAD_PATH = os.environ.get("SCCAD_PY", _TGP.DATA + "/dl/sccad/repo/scCAD.py")
SCCAD_OUT = _TGP.DATA + "/sccad_out/"
_MODS = {}


def _sccad_module(floor):
    if floor not in _MODS:
        src = open(SCCAD_PATH).read()
        # dimension guard: the authors' fast_clustering asks for 40 PCA components even when the data has fewer
        # features (gene matrices always have more); cap at the feature count so 20-d inputs run
        assert "nps = min(40, data.shape[0])" in src
        src = src.replace("nps = min(40, data.shape[0])", "nps = min(40, data.shape[0], data.shape[1])")
        if floor != 10:
            assert "len(id) < 10" in src
            src = src.replace("len(id) < 10", f"len(id) < {floor}")
        mod = types.ModuleType(f"sccad_floor{floor}")
        exec(compile(src, SCCAD_PATH, "exec"), mod.__dict__)
        _MODS[floor] = mod
    return _MODS[floor]


def sccad_raw(Z, floor=10, seed=2023, counts=False):
    """Returns (sub-cluster labels, per-sub-cluster scores) from the authors' scCAD. counts=True: raw counts with the
    authors' own normalisation; otherwise features shifted to be non-negative, normalisation off."""
    os.makedirs(SCCAD_OUT, exist_ok=True)
    X = np.asarray(Z, float)
    if not counts:
        X = X - X.min(0)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res, score, sub, _ = _sccad_module(floor).scCAD(data=X, normalization=counts, seed=seed, save_full=False,
                                                          save_path=SCCAD_OUT)
    return np.asarray(sub), np.asarray(score)


def groups_over_base(Z, groups, base=None, max_size=None):
    lab = (hdb(Z) if base is None else base).copy()
    nxt = lab.max() + 1
    for g in groups:
        g = np.asarray(g)
        if len(g) >= 2 and (max_size is None or len(g) <= max_size):
            lab[g] = nxt; nxt += 1
    return lab


def sccad_sweep(Z, floor=10, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    sub, score = sccad_raw(Z, floor)
    base = hdb(Z)
    return [groups_over_base(Z, [np.flatnonzero(sub == i) for i in np.flatnonzero(score >= t)], base)
            for t in thresholds]


# ---------------- GapClust: NOT REDISTRIBUTED (licensing) ----------------
# The upstream GapClust R package (github.com/fabotao/GapClust) is declared GPL-3 (DESCRIPTION) and has no LICENSE
# file. A line-by-line Python port of its core is a derived work and cannot be distributed under this deposit's MIT
# licence, so the port is not part of this deposit. The functions below keep their names and signatures so that every
# caller still imports; they raise GapClustUnavailable unless you supply your own port.
#
# To use GapClust here: obtain the official package from github.com/fabotao/GapClust, write (or call) your own
# implementation in a Python file, and point the environment variable GAPCLUST_PY at it. That file must define
#   gapclust_raw(Z, k=200, squared=False) -> (skew, top_ave, v1k, idx)
#   gapclust_groups(skew, top_ave, v1k, idx, thr=2.0) -> list of index arrays
# (the signatures used by the callers in this deposit). Results already computed with the original port are kept in
# results/ and in tinybench/results/; they document the comparison but cannot be regenerated from this deposit alone.
class GapClustUnavailable(RuntimeError):
    pass


_GAP_MSG = ("GapClust is not available in this deposit. The official GapClust package (github.com/fabotao/GapClust) is "
            "GPL-3, so a port of it cannot be redistributed under this deposit's MIT licence and was removed. Obtain the "
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


def gapclust_raw(Z, k=200, squared=False):
    """Stub: delegates to the user's port named by GAPCLUST_PY, else raises GapClustUnavailable."""
    return _gapclust_user().gapclust_raw(Z, k, squared)


def gapclust_groups(skew, top_ave, v1k, idx, thr=2.0):
    """Stub: delegates to the user's port named by GAPCLUST_PY, else raises GapClustUnavailable."""
    return _gapclust_user().gapclust_groups(skew, top_ave, v1k, idx, thr)


def gapclust(Z, k=200, thr=2.0, base=None, squared=False):
    """GapClust groups written over `base`; needs the user's port (see above)."""
    return groups_over_base(Z, gapclust_groups(*gapclust_raw(Z, k, squared), thr=thr), base)


def gapclust_sweep(Z, k=200, thrs=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0), squared=False):
    """gapclust over a grid of thresholds; needs the user's port (see above)."""
    raw = gapclust_raw(Z, k, squared)
    base = hdb(Z)
    return [groups_over_base(Z, gapclust_groups(*raw, thr=t), base) for t in thrs]


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


def mccatch_sweep(Z, fracs=(1.0, 0.75, 0.5, 0.25)):
    """Default McCatch, then keep only the top share of its microclusters by score (operating points), plus b = 0.05 /
    0.2 at full share."""
    base = hdb(Z)
    out = []
    groups, sc, _ = mccatch(Z, return_all=True)
    order = np.argsort(-sc) if len(sc) else np.array([], int)
    for f in fracs:
        k = int(math.ceil(f * len(order)))
        out.append(groups_over_base(Z, [groups[i] for i in order[:k]], base, max_size=15))
    for bb in (0.05, 0.2):
        out.append(groups_over_base(Z, mccatch(Z, b=bb), base, max_size=15))
    return out
