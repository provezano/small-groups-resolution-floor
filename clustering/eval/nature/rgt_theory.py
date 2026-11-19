"""
Theory-predicted detectability of a known group (real-group theory branch, stage 9).

Every quantity here is computed from the ground-truth group and the geometry of the scene. No detection method's
output is used. Each function implements one of the theorems:

- boundary_margin  Result 4 / 4b: the oracle local scan (Poisson count in a ball, local intensity and local intrinsic
                   dimension read from the background around the group, Bonferroni over N centres and M sizes).
                   margin > 0 means "above the detection boundary". The best piece of the group is used: a ball
                   around a member holding j >= m/2 members and b background points with j / (m + b) >= 1/2, the
                   condition for any method to recover the group at Jaccard >= 1/2.
                   With d_plug (the plug-in d of a Beta scan) instead of the local slope, the same margin is the
                   prediction for the Beta scan itself; the difference is the curvature term of Theorem 9.1
                   (U / V = (R / r)^(D - d_plug), Corollary 9.1c).
- masking          Result 1 (HDBSCAN): m <= min_samples and a member with an outsider within its core distance whose
                   core distance is not larger.
- mode_persistence identifiability: does the group hold its own density peak (Result 2), and how deep is the dip to
                   the saddle where it merges with a higher peak (kNN density, kNN graph, union-find).
- umap_share       Result 3: median share of a member's UMAP fuzzy-graph weight that stays inside the group, and
                   the budget bound (m - 1) / log2 k.
- subspace_margin  improvement (a): the same oracle margin in a coordinate subspace, with Bonferroni over the
                   family of subspaces.
"""
import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.neighbors import KDTree, NearestNeighbors


def levina_bickel(R):
    """Local MLE of the mass slope from sorted distances R (n x k) to the background: (k - 2) / sum log(R_k / R_j)
    (the unbiased form). Rows with zero distances are floored."""
    R = np.maximum(R, 1e-12)
    k = R.shape[1]
    s = np.log(R[:, -1:] / R[:, :-1]).sum(1)
    return (k - 2) / np.maximum(s, 1e-12)


def _bg(Z, g):
    out = np.setdiff1d(np.arange(len(Z)), g)
    return out, KDTree(Z[out])


def bg_dim(Z, g, k=20, n_pts=30, out=None, tree=None):
    """Local intrinsic dimension of the background near the group: Levina-Bickel (k background neighbours) at each
    of the n_pts background points nearest the group's centroid, median. This is the d of the homogeneous null
    that Result 4 assumes around the group."""
    g = np.asarray(g)
    if out is None:
        out, tree = _bg(Z, g)
    near = tree.query(Z[g].mean(0, keepdims=True), k=min(n_pts, len(out)))[1][0]
    R = tree.query(Z[out][near], k=k + 1)[0][:, 1:]
    return float(np.median(levina_bickel(R)))


def boundary_margin(Z, g, alpha=0.05, M=14, k=20, d_plug=None, n_far=400, tree=None, out=None, detail=False):
    """log(alpha / (N M)) - log p_oracle for the best recoverable piece of the group (natural log units).
    d is the local Levina-Bickel slope of the background mass around each member (k background neighbours), unless
    d_plug is given."""
    g = np.asarray(g)
    N, m = len(Z), len(g)
    if out is None:
        out, tree = _bg(Z, g)
    kq = min(n_far, len(out))
    Rb = tree.query(Z[g], k=kq)[0]                                    # sorted background distances per member
    dl = levina_bickel(Rb[:, :k])
    d_grp = float(np.median(dl))
    Dg = np.linalg.norm(Z[g][:, None] - Z[g][None], axis=2)
    thr = math.log(alpha / (N * M))
    best = (-np.inf, None)
    for a in range(m):
        dg = np.sort(np.delete(Dg[a], a))
        dd = d_grp if d_plug is None else d_plug
        for j in range(max(2, math.ceil(m / 2)), m + 1):
            r = dg[j - 2]
            b = int(np.searchsorted(Rb[a], r, side="right"))
            if b >= kq or j / (m + b) < 0.5:
                continue
            mu = k * (max(r, 1e-300) / max(Rb[a, k - 1], 1e-300)) ** dd
            n_obs = j - 1 + b
            lp = float(poisson.logsf(n_obs - 1, mu))
            mg = thr - lp
            if mg > best[0]:
                best = (mg, dict(member=int(a), j=j, b=b, r=float(r), mu=float(mu), d=float(dd)))
    if detail:
        return best[0], best[1], d_grp
    return best[0]


def rho_ratio(m, d, mu, N, M=14, alpha=0.05):
    """rho / rho*(m, d, N) for a ball with null mass mu = rho^d holding all m members (Result 4)."""
    rho = mu ** (1 / d)
    rs = math.exp((gammaln(m) + math.log(alpha / (N * M))) / (d * (m - 1)))
    return rho / rs


def masking(Z, g, ms):
    """Result 1. Returns (condition_holds, masked) where masked = condition and m <= ms."""
    g = np.asarray(g)
    nn = NearestNeighbors(n_neighbors=ms + 1).fit(Z)
    c = nn.kneighbors(Z)[0][:, -1]
    ing = np.zeros(len(Z), bool); ing[g] = True
    cond = False
    for x in g:
        idx = nn.radius_neighbors(Z[x:x + 1], radius=c[x], return_distance=False)[0]
        o = idx[~ing[idx]]
        if len(o) and (c[o] <= c[x]).any():
            cond = True
            break
    return cond, bool(cond and len(g) <= ms)


def mode_persistence(Z, g, d, kd=5, kg=10):
    """Identifiability (Result 2 + the dip argument). Log density f = -d log r_kd on the symmetrised kNN graph (kg).
    Returns (has_peak, persistence, purity_at_death): has_peak = the group's densest member is a graph local max;
    persistence = log f(peak) - log f(saddle) where its component merges into a component with a higher peak;
    purity = share of that dying component that is group members."""
    g = np.asarray(g)
    n = len(Z)
    kd = max(1, min(kd, len(g) - 1))
    nn = NearestNeighbors(n_neighbors=max(kd, kg) + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    f = -d * np.log(np.maximum(dist[:, kd], 1e-300))
    nb = [set(r[1:kg + 1]) for r in idx]
    for i in range(n):
        for j in idx[i, 1:kg + 1]:
            nb[j].add(i)
    peak = g[np.argmax(f[g])]
    if any(f[j] > f[peak] for j in nb[peak]):
        return False, 0.0, 0.0
    order = np.argsort(-f)
    parent = -np.ones(n, int)
    top = np.arange(n)
    size_g = np.zeros(n, int); size = np.zeros(n, int)
    ing = np.zeros(n, bool); ing[g] = True

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in order:
        parent[i] = i; size[i] = 1; size_g[i] = int(ing[i])
        for j in nb[i]:
            if parent[j] < 0:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # the component with the lower peak dies at level f[i]
            lo, hi = (ri, rj) if f[top[ri]] < f[top[rj]] else (rj, ri)
            if top[lo] == peak:
                return True, float(f[peak] - f[i]), float(size_g[lo] / size[lo])
            parent[lo] = hi; size[hi] += size[lo]; size_g[hi] += size_g[lo]
    return True, float(np.inf), 1.0


def umap_share(Z, g, k=15, seed=0, graph=None):
    """Result 3. Median internal share of members' fuzzy-graph weight, and the budget bound."""
    if graph is None:
        from umap.umap_ import fuzzy_simplicial_set
        graph = fuzzy_simplicial_set(Z, k, np.random.RandomState(seed), "euclidean")[0].tocsr()
    g = np.asarray(g)
    W = graph[g]
    tot = np.asarray(W.sum(1)).ravel()
    inside = np.asarray(W[:, g].sum(1)).ravel()
    share = inside / np.maximum(tot, 1e-12)
    return float(np.median(share)), min(1.0, (len(g) - 1) / math.log2(k)), graph


def subspace_margin(Z, g, subsets, alpha=0.05, M=14, k=20):
    """Improvement (a): the oracle margin in each coordinate subset S, with Bonferroni over the family
    (threshold alpha / (N M |F|)). Returns (best margin, best subset, margin in the full space)."""
    F = len(subsets)
    best = (-np.inf, None)
    for S in subsets:
        mg = boundary_margin(Z[:, list(S)], g, alpha=alpha / F, M=M, k=k)
        if mg > best[0]:
            best = (mg, tuple(int(s) for s in S))
    return best
