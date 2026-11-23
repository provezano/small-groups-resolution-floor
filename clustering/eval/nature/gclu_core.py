"""
GCLU: a general-purpose clusterer built from the CTS ideas, with a calibrated answer at every scale.

Two scales, one test each (scale-split certification):
1. Tiny groups (m <= 15): the CTS scan (exact local Beta null, fringe-fixed collapse) on the FULL data at alpha / 2.
   The candidate family is every kNN ball, which is fixed before looking at the data, and for that reason no split
   is needed. But this branch delegates to cts_fringe, which divides by the count of pairs the collapse leaves
   eligible, not by that fixed family; the free-selection theorem of the paper (Theorem 16 as submitted) asks for
   the fixed family, so the premise does not hold as run and the paper does not bound the cost in level.
2. Large clusters: Poisson thinning. A random half X1 proposes clusters (HDBSCAN, leaf selection, over-segmenting);
   the other half X2 certifies each split with an exact, intensity-free dip test. Under an inhomogeneous Poisson
   model the two halves are independent Poisson processes with the same intensity shape (thinning theorem), so any
   hypothesis built from X1 is a fixed hypothesis for X2.
   Dip test for a split between adjacent X1 clusters P, Q (adjacent = joined by an edge of X1's mutual-reachability
   MST with no cluster in between, Kruskal order): saddle s = midpoint of the MST edge where they join; a, b = the
   densest X1 point (smallest core distance) of P and of Q; three disjoint balls of the same radius h (from X1).
   H0: mean intensity in B(s, h) >= min(mean intensity in B(a, h), B(b, h)) ("no dip").
   X2 counts N_s, N_a, N_b are independent Poisson; conditional on N_s + N_a, N_s ~ Bin(., >= 1/2) under
   lambda_s >= lambda_a. p = max(p_a, p_b) (intersection-union) is valid under H0. Radii from k_t in K_TEST expected
   points (Bonferroni over the radii and the K - 1 tests). Splits that fail are merged (union-find).
   The two stages share alpha by a union bound.

Also here: the pruned direction (a multi-scale Beta scan up to m = n / 2), the invalid same-data ablation, and the
significance-tested baselines (G-means, GMM-BIC) used in the stage-12 gclu write-up.
"""
import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta as beta_dist, binom, norm
from sklearn.neighbors import KDTree, NearestNeighbors

K_TEST = (10, 30, 90)


# ---------------------------------------------------------------- the large-scale stage
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a
        return a


def dip_pvalue(n_s, n_a, n_b):
    """Exact p-value of H0: lambda_s >= min(lambda_a, lambda_b) for three independent Poisson counts in equal
    volumes (conditional binomial tests, combined by intersection-union)."""
    pa = binom.cdf(n_s, n_s + n_a, 0.5) if n_s + n_a > 0 else 1.0
    pb = binom.cdf(n_s, n_s + n_b, 0.5) if n_s + n_b > 0 else 1.0
    return float(max(pa, pb))


def lune_counts(t2, X2, s, a, b, h):
    """X2 counts in B_s minus B_a, B_a minus B_s, B_s minus B_b, B_b minus B_s (all of radius h)."""
    ids = t2.query_radius(np.vstack([s, a, b]), r=h)
    S_, A_, B_ = set(ids[0].tolist()), set(ids[1].tolist()), set(ids[2].tolist())
    return len(S_ - A_), len(A_ - S_), len(S_ - B_), len(B_ - S_)


def propose(X1, ms=5, mcs=None, sel="leaf"):
    import hdbscan
    n1 = len(X1)
    mcs = mcs or max(5, n1 // 100)
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=sel,
                         gen_min_span_tree=True).fit(X1.astype(np.float64))
    mst = cl.minimum_spanning_tree_.to_numpy()
    core = NearestNeighbors(n_neighbors=ms + 1).fit(X1).kneighbors(X1)[0][:, -1]
    return cl.labels_.copy(), mst, core


def certify(X1, X2, lab1, mst, core, alpha=0.025, k_test=K_TEST, same_data=False, ref="far", side="cluster", pick="core", mult="fallback", h_rule="min"):
    """Kruskal over the MST edges; each edge that first joins two cluster-holding components gets one dip test
    between the nearest cluster on each side. Returns the merged X1 labels and the list of tests."""
    K = int(lab1.max()) + 1
    if K <= 1:
        return lab1.copy(), []
    modes = []
    for c in range(K):
        mem = np.flatnonzero(lab1 == c)
        modes.append(mem[np.argmin(core[mem])])
    modes = np.array(modes)
    k_test = tuple(k for k in k_test if k < len(X1)) or (len(X1) - 1,)
    nn1 = NearestNeighbors(n_neighbors=max(k_test) + 1).fit(X1)
    rk = nn1.kneighbors(X1[modes])[0]                         # X1 k-NN radii at the modes
    t2 = KDTree(X2)
    t1 = KDTree(X1)
    rng_pick = np.random.default_rng(0)
    order = np.argsort(mst[:, 2], kind="stable")
    n1 = len(X1)
    uf = _UF(n1)
    # each X1 cluster is a connected subtree of the MST (it is a component at its birth level), so it is contracted first
    for c in range(K):
        mem = np.flatnonzero(lab1 == c)
        for x in mem[1:]:
            uf.union(int(mem[0]), int(x))
    clus = {i: [] for i in range(n1)}
    for c in range(K):
        clus[uf.find(int(np.flatnonzero(lab1 == c)[0]))] = [c]
    pending = []
    for e in order:
        u, v = int(mst[e, 0]), int(mst[e, 1])
        ru, rv = uf.find(u), uf.find(v)
        if ru == rv:
            continue
        cu, cv = clus[ru], clus[rv]
        if cu and cv:
            s = 0.5 * (X1[u] + X1[v])
            # nearest cluster (by mode) on each side of the saddle
            P = min(cu, key=lambda c: np.linalg.norm(X1[modes[c]] - s))
            Q = min(cv, key=lambda c: np.linalg.norm(X1[modes[c]] - s))
            # X1's own estimate of the dip depth: saddle reachability over the sparser side's best core distance
            depth = float(mst[e, 2]) / max(core[modes[cu]].min(), core[modes[cv]].min(), 1e-300)
            pending.append((P, Q, s, depth, list(cu), list(cv)))
        r = uf.union(ru, rv)
        clus[r] = cu + cv
        if r != ru:
            clus.pop(ru, None)
        if r != rv:
            clus.pop(rv, None)
    n_tests = max(1, len(pending)) * len(k_test)
    tests = []
    merge = _UF(K)
    # multiplicity over the K - 1 splits. The order (deepest X1 dip first) is fixed by X1 alone, so it is a valid
    # a-priori order for X2. "bonferroni": alpha / (K - 1) each. "fallback" (Wiens 2003): split i gets weight w_i
    # (harmonic, summing to 1) plus the level carried over from an immediately preceding rejection; FWER <= alpha.
    pending.sort(key=lambda t: -t[3])
    L = len(pending)
    wts = 1.0 / np.arange(1, L + 1)
    wts /= wts.sum()
    carry = 0.0
    for ti, (P, Q, s, w, cu, cv) in enumerate(pending):
        best = 1.0
        rows = []
        if side == "component":
            # any reference point on each side is valid: every point of a Kruskal component is joined to the saddle
            # edge by a path whose MST edges are all lower (denser) than the saddle edge
            memP, memQ = np.flatnonzero(np.isin(lab1, cu)), np.flatnonzero(np.isin(lab1, cv))
        else:
            memP, memQ = np.flatnonzero(lab1 == P), np.flatnonzero(lab1 == Q)
        grid = [(kt, hr) for kt in k_test for hr in (("min", "max") if h_rule == "both" else (h_rule,))]
        for kt, hr in grid:
            if side == "component":
                h = min(rk[cu, kt].min(), rk[cv, kt].min())
            else:
                # "min" (frozen v1): the denser cluster's scale, which leaves the sparser side's ball nearly empty
                # when densities differ; "max" (post-hoc v2): both reference balls expect >= k_t points
                h = min(rk[P, kt], rk[Q, kt]) if hr == "min" else max(rk[P, kt], rk[Q, kt])
            if ref == "lune":
                # the clusters' modes, any radius: when balls overlap, compare the two lunes B_s minus B_a and
                # B_a minus B_s (disjoint, congruent, so equal volume in any dimension)
                a, b = X1[modes[P]], X1[modes[Q]]
            elif ref == "mode":
                a, b = X1[modes[P]], X1[modes[Q]]
                h = min(h, 0.999 * min(np.linalg.norm(s - a), np.linalg.norm(s - b), 0.5 * np.linalg.norm(a - b)))
            else:
                # "far": the densest point of each cluster at least 2h from the saddle (balls disjoint by construction)
                fp = memP[np.linalg.norm(X1[memP] - s, axis=1) >= 2 * h]
                fq = memQ[np.linalg.norm(X1[memQ] - s, axis=1) >= 2 * h]
                if not len(fp) or not len(fq):
                    continue
                if pick == "count":
                    # the reference with the most X1 points within h (the test's own scale; a small-scale density
                    # such as the core distance regresses badly in X2: winner's curse)
                    fp = fp if len(fp) <= 400 else rng_pick.choice(fp, 400, replace=False)
                    fq = fq if len(fq) <= 400 else rng_pick.choice(fq, 400, replace=False)
                    a = X1[fp[np.argmax(t1.query_radius(X1[fp], r=h, count_only=True))]]
                    b = X1[fq[np.argmax(t1.query_radius(X1[fq], r=h, count_only=True))]]
                else:
                    a, b = X1[fp[np.argmin(core[fp])]], X1[fq[np.argmin(core[fq])]]
                if np.linalg.norm(a - b) < 2 * h:
                    continue
            if h <= 0:
                continue
            if ref == "lune":
                ns_a, na, ns_b, nb = lune_counts(t2, X2, s, a, b, h)
                if same_data:
                    na, nb = max(na - 1, 0), max(nb - 1, 0)
                pa = binom.cdf(ns_a, ns_a + na, 0.5) if ns_a + na > 0 else 1.0
                pb = binom.cdf(ns_b, ns_b + nb, 0.5) if ns_b + nb > 0 else 1.0
                p = float(max(pa, pb))
                cnt = [ns_a, na, ns_b, nb]
            else:
                cnt = t2.query_radius(np.vstack([s, a, b]), r=h, count_only=True)
                if same_data:          # the invalid ablation: the proposal points themselves (minus the centres)
                    cnt = np.maximum(cnt - np.array([0, 1, 1]), 0)
                p = dip_pvalue(int(cnt[0]), int(cnt[1]), int(cnt[2]))
            rows.append((kt, float(h), [int(c) for c in cnt], p))
            best = min(best, p)
        if mult == "bonferroni":
            level = alpha / max(1, L)
        else:
            level = alpha * wts[ti] + carry
        sig = best * len(k_test) * (2 if h_rule == "both" else 1) <= level
        carry = level if (sig and mult == "fallback") else 0.0
        tests.append({"P": int(P), "Q": int(Q), "p": best, "sig": bool(sig), "rows": rows})
        if not sig:
            merge.union(P, Q)
    root = np.array([merge.find(c) for c in range(K)])
    _, newid = np.unique(root, return_inverse=True)
    out = np.where(lab1 >= 0, newid[np.maximum(lab1, 0)], -1)
    return out, tests


def gclu(X, alpha=0.05, seed=0, ms=10, mcs=None, tiny=True, k_test=K_TEST, k_vote=5, same_data=False,
         noise="assign", ref="far", side="cluster", pick="core", mult="fallback", sel="leaf", h_rule="min", return_info=False):
    """noise='assign': every point takes the majority label of its k_vote nearest clustered X1 points (a complete
    partition); noise='keep': HDBSCAN-style noise stays -1 (X2 points vote among all X1 points)."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if same_data == "full":            # invalid ablation without the split: propose and test on all the data
        A, B = np.arange(n), np.arange(0)
    X1 = X[A]
    X2 = X1 if same_data else X[B]
    mcs = mcs or max(5, n // 100)
    if same_data == "full":
        mcs = 2 * mcs          # frozen on seeds 600-619: 1% of n (2% of the proposal half), ms = 10
    lab1, mst, core = propose(X1, ms=ms, mcs=mcs, sel=sel)
    a_large = alpha / 2 if tiny else alpha
    lab1m, tests = certify(X1, X2, lab1, mst, core, alpha=a_large, k_test=k_test, same_data=same_data, ref=ref, side=side, pick=pick, mult=mult, h_rule=h_rule)
    lab = np.empty(n, np.int64)
    lab[A] = lab1m
    if noise == "assign" and (lab1m >= 0).sum() >= k_vote:
        src = np.flatnonzero(lab1m >= 0)
        tgt = np.r_[A[lab1m < 0], B]
    else:
        src, tgt = np.arange(len(A)), B
    if len(tgt):
        idx = NearestNeighbors(n_neighbors=min(k_vote, len(src))).fit(X1[src]).kneighbors(X[tgt], return_distance=False)
        votes = lab1m[src][idx]
        for t, row in enumerate(votes):
            vals, cnt = np.unique(row, return_counts=True)
            lab[tgt[t]] = vals[np.argmax(cnt)]
    if h_rule != "min" and lab.max() < 0:
        lab[:] = 0                     # v2: no proposal at all means one cluster (the "no structure" answer), not all noise
    n_large = int(lab.max()) + 1
    groups = []
    if tiny:
        from .fringe_cts import cts_fringe
        _, inf = cts_fringe(X, fix="joint", alpha=alpha / 2, return_info=True)
        groups = inf["groups"]
        nxt = lab.max() + 1
        for g in groups:
            lab[g] = nxt; nxt += 1
    info = {"proposed": int(lab1.max()) + 1, "large": n_large, "tiny": len(groups), "tests": tests}
    return (lab, info) if return_info else lab


# ---------------------------------------------------------------- direction 1 (pruned): multi-scale Beta scan
def twonn(X):
    d = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)[0]
    mu = np.log(np.maximum(d[:, 2], 1e-12) / np.maximum(d[:, 1], 1e-12))
    mu = mu[mu > 0]
    return float((len(mu) - 1) / mu.sum())


def m_grid(n, small=15):
    g = list(range(2, small + 1))
    m = small
    while True:
        m = int(round(m * 1.5))
        if 2 * m - 1 > n - 1 or m > n // 2:
            break
        g.append(m)
    return g


def mscan(X, alpha=0.05, k_min=10, return_info=False):
    """Exact Beta scan over every point and m up to n / 2. Ring k = max(k_min, m) (ratio r_(m-1) / r_(m-1+k)),
    Bonferroni over all (point, m), greedy non-overlapping selection, the rest noise."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    grid = m_grid(n)
    kmax = min(n - 1, max(m - 1 + max(k_min, m) for m in grid))
    D, I = NearestNeighbors(n_neighbors=kmax + 1).fit(X).kneighbors(X)
    rank = int(np.linalg.matrix_rank(X - X.mean(0)))
    d = min(max(1.0, twonn(X)), rank)
    S, P, Ms = [], [], []
    n_tests = 0
    for m in grid:
        k = max(k_min, m)
        if m - 1 + k > kmax:
            continue
        r, R = D[:, m - 1], D[:, m - 1 + k]
        ok = (r > 0) & (R > r)
        u = np.where(ok, (r / np.where(ok, R, 1)) ** d, 1.0)
        s = -beta_dist.logcdf(u, m - 1, k)
        n_tests += n
        S.append(s); P.append(np.arange(n)); Ms.append(np.full(n, m))
    S, P, Ms = np.concatenate(S), np.concatenate(P), np.concatenate(Ms)
    thr = -np.log(alpha / n_tests)
    keep = np.flatnonzero(S >= thr)
    keep = keep[np.lexsort((Ms[keep], P[keep], -S[keep]))]
    used = np.zeros(n, bool)
    lab = -np.ones(n, np.int64)
    c = 0
    for o in keep:
        mem = I[P[o], :Ms[o]]
        if not used[mem].any():
            used[mem] = True; lab[mem] = c; c += 1
    info = {"d": d, "n_tests": n_tests, "groups": c, "sizes": np.bincount(lab[lab >= 0]).tolist() if c else []}
    return (lab, info) if return_info else lab


# ---------------------------------------------------------------- significance-tested baselines
def _ad_pvalue(x):
    """Anderson-Darling normality test with estimated mean and variance; Stephens' A* and the D'Agostino-Stephens
    p-value approximation."""
    x = np.sort((x - x.mean()) / (x.std(ddof=1) + 1e-300))
    n = len(x)
    if n < 8:
        return 1.0
    F = np.clip(norm.cdf(x), 1e-15, 1 - 1e-15)
    i = np.arange(1, n + 1)
    A2 = -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    A = A2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if A >= 0.6:
        return float(math.exp(1.2937 - 5.709 * A + 0.0186 * A ** 2))
    if A >= 0.34:
        return float(math.exp(0.9177 - 4.279 * A - 1.38 * A ** 2))
    if A >= 0.2:
        return float(1 - math.exp(-8.318 + 42.796 * A - 59.938 * A ** 2))
    return float(1 - math.exp(-13.436 + 101.14 * A - 223.73 * A ** 2))


def gmeans(X, alpha=1e-4, k_max=30, seed=0):
    """G-means (Hamerly & Elkan 2003): split a cluster in two with 2-means, project on the line between the two
    centres, keep the split if the projection fails an Anderson-Darling normality test at alpha (their 0.0001)."""
    from sklearn.cluster import KMeans
    X = np.asarray(X, dtype=np.float64)
    centers = X.mean(0, keepdims=True)
    while True:
        km = KMeans(len(centers), init=centers, n_init=1, random_state=seed).fit(X)
        lab, centers = km.labels_, km.cluster_centers_
        new, changed = [], False
        for c in range(len(centers)):
            Y = X[lab == c]
            if len(Y) < 16 or len(centers) + len(new) >= k_max:
                new.append(centers[c]); continue
            k2 = KMeans(2, n_init=2, random_state=seed).fit(Y)
            v = k2.cluster_centers_[0] - k2.cluster_centers_[1]
            if np.linalg.norm(v) == 0:
                new.append(centers[c]); continue
            if _ad_pvalue(Y @ v / (v @ v)) < alpha:
                new += list(k2.cluster_centers_); changed = True
            else:
                new.append(centers[c])
        centers = np.array(new)
        if not changed:
            return KMeans(len(centers), init=centers, n_init=1, random_state=seed).fit(X).labels_


def gmm_bic(X, k_max=10, seed=0):
    """Two-part-code (MDL / BIC) choice of the number of Gaussian components, k = 1..k_max."""
    from sklearn.mixture import GaussianMixture
    best, lab = np.inf, None
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type="full", random_state=seed).fit(X)
        b = g.bic(X)
        if b < best:
            best, lab = b, g.predict(X)
    return lab


# ---------------------------------------------------------------- scenes
def null_scene(kind, d, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "torus":           # flat torus, embedded isometrically up to scale: no boundary, homogeneous
        U = rng.random((n, d))
        return np.hstack([np.cos(2 * np.pi * U), np.sin(2 * np.pi * U)]) / (2 * np.pi)
    if kind == "cube":
        return rng.random((n, d))
    if kind == "gauss":
        return rng.normal(size=(n, d))
    raise ValueError(kind)


def rho_star(m, d, N, M=14, alpha=0.05):
    """Detection boundary radius (Result 4), in background-spacing units."""
    return math.exp((gammaln(m) + math.log(alpha / (N * M))) / (d * (m - 1)))


def plant_boundary(X, y, rng, n_groups=6, k_loc=20, factors=(0.5, 0.5, 0.5, 2.0, 2.0, 2.0)):
    """Plant tiny groups (3-8 points, uniform in a ball) centred on random data points, with radius set relative to
    the detection boundary at that spot: factor 0.5 x rho* (above the boundary: detectable) or 2 x rho* (below)."""
    n, d = X.shape
    vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    rk = NearestNeighbors(n_neighbors=k_loc + 1).fit(X)
    Xs, ys, gl, info = [X], [y], [], []
    nxt, start = y.max() + 1, n
    for g in range(n_groups):
        m = int(rng.integers(3, 9))
        c = X[rng.integers(n)]
        r = rk.kneighbors(c[None])[0][0, -1]
        spacing = r / k_loc ** (1 / d)          # lambda v_d spacing^d = 1 with lambda = k / (v_d r^d)
        rad = factors[g % len(factors)] * rho_star(m, d, n) * spacing
        u = rng.normal(size=(m, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        pts = c + u * rad * rng.random((m, 1)) ** (1 / d)
        Xs.append(pts); ys.append(np.full(m, nxt + g))
        gl.append(np.arange(start, start + m)); start += m
        info.append({"m": m, "factor": factors[g % len(factors)], "radius": float(rad)})
    _ = vd
    return np.vstack(Xs), np.concatenate(ys), gl, info


def n_clusters(lab):
    return int(len(set(np.asarray(lab).tolist()) - {-1}))
