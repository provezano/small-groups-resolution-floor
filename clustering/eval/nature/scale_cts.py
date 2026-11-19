"""
Scalable CTS v5: the same collapse-then-scan as `cts3(X, dim_est="local", scale_min=True, border=True)`, rebuilt so it
runs on 1e6 points and can be split across machines.

What changes against tiny_methods.cts3:
1. kNN rows are computed once (exact kd-tree, approximate NN-descent, or the partitioned exact form below), stored as
   float32/int32 where safe, and every later step reads only these rows.
2. The collapse reads the same rows. Its border join and attach step are exact from the rows (see `collapse_rows`);
   the family links are the eps-edges inside the rows instead of the full eps radius graph (the one approximation,
   measured in the stage-9 doc).
3. The scan is vectorised by size m over chunks of rows; only candidates with p <= alpha are kept (enough for both
   Bonferroni and BH, whose cut-offs never exceed alpha).
4. Partitioned / map-reduce form: space is cut into k-means Voronoi cells; each cell gets a halo of ghost points.
   Map 1 computes the kNN rows of its owned points and local summary statistics; reduce 1 forms the global quantities
   (TwoNN d, rank, per-size medians); map 2 scores the owned points' candidates; reduce 2 sums the test counts,
   applies Bonferroni or BH, and runs the greedy non-overlapping selection.

Exactness (proved in: if each cell's halo radius h is at least
max over owned i of max over j in N_kk(i) of [d(i, j) + r_2(j)] (which is <= r_kk(i) + max r_2), every owned row and
every ring statistic equals its single-machine value, so the partitioned output equals the single-machine output.
The code checks the condition a posteriori per cell (certificate) and doubles h where it fails.
"""
import math
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import betainc
from scipy.stats import beta as beta_dist, binom, chi2, ncx2
from sklearn.neighbors import KDTree

MAX_M, K_BG, N_LOC, ALPHA = 15, 10, 30, 0.05


def kk_for(n, max_m=MAX_M, k_bg=K_BG, n_loc=N_LOC):
    return min(n - 1, max_m + k_bg + n_loc)


# ---------------------------------------------------------------- kNN rows
def _sort_rows(dist, idx, self_ids):
    """Sort each row by (distance, not-self, global id): deterministic ties, the point itself first."""
    notself = (idx != self_ids[:, None]).astype(np.int64)
    order = np.lexsort((idx, notself, dist), axis=1)
    return np.take_along_axis(dist, order, 1), np.take_along_axis(idx, order, 1)


def knn_exact(Z, k, chunk=20000, tree=None, query=None, ids=None):
    """Exact (k+1)-NN rows (self in column 0) with a kd-tree; distances are computed pair by pair, so the same pair
    gives the same bits in any tree (needed for the partitioned == single check). k+3 neighbours are fetched and the
    rows re-sorted by (distance, id) so ties are broken the same way everywhere."""
    if tree is None:
        tree = KDTree(Z, leaf_size=40)
    Q = Z if query is None else query
    ids = np.arange(len(Q)) if ids is None else ids
    kq = min(k + 3, tree.data.shape[0])
    D = np.empty((len(Q), k + 1)); I = np.empty((len(Q), k + 1), np.int64)
    for s in range(0, len(Q), chunk):
        d, i = tree.query(Q[s:s + chunk], k=kq)
        d, i = _sort_rows(d, i, ids[s:s + chunk])
        D[s:s + chunk], I[s:s + chunk] = d[:, :k + 1], i[:, :k + 1]
    return D, I


def knn_ann(Z, k, seed=0, n_jobs=2):
    """Approximate rows by NN-descent (pynndescent). Distances are recomputed exactly for the returned ids."""
    from pynndescent import NNDescent
    nd = NNDescent(Z, n_neighbors=k + 1, random_state=seed, n_jobs=n_jobs, low_memory=True, verbose=False)
    I, _ = nd.neighbor_graph
    I = I.astype(np.int64)
    ids = np.arange(len(Z))
    # make sure the point itself is in the row (NN-descent nearly always returns it first)
    miss = ~(I == ids[:, None]).any(1)
    I[miss, -1] = ids[miss]
    D = np.empty(I.shape)
    for s in range(0, len(Z), 50000):
        D[s:s + 50000] = np.sqrt(((Z[I[s:s + 50000]] - Z[s:s + 50000, None]) ** 2).sum(-1))
    return _sort_rows(D, I, ids)


def _local_rows(Z, k, own, loc):
    """Rows of the owned points `own` computed only from the points `loc` (own first), and every local point's
    distances to its first two neighbours (for the ring statistic)."""
    tree = KDTree(Z[loc], leaf_size=40)
    if len(loc) < k + 1:                    # too few local points: an infinite row fails the certificate, h doubles
        d = np.full((len(own), k + 1), np.inf); i = np.zeros((len(own), k + 1), np.int64)
        d[:, 0] = 0; i[:, 0] = np.arange(len(own))
        dg = tree.query(Z[loc], k=min(3, len(loc)))[0]
        return d, own[i], np.pad(dg, ((0, 0), (0, 3 - dg.shape[1])), constant_values=np.inf)
    d, i = knn_exact(None, k, tree=tree, query=Z[own], ids=np.arange(len(own)))
    gi = loc[i]
    d, gi = _sort_rows(d, gi, own)
    dg = tree.query(Z[loc], k=min(3, len(loc)))[0]
    return d, gi, dg


def knn_partitioned(Z, k, n_cells=16, seed=0, h0=None, max_rounds=12, return_cells=False, q_h0=0.9):
    """Exact rows by the partitioned form. Cells: k-means Voronoi cells (fit on a sample). Stage 1 of cell c: ghosts are
    the points whose distance to c's Voronoi region can be <= h, via the bisector bound
    (|x - C_c|^2 - |x - C_a|^2) / (2 |C_c - C_a|) for x in cell a (a lower bound on the distance from x to region c).
    Certificate per owned point i: r_k(i) <= h and d(i, j) + r_2(j) <= h for every j in its row (r_2 measured inside
    the cell: an over-estimate for a ghost, so the check is conservative). Owned points that fail (isolated points
    whose neighbours are far) are redone with h doubled and a ball halo around them (centre c_F, radius
    max |f - c_F| + h), until certified. h0 defaults to the q_h0 quantile of r_k + r_2 on a 2,000-point sample."""
    from sklearn.cluster import MiniBatchKMeans
    n = len(Z)
    rng = np.random.default_rng(seed)
    samp = Z[rng.choice(n, min(n, 20000), replace=False)]
    C = MiniBatchKMeans(n_cells, random_state=seed, n_init=3, batch_size=4096).fit(samp).cluster_centers_
    sq = (Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None]
    cell = np.argmin(sq, 1)
    own_sq = sq[np.arange(n), cell]
    if h0 is None:
        pick = rng.choice(n, min(n, 2000), replace=False)
        d, _ = knn_exact(Z, k, query=Z[pick], ids=pick)
        h0 = float(np.quantile(d[:, -1], q_h0) + np.quantile(d[:, 2], q_h0))
    D = np.empty((n, k + 1)); I = np.empty((n, k + 1), np.int64)
    info = {"n_cells": n_cells, "h0": h0, "ghosts": [], "rounds": [], "redone": 0}
    cells = []
    for c in range(n_cells):
        own = np.flatnonzero(cell == c)
        if not len(own):
            continue
        h = h0
        den = 2 * np.linalg.norm(C[cell] - C[c], axis=1)
        lb = np.where(cell == c, 0.0, (sq[:, c] - own_sq) / np.maximum(den, 1e-300))
        ghost = np.flatnonzero((cell != c) & (lb <= h))
        todo, n_ghost, rnd = own, len(ghost), 1
        loc = np.r_[own, ghost]
        ref_ids, ref_mu = [], []
        while True:
            d, gi, dg = _local_rows(Z, k, todo, loc)
            r2loc = np.full(n, np.inf); r2loc[loc] = dg[:, -1]
            ok = ((d[:, -1] <= h) & ((d + r2loc[gi]).max(1) <= h)) | (len(loc) == n)
            D[todo[ok]], I[todo[ok]] = d[ok], gi[ok]
            used = np.unique(gi[ok])
            mu_loc = np.full(n, np.nan)
            mu_loc[loc] = np.log(np.maximum(dg[:, 2], 1e-12) / np.maximum(dg[:, 1], 1e-12))
            ref_ids.append(used); ref_mu.append(mu_loc[used])
            if ok.all() or rnd >= max_rounds:
                if not ok.all():
                    raise RuntimeError("certificate not reached")
                break
            todo = todo[~ok]
            info["redone"] += len(todo)
            h *= 2
            cF = Z[todo].mean(0)
            rad = np.linalg.norm(Z[todo] - cF, axis=1).max() + h
            near = np.flatnonzero(np.linalg.norm(Z - cF, axis=1) <= rad)
            loc = np.r_[todo, np.setdiff1d(near, todo)]
            n_ghost += len(loc) - len(todo)
            rnd += 1
        info["ghosts"].append(int(n_ghost)); info["rounds"].append(rnd)
        if return_cells:
            ids = np.concatenate(ref_ids); mus = np.concatenate(ref_mu)
            cells.append({"own": own, "D": D[own], "I": I[own], "ref_ids": ids, "ref_mu": mus})
    info["ghost_ratio"] = float(np.sum(info["ghosts"]) / n)
    return (D, I, info, cells) if return_cells else (D, I, info)


# ---------------------------------------------------------------- collapse from rows
def collapse_rows(D, I, Z, big=20, min_core=10, tight=0.25, attach=0.5, return_members=False):
    """cts3 v5 collapse (border=True) from kNN rows. Exact parts: core flags, eps, ref, border join (a non-core point has
    r_10 > eps, so a row with no core neighbour has its nearest core beyond eps; a row with no non-core neighbour has
    its nearest non-core beyond the nearest core), attach (rows without a collapsed neighbour whose last distance is
    below attach x eps are re-queried exactly). Approximate part: family links are the eps-edges inside the rows.
    return_members=True adds a fourth output: the member indices of each collapsed family, in the order of the
    representatives in R (needed by the fringe fix)."""
    n = len(Z)
    r = D[:, min_core]
    eps = float(np.median(r))
    core = r <= eps
    nb, nd = I[:, 1:], D[:, 1:]
    m = core[:, None] & core[nb] & (nd <= eps)
    rows = np.repeat(np.arange(n), nb.shape[1])[m.ravel()]
    cols = nb.ravel()[m.ravel()]
    comp = -np.ones(n, np.int64)
    ci = np.flatnonzero(core)
    if len(ci):
        pos = -np.ones(n, np.int64); pos[ci] = np.arange(len(ci))
        g = coo_matrix((np.ones(len(rows)), (pos[rows], pos[cols])), shape=(len(ci),) * 2)
        comp[ci] = connected_components(g, directed=False)[1]
    ref = float(np.median(r[~core])) if (~core).any() else np.inf
    nc = np.flatnonzero(~core)
    if len(nc):
        cm = core[nb[nc]]
        has_c = cm.any(1)
        fc = np.argmax(cm, 1)
        dd = np.where(has_c, nd[nc, fc], np.inf)
        jc = nb[nc, fc]
        ncm = ~cm
        has_n = ncm.any(1)
        dn = np.where(has_n, nd[nc, np.argmax(ncm, 1)], np.inf)
        join = (dd <= eps) & (dd <= dn)
        comp[nc[join]] = comp[jc[join]]
    collapsed = np.zeros(n, bool)
    if (comp >= 0).any():
        labs = np.unique(comp[core])
        cc = comp[comp >= 0]
        sizes = np.bincount(cc)
        order = np.argsort(cc, kind="stable")
        bounds = np.r_[0, np.cumsum(sizes)]
        pts = np.flatnonzero(comp >= 0)[order]
        for c in labs:
            if sizes[c] > big:
                mem = pts[bounds[c]:bounds[c + 1]]
                if np.median(r[mem]) <= tight * ref:
                    collapsed[mem] = True
    if collapsed.any():
        free = np.flatnonzero(~collapsed)
        cmk = collapsed[nb[free]]
        has = cmk.any(1)
        f = np.argmax(cmk, 1)
        dd = np.where(has, nd[free, f], np.inf)
        jj = nb[free, f]
        # exact re-query where the row is too short to rule out a collapsed point within attach x eps
        amb = np.flatnonzero(~has & (nd[free, -1] < attach * eps))
        if len(amb):
            cidx = np.flatnonzero(collapsed)
            d2, j2 = KDTree(Z[cidx]).query(Z[free[amb]], k=1)
            dd[amb], jj[amb] = d2[:, 0], cidx[j2[:, 0]]
        near = dd <= attach * eps
        comp[free[near]] = comp[jj[near]]
        collapsed[free[near]] = True
    keep = np.flatnonzero(~collapsed)
    reps, members = [], []
    cp = np.flatnonzero(collapsed)
    if len(cp):
        cp = cp[np.argsort(comp[cp], kind="stable")]
        cuts = np.flatnonzero(np.diff(comp[cp])) + 1
        for mem in np.split(cp, cuts):        # same family order and members as cts3 (ascending ids per family)
            reps.append(mem[np.argmin(np.linalg.norm(Z[mem] - Z[mem].mean(0), axis=1))])
            members.append(mem)
    R = np.r_[keep, np.array(reps, np.int64)]
    is_rep = np.r_[np.zeros(len(keep), bool), np.ones(len(reps), bool)]
    info = {"eps": eps, "families": len(reps), "collapsed": int(collapsed.sum())}
    return (R, is_rep, info, members) if return_members else (R, is_rep, info)


# ---------------------------------------------------------------- global statistics (reduce 1)
def local_stats(D, I, is_rep_rows, max_m=MAX_M, k_bg=K_BG):
    """Per-point quantities a cell sends to the reducer: TwoNN log-ratios and per-size log(r / R_k) of non-rep points."""
    mu = np.log(np.maximum(D[:, 2], 1e-12) / np.maximum(D[:, 1], 1e-12))
    kk = D.shape[1] - 1
    lr = {}
    free = ~is_rep_rows
    for m in range(2, max_m + 1):
        if m - 1 + k_bg > kk:
            break
        r, Rk = D[free, m - 1], D[free, m - 1 + k_bg]
        ok = (r > 0) & (Rk > r)
        lr[m] = np.log(r[ok] / Rk[ok])
    return mu, lr


def reduce_stats(mus, lrs, scatter, max_m=MAX_M, k_bg=K_BG, q=0.5):
    """Global d (TwoNN, exact sum with fsum so the order of cells does not matter), rank, per-size d_m (medians)."""
    mu = np.concatenate(mus)
    mu = mu[mu > 0]
    dim = (len(mu) - 1) / math.fsum(mu.tolist())
    S, s, nn, amax = scatter
    cov = S - np.outer(s, s) / nn
    sv = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0, None))
    rank = int((sv > 1e-8 * amax).sum())
    dims = {}
    for m in lrs[0]:
        v = np.concatenate([l[m] for l in lrs])
        dims[m] = float(np.log(beta_dist.ppf(q, m - 1, k_bg)) / np.quantile(v, q))
    dim = min(max(1.0, dim), rank)
    return dim, rank, dims


def scatter_of(Zr):
    return (Zr.T @ Zr, Zr.sum(0), len(Zr), float(np.abs(Zr).max()))


def add_scatter(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], max(a[3], b[3]))


# ---------------------------------------------------------------- scan (map 2)
def scan_rows(D, I, row_ids, is_rep, mu_of, dim, rank, dims, max_m=MAX_M, k_bg=K_BG, n_loc=N_LOC, alpha=ALPHA,
              chunk=100000, local=True):
    """Vectorised v5 scan over the given rows. D, I: rows (global ids in I, positions in the reduced set);
    row_ids: the reduced-set position of each row; mu_of: TwoNN log-ratio of any reduced-set position (array).
    The d plugged in at size m is min(dim, dims[m]) (pass dims[m] = inf for a global d only); local=True (v5) raises
    it to the local one-sided lower bound from the ring's TwoNN log-ratios, local=False skips that step (cts3's
    dim_est="lb" / "twonn"). Both are capped at rank.
    Returns (n_tests, candidates with p <= alpha as arrays score, row position, m)."""
    kk = D.shape[1] - 1
    lo_q = chi2.ppf(0.001, 2 * n_loc) / 2
    n_tests = 0
    S, P, Ms = [], [], []
    keep_thr = -np.log(alpha)
    for s0 in range(0, len(D), chunk):
        d, ii, rid = D[s0:s0 + chunk], I[s0:s0 + chunk], row_ids[s0:s0 + chunk]
        live = ~is_rep[rid]
        d, ii, rid = d[live], ii[live], rid[live]
        rep_cols = is_rep[ii[:, :max_m]]
        first = np.where(rep_cols.any(1), np.argmax(rep_cols, 1), max_m)   # m valid iff m <= first
        csum = np.concatenate([np.zeros((len(ii), 1)), np.cumsum(mu_of[ii], 1)], 1)
        for m in range(2, max_m + 1):
            if m - 1 + k_bg > kk:
                break
            valid = first >= m
            n_tests += int(valid.sum())
            r, Rk = d[:, m - 1], d[:, m - 1 + k_bg]
            ok = valid & (Rk > r) & (r > 0)
            if not ok.any():
                continue
            dd = min(dim, dims[m])
            if local:
                ring = csum[ok, min(m + n_loc, kk + 1)] - csum[ok, m]
                dd = np.minimum(np.maximum(dd, lo_q / np.maximum(ring, 1e-12)), rank)
            else:
                dd = min(dd, rank)
            u = (r[ok] / Rk[ok]) ** dd
            with np.errstate(divide="ignore"):
                sc = -np.log(betainc(m - 1, k_bg, u))
            sel = sc >= keep_thr
            S.append(sc[sel]); P.append(rid[ok][sel]); Ms.append(np.full(int(sel.sum()), m, np.int16))
    cat = lambda L, t: np.concatenate(L) if L else np.zeros(0, t)
    return n_tests, cat(S, float), cat(P, np.int64), cat(Ms, np.int16)


# ---------------------------------------------------------------- fringe fix (stage 11): joint family test
Q_UCB, K_FAM, FAR = 0.999, 3, 12.0


def family_models(Z, R, is_rep, comp_of_rep_members, q_ucb=Q_UCB, dim=None):
    """Fit one isotropic model per collapsed family: centre c = member mean, spread s2 = mean squared distance to c
    per intrinsic dimension, taken at its q_ucb upper confidence bound (sum of squares over M members has about
    dim (M - 1) degrees of freedom: s2_U = SS / chi2.ppf(1 - q_ucb, dim (M - 1))).
    comp_of_rep_members: list of member index arrays, one per family (same order as the reps in R)."""
    cs, s2, Ms = [], [], []
    for mem in comp_of_rep_members:
        X = Z[mem]
        c = X.mean(0)
        ss = float(((X - c) ** 2).sum())
        M = len(mem)
        df = max(1.0, dim * (M - 1))
        cs.append(c); s2.append(ss / chi2.ppf(1 - q_ucb, df)); Ms.append(M)
    return np.array(cs), np.array(s2), np.array(Ms)


def joint_pvalues(Z, R, cand_pos, cand_m, cand_r, fam_c, fam_s2, fam_M, dim, k_fam=K_FAM, m_extra=True, far=FAR):
    """Joint (tail x ball) family test of scan candidates (stage-11 fringe doc, Proposition 11.2).
    Null for family F: the candidate's centre x and the other points near it are members of F, whose points are iid
    N(c_F, s2_F I_dim) (n_F = M_F - 1 + m others when m_extra). Two exact uniforms under this null (Palm: given that x
    is a member, the others are iid):
      U = tau(rho) = chi2.sf(rho^2 / s2_F, dim), rho = |x - c_F|   (probability integral transform of the radius);
      V = P(Bin(n_F, pi) >= m - 1), pi = ncx2.cdf(r^2 / s2_F, dim, rho^2 / s2_F)   (the conditional CDF of the
          (m - 1)-th neighbour distance given rho, at the observed r).
    U ~ Uniform(0, 1), and given rho V ~ Uniform(0, 1), so U and V are independent uniforms and b = U V has
    P(b <= t) = t (1 - ln t). p_F = b (1 - ln b). The null is the union over the k_fam nearest families: p = max_F p_F.
    Families farther than `far` spreads beyond r give p_F = 0. The scan's r is measured in the reduced set (collapsed
    members removed), so it is >= the full-data distance and V only grows (conservative)."""
    p = np.zeros(len(cand_pos))
    if len(fam_c) == 0 or len(cand_pos) == 0:
        return p
    x = Z[R[cand_pos]]
    kq = min(k_fam, len(fam_c))
    dist, jj = KDTree(fam_c).query(x, k=kq)
    for t in range(kq):
        f = jj[:, t]
        s = np.sqrt(fam_s2[f])
        near = dist[:, t] - cand_r <= far * s
        if not near.any():
            continue
        i = np.flatnonzero(near)
        lam = (dist[i, t] / s[i]) ** 2
        U = chi2.sf(lam, dim)
        pi = ncx2.cdf((cand_r[i] / s[i]) ** 2, dim, lam)
        nF = fam_M[f[i]] - 1 + (cand_m[i] if m_extra else 0)
        V = binom.sf(cand_m[i] - 2, nF, pi)
        b = np.clip(U * V, 1e-300, 1.0)
        p[i] = np.maximum(p[i], np.minimum(1.0, b * (1 - np.log(b))))
    return p


def fringe_scores(Z, R, is_rep, members, dim, score, cand_pos, cand_m, cand_r, q_ucb=Q_UCB, k_fam=K_FAM):
    """The fringe fix (fix="joint"): each candidate's score becomes min(score_Beta, -log p_family), with p_family the
    joint family test against its k_fam nearest collapsed families (intersection-union: a candidate is reported only
    if it is unusual both against the background ring and as the edge of a nearby family). A candidate with no family
    within reach keeps its Beta score (p_family = 0). With no collapsed family the scores are returned unchanged.
    The test count and the selection rule are not touched. Returns (new scores, number of candidates whose score
    went down)."""
    score = np.asarray(score, float)
    if not len(members) or not len(score):
        return score, 0
    fam_c, fam_s2, fam_M = family_models(Z, R, is_rep, members, q_ucb=q_ucb, dim=dim)
    pj = joint_pvalues(Z, R, np.asarray(cand_pos, np.int64), np.asarray(cand_m, np.int64), np.asarray(cand_r, float),
                       fam_c, fam_s2, fam_M, dim, k_fam=k_fam)
    with np.errstate(divide="ignore"):
        sf = np.where(pj > 0, -np.log(np.maximum(pj, 1e-300)), np.inf)
    new = np.minimum(score, sf)
    return new, int((new < score).sum())


# ---------------------------------------------------------------- selection (reduce 2)
def select(n_tests, S, P, Ms, Irows_of, alpha=ALPHA, rule="bonferroni"):
    """Bonferroni, BH or BY over all n_tests (p > alpha candidates were dropped: no cut-off can reach them), then the
    greedy non-overlapping pass in the order (score desc, row position, m): cts3's stable sort of (i, m)-ordered
    candidates by score gives exactly this order. Irows_of(p) returns the reduced-set row of position p."""
    thr = -np.log(alpha / max(1, n_tests))
    if rule not in ("bonferroni", "bh", "by"):
        raise ValueError(rule)
    if rule in ("bh", "by") and len(S):
        # BY divides the BH line by sum_{i<=n} 1/i (same approximation as tiny_methods._scan_select)
        c = 1.0 if rule == "bh" else float(np.log(max(1, n_tests)) + np.euler_gamma + 1 / (2 * max(1, n_tests)))
        sc = np.sort(S)[::-1]
        ok = np.flatnonzero(sc >= -np.log(alpha * np.arange(1, len(sc) + 1) / (max(1, n_tests) * c)))
        thr = sc[ok[-1]] if len(ok) else np.inf
    sig = S >= thr
    S, P, Ms = S[sig], P[sig], Ms[sig]
    order = np.lexsort((Ms, P, -S))
    used = set()
    groups = []
    for o in order:
        mem = Irows_of(P[o])[:Ms[o]]
        if used.isdisjoint(mem.tolist()):
            groups.append(np.sort(mem)); used.update(mem.tolist())
    return groups, float(thr), int(sig.sum())


# ---------------------------------------------------------------- full pipelines
def _reduced_rows(Z, R, kk, knn, seed, n_cells):
    Zr = Z[R]
    if callable(knn):
        return knn(Zr, kk) + ({},)
    if knn == "exact":
        return knn_exact(Zr, kk) + ({},)
    if knn == "ann":
        return knn_ann(Zr, kk, seed=seed) + ({},)
    raise ValueError(knn)


def cts_scale(Z, rule="bonferroni", knn="exact", alpha=ALPHA, max_m=MAX_M, k_bg=K_BG, seed=0, rows=None,
              base=None, collapse=True, return_info=False, force_rep=None, fix="joint", count_family_points=False):
    """Single-machine scalable CTS. Returns labels (base, or -1, with groups written over) and info.
    fix="joint" (default): the stage-11 fringe fix; candidates also have to pass the joint family test against the
    nearby collapsed families (`fringe_scores`). Scenes with no collapsed family give exactly the v5 output.
    fix=None: CTS v5 exactly, as frozen before stage 11.
    count_family_points (stage 13 conf, Proposition 13.3; default False = unchanged): with fix="joint", the selection
    rule counts T + (max_m - 1) sum_F M_F tests (every family point is a potential centre at max_m - 1 sizes). The
    candidates near a family are selected by the collapse, so the family-wise bound needs the union over all family
    points; without it the level is at most alpha (1 + (max_m - 1) sum M_F / T). Only the threshold changes; scenes
    with no collapsed family are unchanged. Same option as tiny_methods.cts3(count_family_points=)."""
    if fix not in (None, "joint"):
        raise ValueError("fix must be 'joint' or None")
    t0 = time.time()
    n = len(Z)
    kk = kk_for(n, max_m, k_bg)
    if rows is None:
        rows = knn(Z, kk) if callable(knn) else knn_exact(Z, kk) if knn == "exact" else knn_ann(Z, kk, seed=seed)
    D, I = rows
    t_knn = time.time() - t0
    members = []
    if collapse:
        R, is_rep, cinfo, members = collapse_rows(D, I, Z, return_members=True)
    else:
        R, is_rep, cinfo = np.arange(n), np.zeros(n, bool), {}
    if force_rep is not None:
        is_rep = is_rep | force_rep[R]               # exact-duplicate families removed upstream (cts_scale_dedup)
    if len(R) == n and not is_rep.any():
        Dr, Ir = D, I                               # nothing collapsed: reduced set = full set, reuse the rows
    else:
        Dr, Ir, _ = _reduced_rows(Z, R, kk_for(len(R), max_m, k_bg), knn, seed, None)
    t_c = time.time() - t0 - t_knn
    mu, lr = local_stats(Dr, Ir, is_rep, max_m, k_bg)
    dim, rank, dims = reduce_stats([mu], [lr], scatter_of(Z[R]), max_m, k_bg)
    nt, S, P, Ms = scan_rows(Dr, Ir, np.arange(len(R)), is_rep, mu, dim, rank, dims, max_m, k_bg, alpha=alpha)
    n_gated = 0
    nt_sel = nt
    if count_family_points and fix == "joint" and len(members):
        nt_sel = nt + (max_m - 1) * int(sum(len(m) for m in members))
    if fix == "joint" and len(members) and len(S):
        thr0 = -np.log(alpha / max(1, nt_sel))
        S0 = S
        S, _ = fringe_scores(Z, R, is_rep, members, dim, S, P, Ms, Dr[P, Ms.astype(np.int64) - 1])
        n_gated = int(((S0 >= thr0) & (S < thr0)).sum())
    groups, thr, nsig = select(nt_sel, S, P, Ms, lambda p: Ir[p], alpha, rule)
    lab = (-np.ones(n, np.int64) if base is None else np.asarray(base).copy())
    nxt = lab.max() + 1
    out_groups = []
    for g in groups:
        pts = R[g]
        lab[pts] = nxt; nxt += 1
        out_groups.append(pts)
    info = {"n": n, "reduced": len(R), "n_tests": nt, "thr": thr, "n_sig": nsig, "n_kept_p_le_alpha": int(len(S)),
            "dim": dim, "rank": rank, "dims": dims, "groups": out_groups, "fix": fix, "n_gated": n_gated, "n_tests_sel": nt_sel,
            "t_knn": t_knn, "t_collapse": t_c,
            "t_total": time.time() - t0, **cinfo}
    return (lab, info) if return_info else lab


def cts_partitioned(Z, rule="bonferroni", n_cells=16, alpha=ALPHA, max_m=MAX_M, k_bg=K_BG, seed=0, collapse=True):
    """Map-reduce CTS v5 (no fringe fix yet: it equals cts_scale(..., fix=None)). Map 1 (per cell, with halo): exact kNN rows of owned points. Reduce 1: collapse (global
    eps/ref medians, union of edge lists = connected components, family statistics). Map 1': kNN rows on the reduced
    set, again per cell with halo, plus per-cell TwoNN log-ratios, size-wise log ratios, scatter. Reduce 1': d, rank,
    d_m. Map 2 (per cell): scan of owned rows, using ghost ring statistics computed inside the cell. Reduce 2: sum of
    test counts, Bonferroni/BH, merge, greedy selection. Returns (groups as arrays of indices of Z, info)."""
    t0 = time.time()
    n = len(Z)
    kk = kk_for(n, max_m, k_bg)
    D, I, i1 = knn_partitioned(Z, kk, n_cells, seed)
    if collapse:
        R, is_rep, cinfo = collapse_rows(D, I, Z)
    else:
        R, is_rep, cinfo = np.arange(n), np.zeros(n, bool), {}
    Zr = Z[R]
    kr = kk_for(len(R), max_m, k_bg)
    Dr, Ir, i2, cells = knn_partitioned(Zr, kr, n_cells, seed, return_cells=True)
    # map 1': per-cell statistics from owned rows only
    mus, lrs, sc = [], [], None
    for c in cells:
        mu, lr = local_stats(c["D"], c["I"], is_rep[c["own"]], max_m, k_bg)
        mus.append(mu); lrs.append(lr)
        s = scatter_of(Zr[c["own"]])
        sc = s if sc is None else add_scatter(sc, s)
    dim, rank, dims = reduce_stats(mus, lrs, sc, max_m, k_bg)
    # map 2: each cell scores its owned rows; ring statistics of ghost points come from the cell's own data
    nt, Ss, Ps, Mss = 0, [], [], []
    for c in cells:
        mu_of = np.full(len(R), np.nan)
        mu_of[c["ref_ids"]] = c["ref_mu"]          # ring statistics measured inside the cell (certified exact)
        t, S, P, Ms = scan_rows(c["D"], c["I"], c["own"], is_rep, mu_of, dim, rank, dims, max_m, k_bg, alpha=alpha)
        nt += t; Ss.append(S); Ps.append(P); Mss.append(Ms)
    S, P, Ms = np.concatenate(Ss), np.concatenate(Ps), np.concatenate(Mss)
    groups, thr, nsig = select(nt, S, P, Ms, lambda p: Ir[p], alpha, rule)
    return [R[g] for g in groups], {"n_tests": nt, "thr": thr, "n_sig": nsig, "dim": dim, "rank": rank, "dims": dims,
                                     "knn1": i1, "knn2": i2, "t_total": time.time() - t0, **cinfo}


def dedup_dither(Z_raw, key=None, big=20, dither=1e-3, seed=0):
    """Exact-duplicate pre-collapse for data with huge identical masses (KDD: ~293k records in 434 identical sets).
    Dithering them (run_kdd's recipe) turns each mass into a full-dimensional blob, the worst case for any kNN index
    (50k KDD records took 11 s, the full file did not finish in 15 min). A set of > big identical records is a
    near-duplicate family by definition (r_10 = 0, so v5 would collapse it), so it is replaced by one representative
    before dithering; sets of <= big identical records stay and are dithered (a tiny group can be identical records,
    e.g. KDD 'land'). key: identity of each record (default: identical coordinates).
    Returns (Zk dithered, keep indices into Z_raw, rep mask over keep)."""
    n = len(Z_raw)
    _, first, inv, cnt = np.unique(Z_raw if key is None else np.asarray(key).reshape(-1, 1), axis=0,
                                   return_index=True, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    fam = cnt[inv] > big
    keep = np.flatnonzero(~fam | (np.arange(n) == first[inv]))
    rep = fam[keep]
    rng = np.random.default_rng(seed)
    Zk = Z_raw[keep]
    pick = rng.choice(len(Zk), min(len(Zk), 20000), replace=False)
    d1 = knn_exact(Zk, 1, query=Zk[pick], ids=pick)[0][:, 1]
    Zk = Zk + rng.normal(0, dither * np.median(d1[d1 > 0]), Zk.shape)
    return Zk, keep, rep


def cts_scale_dedup(Z_raw, key=None, big=20, dither=1e-3, seed=0, **kw):
    """cts_scale after dedup_dither; the representatives are never part of a group. Labels on Z_raw."""
    Zk, keep, rep = dedup_dither(Z_raw, key, big, dither, seed)
    _, info = cts_scale(Zk, force_rep=rep, return_info=True, seed=seed, **kw)
    groups = [keep[g] for g in info["groups"]]
    info["groups"] = groups
    info["dedup_removed"] = int(len(Z_raw) - len(keep))
    return groups_to_labels(groups, len(Z_raw)), info


def groups_to_labels(groups, n, base=None):
    lab = -np.ones(n, np.int64) if base is None else np.asarray(base).copy()
    nxt = lab.max() + 1
    for g in groups:
        lab[g] = nxt; nxt += 1
    return lab
