"""
Methods for tiny groups. Baselines: HDBSCAN (default and min_cluster_size 3), our micro step, and three outlier scorers
(isolation forest, LOF, HDBSCAN's GLOSH) whose top 5% are grouped by mutual 3-nearest neighbours, the way McCatch
groups its outliers (McCatch itself is not on PyPI; its code would have to come from the authors' repository).

New: collapse-then-scan (CTS).
1. Collapse: core points (>= 10 others within eps = 1.5 x the median nearest-neighbour distance, as in DBSCAN) linked
   within eps form pieces; pieces of more than 20 points (near-duplicate families) are replaced by one representative
   each, counted once. A group of 2-9 points has no core points, so it is never collapsed or chained into a family
   (revision 2: v1 linked all points and chained 1/3 of tight inside groups into the mass); the near-duplicate mass
   stops masking what sits inside it.
2. Scan: on the reduced set, for each point and each size m = 2..15, the candidate group is the point and its m - 1
   nearest neighbours, radius r. The local background density comes from the ring between r and the distance to the
   (m - 1 + 10)th neighbour, in the intrinsic dimension (Levina-Bickel estimate). Score = -log P(Poisson(expected) >=
   m - 1). Candidates containing a representative are skipped (that is the mass itself).
3. Select: greedy, highest score first, no overlaps, above a Bonferroni threshold (alpha 0.05 over all candidates).
Groups are written over HDBSCAN's default labels, so big structure is unchanged.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta as beta_dist, chi2, poisson
from sklearn.neighbors import NearestNeighbors

from . import methods as M
from .ideas_select import hdbscan_micro


def _mcs(n):
    return max(8, n // 250)


def hdb(Z):
    return M.hdbscan_labels(Z, _mcs(len(Z)), max(3, _mcs(len(Z)) // 2))


def hdb3(Z):
    return M.hdbscan_labels(Z, 3, 3)


def _group_outliers(Z, out_idx, base, k=3, max_size=15):
    lab = base.copy()
    if len(out_idx) < 2:
        return lab
    k = min(k, len(out_idx) - 1)
    dn, nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[out_idx]).kneighbors(Z[out_idx])
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


def scored(kind, q=0.05):
    def f(Z):
        base = hdb(Z)
        if kind == "iforest":
            from sklearn.ensemble import IsolationForest
            s = -IsolationForest(random_state=0).fit(Z).score_samples(Z)
        elif kind == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            l = LocalOutlierFactor(n_neighbors=20).fit(Z); s = -l.negative_outlier_factor_
        else:
            s = np.nan_to_num(M.hdbscan_fit(Z, _mcs(len(Z)), max(3, _mcs(len(Z)) // 2)).outlier_scores_)
        return _group_outliers(Z, np.flatnonzero(s >= np.quantile(s, 1 - q)), base)
    return f


def intrinsic_dim(Z, k=10):
    d = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[0][:, 1:]
    d = np.maximum(d, 1e-12)
    m = (k - 1) / np.log(d[:, -1:] / d[:, :-1]).sum(1)
    return float(np.median(m))


def twonn_dim(Z):
    """TwoNN (Facco et al. 2017) pooled MLE: d = (n - 1) / sum log(r_2 / r_1). It reads the smallest scale, where the
    tail of the scan statistic lives; the 10-NN estimate reads a larger scale and overstated d on curved nulls."""
    d = NearestNeighbors(n_neighbors=3).fit(Z).kneighbors(Z)[0]
    mu = np.log(np.maximum(d[:, 2], 1e-12) / np.maximum(d[:, 1], 1e-12))
    mu = mu[mu > 0]
    return float((len(mu) - 1) / mu.sum())


def cts(Z, eps_mult=1.5, big=20, max_m=15, k_bg=10, alpha=0.05, min_core=10, return_scores=False):
    n = len(Z)
    d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    eps = eps_mult * np.median(d1)
    pairs = NearestNeighbors(radius=eps).fit(Z).radius_neighbors_graph(Z, mode="connectivity")
    core = np.asarray(pairs.sum(1)).ravel() - 1 >= min_core
    pc = pairs[core][:, core]
    comp = -np.ones(n, int)
    comp[core] = connected_components(pc, directed=False)[1]
    size = np.bincount(comp[core]) if core.any() else np.array([0])
    collapsed = core & (size[np.maximum(comp, 0)] > big)
    keep = [i for i in range(n) if not collapsed[i]]
    reps = []
    for c in np.flatnonzero(size > big):
        mem = np.flatnonzero(collapsed & (comp == c))
        reps.append(mem[np.argmin(np.linalg.norm(Z[mem] - Z[mem].mean(0), axis=1))])
    R = np.array(keep + reps)
    is_rep = np.r_[np.zeros(len(keep), bool), np.ones(len(reps), bool)]
    Zr = Z[R]
    dim = max(1.0, intrinsic_dim(Zr))
    kk = min(len(R) - 1, max_m + k_bg)
    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Zr).kneighbors(Zr)
    cands = []
    for i in range(len(R)):
        if is_rep[i]:
            continue
        for m in range(2, max_m + 1):
            members = idx[i, :m]
            if is_rep[members].any():
                break
            r, Rk = dist[i, m - 1], dist[i, m - 1 + k_bg]
            if Rk <= r or r <= 0:
                continue
            dens = k_bg / (Rk ** dim - r ** dim)
            mu = dens * r ** dim
            s = -poisson.logsf(m - 2, mu)           # P(X >= m - 1)
            cands.append((s, tuple(sorted(members))))
    thr = -np.log(alpha / max(1, len(cands)))
    cands.sort(key=lambda t: -t[0])
    used, groups = set(), []
    for s, mem in cands:
        if s < thr:
            break
        if used.isdisjoint(mem):
            groups.append(mem); used.update(mem)
    lab = hdb(Z).copy()
    nxt = lab.max() + 1
    for mem in groups:
        lab[R[list(mem)]] = nxt; nxt += 1
    return (lab, {"dim": dim, "eps": eps, "reps": len(reps), "groups": len(groups)}) if return_scores else lab


METHODS = {"hdbscan": hdb, "hdbscan_mcs3": hdb3, "hdbscan_micro": lambda Z: hdbscan_micro(Z, ratio=0.7),
           "iforest+group": scored("iforest"), "lof+group": scored("lof"), "glosh+group": scored("glosh"),
           "cts": cts}


def cts3(Z, big=20, max_m=15, k_bg=10, alpha=0.05, min_core=10, tight=0.25, attach=0.5, null="beta", dim_est="lb", closure=False, count="all", respect_base=False, scale_min=False, select="bonferroni", border=False, order="score", fix="joint", base=None, weights=None, flt=False, return_cands=False, vectorised=True,
         count_family_points=False, knn="auto", group_shape=("trunc", 16), d_rule="v5", alpha_base=0.05, base_seed=0,
         reference=None, union_rule=None, _thr_of_n=None, tmax_mode="max"):
    """CTS v3. Near-duplicate families are found on the 10-NN scale, so the rule does not depend on dimension:
    core = r_10(x) <= eps, eps = median r_10 over all points; core points linked within eps form families; a family is
    collapsed only if it holds > big points and is near-duplicate, i.e. its median r_10 is <= tight x the median r_10
    of the non-core points (at least 4x tighter than its surroundings, fixed from Result 5, not tuned). Non-core points
    within attach x eps of a collapsed family join it (v2 left the mass border free). v2's eps (1.5 x median 1-NN)
    made only 53-67% of mass points core at d = 2-5, and the rest became false groups.

    CTS v5 is cts3(Z, dim_est="local", scale_min=True, border=True).

    fix="joint" (default, stage 11): the fringe fix. Near-copies left just outside a collapsed family (mutual edge
    pairs that the border guard misses) were reported as false groups at a fixed rate per family, so O(n) overall.
    Each candidate's score becomes min(score_Beta, -log p_family), where p_family is the exact joint (tail x ball)
    test against the 3 nearest collapsed families under a Gaussian family model (scale_cts.fringe_scores). The test
    count and the selection rule are unchanged; a scene with no collapsed family gives exactly the v5 output.
    Cost: recall on groups planted inside the near-duplicate mass drops at d = 10 (stage-11 fringe doc).
    fix=None: the old behaviour, exactly (v5 as frozen before stage 11).

    base: the labels the groups are written over. None = the default HDBSCAN base (hdb); "hdb3" = HDBSCAN with min
    cluster size 3, min samples 3; an array = those labels. The scan's false-group guarantee covers only the groups
    the scan adds: the base's own small clusters are not tested, and "hdb3" alone gives about 126 small clusters on a
    1,600-point homogeneous Poisson scene at d = 2 (stage-11 hdbvs doc). base="hdb3" has no false-group guarantee.

    base="certified" (stage 14 core): core_certified.certified_base(Z, alpha_base, base_seed, "v2"), the gclu base with
    tested splits (tiny=False, h_rule="both", sel="eom").

    weights, flt, return_cands, vectorised: see _scan_select (weighted Bonferroni options; flt=True applies the frozen stage-12
    independent-filter weights). Options only; the defaults keep the method unchanged.

    Stage-15 options (all off by default; see _scan_select): count_family_points (Prop. 13.3 test count), knn="kd"
    (exact kd-tree rows for the collapse and the scan), select="group" with group_shape (Theorem 13.1 group step-up),
    d_rule="adapt" (stage-14 adaptive outer annulus) or "ccmin_lg" (stage-15 validity-first exponent) with
    tmax_mode ("max" default, "level", or "auto" = as frozen before integ stage 3; see TMAX_MODES), respect_base as a number (base-size factor).

    reference (stage 14 two / 15): an optional reference sample A (array, same columns as Z) from the background
    without the groups. CTS and the exact two-sample scan (two_scan.two_cands on (A, Z)) both run, and their rejected
    groups are combined by union_rule:
      None = "V6" (stage 15 union, the default since stage 15): Tarone's common threshold over both families
        (two_scan.tarone_threshold with n_1 = CTS's own test count; scan sizes that cannot reach it take no share of
        alpha) and the coverage rule (two_scan.union_cover: a scan group replaces the CTS groups it overlaps iff they
        cover less than half of it). Needs plain Bonferroni (select="bonferroni", count="all", no weights / flt);
        with other settings None falls back to "V5";
      "V5" = the coverage rule with alpha / 2 to each family;
      "V0" = stage 14's rule: alpha / 2 each, CTS groups first, then scan groups that overlap none of them
        (two_scan.union_groups);
      a callable (alpha / 2 each)
    f(cts_list, scan_list) -> list of member arrays, where each list holds (members, margin) with margin =
    -log p - its own threshold, in nats, best first; any choice among rejected groups keeps the alpha bound (the
    hook for the overlap rules of the union branch). With return_cands=True the dict also holds "two_groups",
    "cts_groups", "two_thr" (the scan's threshold, nats) and "union_variant"."""
    if fix not in (None, "joint"):
        raise ValueError("fix must be 'joint' or None")
    if knn not in ("auto", "kd"):
        raise ValueError("knn must be 'auto' or 'kd'")
    if reference is not None:
        return _cts_two(Z, reference, alpha, union_rule, return_cands, dict(
            big=big, max_m=max_m, k_bg=k_bg, min_core=min_core, tight=tight, attach=attach, null=null, dim_est=dim_est,
            closure=closure, count=count, respect_base=respect_base, scale_min=scale_min, select=select, border=border,
            order=order, fix=fix, base=base, weights=weights, flt=flt, vectorised=vectorised,
            count_family_points=count_family_points, knn=knn, group_shape=group_shape, d_rule=d_rule,
            alpha_base=alpha_base, base_seed=base_seed, tmax_mode=tmax_mode))
    NN = _nn_factory(knn)
    n = len(Z)
    r = NN(n_neighbors=min_core + 1).fit(Z).kneighbors(Z)[0][:, -1]
    eps = float(np.median(r))
    core = r <= eps
    comp = -np.ones(n, int)
    pairs = NN(radius=eps).fit(Z[core]).radius_neighbors_graph(Z[core], mode="connectivity")
    comp[core] = connected_components(pairs, directed=False)[1]
    ref = np.median(r[~core]) if (~core).any() else np.inf
    if border and (~core).any():
        # DBSCAN border points, with one guard. Removing a collapsed family's core leaves a hole, so its fringe (the
        # non-core shell, about half of a 30-copy family at d = 10) looked like a tight group with an empty ring and
        # was cut into false groups. A non-core point within eps of a core point joins that family when the core
        # point is at least as close as its nearest non-core point; a tight group beside a template (its members are
        # each other's nearest points) stays free.
        nc = np.flatnonzero(~core)
        dd, jj = NN(n_neighbors=1).fit(Z[core]).kneighbors(Z[nc])
        dn = NN(n_neighbors=2).fit(Z[nc]).kneighbors(Z[nc])[0][:, 1] if len(nc) > 1 else np.full(len(nc), np.inf)
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
        dd, jj = NN(n_neighbors=1).fit(Z[cidx]).kneighbors(Z[free])
        near = dd[:, 0] <= attach * eps
        comp[free[near]] = comp[cidx[jj[near, 0]]]
        collapsed[free[near]] = True
    keep = np.flatnonzero(~collapsed)
    reps, fams = [], []
    for c in np.unique(comp[collapsed]):
        mem = np.flatnonzero(collapsed & (comp == c))
        reps.append(mem[np.argmin(np.linalg.norm(Z[mem] - Z[mem].mean(0), axis=1))])
        fams.append(mem)
    R = np.r_[keep, np.array(reps, int)]
    is_rep = np.r_[np.zeros(len(keep), bool), np.ones(len(reps), bool)]
    if isinstance(base, str):
        if base not in ("hdb", "hdb3", "certified"):
            raise ValueError("base must be None, 'hdb', 'hdb3', 'certified' or an array of labels")
        if base == "certified":
            from .core_certified import certified_base
            base = certified_base(Z, alpha_base, base_seed, "v2")
        else:
            base = hdb3(Z) if base == "hdb3" else None
    return _scan_select(Z, R, is_rep, max_m, k_bg, alpha, base=base, null=null, dim_est=dim_est, closure=closure, count=count, respect_base=respect_base, scale_min=scale_min, select=select, order=order, fix=fix, families=fams, weights=weights, flt=flt, return_cands=return_cands, vectorised=vectorised,
                        count_family_points=count_family_points, knn=knn, group_shape=group_shape, d_rule=d_rule,
                        thr_of_n=_thr_of_n, tmax_mode=tmax_mode)


def _nn_factory(knn):
    """NearestNeighbors with sklearn's default algorithm ("auto": brute force above 15 columns, whose dot-product
    distances carry rounding error) or with the kd-tree (knn="kd": distances computed pair by pair)."""
    if knn == "kd":
        return lambda **kw: NearestNeighbors(algorithm="kd_tree", **kw)
    return NearestNeighbors


ADAPT_RULE = ("adapt", 0.3, (20, 40, 80, 160), 0.01)   # stage 14 core (post-hoc), as frozen in leak_eval / core_run
ADAPT_LS = (10, 20, 40, 80, 160)                       # core_cands.LS: outer radii kept per candidate
# stage 15 leak, frozen 2026-11-11 (reference: leak_theory.rule_dhat)
CCMIN_LG_RULE = ("ccmin_lg", (20, 40, 80, 160), 0.01, 20, 0.3)
D_RULES = {"v5": None, "adapt": ADAPT_RULE, "ccmin_lg": CCMIN_LG_RULE}
# tmax of the CC rules (adapt, ccmin_lg). Theorem 14.1 step 3 needs tmax >= t, the level the selection uses
# (t = alpha / n_sel; Tarone: exp(-thr)). "auto" = tmax_default(n_tests) = 0.47 / n_tests (rounds 14-15 as frozen;
# below t at alpha >= 1 or 3), "level" = t (cost branch's primary_tlevel), "max" = max(auto, t) (default since
# stage 15 integ stage 3: unchanged wherever the frozen value already meets the condition).
TMAX_MODES = ("max", "level", "auto")


def _scored_groups(cands_groups, thr):
    """(score, centre, members) tuples -> [(members array, margin = score - thr)]."""
    return [(np.array(g[2], np.int64), float(g[0] - thr)) for g in cands_groups]


UNION_VARIANTS = {"V0": ("first", "half"), "V5": ("cover", "half"), "V6": ("cover", "tarone")}   # union_rules names


def _cts_two(Z, A, alpha, union_rule, return_cands, kw):
    """CTS plus the exact two-sample scan, combined by union_rule (see cts3, reference=)."""
    from . import two_scan as W
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[1] != Z.shape[1]:
        raise ValueError("reference must be an array with the same number of columns as Z")
    plain = kw["select"] == "bonferroni" and kw["count"] == "all" and kw["weights"] is None and not kw["flt"]
    if union_rule is None:
        variant = "V6" if plain else "V5"
    elif isinstance(union_rule, str):
        if union_rule not in UNION_VARIANTS:
            raise ValueError(f"union_rule must be None, a callable or one of {sorted(UNION_VARIANTS)}")
        if union_rule == "V6" and not plain:
            raise ValueError("union_rule='V6' (Tarone) needs select='bonferroni', count='all' and no weights / flt")
        variant = union_rule
    else:
        variant = "callable"
    rule, alloc = UNION_VARIANTS.get(variant, (union_rule, "half"))
    C = W.two_cands(A, Z)
    if alloc == "tarone":
        nA = len(A)
        out = cts3(Z, alpha=alpha, return_cands=True, _thr_of_n=lambda n1: W.tarone_threshold(max(1, n1), C, nA, alpha),
                   **kw)
        thr2 = out["thr"]                       # one common threshold for both families
    else:
        out = cts3(Z, alpha=alpha / 2, return_cands=True, **kw)
        thr2 = float(np.log(max(1, C["n_tests"]) / (alpha / 2)))
    c_list = [(m, mg) for (m, mg), w in zip(_scored_groups(out["groups"], out["thr"]), out["written"]) if w]
    sig = np.flatnonzero(C["score"] >= thr2)
    order = sig[np.lexsort((C["m"][sig], -C["score"][sig]))]          # two_scan.two_groups' order
    used, s_list = set(), []
    for o in order:
        mem = C["I"][C["centre"][o], :C["m"][o]].tolist()
        if used.isdisjoint(mem):
            s_list.append((np.sort(np.array(mem, np.int64)), float(C["score"][o] - thr2))); used.update(mem)
    first, second = [m for m, _ in c_list], [m for m, _ in s_list]
    if rule == "first":
        groups = W.union_groups(first, second)
    elif rule == "cover":
        groups = W.union_groups(first, second, rule=lambda f, g: W.union_cover(f, g, [mg for _, mg in s_list]))
    else:
        groups = [np.asarray(g, np.int64) for g in union_rule(c_list, s_list)]
    lab = out["base"].copy()                   # the base CTS wrote over; the union's groups are written in its order
    nxt = lab.max() + 1
    for g in groups:
        lab[g] = nxt; nxt += 1
    if return_cands:
        return {**out, "labels": lab, "cts_groups": c_list, "two_groups": s_list, "two_thr": thr2,
                "two_n_tests": int(C["n_tests"]), "union": [np.asarray(g) for g in groups], "union_variant": variant}
    return lab


FLT_Q, FLT_ETA = 0.1, 0.5      # frozen stage-12 filter: half of alpha on the 10% largest ring radii per size m


def flt_weights(info, q=FLT_Q, eta=FLT_ETA):
    """Stage-12 independent-filter weights (flt_cts.weights with stat="ring"). A candidate passes when its ring radius
    r_(m-1+k_bg) is among the q share of largest ring radii of its size m (ties broken by candidate order);
    w = (1 - eta) + eta T 1{passes} / T_f, T = n_tests, T_f = number that pass. Sum w <= T, and the ring radius is
    independent of the Beta ratio under the local Poisson null (Result F1), so weighted Bonferroni keeps the FWER."""
    m, x = info["m"], np.log(info["Rk"])
    qr = np.empty(len(x))
    for mm in np.unique(m):
        s = np.flatnonzero(m == mm)
        o = np.argsort(-x[s], kind="stable")
        rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
        qr[s] = rk / len(s)
    passed = qr <= q
    T = max(1, info["n_tests"])
    return (1 - eta) + eta * passed * (T / max(1, int(passed.sum())))


def _scan_select(Z, R, is_rep, max_m=15, k_bg=10, alpha=0.05, base=None, dim=None, null="beta", q=0.5, dim_est="lb", closure=False, count="all", respect_base=False, scale_min=False, select="bonferroni", order="score", fix=None, families=(), weights=None, flt=False, return_cands=False, vectorised=True, rings=None,
                 count_family_points=False, knn="auto", group_shape=("trunc", 16), d_rule="v5", thr_of_n=None,
                 tmax_mode="max"):
    """null='beta' (Result 6): under a locally homogeneous Poisson process, U = (r_(m-1) / r_(m-1+k_bg))^d is exactly
    Beta(m - 1, k_bg) (ratios of partial sums of exponential ball volumes), so p = BetaCDF(U) needs no density
    estimate. null='poisson' is the v2 plug-in test (local intensity estimated from the ring), which T6 found ~10x
    anti-conservative.

    Speed: with null='beta', closure=False, count='all' and order='score' (v5 and every frozen variant), the scan is
    scale_cts.scan_rows (vectorised by size, same candidates, scores equal up to rounding, same groups); the other
    settings use the original loop. vectorised=False forces the loop (reference for the tests).

    fix='joint' with `families` (member index arrays of the collapsed families, in the order of the representatives
    in R): the fringe fix, score = min(score_Beta, -log p_family) (scale_cts.fringe_scores). fix=None: unchanged.

    weights: weighted Bonferroni. Candidate i is significant when p_i <= alpha w_i / n_tests, i.e. its score is at
    least -log(alpha w_i / n_tests); the greedy pass runs on score + log w_i (the margin over its own threshold, as in
    flt_cts.flt_labels). With select='bh'/'by' this is weighted BH/BY (p_i / w_i). The weights must not depend on the
    candidates' own Beta ratios, and their sum over all tests must be <= n_tests, or the guarantee is lost. Either an
    array aligned with the scored candidates (centre-major, then size, those with 0 < r < R_k: the order of
    hdbvs_cts.v5_parts), or a callable f(info) -> array, where info holds n_tests, centre (reduced-set position), m,
    r, Rk and R. None = unchanged. flt=True uses flt_weights (the frozen stage-12 option; ignored if weights is given).

    rings: reserved for a multi-ring option (several ring sizes k per candidate), being tested on a separate branch;
    not implemented here. It would change only the per-candidate score (the block marked "score hook" below); the
    test count, the fringe fix, the weights and the selection would stay as they are.

    Stage-15 options (defaults unchanged):
    - count_family_points=False. True (stage 13 conf, Proposition 13.3): with fix="joint" and at least one collapsed
      family, the selection counts T_eff = n_tests + (max_m - 1) * (points in collapsed families) tests, because
      every family point is a potential centre at max_m - 1 sizes. Without it the family-wise level is at most
      alpha (1 + (max_m - 1) sum M_F / T). Only the threshold changes; scenes with no family are unchanged.
    - knn="auto" | "kd". "kd": the scan rows come from scale_cts.knn_exact (kd-tree, pair-by-pair distances, rows
      sorted by (distance, not-self, id), the point itself first), the scalable path's rows. "auto" keeps sklearn's
      default (brute force above 15 columns).
    - select="group" (stage 13 prf, Theorem 13.1): the group-count step-up prf_theory.group_stepup_any with shape
      group_shape (default ("trunc", 16)) over the candidates in the greedy order; group-level FDR <= alpha T0 / T
      under any dependence. With respect_base, candidates the base would refuse are dropped before the path
      (Remark ii: a fixed filter that does not use the candidates' scores). Not combined with weights or order="gap".
    - d_rule="v5" | "adapt". "adapt" (stage 14 core, post-hoc): each candidate's exponent is the v5 one corrected
      from the widest outer annulus L in (20, 40, 80, 160) not seen to saturate (core_cands.rule_dhat with
      ADAPT_RULE); needs the v5 settings (null="beta", dim_est="local", scale_min=True). Lower d only, so it never
      adds a candidate v5 would not test.
      "ccmin_lg" (stage 15 leak, frozen; validity-first): core_cands.rule_dhat with CCMIN_LG_RULE, the smallest CC
      exponent over the annuli (20, 40, 80, 160), lowered to LCB-d on L = 20 where that annulus saturates. Valid
      whenever (C2') holds on at least one annulus (Thm 15.1), eta <= 0.18. Exact rate <= 0.81 t on Gaussian clusters
      of >= 60 points at d = 2/5/10. Not covered for clusters of about n + 20 points or fewer (Props 14.4, 15.5).
      Same settings as "adapt"; lower d only.
    - respect_base: False, True (= factor 4, unchanged) or a number f > 0: write a group over a base cluster only if
      that cluster holds at least f x the group's size. For pooled replicates (concatenated copies) the base clusters
      grow with the number of replicates, so f should scale with it or respect_base be False (stage 13 rep).

    return_cands=True returns a dict instead of labels: labels, cands (every significant candidate, best first, as
    (score, centre index, sorted member indices)), groups (the greedy subset, same tuples), written (per group:
    was it written over the base), base (the base labels), n_tests, n_tests_sel (the count the selection used),
    thr, R, dim, fams (the collapsed families)."""
    if order != "score" and select in ("bh", "by"):
        # any rule that reports only candidates above the Bonferroni threshold keeps FWER control (whatever the order);
        # under BH the threshold itself depends on the ranking, so a custom order loses the guarantee
        raise ValueError("order='gap' is only valid with select='bonferroni'")
    if rings is not None:
        raise NotImplementedError("rings= is reserved for the multi-ring option (separate branch)")
    if flt and weights is None:
        weights = flt_weights
    if weights is not None and order != "score":
        raise ValueError("weights are only supported with order='score'")
    if select == "group" and (weights is not None or order != "score"):
        raise ValueError("select='group' is not combined with weights or order='gap'")
    if select not in ("bonferroni", "bh", "by", "group"):
        raise ValueError("select must be 'bonferroni', 'bh', 'by' or 'group'")
    if tmax_mode not in TMAX_MODES:
        raise ValueError(f"tmax_mode must be one of {TMAX_MODES}")
    if d_rule not in D_RULES:
        raise ValueError("d_rule must be 'v5', 'adapt' or 'ccmin_lg'")
    if d_rule != "v5" and not (null == "beta" and dim_est == "local" and scale_min and not closure and count == "all"
                                  and order == "score"):
        raise ValueError(f"d_rule={d_rule!r} needs the v5 settings (null='beta', dim_est='local', scale_min=True)")
    if knn not in ("auto", "kd"):
        raise ValueError("knn must be 'auto' or 'kd'")
    if respect_base is True:
        rb_fac = 4.0
    elif respect_base is False or respect_base is None:
        rb_fac = 0.0
    else:
        rb_fac = float(respect_base)
        if not rb_fac > 0:
            raise ValueError("respect_base must be a bool or a number > 0")
    Zr = Z[R]
    # d can never exceed the data's rank; quantised coordinates (ties in r1, r2) inflate TwoNN, e.g. 5.7 on 2-D sets
    rank = int(np.linalg.matrix_rank(Zr - Zr.mean(0), tol=1e-8 * np.abs(Zr).max()))
    if dim is None:
        dim = max(1.0, twonn_dim(Zr) if dim_est in ("twonn", "local") else intrinsic_dim(Zr))
        dim = min(dim, rank)
    n_loc = 30
    kk = min(len(R) - 1, max_m + k_bg + (n_loc if dim_est == "local" else 0))
    kq = min(len(R) - 1, max_m + k_bg + max(ADAPT_LS)) if d_rule != "v5" else kk
    if knn == "kd":
        from .scale_cts import knn_exact
        dist_x, idx_x = knn_exact(Zr, kq)
    else:
        dist_x, idx_x = NearestNeighbors(n_neighbors=kq + 1).fit(Zr).kneighbors(Zr)
    dist, idx = dist_x[:, :kk + 1], idx_x[:, :kk + 1]
    if dim_est == "local":
        # Local d, one-sided: TwoNN log-ratios mu = log(r2/r1) are Exp(d) under local homogeneity, so over n points
        # 2 d S ~ chi2(2n) and d_lo = chi2.ppf(0.001, 2n) / (2S) is an exact 99.9% lower bound. Pooled over the n_loc
        # points just beyond the candidate; the global d is kept wherever the local bound is lower.
        mu_all = np.log(np.maximum(dist[:, 2], 1e-12) / np.maximum(dist[:, 1], 1e-12))
        lo_q = chi2.ppf(0.001, 2 * n_loc) / 2
    if null == "self" or scale_min:
        dims = _self_dims(dist, is_rep, max_m, k_bg, kk, q)
    fast = vectorised and null == "beta" and not closure and count == "all" and order == "score"
    if fast:
        from .scale_cts import scan_rows
        dims_use = dims if scale_min else {m: np.inf for m in range(2, max_m + 1)}
        # p <= alpha is enough for Bonferroni, BH and BY (their cut-offs never exceed alpha); weights need every
        # scored candidate (alpha = 1 keeps them all: scores are >= 0)
        n_tests, S, P, Ms = scan_rows(dist, idx, np.arange(len(R)), is_rep,
                                      mu_all if dim_est == "local" else np.zeros(len(R)), dim, rank, dims_use, max_m,
                                      k_bg, n_loc, alpha=1.0 if weights is not None else alpha,
                                      local=dim_est == "local")
        o = np.lexsort((Ms, P))                   # the loop's order (centre-major, then size): ties break the same way
        S, P, Ms = S[o], P[o], Ms[o].astype(np.int64)
        G = None
    else:
        S, P, Ms, G, n_tests = [], [], [], [], 0
        for i in range(len(R)):
            if is_rep[i]:
                continue
            for m in range(2, max_m + 1):
                if m - 1 + k_bg > kk:
                    break
                members = idx[i, :m]
                if is_rep[members].any():
                    break
                n_tests += 1
                if closure and any(set(idx[j, :m]) != set(members) for j in members[1:]):
                    continue          # only sets closed under m-NN (each member's m nearest = the set); Bonferroni count unchanged
                r, Rk = dist[i, m - 1], dist[i, m - 1 + k_bg]
                if Rk <= r or r <= 0:
                    continue
                gscore = 0.0
                if null == "gap":
                    # Result 6c: ball volumes are a unit Poisson process, spacings iid Exp(1). The gap after the group,
                    # over the k spacings after r_(m-1), is Beta(1, k - 1) and independent of the ratio U (Dirichlet
                    # aggregation), so Fisher's combination is exactly chi2(4).
                    v0, v1, vk = r ** dim, dist[i, m] ** dim, Rk ** dim
                    lp1 = beta_dist.logcdf(v0 / vk, m - 1, k_bg)
                    lp2 = (k_bg - 1) * np.log1p(-min((v1 - v0) / (vk - v0), 1 - 1e-300))
                    score = -chi2.logsf(-2 * (lp1 + lp2), 4)
                elif null in ("beta", "self"):
                    dd = dims[m] if null == "self" else dim
                    if scale_min:
                        # conservative both ways: the smallest-scale d (TwoNN) overstates d where the bulk is curved
                        # outward (torus), the test-scale d (median match at size m) overstates it where structures
                        # thin out with scale (tubes, curves); the smaller of the two is used
                        dd = min(dd, dims[m])
                    if dim_est == "local":
                        ring = idx[i, m:m + n_loc]
                        dd = min(max(dd, lo_q / max(mu_all[ring].sum(), 1e-12)), rank)
                    score = -beta_dist.logcdf((r / Rk) ** dd, m - 1, k_bg)
                    if order == "gap":
                        # order significant candidates by the gap after the set (Result 6c's spacing term, -log p),
                        # not by the ratio score: a larger ball that swallows cluster points scores higher on the ratio
                        # but ends inside the cluster, with no gap, and used to block the set that matches the group
                        v0, v1, vk = r ** dd, dist[i, m] ** dd, Rk ** dd
                        gscore = -(k_bg - 1) * np.log1p(-min((v1 - v0) / (vk - v0), 1 - 1e-300))
                else:
                    mu = k_bg / (Rk ** dim - r ** dim) * r ** dim
                    score = -poisson.logsf(m - 2, mu)
                S.append(score); P.append(i); Ms.append(m); G.append(gscore)
        S, P, Ms, G = np.array(S, float), np.array(P, np.int64), np.array(Ms, np.int64), np.array(G, float)
    # score hook: S holds one -log p per scored candidate (centre P, size Ms), in the loop's order; a future rings=
    # option would replace S here, before the fringe fix and the weights
    n_sel = n_tests
    if count_family_points and fix == "joint" and len(families):
        n_sel = n_tests + (max_m - 1) * int(sum(len(f) for f in families))
    t_level = float(np.exp(-thr_of_n(n_sel))) if thr_of_n is not None else alpha / max(1, n_sel)
    tmax_used = None
    if d_rule != "v5":
        from .core_cands import tmax_default
        tmax_used = {"auto": tmax_default(n_tests), "level": t_level,
                     "max": max(tmax_default(n_tests), t_level)}[tmax_mode]
    if d_rule != "v5" and len(S):
        S = _adapt_scores(dist_x, idx, P, Ms, kq, dim, dims, rank, mu_all, lo_q, n_loc, k_bg, n_tests,
                          rule=D_RULES[d_rule], tmax=None if tmax_mode == "auto" else tmax_used)
    if fix == "joint" and len(families) and len(S):
        from .scale_cts import fringe_scores
        S, _ = fringe_scores(Z, R, is_rep, families, dim, S, P, Ms, dist[P, Ms - 1])
    if weights is not None:
        w = weights({"n_tests": n_tests, "centre": P, "m": Ms, "r": dist[P, Ms - 1], "Rk": dist[P, Ms - 1 + k_bg],
                     "R": R}) if callable(weights) else np.asarray(weights, float)
        if len(w) != len(S):
            raise ValueError(f"weights has {len(w)} entries, the scan scored {len(S)} candidates")
        with np.errstate(divide="ignore"):
            S = S + np.log(w)                       # p / w on the -log scale; w = 0 never rejects
    thr = -np.log(alpha / max(1, n_sel if count == "all" else len(S)))
    if thr_of_n is not None:
        # internal (cts3 reference=, union "V6"): a threshold in nats set from the test count, e.g. Tarone's common
        # threshold over CTS and the two-sample scan (two_scan.tarone_threshold); plain Bonferroni only
        if select != "bonferroni" or count != "all" or weights is not None:
            raise ValueError("thr_of_n needs select='bonferroni', count='all' and no weights")
        thr = float(thr_of_n(n_sel))
    if select in ("bh", "by") and len(S):
        # Benjamini-Hochberg at level alpha over all n_tests candidates (scores are -log p); untested ones count as p=1.
        # The scan's p-values are not PRDS, and BH's FDR over reported groups runs about 5x its target, so BH is
        # exploratory only. BY divides the line by sum_{i<=n} 1/i, valid under any dependence at the candidate level.
        c = 1.0 if select == "bh" else float(np.log(max(1, n_sel)) + np.euler_gamma + 1 / (2 * max(1, n_sel)))
        sc = np.sort(S)[::-1]
        ok = np.flatnonzero(sc >= -np.log(alpha * np.arange(1, len(sc) + 1) / (max(1, n_sel) * c)))
        thr = sc[ok[-1]] if len(ok) else np.inf
    lab = (hdb(Z) if base is None else base).copy()
    ref = lab.copy()
    sizes = np.bincount(ref[ref >= 0]) if (ref >= 0).any() else np.zeros(0, np.int64)

    def base_refuses(pts):
        # keep the base where it already isolates the group: write only over noise, or inside a base cluster at
        # least rb_fac x the group (the masking case); a group carved out of a small base cluster would split it
        b, c = np.unique(ref[pts], return_counts=True)
        top = b[np.argmax(c)]
        return top >= 0 and sizes[top] < rb_fac * len(pts)

    if select == "group":
        from .prf_theory import group_stepup_any
        # every step-up line is at least -log(alpha) (beta(r) <= T); the fast scan keeps exactly the p <= alpha rows
        sig = np.flatnonzero(S >= -np.log(alpha))
        sig = sig[np.argsort(-S[sig], kind="stable")]
        pool = []
        for j in sig:
            mem = tuple(sorted(idx[P[j], :Ms[j]].tolist()))
            if rb_fac and base_refuses(R[list(mem)]):
                continue
            pool.append((float(S[j]), mem, j))
        _, thr, n_rej = group_stepup_any(pool, n_sel, alpha, group_shape)
        sig = np.array([c[2] for c in pool[:n_rej]], np.int64)      # the step-up's prefix of the score order
    else:
        sig = np.flatnonzero(S >= thr)
    if order == "gap":
        sig = sig[np.argsort(-G[sig], kind="stable")]
    else:
        sig = sig[np.argsort(-S[sig], kind="stable")]
    used, groups, cands, gcands = set(), [], [], []
    for j in sig:
        mem = tuple(sorted(idx[P[j], :Ms[j]].tolist()))
        if return_cands:
            cands.append((float(S[j]), int(R[P[j]]), tuple(sorted(int(t) for t in R[list(mem)]))))
        if used.isdisjoint(mem):
            groups.append(mem); used.update(mem)
            if return_cands:
                gcands.append(cands[-1])
    nxt = lab.max() + 1
    written = []
    for mem in groups:
        pts = R[list(mem)]
        if rb_fac and base_refuses(pts):
            written.append(False)
            continue
        lab[pts] = nxt; nxt += 1
        written.append(True)
    if return_cands:
        return {"labels": lab, "cands": cands, "groups": gcands, "written": written, "base": ref,
                "n_tests": n_tests, "n_tests_sel": int(n_sel), "thr": float(thr), "level": float(t_level),
                "tmax": tmax_used, "R": R, "dim": float(dim),
                "fams": list(families)}
    return lab


def _adapt_scores(dist, idx, P, Ms, kq, dim, dims, rank, mu_all, lo_q, n_loc, k_bg, n_tests, rule=None, tmax=None):
    """Scores under d_rule="adapt" (rule=None = ADAPT_RULE) or any core_cands rule spec (d_rule="ccmin_lg" passes
    CCMIN_LG_RULE): the v5 exponent of each candidate (min(dim, d_m), raised to the local lower bound,
    capped at rank, exactly as in the scan; the ring uses the scan's own rows idx), corrected by
    core_cands.rule_dhat(ADAPT_RULE); radii from the extended rows dist (kq + 1 columns, core_cands.scan_table's
    layout)."""
    from .core_cands import _score, rule_dhat
    Ms = np.asarray(Ms, np.int64)
    d5 = np.empty(len(P))
    for m in np.unique(Ms):
        s = np.flatnonzero(Ms == m)
        dd = min(dim, dims[m])
        ring = mu_all[idx[P[s], m:m + n_loc]].sum(1)
        d5[s] = np.minimum(np.maximum(dd, lo_q / np.maximum(ring, 1e-12)), rank)
    n = Ms - 1 + k_bg
    tab = {"m": Ms.astype(float), "d_v5": d5, "r": dist[P, Ms - 1], "rn": dist[P, n],
           "Lmax": (np.minimum(n + max(ADAPT_LS), kq) - n).astype(float)}
    for L in ADAPT_LS:
        tab[f"r{L}"] = dist[P, np.minimum(n + L, kq)]
    rule = ADAPT_RULE if rule is None else rule
    dh = rule_dhat(tab, rule, n_tests, k_bg) if tmax is None else rule_dhat(tab, rule, n_tests, k_bg, tmax_auto=tmax)
    return _score(tab["r"], tab["rn"], dh, tab["m"], k_bg)


def _self_dims(dist, is_rep, max_m, k_bg, kk, q=0.5):
    """null='self': no intrinsic-dimension estimate. For each m, d_m is set so that the median over all points of
    (r_(m-1) / r_(m-1+k_bg))^d_m equals the median of Beta(m - 1, k_bg). The scan statistic calibrates itself on its
    own law (the Gride ratio law), and a median is untouched by a few percent of planted points."""
    out = {}
    free = dist[~is_rep]
    for m in range(2, max_m + 1):
        if m - 1 + k_bg > kk:
            break
        r, Rk = free[:, m - 1], free[:, m - 1 + k_bg]
        ok = (r > 0) & (Rk > r)
        out[m] = float(np.log(beta_dist.ppf(q, m - 1, k_bg)) / np.quantile(np.log(r[ok] / Rk[ok]), q))
    return out


def scan_only(Z, alpha=0.05, base=None, null="beta", dim_est="lb", **kw):
    """The scan without any collapse (for the collapse ablation and the null-calibration test)."""
    return _scan_select(Z, np.arange(len(Z)), np.zeros(len(Z), bool), alpha=alpha, base=base, null=null, dim_est=dim_est, **kw)


def cts3_hybrid(Z, alpha=0.05, q=0.01, **kw):
    """CTS v5 + BH, plus an outlier-clump arm: the top q LOF points grouped by mutual 3-NN (McCatch-style), added
    where they do not overlap a CTS group. The arm catches clumps of tail points that are not denser than their
    surroundings (touching groups at low contrast, d = 10); it is not calibrated, so it is an option, not the default."""
    base = hdb(Z)
    lab = cts3(Z, alpha=alpha, dim_est="local", scale_min=True, border=True, respect_base=True, select="bh", **kw)
    extra = scored("lof", q)(Z)
    nxt = lab.max() + 1
    for c in np.unique(extra):
        if c <= base.max():
            continue
        pts = np.flatnonzero(extra == c)
        if (lab[pts] <= base.max()).all():
            lab[pts] = nxt; nxt += 1
    return lab
