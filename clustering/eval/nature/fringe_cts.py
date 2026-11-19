"""
Fringe leak of the near-duplicate collapse (stage 11), and a fix.

The problem. On structured data with nothing planted, CTS v5's false groups grow linearly with n (34 / 246 / 615 at
1M points for d = 2 / 5 / 10). Nearly all are pairs of near-copies left outside their collapsed family: a fixed leak
per family, which no multiple-testing correction over N can remove.

This file holds:
- `make_scaled_fam`: the scale branch's scene generator (`scale_bench.make_scaled`, same output bit for bit) plus the
  family (template) id of each mass point, for error analysis;
- `collapse_rows_diag`: `scale_cts.collapse_rows` with its internal flags returned;
- `family_models`: a per-family model (centre, spread with an upper confidence bound) fitted on the collapsed members
  (now in scale_cts, re-exported here);
- `family_pvalues`: the family-conditional (Palm) p-value of each scan candidate;
- `cts_fringe`: scalable CTS v5 with the family test added as an intersection-union gate (see the stage-11 doc for
  the validity argument), plus the other candidate fixes used in the tree of thoughts.
"""
import time

import numpy as np
from scipy.special import betainc
from scipy.stats import binom, chi2, ncx2

from . import scale_cts as S
from .tiny_bench import D_AMBIENT, make


# ---------------------------------------------------------------- scenes with family ids
def make_scaled_fam(placement, n_target, d=5, seed=0, n_groups=48, contrast="mid", spacing=150.0, tpl=30, copies=30):
    """Same as scale_bench.make_scaled; also returns fam (global family id of each mass point, -1 otherwise)."""
    rng = np.random.default_rng(seed)
    tiles = max(1, int(round(n_target / 1600)))
    g_tiles = min(tiles, int(np.ceil(n_groups / 12)))
    A, kinds, gids, fams = [], [], [], []
    centres = []
    goff = 0
    for t in range(tiles):
        a, kind, gid = make(placement, d=d, seed=seed * 100003 + t, n_groups=12 if t < g_tiles else 0,
                            contrast=contrast)
        while True:
            c = rng.normal(size=D_AMBIENT)
            c *= spacing * (1 + tiles) ** (1 / D_AMBIENT) * 3 / np.linalg.norm(c) * rng.uniform(0.5, 1.0) ** (1 / D_AMBIENT)
            if not centres or np.min(np.linalg.norm(np.array(centres) - c, axis=1)) > spacing:
                break
        centres.append(c)
        A.append(a + c); kinds.append(kind)
        gids.append(np.where(gid >= 0, gid + goff, -1)); goff += 12 if t < g_tiles else 0
        f = -np.ones(len(a), np.int64)
        f[:tpl * copies] = t * tpl + np.arange(tpl * copies) // copies     # make() writes the mass first, by template
        fams.append(f)
    return np.vstack(A), np.concatenate(kinds), np.concatenate(gids), np.concatenate(fams)


def make_fam(placement, d=5, seed=0, n_groups=12, contrast="mid", tpl=30, copies=30):
    """tiny_bench.make plus family ids."""
    A, kind, gid = make(placement, d=d, seed=seed, n_groups=n_groups, contrast=contrast)
    f = -np.ones(len(A), np.int64)
    f[:tpl * copies] = np.arange(tpl * copies) // copies
    return A, kind, gid, f


# ---------------------------------------------------------------- collapse with diagnostics
def collapse_rows_diag(D, I, Z, big=20, min_core=10, tight=0.25, attach=0.5):
    """scale_cts.collapse_rows, returning (R, is_rep, info, diag) where diag holds the per-point flags: r10, core,
    comp (after the border join and attach), collapsed, joined (border join), attached, and for non-core points the
    guard inputs dd (nearest core in the row) and dn (nearest non-core in the row)."""
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
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(ci):
        pos = -np.ones(n, np.int64); pos[ci] = np.arange(len(ci))
        g = coo_matrix((np.ones(len(rows)), (pos[rows], pos[cols])), shape=(len(ci),) * 2)
        comp[ci] = connected_components(g, directed=False)[1]
    ref = float(np.median(r[~core])) if (~core).any() else np.inf
    nc = np.flatnonzero(~core)
    ddv = np.full(n, np.nan); dnv = np.full(n, np.nan)
    joined = np.zeros(n, bool)
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
        ddv[nc], dnv[nc] = dd, dn
        joined[nc[join]] = True
    collapsed = np.zeros(n, bool)
    if (comp >= 0).any():
        for c in np.unique(comp[core]):
            mem = np.flatnonzero(comp == c)
            if len(mem) > big and np.median(r[mem]) <= tight * ref:
                collapsed[mem] = True
    attached = np.zeros(n, bool)
    if collapsed.any():
        from sklearn.neighbors import KDTree
        free = np.flatnonzero(~collapsed)
        cidx = np.flatnonzero(collapsed)
        d2, j2 = KDTree(Z[cidx]).query(Z[free], k=1)
        near = d2[:, 0] <= attach * eps
        comp[free[near]] = comp[cidx[j2[near, 0]]]
        collapsed[free[near]] = True
        attached[free[near]] = True
    diag = {"r10": r, "core": core, "comp": comp, "collapsed": collapsed, "joined": joined, "attached": attached,
            "dd": ddv, "dn": dnv, "eps": eps, "ref": ref}
    return diag


# ---------------------------------------------------------------- per-family model
# family_models and joint_pvalues (the adopted fix) now live in scale_cts, shared with tiny_methods.cts3 and
# scale_cts.cts_scale (fix="joint", the default); they are re-exported here unchanged.
family_models = S.family_models


# ---------------------------------------------------------------- the family-conditional (Palm) p-value
def family_pvalues(Z, R, cand_pos, cand_m, cand_r, fam_c, fam_s2, fam_M, dim, k_fam=3, far=12.0):
    """Family test of scan candidates (centre Z[R[cand_pos]], size m, radius r = r_(m-1)).
    Null: the centre and the other points near it are members of the collapsed families nearby, each family an
    isotropic Gaussian N(c_F, s2_F I_dim) with M_F members. Given that the centre is a family point (Palm), the number
    of other family points in B(x, r) is a sum of independent binomials Bin(M_F - 1, pi_F), pi_F = P(|Y - x| <= r) =
    ncx2.cdf(r^2 / s2_F, dim, |x - c_F|^2 / s2_F). {r_(m-1) <= r} is {count >= m - 1}, so
    p_fam = P(count >= m - 1); the binomials are replaced by Poisson(sum (M_F - 1) pi_F), whose upper tail is at least
    as heavy at the counts that matter (conservative). Families farther than `far` spreads (beyond r) are skipped
    (pi < 1e-20). Returns p_fam (1 where there is no family nearby is not used: far candidates get p_fam = 0)."""
    from sklearn.neighbors import KDTree
    from scipy.stats import poisson
    p = np.zeros(len(cand_pos))
    if len(fam_c) == 0 or len(cand_pos) == 0:
        return p
    x = Z[R[cand_pos]]
    kq = min(k_fam, len(fam_c))
    dist, jj = KDTree(fam_c).query(x, k=kq)
    mu = np.zeros(len(cand_pos))
    for t in range(kq):
        s = np.sqrt(fam_s2[jj[:, t]])
        near = dist[:, t] - cand_r <= far * s
        if not near.any():
            continue
        idx = np.flatnonzero(near)
        lam = (dist[idx, t] / s[idx]) ** 2
        q = (cand_r[idx] / s[idx]) ** 2
        pi = ncx2.cdf(q, dim, lam)
        mu[idx] += (fam_M[jj[idx, t]] - 1) * pi
    has = mu > 0
    p[has] = poisson.sf(cand_m[has] - 2, mu[has])
    return p


def radial_pvalues(Z, R, Ir, cand_pos, cand_m, fam_c, fam_s2, fam_M, dim, k_fam=3, m_extra=True):
    """Radial tail test of scan candidates against the nearby collapsed families.
    Null for family F: the candidate's m points are members of F, whose M_F + m points are iid N(c_F, s2_F I_dim).
    Let rho = the smallest distance from a candidate member to c_F. The candidate puts m points at radius >= rho, so the
    m-th largest radius among the family's points is >= rho, and P(R_(m) >= rho) = P(Bin(M_F + m, tau) >= m) with
    tau = chi2.sf(rho^2 / s2_F, dim). Taking rho as the candidate's own minimum only makes p larger (conservative).
    The null is the union over the k_fam nearest families, so p_radial = max over them. Far families give p ~ 0."""
    from sklearn.neighbors import KDTree
    p = np.zeros(len(cand_pos))
    if len(fam_c) == 0 or len(cand_pos) == 0:
        return p
    kq = min(k_fam, len(fam_c))
    x = Z[R[cand_pos]]
    _, jj = KDTree(fam_c).query(x, k=kq)
    mm = int(cand_m.max())
    memb = Ir[cand_pos, :mm]                       # reduced-set positions of the members (first m are used)
    valid = np.arange(mm)[None, :] < cand_m[:, None]
    for t in range(kq):
        f = jj[:, t]
        dz = np.linalg.norm(Z[R[memb]] - fam_c[f][:, None, :], axis=2)
        rho = np.where(valid, dz, np.inf).min(1)
        tau = chi2.sf(rho ** 2 / fam_s2[f], dim)
        n_f = fam_M[f] + (cand_m if m_extra else 0)
        p = np.maximum(p, binom.sf(cand_m - 1, n_f, tau))
    return p


joint_pvalues = S.joint_pvalues


def _collapse_full(D, I, Z, big=20, min_core=10, tight=0.25, attach=0.5):
    """collapse_rows_diag plus R, is_rep and the member list of each family (same R order as scale_cts)."""
    dg = collapse_rows_diag(D, I, Z, big, min_core, tight, attach)
    return dg


def _reduce(Z, collapsed, comp):
    keep = np.flatnonzero(~collapsed)
    reps, members = [], []
    cp = np.flatnonzero(collapsed)
    if len(cp):
        cp = cp[np.argsort(comp[cp], kind="stable")]
        cuts = np.flatnonzero(np.diff(comp[cp])) + 1
        for mem in np.split(cp, cuts):
            reps.append(mem[np.argmin(np.linalg.norm(Z[mem] - Z[mem].mean(0), axis=1))])
            members.append(mem)
    R = np.r_[keep, np.array(reps, np.int64)]
    is_rep = np.r_[np.zeros(len(keep), bool), np.ones(len(reps), bool)]
    return R, is_rep, members


def cts_fringe(Z, fix="palm", rule="bonferroni", alpha=S.ALPHA, max_m=S.MAX_M, k_bg=S.K_BG, knn="exact", seed=0,
               rows=None, q_ucb=0.999, fam_dim="global", absorb_q=1e-4, two_scale=1.5, sep=1.0, k_fam=3, m_extra=True, return_info=False):
    """Scalable CTS v5 with a fringe fix. fix:
    - "none": v5 exactly (same collapse, scan and selection as scale_cts.cts_scale(..., fix=None));
    - "palm": family-conditional gate. Each candidate gets p = max(p_beta, p_fam) (intersection-union: it is reported
      only if it is unusual both as background and as part of the nearby collapsed families), then Bonferroni / BH
      and the greedy pass as in v5;
    - "absorb": extended collapse. A free point joins a collapsed family when |x - c_F|^2 <= s2_F chi2_dim(1 - absorb_q)
      (the family's own spread, upper-bounded), iterated once;
    - "two_scale": a second border pass at a looser core scale: a free point with r_10 <= two_scale x eps that lies
      within eps of a collapsed point joins its family;
    - "sep": drop candidates whose gap to the nearest collapsed point is below sep x r_(m-1) (a selection filter,
      valid by Theorem 2a under Bonferroni)."""
    t0 = time.time()
    n = len(Z)
    kk = S.kk_for(n, max_m, k_bg)
    if rows is None:
        rows = S.knn_exact(Z, kk) if knn == "exact" else S.knn_ann(Z, kk, seed=seed)
    D, I = rows
    dg = collapse_rows_diag(D, I, Z)
    collapsed, comp, eps, r10 = dg["collapsed"].copy(), dg["comp"].copy(), dg["eps"], dg["r10"]
    from sklearn.neighbors import KDTree
    if fix == "two_scale" and collapsed.any():
        free = np.flatnonzero(~collapsed & (r10 <= two_scale * eps))
        if len(free):
            cidx = np.flatnonzero(collapsed)
            d2, j2 = KDTree(Z[cidx]).query(Z[free], k=1)
            ok = d2[:, 0] <= eps
            comp[free[ok]] = comp[cidx[j2[ok, 0]]]; collapsed[free[ok]] = True
    R, is_rep, members = _reduce(Z, collapsed, comp)
    # the plug-in d of the reduced set (needed by the family model when fam_dim == "global")
    if len(R) == n and not is_rep.any():
        Dr, Ir = D, I
    else:
        Dr, Ir, _ = S._reduced_rows(Z, R, S.kk_for(len(R), max_m, k_bg), knn, seed, None)
    mu, lr = S.local_stats(Dr, Ir, is_rep, max_m, k_bg)
    dim, rank, dims = S.reduce_stats([mu], [lr], S.scatter_of(Z[R]), max_m, k_bg)
    fd = dim if fam_dim == "global" else float(fam_dim)
    fam_c, fam_s2, fam_M = family_models(Z, R, is_rep, members, q_ucb=q_ucb, dim=fd)
    if fix == "absorb" and len(members):
        free = np.flatnonzero(~collapsed)
        dist, jj = KDTree(fam_c).query(Z[free], k=1)
        lim = fam_s2[jj[:, 0]] * chi2.ppf(1 - absorb_q, fd)
        ok = dist[:, 0] ** 2 <= lim
        repcomp = np.array([comp[m[0]] for m in members])
        comp[free[ok]] = repcomp[jj[ok, 0]]; collapsed[free[ok]] = True
        R, is_rep, members = _reduce(Z, collapsed, comp)
        Dr, Ir, _ = S._reduced_rows(Z, R, S.kk_for(len(R), max_m, k_bg), knn, seed, None)
        mu, lr = S.local_stats(Dr, Ir, is_rep, max_m, k_bg)
        dim, rank, dims = S.reduce_stats([mu], [lr], S.scatter_of(Z[R]), max_m, k_bg)
    nt, Sc, P, Ms = S.scan_rows(Dr, Ir, np.arange(len(R)), is_rep, mu, dim, rank, dims, max_m, k_bg, alpha=alpha)
    n_gated = 0
    if fix == "palm" and len(Sc):
        rr = Dr[P, Ms.astype(np.int64) - 1]
        pf = family_pvalues(Z, R, P, Ms.astype(np.int64), rr, fam_c, fam_s2, fam_M, fd)
        with np.errstate(divide="ignore"):
            sf = -np.log(np.maximum(pf, 1e-300))
        new = np.minimum(Sc, sf)
        n_gated = int(((Sc >= -np.log(alpha / max(1, nt))) & (new < -np.log(alpha / max(1, nt)))).sum())
        Sc = new
    if fix == "joint" and len(Sc) and len(members):
        rr = Dr[P, Ms.astype(np.int64) - 1]
        pj = joint_pvalues(Z, R, P, Ms.astype(np.int64), rr, fam_c, fam_s2, fam_M, fd, k_fam=k_fam, m_extra=m_extra)
        with np.errstate(divide="ignore"):
            sf = -np.log(np.maximum(pj, 1e-300))
        new = np.minimum(Sc, sf)
        thr0 = -np.log(alpha / max(1, nt))
        n_gated = int(((Sc >= thr0) & (new < thr0)).sum())
        Sc = new
    if fix == "radial" and len(Sc) and len(members):
        pr = radial_pvalues(Z, R, Ir, P, Ms.astype(np.int64), fam_c, fam_s2, fam_M, fd, k_fam=k_fam, m_extra=m_extra)
        with np.errstate(divide="ignore"):
            sf = -np.log(np.maximum(pr, 1e-300))
        new = np.minimum(Sc, sf)
        thr0 = -np.log(alpha / max(1, nt))
        n_gated = int(((Sc >= thr0) & (new < thr0)).sum())
        Sc = new
    if fix == "sep" and len(Sc) and collapsed.any():
        cidx = np.flatnonzero(collapsed)
        gap = np.full(len(R), np.inf)
        fr = np.flatnonzero(~is_rep)
        gap[fr] = KDTree(Z[cidx]).query(Z[R[fr]], k=1)[0][:, 0]
        rr = Dr[P, Ms.astype(np.int64) - 1]
        ming = np.array([gap[Ir[p, :m]].min() for p, m in zip(P, Ms)])
        bad = ming < sep * rr
        n_gated = int(bad.sum())
        Sc = np.where(bad, 0.0, Sc)
    groups, thr, nsig = S.select(nt, Sc, P, Ms, lambda p: Ir[p], alpha, rule)
    lab = -np.ones(n, np.int64)
    out = []
    for k, g in enumerate(groups):
        lab[R[g]] = k; out.append(R[g])
    info = {"n": n, "reduced": len(R), "n_tests": nt, "thr": thr, "dim": dim, "fam_dim": fd, "families": len(members),
            "collapsed": int(collapsed.sum()), "groups": out, "n_gated": n_gated, "eps": eps,
            "t_total": time.time() - t0}
    return (lab, info) if return_info else lab


# ---------------------------------------------------------------- robustness scenes: non-Gaussian families
def make_shape(placement, d=5, seed=0, n_groups=12, tpl=30, copies=30, copy_sd=0.02, contrast="mid", shape="gauss"):
    """tiny_bench.make with the near-copy law changed (same seeds, same layout otherwise):
    shape = "gauss" (as make), "t3" (Student t with 3 df, scaled to the same sd: heavy tails), "aniso" (axis sds
    spread log-uniformly over [1/2, 2] x copy_sd, random rotation per family), "uniform" (uniform ball, same sd)."""
    from .tiny_bench import CONTRAST
    rng = np.random.default_rng(seed)
    X, kind, gid = [], [], []
    tc = rng.normal(0, 1.0, (tpl, d))
    srng = np.random.default_rng(seed + 7919)
    for c in tc:
        if shape == "gauss":
            e = rng.normal(0, copy_sd, (copies, d))
        else:
            rng.normal(0, copy_sd, (copies, d))          # keep the main stream aligned with make()
            if shape == "t3":
                e = srng.standard_t(3, (copies, d)) * copy_sd / np.sqrt(3.0)
            elif shape == "aniso":
                sc = np.exp(srng.uniform(np.log(0.5), np.log(2.0), d)); sc /= np.sqrt((sc ** 2).mean())
                Q = np.linalg.qr(srng.normal(size=(d, d)))[0]
                e = (srng.normal(0, copy_sd, (copies, d)) * sc) @ Q.T
            elif shape == "uniform":
                u = srng.normal(size=(copies, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
                rad = srng.uniform(0, 1, copies) ** (1 / d) * copy_sd * np.sqrt(d + 2)
                e = u * rad[:, None]
            else:
                raise ValueError(shape)
        X.append(c + e); kind += ["mass"] * copies; gid += [-1] * copies
    bc = []
    for _ in range(8):
        while True:
            c = rng.normal(0, 6.0, d)
            if np.linalg.norm(c) > 4 and all(np.linalg.norm(c - b[0]) > 3 for b in bc):
                break
        bc.append(c)
        m, sd = int(rng.integers(30, 151)), float(rng.uniform(0.3, 0.8))
        X.append(c + rng.normal(0, sd, (m, d))); kind += ["background"] * m; gid += [-1] * m
        bc[-1] = (c, sd)
    allpts = np.vstack(X)
    lo, hi = allpts.min(0) - 2, allpts.max(0) + 2
    nn = int(0.02 * len(allpts))
    X.append(rng.uniform(lo, hi, (nn, d))); kind += ["noise"] * nn; gid += [-1] * nn
    for g in range(n_groups):
        m = int(rng.integers(2, 10))
        if placement == "isolated":
            while True:
                c = rng.uniform(lo, hi)
                if np.min(np.linalg.norm(allpts - c, axis=1)) > 2.0:
                    break
            sd = 0.5 * CONTRAST[contrast]
        elif placement == "touching":
            b, bsd = bc[rng.integers(len(bc))]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = b + u * bsd * (np.sqrt(d) + 1.5), bsd * CONTRAST[contrast]
        else:
            a = tc[rng.integers(tpl)]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = a + u * 6 * copy_sd, copy_sd * CONTRAST[contrast] * 2
        X.append(c + rng.normal(0, sd, (m, d))); kind += ["group"] * m; gid += [g] * m
    Z = np.vstack(X)
    Q = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_AMBIENT)))[0][:, :d]
    A = Z @ Q.T + rng.normal(0, 0.002, (len(Z), D_AMBIENT))
    return A, np.array(kind), np.array(gid)


def make_scaled_shape(n_target, d=5, seed=0, shape="gauss", spacing=150.0):
    """Structured null at scale (no planted groups) with non-Gaussian families; tiles placed as in make_scaled."""
    rng = np.random.default_rng(seed)
    tiles = max(1, int(round(n_target / 1600)))
    A, kinds, centres = [], [], []
    for t in range(tiles):
        a, kind, _ = make_shape("isolated", d=d, seed=seed * 100003 + t, n_groups=0, shape=shape)
        while True:
            c = rng.normal(size=D_AMBIENT)
            c *= spacing * (1 + tiles) ** (1 / D_AMBIENT) * 3 / np.linalg.norm(c) * rng.uniform(0.5, 1.0) ** (1 / D_AMBIENT)
            if not centres or np.min(np.linalg.norm(np.array(centres) - c, axis=1)) > spacing:
                break
        centres.append(c); A.append(a + c); kinds.append(kind)
    return np.vstack(A), np.concatenate(kinds)
