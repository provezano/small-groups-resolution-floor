"""
CTS with a family of ring sizes (stage 12,.

v5 compares a candidate's radius r_(m-1) with the radius of a fixed ring, the (m - 1 + k)-th neighbour, k = 10:
U_k = (r_(m-1) / r_(m-1+k))^d ~ Beta(m - 1, k) under a locally homogeneous Poisson null. Here each candidate is tested
at several ring sizes K = {k_1, ..., k_J} and the smallest p-value is kept. Two exact ways to pay for the choice:
- calib="bonf": Bonferroni over J (score - log J);
- calib="minp": the null law of the vector (U_k)_k is pivotal (it depends only on m and K: A ~ Gamma(m - 1) and the
  ring's partial sums B_k ~ Gamma(k) of one unit Poisson process), so P(min_k p_k <= t) is the same function of t
  in every scene. As t -> 0 it equals c_m t with c_m = E[max_k B_k^(m-1) / E B_k^(m-1)] (Result R2 in the doc).
  The score is -log(c_m min_k p_k); c_m is computed once per (m, K) by simulation (2e5 draws, fixed seed).

Optionally the stage-11 fringe gate (`fringe_cts.joint_pvalues`, intersection-union) is applied on top
(fix="joint"). Everything else (collapse, plug-in d, local d guard, selection) is v5 as in `scale_cts` /
`fringe_cts.cts_fringe`. With rings=(10,) and fix="none" the output equals cts_fringe(fix="none"), checked in the tests.
"""
import functools

import numpy as np
from scipy.special import betainc, gammaln
from scipy.stats import chi2

from . import fringe_cts as F
from . import scale_cts as S

MAX_M, K_BG, N_LOC, ALPHA = S.MAX_M, S.K_BG, S.N_LOC, S.ALPHA


@functools.lru_cache(maxsize=None)
def c_minp(m, rings, n=200000, seed=20261110):
    """Small-t constant of the pivotal min-p null: P(min_k p_k <= t) ~ c_m t (Result R2). c = 1 for one ring."""
    rings = tuple(sorted(rings))
    if len(rings) == 1 or m < 2:
        return 1.0
    rng = np.random.default_rng(seed + 1000 * m + sum(rings))
    Sm = np.cumsum(rng.exponential(size=(n, max(rings))), 1)
    lb = np.stack([(m - 1) * np.log(Sm[:, k - 1]) - (gammaln(k + m - 1) - gammaln(k)) for k in rings], 1)
    return float(np.exp(lb.max(1)).mean())


def kk_ring(n, rings, max_m=MAX_M):
    return min(n - 1, max(S.kk_for(n, max_m, K_BG), max_m - 1 + max(rings)))


def ring_parts(Z, rings=(10, 25, 50), rows=None, max_m=MAX_M, keep_alpha=0.5, family=True):
    """Collapse (v5, on v5's own 55-NN rows) + the scan at every ring size in `rings`.
    Returns a dict with per-candidate arrays: P (reduced-set row), m, r, scores (n_cand x len(rings), -log p, nan where
    the ring does not fit), the joint family p-value (if family), plus n_tests and the reduced rows."""
    rings = tuple(sorted(set(int(k) for k in rings) | {K_BG}))
    n = len(Z)
    kk0 = S.kk_for(n, max_m, K_BG)
    kk = kk_ring(n, rings, max_m)
    if rows is None:
        rows = S.knn_exact(Z, kk)
    D, I = rows
    dg = F.collapse_rows_diag(D[:, :kk0 + 1], I[:, :kk0 + 1], Z)
    R, is_rep, members = F._reduce(Z, dg["collapsed"], dg["comp"])
    kr0 = S.kk_for(len(R), max_m, K_BG)
    kr = kk_ring(len(R), rings, max_m)
    if len(R) == n and not is_rep.any() and D.shape[1] - 1 >= kr:
        Dr, Ir = D[:, :kr + 1], I[:, :kr + 1]
    else:
        Dr, Ir = S.knn_exact(Z[R], kr)
    mu, lr = S.local_stats(Dr[:, :kr0 + 1], Ir[:, :kr0 + 1], is_rep, max_m, K_BG)
    dim, rank, dims = S.reduce_stats([mu], [lr], S.scatter_of(Z[R]), max_m, K_BG)
    lo_q = chi2.ppf(0.001, 2 * N_LOC) / 2
    live = ~is_rep
    rid = np.flatnonzero(live)
    d, ii = Dr[live], Ir[live]
    rep_cols = is_rep[ii[:, :max_m]]
    first = np.where(rep_cols.any(1), np.argmax(rep_cols, 1), max_m)
    csum = np.concatenate([np.zeros((len(ii), 1)), np.cumsum(mu[ii], 1)], 1)
    n_tests = 0
    out = {k: [] for k in ("P", "m", "r", "sc", "wq")}
    keep = -np.log(keep_alpha)
    for m in range(2, max_m + 1):
        if m - 1 + K_BG > kr0:
            break
        valid = first >= m
        n_tests += int(valid.sum())
        r, R10 = d[:, m - 1], d[:, m - 1 + K_BG]
        ok = valid & (R10 > r) & (r > 0)            # v5's own validity condition (ring of 10)
        if not ok.any():
            continue
        ring = csum[ok, min(m + N_LOC, kr0 + 1)] - csum[ok, m]
        dd = min(dim, dims[m])
        dd = np.minimum(np.maximum(dd, lo_q / np.maximum(ring, 1e-12)), rank)
        sc = np.full((int(ok.sum()), len(rings)), np.nan)
        wq = np.full((int(ok.sum()), len(rings)), np.nan)
        v0, v10 = r[ok] ** dd, R10[ok] ** dd
        for j, k in enumerate(rings):
            if m - 1 + k > kr:
                continue
            Rk = d[ok, m - 1 + k]
            g = Rk > r[ok]
            u = np.where(g, (r[ok] / np.where(g, Rk, 1.0)) ** dd, 1.0)
            with np.errstate(divide="ignore"):
                sc[:, j] = np.where(g, -np.log(np.maximum(betainc(m - 1, k, u), 1e-320)), 0.0)
            if k > K_BG:
                # ring shape: W = (V(R_10) - V(r)) / (V(R_k) - V(r)) ~ Beta(10, k - 10) under the null, independent
                # of U_k (Dirichlet aggregation); its CDF is the gate statistic (Result R4)
                w = (v10 - v0) / np.maximum(Rk ** dd - v0, 1e-300)
                wq[:, j] = betainc(K_BG, k - K_BG, np.clip(w, 0.0, 1.0))
        sel = np.nanmax(sc, 1) >= keep
        out["P"].append(rid[ok][sel]); out["m"].append(np.full(int(sel.sum()), m, np.int16))
        out["r"].append(r[ok][sel]); out["sc"].append(sc[sel]); out["wq"].append(wq[sel])
    cat = lambda L, t, sh=None: np.concatenate(L) if L else np.zeros((0,) + (sh or ()), t)
    P, Ms, rr = cat(out["P"], np.int64), cat(out["m"], np.int16), cat(out["r"], float)
    SC = cat(out["sc"], float, (len(rings),))
    WQ = cat(out["wq"], float, (len(rings),))
    pj = None
    if family and len(members) and len(P):
        fam_c, fam_s2, fam_M = F.family_models(Z, R, is_rep, members, q_ucb=0.999, dim=dim)
        pj = F.joint_pvalues(Z, R, P, Ms.astype(np.int64), rr, fam_c, fam_s2, fam_M, dim)
    return dict(n=n, R=R, is_rep=is_rep, Ir=Ir, Dr=Dr, rings=rings, n_tests=n_tests, P=P, m=Ms, r=rr, sc=SC, wq=WQ,
                pj=pj, dim=dim, families=len(members))


def ring_scores(parts, rings=(10,), calib="bonf", fix="none"):
    """Combined -log p of every candidate for a ring subset. calib: "bonf" (- log J) or "minp" (- log c_m), or
    calib=("gate", q) with rings=(10, k): use ring k when the ring-shape CDF is >= 1 - q (the first 10 ring points
    are not front-loaded, as they are where the density falls outward), else ring 10; no correction. Valid without
    paying for the choice (Result R4): U_k is independent of the shape, and on the fallback event the ring of 10 is
    stochastically closer, so p_10 is conservative there. Under a homogeneous null ring k is used in a share q."""
    rings = tuple(sorted(rings))
    if isinstance(calib, (tuple, list)) and calib[0] == "gate":
        assert len(rings) == 2 and rings[0] == K_BG
        j10, jk = parts["rings"].index(K_BG), parts["rings"].index(rings[1])
        use = parts["wq"][:, jk] >= 1 - calib[1]
        best = np.where(use & ~np.isnan(parts["sc"][:, jk]), parts["sc"][:, jk], parts["sc"][:, j10])
        if fix == "joint" and parts["pj"] is not None:
            with np.errstate(divide="ignore"):
                best = np.minimum(best, -np.log(np.maximum(parts["pj"], 1e-300)))
        return best
    cols = [parts["rings"].index(k) for k in rings]
    sc = parts["sc"][:, cols]
    best = np.nanmax(np.where(np.isnan(sc), -np.inf, sc), 1) if len(sc) else np.zeros(0)
    if len(rings) > 1:
        if calib == "bonf":
            best = best - np.log(len(rings))
        elif calib == "minp":
            cm = {int(m): c_minp(int(m), rings) for m in np.unique(parts["m"])}
            best = best - np.log(np.array([cm[int(m)] for m in parts["m"]]))
        else:
            raise ValueError(calib)
    if fix == "joint" and parts["pj"] is not None:
        with np.errstate(divide="ignore"):
            best = np.minimum(best, -np.log(np.maximum(parts["pj"], 1e-300)))
    return best


def ring_groups(parts, alpha=ALPHA, rings=(10,), calib="bonf", fix="none", rule="bonferroni"):
    """Groups (arrays of original point ids) selected as in v5: Bonferroni over the n_tests tests, greedy pass."""
    sc = ring_scores(parts, rings, calib, fix)
    groups, thr, nsig = S.select(parts["n_tests"], sc, parts["P"], parts["m"], lambda p: parts["Ir"][p], alpha, rule)
    return [parts["R"][g] for g in groups]


def ring_labels(parts, alpha=ALPHA, rings=(10,), calib="bonf", fix="none", base=None):
    return S.groups_to_labels(ring_groups(parts, alpha, rings, calib, fix), parts["n"], base)


def cts_ring(Z, alpha=ALPHA, rings=(10, 25, 50), calib="minp", fix="joint", base=None):
    """One-call entry point."""
    return ring_labels(ring_parts(Z, rings, family=(fix == "joint")), alpha, rings, calib, fix, base)
