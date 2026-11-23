"""
CTS v5 vs plain HDBSCAN: a vectorised, instrumented copy of CTS v5 and the variants built from the error analysis.

`v5_parts(Z)` runs the collapse and the scan once and returns every candidate (centre, size, members, ratio score,
the d plugged in, and the separation score), so one scene can be thresholded at many alpha values and inspected.
`v5_labels(parts, alpha)` reproduces `cts3(Z, alpha, dim_est="local", scale_min=True, border=True)` exactly
(checked in the tests).

Separation family (Result S1,. Under the same local Poisson model as the
Beta null, the scaled ball volumes v_j = lambda V_d r_j^d around a point are the arrival times of a unit Poisson
process, so v_(m-1) / v_m = (r_(m-1) / r_m)^d is Beta(m - 1, 1): P(U <= u) = u^(m-1). This is the Beta ratio test with
a ring of one point (k = 1): it asks for a gap right after the group, not for a dense group against a ring of ten.
It is exact for any intensity; any d below the true d keeps it valid (same argument as for the ratio test).

Variants (all write over an HDBSCAN base; none uses labels):
- "v5": the frozen method.
- base3: the HDBSCAN base with min cluster size 3, min samples 3 (the setting plain HDBSCAN uses at its frontier).
- sep: the separation family added as a second Bonferroni family (both families counted, threshold alpha / (2 T)).
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta as beta_dist, chi2
from sklearn.neighbors import NearestNeighbors

from . import methods as M
from .tiny_methods import hdb, intrinsic_dim, twonn_dim  # noqa: F401  (intrinsic_dim kept for parity)

MAX_M, K_BG, N_LOC = 15, 10, 30


def collapse(Z, big=20, min_core=10, tight=0.25, attach=0.5):
    """cts3's collapse with border=True (copied, unchanged). Returns R (indices kept + representatives), is_rep."""
    n = len(Z)
    r = NearestNeighbors(n_neighbors=min_core + 1).fit(Z).kneighbors(Z)[0][:, -1]
    eps = float(np.median(r))
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
    return R, is_rep, collapsed


def _self_dims(dist, is_rep, kk, q=0.5):
    out = {}
    free = dist[~is_rep]
    for m in range(2, MAX_M + 1):
        if m - 1 + K_BG > kk:
            break
        r, Rk = free[:, m - 1], free[:, m - 1 + K_BG]
        ok = (r > 0) & (Rk > r)
        out[m] = float(np.log(beta_dist.ppf(q, m - 1, K_BG)) / np.quantile(np.log(r[ok] / Rk[ok]), q))
    return out


def v5_parts(Z, base=None):
    """Collapse + scan of CTS v5 (dim_est='local', scale_min=True, border=True), every candidate kept."""
    R, is_rep, collapsed = collapse(Z)
    Zr = Z[R]
    rank = int(np.linalg.matrix_rank(Zr - Zr.mean(0), tol=1e-8 * np.abs(Zr).max()))
    dim = min(max(1.0, twonn_dim(Zr)), rank)
    kk = min(len(R) - 1, MAX_M + K_BG + N_LOC)
    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Zr).kneighbors(Zr)
    mu_all = np.log(np.maximum(dist[:, 2], 1e-12) / np.maximum(dist[:, 1], 1e-12))
    lo_q = chi2.ppf(0.001, 2 * N_LOC) / 2
    dims = _self_dims(dist, is_rep, kk)
    isrep_nb = is_rep[idx]                                   # (n, kk+1)
    # first column holding a representative: members idx[i, :m] are rep-free iff m <= firstrep
    has = isrep_nb.any(1)
    firstrep = np.where(has, np.argmax(isrep_nb, 1), kk + 1)
    cms = np.c_[np.zeros(len(R)), np.cumsum(mu_all[idx], 1)]  # cms[i, j] = sum mu_all[idx[i, :j]]
    rows = []
    for m in range(2, MAX_M + 1):
        if m - 1 + K_BG > kk:
            break
        ok = (~is_rep) & (firstrep >= m)
        ii = np.flatnonzero(ok)
        r, Rk, rn = dist[ii, m - 1], dist[ii, m - 1 + K_BG], dist[ii, m]
        hi = min(m + N_LOC, kk + 1)
        ring_sum = cms[ii, hi] - cms[ii, m]
        dd = np.minimum(dim, dims[m])
        dd = np.minimum(np.maximum(dd, lo_q / np.maximum(ring_sum, 1e-12)), rank)
        rows.append((ii, np.full(len(ii), m), r, Rk, rn, dd))
    ii = np.concatenate([x[0] for x in rows]); mm = np.concatenate([x[1] for x in rows])
    r = np.concatenate([x[2] for x in rows]); Rk = np.concatenate([x[3] for x in rows])
    rn = np.concatenate([x[4] for x in rows]); dd = np.concatenate([x[5] for x in rows])
    # restore cts3's order (centre-major, then size) so ties break the same way
    o = np.lexsort((mm, ii))
    ii, mm, r, Rk, rn, dd = ii[o], mm[o], r[o], Rk[o], rn[o], dd[o]
    n_tests = len(ii)
    good = (Rk > r) & (r > 0)
    ii, mm, r, Rk, rn, dd = ii[good], mm[good], r[good], Rk[good], rn[good], dd[good]
    score = -beta_dist.logcdf((r / Rk) ** dd, mm - 1, K_BG)
    # separation family: U1 = (r_(m-1) / r_m)^d ~ Beta(m - 1, 1), -log p = -(m - 1) d log(r_(m-1) / r_m)
    sep = np.where(rn > r, -(mm - 1) * dd * np.log(np.maximum(r, 1e-300) / np.maximum(rn, 1e-300)), 0.0)
    members = np.full((len(ii), MAX_M), -1, int)
    for m in range(2, MAX_M + 1):
        s = mm == m
        members[s, :m] = idx[ii[s], :m]
    return dict(R=R, is_rep=is_rep, collapsed=collapsed, dim=dim, rank=rank, dims=dims, n_tests=n_tests,
                centre=ii, m=mm, r=r, Rk=Rk, rn=rn, dd=dd, score=score, sep=sep, members=members,
                base=hdb(Z) if base is None else base)


def greedy(order_scores, ok, members):
    """Greedy, non-overlapping, highest score first among candidates with ok == True (stable order)."""
    sel = np.flatnonzero(ok)
    sel = sel[np.argsort(-order_scores[sel], kind="stable")]
    used, groups = set(), []
    for c in sel:
        mem = members[c][members[c] >= 0]
        s = set(mem.tolist())
        if used.isdisjoint(s):
            groups.append(c); used.update(s)
    return groups


def write(parts, groups, base=None):
    lab = (parts["base"] if base is None else base).copy()
    nxt = lab.max() + 1
    for c in groups:
        mem = parts["members"][c]
        lab[parts["R"][mem[mem >= 0]]] = nxt; nxt += 1
    return lab


def v5_labels(parts, alpha, base=None):
    thr = -np.log(alpha / max(1, parts["n_tests"]))
    return write(parts, greedy(parts["score"], parts["score"] >= thr, parts["members"]), base)


def sep_labels(parts, alpha, base=None, families=("ratio", "sep")):
    """Two exact families, Bonferroni over both: a candidate is significant if its ratio p or its separation p is below
    alpha / (F T). Greedy order: the larger of the two -log p. The order is free under Bonferroni by the
    free-selection theorem of the paper (Theorem 16 as submitted) only for a candidate family fixed before the
    scores are seen; T here is parts["n_tests"], the realised count over the reduced set the collapse leaves, so
    that premise does not hold as run."""
    F = len(families)
    thr = -np.log(alpha / (F * max(1, parts["n_tests"])))
    s = np.zeros(len(parts["score"]))
    if "ratio" in families:
        s = np.maximum(s, parts["score"])
    if "sep" in families:
        s = np.maximum(s, parts["sep"])
    return write(parts, greedy(s, s >= thr, parts["members"]), base)


def hdb3_base(Z):
    return M.hdbscan_labels(Z, 3, 3)


def cts_hdbvs(Z, alpha=0.05, families=("ratio",), base="hdb8"):
    """One-call entry point. families=("ratio",) is v5's scan; ("ratio", "sep") adds the exact separation family
    (stage 11 verdict: no gain beyond noise). base="hdb3" writes over HDBSCAN 3/3 instead of the default base: +0.02 to
    +0.07 recall on the bench, but that base is uncalibrated (about 126 small clusters per 1,600-point homogeneous
    Poisson scene at d = 2, against 24 for the default base), so it is an option, not the default."""
    b = hdb3_base(Z) if base == "hdb3" else None
    P = v5_parts(Z, base=b)
    return sep_labels(P, alpha, families=families)
