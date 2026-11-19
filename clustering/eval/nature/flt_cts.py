"""
CTS with independent filtering: weighted Bonferroni whose weights come from statistics that are independent of each
candidate's Beta ratio under the local Poisson null (stage 12,.

Why it is valid (Result F1 in the doc). Around a centre x, the ball B(x, r_n), n = m - 1 + k, is a stopping set of the
point process. Under a homogeneous Poisson null (any intensity), given r_n the n - 1 points inside are i.i.d. uniform
in the ball and the points outside form an independent Poisson process. The Beta ratio U = (r_(m-1) / r_n)^d depends
only on the inside points' normalised radii, whose law does not involve r_n. So U is independent of r_n and of every
function of the outside configuration: the ring radius itself, radii further out, the density beyond the ring.

The weights. w_i = (1 - eta) + eta * T * 1{candidate i passes the filter} / T_f, with T the number of tests and T_f the
number that pass. Sum_i w_i = T exactly, so weighted Bonferroni (reject when p_i <= alpha w_i / T) keeps the family-wise
error at alpha whenever each null p_i is independent of its own weight w_i (Genovese, Roeder & Wasserman 2006). It is
exact for the filter indicator; T_f depends weakly on the inside points of i through neighbouring candidates, so the
guarantee for the data-normalised version is approximate (checked by simulation on homogeneous and bench nulls).

Filters (all use no labels):
- "ring": the candidate's ring radius r_n is among the largest q share of ring radii of the same size m (local sparsity
  at the candidate's own scale, relative to the scene).
- "far": the same with the radius to the farthest neighbour computed (index kk = 55), i.e. the density further out.
- "sfhi": the upper tail of the same statistic (the density rises beyond the ring: the candidate sits in a local void),
  ranked like "ring" (top q share, normalised by the count that passes).
- "sf": a scale-free outer statistic, (r_n / r_kk)^d ~ Beta(n, kk - n) exactly under the null; pass if its CDF <= q.
  Pivotal, so its pass rate is exactly q under the null with no normalisation (an exactly valid filter).
Selection among significant candidates is greedy by score, as in v5 (free under any Bonferroni-type rule, Theorem 2a).
"""
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.neighbors import NearestNeighbors

from . import hdbvs_cts as H

MAX_M, K_BG, N_LOC = H.MAX_M, H.K_BG, H.N_LOC
KS = (3, 5, 10, 20, 30)


def _rank_frac(x, m):
    """For each candidate, the share of same-size candidates whose x is >= its x (1/n_m for the largest)."""
    q = np.empty(len(x))
    for mm in np.unique(m):
        s = np.flatnonzero(m == mm)
        o = np.argsort(-x[s], kind="stable")
        r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        q[s] = r / len(s)
    return q


def parts(Z, base=None):
    """v5's collapse + scan (hdbvs_cts.v5_parts) plus the filter statistics for every candidate."""
    P = H.v5_parts(Z, base=base)
    Zr = Z[P["R"]]
    kk = min(len(Zr) - 1, MAX_M + K_BG + N_LOC)
    dist = NearestNeighbors(n_neighbors=kk + 1).fit(Zr).kneighbors(Zr)[0]
    c, m = P["centre"], P["m"]
    far = dist[c, kk]
    P["kk"] = kk
    P["far"] = far
    P["q_ring"] = _rank_frac(np.log(P["Rk"]), m)
    P["q_far"] = _rank_frac(np.log(np.maximum(far, 1e-300)), m)
    n = m - 1 + K_BG
    ok = far > P["Rk"]
    u = np.where(ok, (P["Rk"] / np.maximum(far, 1e-300)) ** P["dd"], 1.0)
    P["p_sf"] = np.where(ok & (kk > n), beta_dist.cdf(u, n, np.maximum(kk - n, 1)), 1.0)
    # upper tail: density rises beyond the ring (the candidate sits in a local void). log sf avoids ties at 1.0
    lsf = np.where(ok & (kk > n), beta_dist.logsf(u, n, np.maximum(kk - n, 1)), 0.0)
    P["q_sfhi"] = _rank_frac(-lsf, m)
    # Cauchy combination over ring sizes k (Liu & Xie 2020), same candidate, same plug-in d
    r = P["r"]
    T_c = np.zeros(len(m)); nk = np.zeros(len(m))
    for k in KS:
        j = m - 1 + k
        okk = j <= kk
        Rk = np.where(okk, dist[c, np.minimum(j, kk)], np.inf)
        g = okk & (Rk > r)
        pk = np.clip(beta_dist.cdf(np.where(g, (r / np.where(g, Rk, 1.0)) ** P["dd"], 1.0), m - 1, k), 1e-300, 1 - 1e-12)
        T_c += np.where(g, np.tan((0.5 - pk) * np.pi), 0.0); nk += g
    T_c = T_c / np.maximum(nk, 1)
    # -log of the Cauchy p-value; for large T_c use the tail 1 / (pi T_c) to avoid cancellation
    big = T_c > 1e6
    pc = np.where(big, 1 / (np.pi * np.maximum(T_c, 1e6)), 0.5 - np.arctan(T_c) / np.pi)
    P["score_c"] = -np.log(np.clip(pc, 1e-300, 1.0))
    return P


def weights(P, q=0.02, eta=0.5, stat="ring", size_w=None):
    """Per-candidate Bonferroni weights (mean 1 over the n_tests tests)."""
    T = max(1, P["n_tests"])
    if stat == "sf":
        passed = P["p_sf"] <= q
        tf = q * T                                        # exact null pass rate, no normalisation needed
    else:
        passed = P["q_" + stat] <= q
        tf = max(1, int(passed.sum()))
    w = (1 - eta) + eta * passed * (T / tf)
    if size_w is not None:
        w = w * np.asarray([size_w.get(int(mm), 1.0) for mm in P["m"]])
    return w


def size_weights(P, prior):
    """Data-independent prior over sizes, normalised so that the weights average to 1 over this scene's tests."""
    m = P["m"]
    raw = np.array([prior.get(int(mm), 1.0) for mm in m])
    return {int(mm): float(prior.get(int(mm), 1.0) * len(m) / raw.sum()) for mm in np.unique(m)}


def flt_labels(P, alpha, q=0.02, eta=0.5, stat="ring", base=None, size_prior=None, return_groups=False,
               score="beta", c=1.0, guide=None):
    """stat=None: no filter (plain Bonferroni). score="cauchy": Cauchy combination over k instead of k = 10.
    c: multiplier on the family size (c < 1 is the Monte Carlo max-score correction, T_eff = c T)."""
    sw = size_weights(P, size_prior) if size_prior else None
    w = np.ones(len(P["m"])) if stat is None else weights(P, q, eta, stat, sw)
    if stat is None and sw is not None:
        w = w * np.asarray([sw.get(int(mm), 1.0) for mm in P["m"]])
    T = max(1, P["n_tests"]) * c
    thr = -np.log(alpha * w / T)
    S = P["score_c"] if score == "cauchy" else P["score"]
    ok = S >= thr
    key = S - thr
    if guide is not None:
        # calibrated use of an HDBSCAN 3/3 tree: among significant candidates only, take first those that match one
        # of its small clusters (Jaccard >= 0.5). Valid: selection among rejections is free (Theorem 2a).
        key = key.copy()
        R = P["R"]
        for i in np.flatnonzero(ok):
            mem = R[P["members"][i][P["members"][i] >= 0]]
            labs = guide[mem]
            lab = np.bincount(labs[labs >= 0]).argmax() if (labs >= 0).any() else -1
            if lab >= 0:
                cl = np.flatnonzero(guide == lab)
                if len(cl) <= 15 and len(np.intersect1d(cl, mem)) / len(np.union1d(cl, mem)) >= 0.5:
                    key[i] += 1e6
    g = H.greedy(key, ok, P["members"])
    return g if return_groups else H.write(P, g, base)


def cts_flt(Z, alpha=0.05, q=0.02, eta=0.5, stat="ring"):
    """One-call entry point: v5 scan with sparsity-filtered weighted Bonferroni."""
    return flt_labels(parts(Z), alpha, q, eta, stat)
