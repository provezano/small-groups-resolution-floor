"""
Oracle scan (the per-tier ceiling), vendored from clustering.eval.nature.tiny_oracle (same project).

The local scan over each point and its m - 1 nearest neighbours (m = 2..15), told the TRUE background density: the
expected count in each ball is read from a 40x sample of the true background model, Poisson tail p-value, Bonferroni
over all tests, greedy non-overlapping selection, groups written over the HDBSCAN base. It is a local-scan ceiling,
not an absolute limit (on inside/low a method that collapses the near-duplicate mass can beat it).
Only defined for shape="iso" scenes (the background model is shape-independent, so it also runs on other shapes, but
the "ceiling" interpretation is for the isotropic scan).
"""
import numpy as np
from scipy.stats import poisson
from sklearn.neighbors import NearestNeighbors

from .base import hdb
from .generator import CONTRAST, D_AMBIENT


def params(placement, d=5, seed=0, contrast="mid", n_groups=12, tpl=30, copies=30, copy_sd=0.02):
    rng = np.random.default_rng(seed)
    comps, P = [], []
    tc = rng.normal(0, 1.0, (tpl, d))
    for c in tc:
        P.append(c + rng.normal(0, copy_sd, (copies, d))); comps.append((copies, c, copy_sd))
    bc = []
    for _ in range(8):
        while True:
            c = rng.normal(0, 6.0, d)
            if np.linalg.norm(c) > 4 and all(np.linalg.norm(c - b[0]) > 3 for b in bc):
                break
        bc.append(c)
        m, sd = int(rng.integers(30, 151)), float(rng.uniform(0.3, 0.8))
        P.append(c + rng.normal(0, sd, (m, d))); bc[-1] = (c, sd); comps.append((m, c, sd))
    allp = np.vstack(P)
    lo, hi = allp.min(0) - 2, allp.max(0) + 2
    nn = int(0.02 * len(allp))
    rng.uniform(lo, hi, (nn, d))
    groups = []
    for _ in range(n_groups):
        m = int(rng.integers(2, 10))
        if placement == "isolated":
            while True:
                c = rng.uniform(lo, hi)
                if np.min(np.linalg.norm(allp - c, axis=1)) > 2.0:
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
        rng.normal(0, sd, (m, d))
        groups.append((m, c, sd))
    Q = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_AMBIENT)))[0][:, :d]
    return {"comps": comps, "box": (nn, lo, hi), "groups": groups, "Q": Q, "d": d}


def background_sample(pr, mult=40, seed=999):
    r = np.random.default_rng(seed)
    nn, lo, hi = pr["box"]
    R = [c + r.normal(0, sd, (w * mult, pr["d"])) for w, c, sd in pr["comps"]]
    R.append(r.uniform(lo, hi, (nn * mult, pr["d"])))
    return np.vstack(R) @ pr["Q"].T, sum(w for w, _, _ in pr["comps"]) + nn


def oracle_labels(X, placement, d, seed, contrast, alphas=(0.05,), max_m=15, mult=40, base=None):
    pr = params(placement, d, seed, contrast)
    R, nbg = background_sample(pr, mult)
    base = hdb(X) if base is None else base
    dist, idx = NearestNeighbors(n_neighbors=max_m).fit(X).kneighbors(X)
    ref = NearestNeighbors().fit(R)
    cands = []
    for i in range(len(X)):
        dd = np.sort(ref.radius_neighbors(X[i:i + 1], radius=dist[i, -1], return_distance=True)[0][0])
        for m in range(2, max_m + 1):
            lam = np.searchsorted(dd, dist[i, m - 1], side="right") / mult * (nbg - 1) / nbg
            cands.append((-poisson.logsf(m - 2, max(lam, 1e-12)), tuple(sorted(idx[i, :m]))))
    n_tests = len(cands)
    cands.sort(key=lambda t: -t[0])
    labs = []
    for alpha in alphas:
        thr = -np.log(alpha / n_tests)
        used, lab, nxt = set(), base.copy(), base.max() + 1
        for s, mem in cands:
            if s < thr:
                break
            if used.isdisjoint(mem):
                lab[list(mem)] = nxt; nxt += 1; used.update(mem)
        labs.append(lab)
    return labs
