"""
Ground truth for the tiny-group benchmark (tiny_bench.make): the generator's parameters, its true background intensity,
and the oracle scan, i.e. the local scan told the true background density. The oracle gives each tier a ceiling: a
method that knows nothing about the background cannot be expected to beat a local scan that knows it exactly.

- params(placement, d, seed, contrast): replays make()'s random stream and returns the background components
  (weight, mean, sd) in latent space, the uniform-noise box, the groups (m, centre, sd) and the latent-to-ambient map Q.
- log_intensity(x, ...): log of the true background intensity (expected points per unit volume) in latent space.
- oracle_labels(X, placement, d, seed, contrast, alphas): the scan over each point and its m - 1 nearest neighbours,
  with the count in the ball compared with a large sample of the true background (40x): exact two-sample tail
  (default since stage 15; tail="plugin" = the old Poisson plug-in), Bonferroni over all tests, greedy
  non-overlapping selection; groups written over HDBSCAN's labels.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
from sklearn.neighbors import NearestNeighbors

from .tiny_bench import CONTRAST, D_AMBIENT
from .tiny_methods import hdb


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


def log_intensity(x, pr):
    d, (nn, lo, hi) = pr["d"], pr["box"]
    terms = [np.log(w) - d * np.log(sd) - d / 2 * np.log(2 * np.pi) - ((x - c) ** 2).sum(1) / (2 * sd ** 2)
             for w, c, sd in pr["comps"]]
    inside = np.all((x >= lo) & (x <= hi), 1)
    terms.append(np.where(inside, np.log(nn) - np.log(hi - lo).sum(), -np.inf))
    return logsumexp(np.vstack(terms), axis=0)


def background_sample(pr, mult=40, seed=999):
    """mult x the background, drawn from the true model and mapped to ambient space like the data."""
    r = np.random.default_rng(seed)
    nn, lo, hi = pr["box"]
    R = [c + r.normal(0, sd, (w * mult, pr["d"])) for w, c, sd in pr["comps"]]
    R.append(r.uniform(lo, hi, (nn * mult, pr["d"])))
    return np.vstack(R) @ pr["Q"].T, sum(w for w, _, _ in pr["comps"]) + nn


def oracle_labels(X, placement, d, seed, contrast, alphas=(0.05,), max_m=15, mult=40, base=None, tail="exact"):
    """tail="exact" (default since stage 15): the exact two-sample tail of two_scan.hyp_logsf with the 40x background
    sample as the reference A and the scene as B. With j = m - 1 other scene points in the ball and a reference points
    strictly inside its radius, p = P(X >= j), X ~ Hyp(N - 1, n_B - 1, a + j), N = n_A + n_B. This is the stage-14
    two-sample scan on a known-background reference, valid at every level conditional on the pooled points.
    tail="plugin": the rounds 11-13 oracle exactly (Poisson tail at lambda = reference count / mult, floored at
    1e-12). It is anti-conservative where the reference ball is empty: a pair gets -log p of about 27.6 nats against
    an exact p of 1/(1 + mult) (stage-14 two report, Section 6), so its ceilings are not used any more."""
    if tail not in ("exact", "plugin"):
        raise ValueError("tail must be 'exact' or 'plugin'")
    pr = params(placement, d, seed, contrast)
    R, nbg = background_sample(pr, mult)
    base = hdb(X) if base is None else base
    dist, idx = NearestNeighbors(n_neighbors=max_m).fit(X).kneighbors(X)
    ref = NearestNeighbors().fit(R)
    if tail == "exact":
        return oracle_labels_ref(X, R, alphas, max_m, base, dist_idx=(dist, idx), ref_nn=ref)
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


def oracle_labels_ref(X, R, alphas=(0.05,), max_m=15, base=None, dist_idx=None, ref_nn=None):
    """The exact-tail oracle scan with any reference sample R of the true background (oracle_labels' tail="exact",
    for generators whose reference comes from elsewhere, e.g. an external generator whose own
    frozen oracle uses the plug-in tail). Candidates: each point and its m - 1 nearest scene points; p from
    two_scan.hyp_logsf; Bonferroni over all candidates; greedy disjoint selection over `base` (None = HDBSCAN)."""
    from .two_scan import hyp_logsf
    base = hdb(X) if base is None else base
    dist, idx = dist_idx if dist_idx is not None else NearestNeighbors(n_neighbors=max_m).fit(X).kneighbors(X)
    ref = ref_nn if ref_nn is not None else NearestNeighbors().fit(R)
    nA, nB = len(R), len(X)
    js, ts, mems = [], [], []
    for i in range(len(X)):
        dd = np.sort(ref.radius_neighbors(X[i:i + 1], radius=dist[i, -1], return_distance=True)[0][0])
        for m in range(2, max_m + 1):
            a = int(np.searchsorted(dd, dist[i, m - 1], side="left"))
            js.append(m - 1); ts.append(a + m - 1); mems.append(tuple(sorted(idx[i, :m])))
    lp = hyp_logsf(np.array(js), nA + nB - 1, nB - 1, np.array(ts)) if js else np.zeros(0)
    cands = sorted([(-float(v), mem) for v, mem in zip(lp, mems)], key=lambda t: -t[0])
    n_tests = len(cands)
    labs = []
    for alpha in alphas:
        thr = -np.log(alpha / n_tests)
        used, lab, nxt = set(), base.copy(), base.max() + 1
        for sc, mem in cands:
            if sc < thr:
                break
            if used.isdisjoint(mem):
                lab[list(mem)] = nxt; nxt += 1; used.update(mem)
        labs.append(lab)
    return labs
