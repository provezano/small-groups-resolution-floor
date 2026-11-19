"""
Checks of the theory in. Each test checks one prediction.
T1 masking (HDBSCAN, ToMATo); T2 scan detection boundary on a torus (no edges); T3 UMAP attraction budget and recall by
group size (conjecture 8, nucleation size); T4 collapse ablation; T6 false groups under pure background (calibration);
T9 Good-Turing / Chao1 estimate of how many groups exist, from repeated subsamples.
Usage: python -m clustering.eval.nature.run_theory T1,T2,...   (default: all). Writes results/theory.json.
"""
import json
import math
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.stats import poisson
from sklearn.neighbors import NearestNeighbors

from .tiny_bench import make
from .tiny_methods import cts3, scan_only
from . import methods as M

OUT = os.path.join(os.path.dirname(__file__), *[".."] * 3, "results/theory.json")


def exact(lab, mg):
    for c in set(lab[mg]) - {-1}:
        if np.array_equal(lab == c, mg):
            return True
    return False


def jac(lab, mg):
    return max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0]) >= 0.5


# ---------------- T1 ----------------
def t1_scene(p, d, seed):
    import hdbscan
    from gudhi.clustering.tomato import Tomato
    X, kind, gid = make(p, d, seed, contrast="mid")
    rows = []
    for s in (3, 4, 6, 9):
        k = s                                            # hdbscan: core = distance to min_samples OTHER points
        dist = NearestNeighbors(n_neighbors=s + 1).fit(X).kneighbors(X)[0]  # (checked against its spanning tree)
        c = dist[:, -1]
        labs = {meth: hdbscan.HDBSCAN(min_cluster_size=2, min_samples=s, cluster_selection_method=meth,
                                      allow_single_cluster=False).fit(X).labels_ for meth in ("eom", "leaf")}
        t = Tomato(k=s, density_type="logDTM"); t.fit(X)
        tl = np.asarray(t.labels_)                       # no merging: the most clusters ToMATo can give
        w, nb = np.asarray(t.weights_), t.neighbors_
        nn = NearestNeighbors(radius=1).fit(X)
        for g in np.unique(gid[gid >= 0]):
            mg = gid == g
            mem = np.flatnonzero(mg)
            m = len(mem)
            masked = False
            if m <= k:
                for x in mem:
                    near = nn.radius_neighbors(X[x:x + 1], radius=c[x], return_distance=False)[0]
                    out = [o for o in near if not mg[o]]
                    if any(c[o] <= c[x] for o in out):
                        masked = True; break
            peak = any(all(w[j] <= w[x] for j in nb[x] if j != x) for x in mem)
            rows.append(dict(p=p, d=d, s=s, m=m, m_le_k=m <= k, masked=masked, peak=peak,
                             **{f"exact_{k_}": exact(l, mg) for k_, l in labs.items()},
                             **{f"jac_{k_}": jac(l, mg) for k_, l in labs.items()},
                             exact_tomato=exact(tl, mg), jac_tomato=jac(tl, mg)))
    return rows


def T1():
    rows = sum(Parallel(n_jobs=6)(delayed(t1_scene)(p, d, s) for p in ("isolated", "touching", "inside")
                                  for d in (2, 5, 10) for s in range(300, 303)), [])
    out = {}
    def rate(sel, key):
        v = [r[key] for r in rows if sel(r)]
        return (round(float(np.mean(v)), 3) if v else None, len(v))
    out["hdbscan"] = {
        "masked: exact recovery (theorem says 0)": {m: rate(lambda r: r["masked"], f"exact_{m}") for m in ("eom", "leaf")},
        "masked: Jaccard>=0.5 recovery": {m: rate(lambda r: r["masked"], f"jac_{m}") for m in ("eom", "leaf")},
        "not masked, m<=k: exact": {m: rate(lambda r: r["m_le_k"] and not r["masked"], f"exact_{m}") for m in ("eom", "leaf")},
        "m>k: exact": {m: rate(lambda r: not r["m_le_k"], f"exact_{m}") for m in ("eom", "leaf")},
        "share masked by placement": {p: rate(lambda r, p=p: r["p"] == p and r["m_le_k"], "masked") for p in ("isolated", "touching", "inside")},
    }
    out["tomato"] = {"no peak: exact (theorem says 0)": rate(lambda r: not r["peak"], "exact_tomato"),
                     "no peak: Jaccard>=0.5": rate(lambda r: not r["peak"], "jac_tomato"),
                     "has peak: exact": rate(lambda r: r["peak"], "exact_tomato"),
                     "share with no peak by placement": {p: rate(lambda r, p=p: r["p"] == p, "peak") for p in ("isolated", "touching", "inside")}}
    return out


# ---------------- T2 / T6 ----------------
def torus(n, d, rng):
    return rng.uniform(0, 1, (n, d))


def embed_torus(U):
    """T^d into R^2d (local distances ~ preserved, no edges), then into 20 dims by a random rotation."""
    E = np.concatenate([np.cos(2 * np.pi * U), np.sin(2 * np.pi * U)], 1) / (2 * np.pi)
    D = max(20, E.shape[1])
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(D, D)))[0][:, :E.shape[1]]
    return E @ Q.T


def rho_star(m, d, N, M=14, alpha=0.05):
    mu = brentq(lambda u: poisson.sf(m - 2, u) - alpha / (N * M), 1e-300, 100)
    return mu ** (1 / d)


def t2_one(m, d, mult, rep, N=1600):
    rng = np.random.default_rng(10_000 * m + 1000 * d + rep)
    vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    s = (N * vd) ** (-1 / d)
    rs = rho_star(m, d, N)
    r = min(mult * rs, 3.0) * s
    U = torus(N - m, d, rng)
    c = rng.uniform(0, 1, d)
    dirs = rng.normal(size=(m - 1, d)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    G = (c + dirs * r * rng.uniform(0, 1, (m - 1, 1)) ** (1 / d)) % 1.0
    X = embed_torus(np.vstack([U, c[None], G]))
    mg = np.r_[np.zeros(N - m, bool), np.ones(m, bool)]
    lab = scan_only(X, base=np.full(len(X), -1))
    found = jac(lab, mg)
    false = sum(1 for q in set(lab) - {-1} if not (mg[lab == q]).any())
    return dict(m=m, d=d, mult=mult, found=found, false=false)


def T2():
    rows = Parallel(n_jobs=6)(delayed(t2_one)(m, d, mult, rep) for m in (2, 3, 5, 9) for d in (2, 5, 10)
                              for mult in (0.25, 0.5, 1, 2, 4) for rep in range(20))
    out = {}
    for m in (2, 3, 5, 9):
        for d in (2, 5, 10):
            cur = {mult: float(np.mean([r["found"] for r in rows if r["m"] == m and r["d"] == d and r["mult"] == mult]))
                   for mult in (0.25, 0.5, 1, 2, 4)}
            out[f"m{m}/d{d}"] = {"rho_star": round(rho_star(m, d, 1600), 4), "detect_by_multiple_of_rho_star": cur}
    return out


def t6_one(d, rep, null, N=1600):
    rng = np.random.default_rng(777 + 100 * d + rep)
    X = embed_torus(torus(N, d, rng))
    lab = scan_only(X, base=np.full(N, -1), null=null)
    return d, null, len(set(lab) - {-1})


def T6():
    rows = Parallel(n_jobs=6)(delayed(t6_one)(d, rep, nl) for d in (2, 5, 10) for rep in range(30) for nl in ("poisson", "beta", "self"))
    return {f"{nl}/d{d}": {"mean_false_groups_per_scene": round(float(np.mean([f for dd, n_, f in rows if dd == d and n_ == nl])), 3),
                           "scenes_with_any": round(float(np.mean([f > 0 for dd, n_, f in rows if dd == d and n_ == nl])), 3),
                           "target": 0.05} for nl in ("poisson", "beta", "self") for d in (2, 5, 10)}


# ---------------- T3 ----------------
def t3_scene(p, seed):
    import umap
    from umap.umap_ import fuzzy_simplicial_set
    X, kind, gid = make(p, 5, seed, contrast="mid")
    rows = []
    for k in (5, 15, 30):
        G = fuzzy_simplicial_set(X, k, np.random.RandomState(0), "euclidean")[0].tocsr()
        Z = umap.UMAP(n_components=20, n_neighbors=k, min_dist=0.0, init="pca", random_state=seed, n_jobs=1).fit_transform(X)
        lab = M.hdbscan_labels(Z, 3, 3)
        for g in np.unique(gid[gid >= 0]):
            mem = np.flatnonzero(gid == g)
            shares = []
            for x in mem:
                row = G.getrow(x); idx, val = row.indices, row.data
                shares.append(val[np.isin(idx, mem)].sum() / max(val.sum(), 1e-12))
            rows.append(dict(p=p, k=k, m=len(mem), max_share=float(max(shares)), mean_share=float(np.mean(shares)),
                             bound=min(1.0, (len(mem) - 1) / math.log2(k)), found=jac(lab, gid == g)))
    return rows


def T3():
    rows = sum(Parallel(n_jobs=6)(delayed(t3_scene)(p, s) for p in ("isolated", "touching", "inside")
                                  for s in range(300, 303)), [])
    viol = [r for r in rows if r["max_share"] > r["bound"] + 1e-6]
    out = {"bound_violations": len(viol), "n_groups": len(rows),
           "worst_ratio_share_over_bound": round(max(r["max_share"] / r["bound"] for r in rows), 3)}
    buckets = {"2": (2, 2), "3": (3, 3), "4-5": (4, 5), "6-9": (6, 9)}
    for k in (5, 15, 30):
        out[f"k{k}"] = {b: {"recall": round(float(np.mean([r["found"] for r in rows if r["k"] == k and lo <= r["m"] <= hi])), 3),
                            "mean_internal_share": round(float(np.mean([r["mean_share"] for r in rows if r["k"] == k and lo <= r["m"] <= hi])), 3),
                            "bound": round(min(1.0, (lo - 1) / math.log2(k)), 2)}
                        for b, (lo, hi) in buckets.items()}
    return out


# ---------------- T4 ----------------
def t4_one(p, d, seed):
    from .tiny_bench import score
    X, kind, gid = make(p, d, seed, contrast="mid")
    return p, {"with_collapse": score(cts3(X), kind, gid), "no_collapse": score(scan_only(X), kind, gid)}


def T4():
    rows = Parallel(n_jobs=6)(delayed(t4_one)(p, d, s) for p in ("isolated", "touching", "inside")
                              for d in (2, 5, 10) for s in range(300, 305))
    return {p: {v: {k: round(float(np.mean([r[v][k] for pp, r in rows if pp == p])), 3) for k in ("recall", "false_per100", "mass_pieces")}
                for v in ("with_collapse", "no_collapse")} for p in ("isolated", "touching", "inside")}


# ---------------- T9 ----------------
def t9_one(p, seed, runs=10, frac=0.8):
    X, kind, gid = make(p, 5, seed, contrast="mid")
    rng = np.random.default_rng(seed)
    found_planted, found_any = [], []
    for _ in range(runs):
        sub = np.sort(rng.choice(len(X), int(frac * len(X)), replace=False))
        lab = np.full(len(X), -1)
        l = cts3(X[sub])
        base_max = M.hdbscan_labels(X[sub], max(8, len(sub) // 250), max(3, max(8, len(sub) // 250) // 2)).max()
        l = np.where(l > base_max, l, -1)                # keep only the scan's groups
        lab[sub] = l
        fp, fa = set(), []
        for q in set(lab) - {-1}:
            mem = frozenset(np.flatnonzero(lab == q))
            gg = [g for g in np.unique(gid[list(mem)]) if g >= 0]
            hit = [g for g in gg if len(mem & set(np.flatnonzero((gid == g)))) / len(mem | (set(np.flatnonzero(gid == g)) & set(sub))) >= 0.5]
            if hit:
                fp.add(hit[0])
            fa.append(mem)
        found_planted.append(fp); found_any.append(fa)
    # planted-only frequencies
    from collections import Counter
    cnt = Counter(g for s in found_planted for g in s)
    f1, f2 = sum(1 for v in cnt.values() if v == 1), sum(1 for v in cnt.values() if v == 2)
    chao = len(cnt) + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    # all groups, matched across runs by overlap
    reps = []
    for fa in found_any:
        for mem in fa:
            for r in reps:
                if len(mem & r[0]) / len(mem | r[0]) >= 0.5:
                    r[1] += 1; break
            else:
                reps.append([mem, 1])
    c_all = [r[1] for r in reps]
    g1, g2 = sum(1 for v in c_all if v == 1), sum(1 for v in c_all if v == 2)
    chao_all = len(c_all) + (g1 * g1 / (2 * g2) if g2 else g1 * (g1 - 1) / 2)
    per_run = np.mean([len(s) for s in found_planted])
    return p, dict(planted=12, found_per_run=float(per_run), found_any_run=len(cnt), chao1_planted=float(chao),
                   distinct_all=len(c_all), chao1_all=float(chao_all))


def T9():
    rows = Parallel(n_jobs=6)(delayed(t9_one)(p, s) for p in ("isolated", "touching", "inside") for s in range(300, 303))
    return {p: {k: round(float(np.mean([r[k] for pp, r in rows if pp == p])), 2) for k in rows[0][1]}
            for p in ("isolated", "touching", "inside")}


def main():
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["T1", "T2", "T3", "T4", "T6", "T9"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for w in which:
        res[w] = globals()[w]()
        print(f"\n== {w}\n" + json.dumps(res[w], indent=1))
        json.dump(res, open(OUT, "w"), indent=1); os.chmod(OUT, 0o600)



# ---------------- T10: stability selection (from T9's failure) ----------------
def t10_one(p, seed, runs=10, frac=0.8):
    X, kind, gid = make(p, 5, seed, contrast="mid")
    rng = np.random.default_rng(seed + 1)
    reps = []                                            # [members, count]
    for _ in range(runs):
        sub = np.sort(rng.choice(len(X), int(frac * len(X)), replace=False))
        base = M.hdbscan_labels(X[sub], max(8, len(sub) // 250), max(3, max(8, len(sub) // 250) // 2))
        l = cts3(X[sub])
        for q in set(l[l > base.max()]):
            mem = frozenset(sub[l == q])
            for r in reps:
                if len(mem & r[0]) / len(mem | r[0]) >= 0.5:
                    r[1] += 1; break
            else:
                reps.append([mem, 1])
    full = cts3(X)
    out = {}
    for t in (1, 3, 5, 7):
        kept = [r[0] for r in reps if r[1] >= t]
        planted = sum(1 for g in np.unique(gid[gid >= 0])
                      if any(len(k & set(np.flatnonzero(gid == g))) / len(k | set(np.flatnonzero(gid == g))) >= 0.5 for k in kept))
        false = sum(1 for k in kept if not (gid[list(k)] >= 0).any())
        out[f">={t}/{runs}"] = (planted / 12, 100 * false / len(X))
    return p, out


def T10():
    rows = Parallel(n_jobs=6)(delayed(t10_one)(p, s) for p in ("isolated", "touching", "inside") for s in range(300, 303))
    res = {}
    for p in ("isolated", "touching", "inside"):
        rs = [r for pp, r in rows if pp == p]
        res[p] = {t: {"recall": round(float(np.mean([r[t][0] for r in rs])), 3), "false_per100": round(float(np.mean([r[t][1] for r in rs])), 3)}
                  for t in rs[0]}
    return res


if __name__ == "__main__":
    main()
