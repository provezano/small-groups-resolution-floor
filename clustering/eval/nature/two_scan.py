"""
Two-sample tiny-group scan (stage 14, branch `two`;.

Question: which tiny groups (2-15 points) are new in a sample B compared with a reference sample A drawn from the same
background? Under H0 (A and B i.i.d. from one law) the labels of the pooled points are exchangeable, so a scan that
only counts labels has an exact null whatever the background's shape.

Scan (`two_cands`). Every B point x is a candidate centre. For j = 1..max_m-1, the candidate group is x and its j
nearest B points; its ball has radius r_j = distance to the j-th B neighbour; a = number of A points closer than r_j.
Walking outward from x over the pooled points, a is the number of A points met before the j-th B point, which is
negative hypergeometric under H0 given the pooled set and x in B. Its exact lower tail is
    p = P(A-count <= a) = P(Hyp(N - 1, n_B - 1, a + j) >= j),
because "at most a A's before the j-th B" is "at least j B's among the first a + j draws". Bonferroni over n_B x
(max_m - 1) tests bounds the family-wise error (the union over centres costs n_B, not N: sum over pooled points of
P(x in B) P(p <= t | x in B) = n_B t). Greedy non-overlapping selection by score, as in CTS.

Baselines here (same scene, same scoring):
- `milo_simple`: Milo-style differential abundance (Dann et al. 2022) without replicates: pooled kNN graph, refined
  index sampling (prop 0.1), neighbourhood = index + its k neighbours, one-sided exact test of B enrichment, Milo's
  graph spatial FDR (weighted BH, weights 1 / k-th neighbour distance); significant neighbourhoods sharing B points are
  merged and their B points reported as one group.
- `c2st_local`: classifier two-sample test made local: out-of-fold extra-trees P(B) for each B point; the top q of
  B points are grouped like LOF + group (tiny_methods._group_outliers). No error guarantee.
- one-sample on B alone: CTS v5 + fringe (tail_cts "full"), GapClust k30, LOF + group, and the known-background oracle.

  python -m clustering.eval.nature.two_scan scenes --seeds 620-624 --out-dir runs/two/tune
  python -m clustering.eval.nature.two_scan nulls  --seeds 620000-620029 --out-dir runs/two/null_tune
  python -m clustering.eval.nature.two_scan merge  --out-dir ... --out ../results/two_x.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gammaln, logsumexp
from scipy.stats import hypergeom

from .tiny_bench import D_AMBIENT, TIERS, make

MAX_M = 15
RATIOS = (1, 5, 40)                     # reference size n_A = r x (background size of B)
ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3, 1.0, 3.0, 10.0)
Q6 = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
MILO_Q = (1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2)
GAP_THR = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
NULL_ALPHAS = (0.05, 0.3, 1.0)
CONTRASTS = ("high", "mid", "low")
DIMS = (2, 5, 10)


# ---------------------------------------------------------------------------------------------------------------- data
def background_draw(pr, n, rng, noise=0.002):
    """n i.i.d. draws from the bench background mixture (component shares = make()'s counts, plus the 2% noise box),
    mapped to ambient space like make() (same Q, same ambient noise). Rows are in random order, so any prefix is
    itself an i.i.d. sample."""
    comps = pr["comps"]
    nn, lo, hi = pr["box"]
    d = pr["d"]
    w = np.array([c[0] for c in comps] + [nn], float)
    cnt = rng.multinomial(n, w / w.sum())
    P = [c + rng.normal(0, sd, (k, d)) for (_, c, sd), k in zip(comps, cnt[:-1])]
    P.append(rng.uniform(lo, hi, (cnt[-1], d)))
    Z = np.vstack(P)
    Z = Z[rng.permutation(len(Z))]
    return Z @ pr["Q"].T + rng.normal(0, noise, (n, D_AMBIENT))


def make_pair(placement, d, seed, contrast, r_max=max(RATIOS)):
    """B = tiny_bench.make(...) exactly (same scenes as every other branch); A = r_max x (B's background size) i.i.d.
    background draws from the same generator parameters, with its own random stream."""
    from .tiny_oracle import params
    B, kind, gid = make(placement, d, seed, contrast=contrast)
    pr = params(placement, d, seed, contrast)
    n_bg = int((kind != "group").sum())
    A = background_draw(pr, r_max * n_bg, np.random.default_rng([seed, 7, d, CONTRASTS.index(contrast),
                                                                  TIERS.index(placement)]))
    return A, B, kind, gid, n_bg


# ----------------------------------------------------------------------------------------------------- exact scan
def hyp_logsf(j, M, n, t):
    """log P(X >= j), X ~ Hyp(M total, n successes, t draws); vectorised over arrays j, t (M, n scalars).
    Summed exactly in log space over the upper support (scipy's logsf loses precision far in the tail)."""
    j, t = np.broadcast_arrays(np.asarray(j, np.int64), np.asarray(t, np.int64))
    out = np.empty(j.shape)
    flat_j, flat_t, flat_o = j.ravel(), t.ravel(), out.ravel()
    lc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    for u in np.unique(np.stack([flat_j, flat_t], 1), axis=0):
        jj, tt = int(u[0]), int(u[1])
        hi = min(tt, n)
        sel = (flat_j == jj) & (flat_t == tt)
        if jj > hi:
            flat_o[sel] = -np.inf
            continue
        if jj <= max(0, tt - (M - n)):
            flat_o[sel] = 0.0
            continue
        x = np.arange(jj, hi + 1)
        flat_o[sel] = min(0.0, float(logsumexp(lc(n, x) + lc(M - n, tt - x) - lc(M, tt))))
    return out


def two_cands(A, B, max_m=MAX_M, treeA=None):
    """Every candidate (centre i in B, j = 1..max_m-1 other B points). Returns dict with score (-log p), centre, m,
    the B-neighbour index matrix and n_tests."""
    nB, nA = len(B), len(A)
    N = nA + nB
    kq = min(max_m, nB)
    dB, iB = cKDTree(B).query(B, k=kq)
    tA = cKDTree(A) if treeA is None else treeA
    J = kq - 1
    a = np.zeros((nB, J), np.int64)
    if nA:
        for j in range(1, kq):
            a[:, j - 1] = tA.query_ball_point(B, np.nextafter(dB[:, j], 0), return_length=True)
    jj = np.broadcast_to(np.arange(1, kq), (nB, J))
    lp = hyp_logsf(jj, N - 1, nB - 1, a + jj) if nA else np.zeros((nB, J))
    return {"score": -lp.ravel(), "centre": np.repeat(np.arange(nB), J), "m": np.tile(np.arange(2, kq + 1), nB),
            "I": iB, "a": a.ravel(), "n_tests": nB * J, "n": nB}


def two_groups(C, alpha):
    thr = np.log(C["n_tests"] / alpha)
    sig = np.flatnonzero(C["score"] >= thr)
    order = sig[np.lexsort((C["m"][sig], -C["score"][sig]))]
    used, groups = set(), []
    for o in order:
        mem = C["I"][C["centre"][o], :C["m"][o]].tolist()
        if used.isdisjoint(mem):
            groups.append(np.sort(mem)); used.update(mem)
    return groups


def to_labels(groups, n, base=None):
    lab = -np.ones(n, int) if base is None else base.copy()
    nxt = lab.max() + 1
    for g in groups:
        lab[np.asarray(g)] = nxt; nxt += 1
    return lab


def union_groups(first, second, rule=None):
    """first's groups, then second's groups that do not overlap them (stage 14's rule; the default).
    rule (stage 15 hook): a callable rule(first, second) -> list of groups replaces it, e.g. an overlap rule by
    adjusted p from union_rules; None = unchanged."""
    if rule is not None:
        return list(rule(first, second))
    used = set(int(v) for g in first for v in g)
    out = list(first)
    for g in second:
        if used.isdisjoint(g.tolist()):
            out.append(g); used.update(g.tolist())
    return out


def union_cover(first, second, second_scores=None, cover=0.5):
    """Stage-15 overlap rule, variant V6 with tarone_threshold).
    first = CTS groups, second = scan groups. Each scan group, by decreasing score, replaces the CTS groups it overlaps
    iff they cover less than `cover` of it, and is dropped otherwise. Reports only rejected groups, so the error bound
    of the two thresholds holds for any such rule (Result U1). Use as union_groups(first, second, rule=union_cover)."""
    kept = [(np.asarray(g), 1) for g in first]
    order = range(len(second)) if second_scores is None else np.argsort(-np.asarray(second_scores), kind="stable")
    for i in order:
        g = np.asarray(second[i]); gs = set(g.tolist())
        ov = [k for k, (h, _) in enumerate(kept) if not gs.isdisjoint(h.tolist())]
        if not ov:
            kept.append((g, 2)); continue
        cov = len(gs & set().union(*[set(kept[k][0].tolist()) for k in ov])) / len(gs)
        if cov < cover:
            kept = [x for k, x in enumerate(kept) if k not in ov] + [(g, 2)]
    return [g for g, _ in kept]


def tarone_threshold(n_cts_tests, C, n_A, alpha):
    """Common rejection threshold (nats) for CTS and the scan (Tarone 1990): k = smallest value with
    n_cts + n_B * #{j : pi_j <= alpha / k} <= k, where pi_j is the scan's smallest p-value at size j + 1 (a = 0).
    Scan sizes that cannot reach alpha / k take no share of alpha; at r = 1, alpha 0.05, CTS gets the whole alpha.
    Replaces the alpha / 2 split: CTS at -log p >= thr, scan at score >= thr."""
    nB = C["n"]
    J = C["n_tests"] // max(nB, 1)
    j = np.arange(1, J + 1)
    lp = np.sort(hyp_logsf(j, n_A + nB - 1, nB - 1, j))
    best = np.inf
    for c in range(J + 1):
        k = float(n_cts_tests + nB * c)
        if c < J:
            k = max(k, float(np.exp(np.log(alpha) - lp[c])) * (1 + 1e-9))
        best = min(best, k)
    return float(np.log(best / alpha))

# ------------------------------------------------------------------------------------------------------ baselines
def milo_simple(A, B, k=10, prop=0.1, seed=0, qs=MILO_Q):
    """Milo-style DA (no replicates): list of B-index groups per q in qs."""
    nB = len(B)
    Z = np.vstack([B, A])
    isB = np.zeros(len(Z), bool); isB[:nB] = True
    N = len(Z)
    tree = cKDTree(Z)
    dist, nbr = tree.query(Z, k=k + 1)
    rng = np.random.default_rng(seed)
    samp = rng.choice(N, max(1, int(prop * N)), replace=False)
    med = np.median(Z[nbr[samp]], axis=1)                          # refined sampling (Milo "refined")
    idx = np.unique(tree.query(med, k=1)[1])
    nb = nbr[idx]
    b = isB[nb].sum(1)
    lp = hyp_logsf(b, N, nB, np.full(len(b), k + 1))
    w = 1.0 / np.maximum(dist[idx, -1], 1e-12)
    o = np.argsort(lp)
    padj = np.exp(lp[o]) * w.sum() / np.cumsum(w[o])               # Milo graphSpatialFDR (weighted BH)
    padj = np.minimum.accumulate(padj[::-1])[::-1]
    out = []
    for q in qs:
        sig = o[(padj <= q) & (b[o] >= 2)]
        out.append(_merge_hoods([nb[s][isB[nb[s]]] for s in sig], nB))
    return out


def _merge_hoods(hoods, nB):
    if not hoods:
        return []
    par = np.arange(nB)

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for h in hoods:
        r0 = f(h[0])
        for v in h[1:]:
            par[f(v)] = r0
    pts = np.unique(np.concatenate(hoods))
    roots = np.array([f(v) for v in pts])
    return [np.sort(pts[roots == r]) for r in np.unique(roots)]


def c2st_scores(A, B, seed=0, n_trees=100, folds=5):
    """Out-of-fold extremely-randomised-trees P(B) for every B point (classifier two-sample test, made local; fully
    grown trees, so leaves can isolate a tiny group). ExtraTrees instead of a random forest for runtime (2x faster)."""
    from sklearn.ensemble import ExtraTreesClassifier as RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    Z = np.vstack([B, A]); y = np.r_[np.ones(len(B)), np.zeros(len(A))]
    s = np.zeros(len(B))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Z, y):
        clf = RandomForestClassifier(n_trees, max_features="sqrt", class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=1).fit(Z[tr], y[tr])
        teB = te[te < len(B)]
        s[teB] = clf.predict_proba(Z[teB])[:, 1]
    return s


def top_q_groups(B, s, base, q):
    from .tiny_methods import _group_outliers
    idx = np.flatnonzero(s >= np.quantile(s, 1 - q))
    return _group_outliers(B, idx, base)


# ---------------------------------------------------------------------------------------------------------- runner
def _hits(lab, gid):
    lab = np.asarray(lab)
    out = []
    for g in np.unique(gid[gid >= 0]):
        mg = gid == g
        best = max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0])
        out.append(int(best >= 0.5))
    return out


def _sc(lab, kind, gid):
    from .tiny_bench import score
    s = score(lab, kind, gid)
    s["hits"] = _hits(lab, gid)
    return s


def scene(p, c, d, s, ratios=RATIOS, one_sample=True):
    from .tiny_methods import hdb, scored
    A40, B, kind, gid, n_bg = make_pair(p, d, s, c)
    base = hdb(B)
    out, secs = {}, {}
    cts_g = None
    if one_sample:
        from . import tail_cts as T
        from .tiny_baselines import gapclust_sweep
        from .tiny_oracle import oracle_labels
        t = time.time(); P = T.tail_parts(B, omegas=(), tilt=False)
        out["cts_v5f"] = [_sc(T.tail_labels(P, a, "full", base=base), kind, gid) for a in ALPHAS]
        cts_g = {a: T.tail_groups(P, a / 2, "full") for a in ALPHAS}
        secs["cts"] = round(time.time() - t, 2)
        t = time.time(); out["gapclust_k30"] = [_sc(l, kind, gid) for l in gapclust_sweep(B, 30)]
        secs["gapclust"] = round(time.time() - t, 2)
        t = time.time(); out["lof+group"] = [_sc(scored("lof", q)(B), kind, gid) for q in Q6]
        secs["lof"] = round(time.time() - t, 2)
        t = time.time(); out["oracle"] = [_sc(l, kind, gid) for l in oracle_labels(B, p, d, s, c, ALPHAS, base=base)]
        secs["oracle"] = round(time.time() - t, 2)
    for r in ratios:
        A = A40[:r * n_bg]
        t = time.time(); C = two_cands(A, B)
        out[f"two_r{r}"] = [_sc(to_labels(two_groups(C, a), len(B), base), kind, gid) for a in ALPHAS]
        if cts_g is not None:
            out[f"two+cts_r{r}"] = [_sc(to_labels(union_groups(cts_g[a], two_groups(C, a / 2)), len(B), base),
                                        kind, gid) for a in ALPHAS]
        secs[f"two_r{r}"] = round(time.time() - t, 2)
        for k in (10, 30):
            t = time.time()
            out[f"milo_k{k}_r{r}"] = [_sc(to_labels(g, len(B), base), kind, gid) for g in milo_simple(A, B, k=k)]
            secs[f"milo_k{k}_r{r}"] = round(time.time() - t, 2)
        t = time.time(); sc = c2st_scores(A, B)
        out[f"c2st_r{r}"] = [_sc(top_q_groups(B, sc, base, q), kind, gid) for q in Q6]
        secs[f"c2st_r{r}"] = round(time.time() - t, 2)
    sizes = [int((gid == g).sum()) for g in np.unique(gid[gid >= 0])]
    return {"placement": p, "contrast": c, "d": d, "seed": s, "n_B": len(B), "n_bg": n_bg, "sizes": sizes,
            "scores": out, "secs": secs}


def grids(ratios=RATIOS):
    g = {"cts_v5f": list(ALPHAS), "gapclust_k30": list(GAP_THR), "lof+group": list(Q6), "oracle": list(ALPHAS)}
    for r in ratios:
        g[f"two_r{r}"] = list(ALPHAS); g[f"two+cts_r{r}"] = list(ALPHAS)
        g[f"milo_k10_r{r}"] = list(MILO_Q); g[f"milo_k30_r{r}"] = list(MILO_Q); g[f"c2st_r{r}"] = list(Q6)
    return g


def null_scene(kind, d, s, ratios=RATIOS):
    """A-vs-A nulls. kind 'iid': B and A both i.i.d. from the background mixture (the exact setting of the theorem);
    'bench': B = a bench scene's background exactly as make() draws it (fixed component counts, groups removed), A
    i.i.d. Counts every reported group (all false): two-sample scan at NULL_ALPHAS and at the full grid, Milo and
    the classifier at their grids, CTS on B alone at NULL_ALPHAS."""
    from .tiny_oracle import params
    pr = params("isolated", d, s, "mid")
    B0, k0, _ = make("isolated", d, s, contrast="mid")
    n_bg = int((k0 != "group").sum())
    rngA = np.random.default_rng([s, 11, d])
    B = B0[k0 != "group"] if kind == "bench" else background_draw(pr, n_bg, np.random.default_rng([s, 13, d]))
    A40 = background_draw(pr, max(ratios) * n_bg, rngA)
    out = {}
    from . import tail_cts as T
    P = T.tail_parts(B, omegas=(), tilt=False)
    out["cts_v5f"] = [len(T.tail_groups(P, a, "full")) for a in NULL_ALPHAS]
    for r in ratios:
        A = A40[:r * n_bg]
        C = two_cands(A, B)
        out[f"two_r{r}"] = [len(two_groups(C, a)) for a in ALPHAS]
        out[f"two_r{r}_minlogp_adj"] = float(np.log(C["n_tests"]) - C["score"].max())
        for k in (10, 30):
            out[f"milo_k{k}_r{r}"] = [sum(1 for g in gs if len(g) <= 15) for gs in milo_simple(A, B, k=k)]
        base = -np.ones(len(B), int)
        sc = c2st_scores(A, B)
        out[f"c2st_r{r}"] = [int(len(set(l[l >= 0]))) for l in (top_q_groups(B, sc, base, q) for q in Q6)]
    return {"kind": kind, "d": d, "seed": s, "n": len(B), "groups": out}


def main(argv=None):
    from .tail_bench import _run
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("scenes", "nulls", "merge"))
    ap.add_argument("--seeds")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--kinds", default="iid,bench")
    a = ap.parse_args(argv)
    shard = tuple(int(v) for v in a.shard.split("/"))
    if a.cmd in ("scenes", "nulls"):
        lo, hi = (int(v) for v in a.seeds.split("-"))
        seeds = range(lo, hi + 1)
    if a.cmd == "scenes":
        tasks = [(p, c, d, s) for s in seeds for p in TIERS for c in CONTRASTS for d in DIMS]
        _run(tasks, scene, a.out_dir, shard)
    elif a.cmd == "nulls":
        tasks = [(k, d, s) for s in seeds for k in a.kinds.split(",") for d in DIMS]
        _run(tasks, null_scene, a.out_dir, shard)
    else:
        rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out_dir, "*.json")))]
        res = {"rows": rows}
        if rows and "scores" in rows[0]:
            res["grids"] = grids()
        with open(a.out, "w") as f:
            json.dump(res, f)
        os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
