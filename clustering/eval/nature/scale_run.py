"""
Stage-9 scale experiments (synthetic part). Subcommands:
  exactness  partitioned == single-machine (and == tiny_methods.cts3 v5) on benchmark scenes
  ann        approximate-kNN recall and its effect on p-values and on the groups found
  one        one timing run (called in a subprocess so peak memory is per run)
  timing     runtime and peak memory vs n, 1e3 .. 1e6
  boundary   detection boundary vs N at 1e3 .. 1e6, against rho* = ((m-1)! alpha / (N M))^(1/(d(m-1)))
  recall_n   planted-group recall vs n on the scaled tiny-bench (all tiers)
Results: results/scale_<name>.json (chmod 600).
"""
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time

import numpy as np

from . import scale_cts as S
from .scale_bench import make_scaled
from .tiny_bench import TIERS, make, score

OUT = os.path.join(os.path.dirname(__file__), *[".."] * 3, "results")


def save(name, obj):
    p = os.path.join(OUT, f"scale_{name}.json")
    json.dump(obj, open(p, "w"), indent=1, default=float)
    os.chmod(p, 0o600)
    print("wrote", p, flush=True)


def _gset(groups):
    return sorted(tuple(int(x) for x in g) for g in groups)


# ---------------------------------------------------------------- exactness
def exactness(seeds=(600, 601, 602, 603, 604)):
    from .tiny_methods import cts3, hdb
    rows = []
    # 1) scalable single-machine vs the original cts3 v5 (1,600-point scenes)
    for s in seeds:
        for tier in TIERS:
            for d in (2, 5, 10):
                A, kind, gid = make(tier, d=d, seed=s, contrast="mid")
                hb = hdb(A)
                lo = cts3(A, dim_est="local", scale_min=True, border=True)
                go = _gset(np.flatnonzero(lo == c) for c in np.unique(lo) if c > hb.max())
                lab, info = S.cts_scale(A, return_info=True)
                gp, ip = S.cts_partitioned(A, n_cells=8, seed=s)
                lob = cts3(A, dim_est="local", scale_min=True, border=True, select="bh")
                gob = _gset(np.flatnonzero(lob == c) for c in np.unique(lob) if c > hb.max())
                labb, infob = S.cts_scale(A, rule="bh", return_info=True)
                rows.append({"scene": f"{tier}/d{d}/s{s}", "n": len(A), "orig_groups": len(go),
                             "scale_eq_orig": _gset(info["groups"]) == go,
                             "scale_eq_orig_bh": _gset(infob["groups"]) == gob,
                             "part_eq_single": _gset(gp) == _gset(info["groups"]),
                             "n_tests_eq": ip["n_tests"] == info["n_tests"],
                             "ghost_ratio": ip["knn2"]["ghost_ratio"]})
                print(rows[-1], flush=True)
    # 2) partitioned vs single on scaled scenes where halos are a small part of the data
    big = []
    for n in (20000, 100000):
        for tier in TIERS:
            A, kind, gid = make_scaled(tier, n, d=5, seed=610)
            for rule in ("bonferroni", "bh"):
                lab, info = S.cts_scale(A, rule=rule, return_info=True)
                for cells in (4, 16, 64):
                    gp, ip = S.cts_partitioned(A, rule=rule, n_cells=cells, seed=1)
                    big.append({"scene": f"{tier}/n{len(A)}", "rule": rule, "cells": cells,
                                "groups": len(info["groups"]), "part_eq_single": _gset(gp) == _gset(info["groups"]),
                                "n_tests_eq": ip["n_tests"] == info["n_tests"], "thr_eq": ip["thr"] == info["thr"],
                                "ghost_ratio": round(ip["knn2"]["ghost_ratio"], 3),
                                "redone_share": ip["knn2"]["redone"] / len(A), "max_rounds": max(ip["knn2"]["rounds"])})
                    print(big[-1], flush=True)
    summ = {"small_scale_eq_orig": float(np.mean([r["scale_eq_orig"] for r in rows])),
            "small_scale_eq_orig_bh": float(np.mean([r["scale_eq_orig_bh"] for r in rows])),
            "small_part_eq_single": float(np.mean([r["part_eq_single"] for r in rows])),
            "big_part_eq_single": float(np.mean([r["part_eq_single"] for r in big])),
            "big_n_tests_eq": float(np.mean([r["n_tests_eq"] for r in big]))}
    print(summ)
    save("exactness", {"summary": summ, "small": rows, "big": big})


# ---------------------------------------------------------------- ANN
def _all_scores(D, I, is_rep=None):
    n = len(D)
    is_rep = np.zeros(n, bool) if is_rep is None else is_rep
    mu, lr = S.local_stats(D, I, is_rep)
    return mu, lr


def ann_effect(Z, planted=None, seed=0, name=""):
    kk = S.kk_for(len(Z))
    t = time.time(); E = S.knn_exact(Z, kk); te = time.time() - t
    t = time.time(); A = S.knn_ann(Z, kk, seed=seed); ta = time.time() - t
    n = len(Z)
    # recall of the neighbour sets (self excluded), and of the columns the scan reads
    rec = np.mean([len(set(E[1][i, 1:]) & set(A[1][i, 1:])) for i in range(0, n, max(1, n // 20000))]) / kk
    exact_rows = (E[1] == A[1]).all(1).mean()
    col_ratio = A[0][:, 1:] / np.maximum(E[0][:, 1:], 1e-300)
    res = {"name": name, "n": n, "t_exact": te, "t_ann": ta, "recall_at_k": float(rec),
           "rows_identical": float(exact_rows),
           "dist_overestimate_median": float(np.median(col_ratio)),
           "dist_overestimate_p99": float(np.quantile(col_ratio, 0.99))}
    # p-values of every (point, size) candidate, each with its own global statistics
    out = {}
    for tag, (D, I) in (("exact", E), ("ann", A)):
        mu, lr = S.local_stats(D, I, np.zeros(n, bool))
        dim, rank, dims = S.reduce_stats([mu], [lr], S.scatter_of(Z))
        nt, sc, P, Ms = S.scan_rows(D, I, np.arange(n), np.zeros(n, bool), mu, dim, rank, dims, alpha=1.0)
        key = P.astype(np.int64) * 100 + Ms
        out[tag] = (dict(zip(key.tolist(), sc.tolist())), nt, dim)
    ke = out["exact"][0]; ka = out["ann"][0]
    common = np.array([k for k in ke if k in ka])
    se = np.array([ke[k] for k in common]); sa = np.array([ka[k] for k in common])
    fin = np.isfinite(se) & np.isfinite(sa)
    thr = -np.log(0.05 / out["exact"][1])
    from scipy.stats import spearmanr
    res.update({"dim_exact": out["exact"][2], "dim_ann": out["ann"][2],
                "score_spearman": float(spearmanr(se[fin], sa[fin]).correlation),
                "score_abs_diff_median": float(np.median(np.abs(se[fin] - sa[fin]))),
                "score_abs_diff_p99": float(np.quantile(np.abs(se[fin] - sa[fin]), 0.99)),
                "sig_exact": int((se >= thr).sum()), "sig_ann": int((sa >= thr).sum()),
                "sig_flip_share": float(((se >= thr) != (sa >= thr)).sum() / max(1, (se >= thr).sum())),
                "ann_lower_p_share": float((sa[fin] > se[fin]).mean())})
    # groups
    le, ie = S.cts_scale(Z, rows=E, knn="exact", return_info=True)
    la, ia = S.cts_scale(Z, rows=A, knn="ann", return_info=True, seed=seed)
    ge, ga = _gset(ie["groups"]), _gset(ia["groups"])
    sa_ = [set(g) for g in ga]
    match = [max([len(set(g) & h) / len(set(g) | h) for h in sa_] or [0]) for g in ge]
    res.update({"groups_exact": len(ge), "groups_ann": len(ga), "groups_identical": len(set(ge) & set(ga)),
                "groups_matched_j50": int(np.sum(np.array(match) >= 0.5))})
    if planted is not None:
        kind, gid = planted
        res["recall_exact"] = score(le, kind, gid)["recall"]; res["recall_ann"] = score(la, kind, gid)["recall"]
        res["false100_exact"] = score(le, kind, gid)["false_per100"]; res["false100_ann"] = score(la, kind, gid)["false_per100"]
    print(res, flush=True)
    return res


def ann(which="synthetic"):
    rows = []
    if which in ("synthetic", "all"):
        for d in (2, 5, 10):
            A, kind, gid = make_scaled("isolated", 100000, d=d, seed=620)
            rows.append(ann_effect(A, (kind, gid), name=f"scaled isolated d{d}"))
        A, kind, gid = make_scaled("inside", 100000, d=5, seed=620)
        rows.append(ann_effect(A, (kind, gid), name="scaled inside d5"))
    save("ann" if which == "synthetic" else f"ann_{which}", rows)


# ---------------------------------------------------------------- timing
def one(n, mode, tier="isolated", d=5, seed=630, rule="bonferroni"):
    A, kind, gid = make_scaled(tier, n, d=d, seed=seed)
    t0 = time.time()
    if mode == "orig":
        from .tiny_methods import cts3
        lab = cts3(A, dim_est="local", scale_min=True, border=True, select=rule)
        info = {}
    elif mode == "partitioned":
        g, info = S.cts_partitioned(A, rule=rule, n_cells=16, seed=1)
        lab = S.groups_to_labels(g, len(A))
        info = {"ghost_ratio": info["knn2"]["ghost_ratio"], "n_tests": info["n_tests"], "thr": info["thr"]}
    else:
        lab, info = S.cts_scale(A, rule=rule, knn=mode, return_info=True)
        info = {k: info[k] for k in ("n_tests", "thr", "t_knn", "dim")}
    t = time.time() - t0
    sc = score(lab, kind, gid)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 if sys.platform != "darwin" else 1024) / 1024
    return {"n": len(A), "mode": mode, "time_s": t, "peak_rss_mb": rss, "recall": sc["recall"],
            "false_per100": sc["false_per100"], "groups_planted": int(len(np.unique(gid[gid >= 0]))), **info}


def timing():
    rows = []
    py = sys.executable
    for n in (1000, 10000, 100000, 300000, 1000000):
        for mode in ("exact", "ann", "partitioned") + (("orig",) if n <= 10000 else ()):
            cmd = [py, "-m", "clustering.eval.nature.scale_run", "one", "--n", str(n), "--mode", mode]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), *[".."] * 3))
            try:
                rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
            except Exception:
                rows.append({"n": n, "mode": mode, "error": r.stderr[-500:]})
            print(rows[-1], flush=True)
            save("timing", rows)


# ---------------------------------------------------------------- detection boundary vs N
def boundary(Ns=(1000, 10000, 100000, 1000000), dms=((2, 3), (5, 3), (5, 5)), reps=60, k=10, alpha=0.05, M=14,
             seed=640):
    from scipy.special import betainc, gammaln
    from sklearn.neighbors import KDTree
    rng = np.random.default_rng(seed)
    res = []
    for d, m in dms:
        Vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
        s = (1 / Vd) ** (1 / d)                         # spacing: lambda Vd s^d = 1 at lambda = 1
        for N in Ns:
            L = N ** (1 / d)
            bg = rng.uniform(0, L, (N, d))
            tree = KDTree(bg)
            thr = -math.log(alpha / (N * M))
            rho_star = math.exp((gammaln(m) + math.log(alpha / (N * M))) / (d * (m - 1)))
            beta_fac = math.exp((math.log(k) * (m - 1) + gammaln(k) - gammaln(k + m - 1)) / (d * (m - 1)))
            grid = rho_star * np.geomspace(0.2, 8, 21)
            det = []
            for rho in grid:
                hits = 0
                for _ in range(reps):
                    c = rng.uniform(0.25 * L, 0.75 * L, d)
                    u = rng.normal(size=(m - 1, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
                    G = np.vstack([c, c + u * (rho * s) * rng.uniform(0, 1, (m - 1, 1)) ** (1 / d)])
                    dbg, _ = tree.query(G, k=m - 1 + k + 1)
                    best = 0.0
                    for j in range(m):
                        dg = np.linalg.norm(G - G[j], axis=1)
                        allr = np.sort(np.r_[dg, dbg[j]])         # includes the point itself (0)
                        r, Rk = allr[m - 1], allr[m - 1 + k]
                        p = betainc(m - 1, k, (r / Rk) ** d)
                        best = max(best, -math.log(max(p, 1e-300)))
                    hits += best >= thr
                det.append(hits / reps)
            det = np.array(det)
            # rho at 50% detection, log-linear interpolation on the decreasing curve
            i = np.flatnonzero((det[:-1] >= 0.5) & (det[1:] < 0.5))
            if len(i):
                i = i[0]
                f = (det[i] - 0.5) / max(det[i] - det[i + 1], 1e-9)
                rho50 = float(np.exp(np.log(grid[i]) + f * (np.log(grid[i + 1]) - np.log(grid[i]))))
            else:
                rho50 = float("nan")
            res.append({"d": d, "m": m, "N": N, "rho_star": rho_star, "beta_factor": beta_fac, "rho50": rho50,
                        "rho50_over_star": rho50 / rho_star, "grid": grid.tolist(), "detect": det.tolist()})
            print({k_: res[-1][k_] for k_ in ("d", "m", "N", "rho_star", "rho50", "rho50_over_star")}, flush=True)
    # slopes of log rho50 vs log N against the predicted -1/(d(m-1))
    slopes = {}
    for d, m in dms:
        r = [x for x in res if x["d"] == d and x["m"] == m and np.isfinite(x["rho50"])]
        if len(r) >= 2:
            b = np.polyfit(np.log([x["N"] for x in r]), np.log([x["rho50"] for x in r]), 1)[0]
            slopes[f"d{d}_m{m}"] = {"fitted": float(b), "predicted": -1 / (d * (m - 1))}
    print(slopes)
    save("boundary", {"rows": res, "slopes": slopes})


# ---------------------------------------------------------------- recall vs n on the scaled bench
def recall_n(seeds=(700, 701), Ns=(1600, 10000, 100000, 1000000)):
    rows = []
    for tier in TIERS:
        for d in (2, 5, 10):
            for n in Ns:
                for s in (seeds if n < 1000000 else seeds[:1]):
                    A, kind, gid = make_scaled(tier, n, d=d, seed=s)
                    knn = "exact"
                    out = {"tier": tier, "d": d, "n": len(A), "seed": s, "knn": knn}
                    for rule in ("bonferroni", "bh"):
                        lab, info = S.cts_scale(A, rule=rule, knn=knn, return_info=True)
                        sc = score(lab, kind, gid)
                        out[rule] = {"recall": sc["recall"], "false_per100": sc["false_per100"],
                                     "n_tests": info["n_tests"], "thr": info["thr"]}
                    rows.append(out)
                    print(out, flush=True)
                    save("recall_n", rows)


# ---------------------------------------------------------------- null calibration at scale
def null_n():
    """False groups on scenes with nothing planted: (a) homogeneous Poisson in d = 2, 5, 10 (any group is false; the
    family-wise rate should stay <= alpha at every N); (b) scaled tiny-bench with no planted groups (structured null:
    clusters, near-duplicate mass, noise)."""
    rows = []
    for d in (2, 5, 10):
        for N, reps in ((1000, 40), (10000, 20), (100000, 6), (1000000, 2)):
            for s in range(reps):
                rng = np.random.default_rng(650 + 1000 * d + s)
                Z = rng.uniform(0, 1, (N, d)) @ np.linalg.qr(rng.normal(size=(20, 20)))[0][:d]
                knn = "exact" if N <= 300000 else "ann"
                for rule in ("bonferroni", "bh"):
                    lab, info = S.cts_scale(Z, rule=rule, knn=knn, return_info=True)
                    rows.append({"null": "poisson", "d": d, "N": N, "seed": s, "rule": rule,
                                 "false_groups": len(info["groups"]), "n_tests": info["n_tests"]})
                print(rows[-2], rows[-1], flush=True)
                save("null_n", rows)
    for d in (2, 5, 10):
        for N, reps in ((1600, 10), (100000, 3), (1000000, 1)):
            for s in range(reps):
                A, kind, gid = make_scaled("isolated", N, d=d, seed=660 + s, n_groups=0)
                knn = "exact" if N <= 300000 else "ann"
                for rule in ("bonferroni", "bh"):
                    lab, info = S.cts_scale(A, rule=rule, knn=knn, return_info=True)
                    sc = score(lab, kind, gid)
                    rows.append({"null": "tinybench", "d": d, "N": len(A), "seed": s, "rule": rule,
                                 "false_groups": len(info["groups"]), "false_per100": sc["false_per100"],
                                 "n_tests": info["n_tests"]})
                print(rows[-2], rows[-1], flush=True)
                save("null_n", rows)


# ---------------------------------------------------------------- degraded kNN: how much ANN error is tolerable
def ann_degrade(qs=(0.0, 0.01, 0.05, 0.1, 0.2, 0.4), seed=700):
    """Simulated ANN misses: each true neighbour (not the point itself) is dropped with probability q and the row is
    refilled from the next true neighbours (a missed neighbour pushes later ones forward, as in a real ANN row).
    Uniform misses are pessimistic (real ANN misses are mostly the farthest columns). Measures recall of the rows, the
    groups found (identical / matched at Jaccard >= 0.5) and planted recall, against q = 0."""
    rows = []
    for tier, d in (("isolated", 5), ("inside", 5), ("isolated", 10)):
        A, kind, gid = make_scaled(tier, 100000, d=d, seed=seed)
        kk = S.kk_for(len(A))
        D, I = S.knn_exact(A, kk)
        ref = None
        for q in qs:
            rng = np.random.default_rng(seed + int(q * 1000))

            def degraded(Zx, k, rng=rng, q=q):
                Dx, Ix = S.knn_exact(Zx, k + 40)
                drop = rng.random(Ix.shape) < q
                drop[:, 0] = False
                order = np.argsort(drop, axis=1, kind="stable")   # kept columns first, in their original order
                return np.take_along_axis(Dx, order, 1)[:, :k + 1], np.take_along_axis(Ix, order, 1)[:, :k + 1]
            Dq, Iq = degraded(A, kk)
            rec = float(np.mean([len(set(Iq[i, 1:]) & set(I[i, 1:])) for i in range(0, len(A), 50)]) / kk)
            lab, info = S.cts_scale(A, knn=degraded, return_info=True)
            g = _gset(info["groups"])
            if ref is None:
                ref = g
            sr = [set(x) for x in g]
            match = [max([len(set(x) & h) / len(set(x) | h) for h in sr] or [0]) for x in ref]
            sc = score(lab, kind, gid)
            rows.append({"scene": f"{tier}/d{d}", "q": q, "row_recall": rec, "groups": len(g),
                         "identical_to_exact": len(set(g) & set(ref)) / max(1, len(ref)),
                         "matched_j50": float(np.mean(np.array(match) >= 0.5)) if ref else 1.0,
                         "recall": sc["recall"], "false_per100": sc["false_per100"]})
            print(rows[-1], flush=True)
    save("ann_degrade", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--mode", default="exact")
    ap.add_argument("--which", default="synthetic")
    a = ap.parse_args()
    if a.cmd == "one":
        print(json.dumps(one(a.n, a.mode), default=float))
    elif a.cmd == "ann":
        ann(a.which)
    else:
        globals()[a.cmd]()
