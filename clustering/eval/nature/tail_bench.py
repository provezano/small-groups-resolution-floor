"""
Resumable runner for the directional-ring branch (stage 13,.

Every scene is one task and writes its own JSON file into --out-dir (the checkpoint); finished files are skipped, so a
killed run resumes by re-running the same command. A STATUS.txt in --out-dir counts done / total.

  scenes  tiny-bench scenes (all tiers): every tail_cts variant on one alpha grid, plus GapClust k30, LOF + group and
          the oracle; per grid point the score dict and the per-group hit vector (for the error analysis);
  nulls   homogeneous torus / square samples and bench background-only scenes: number of scan groups per variant and
          alpha (scan groups only, no base clustering, so every group is false);
  merge   collect the per-scene files of a dir into one JSON (rows in the p3_perscene format).

  python -m clustering.eval.nature.tail_bench scenes --seeds 600-619 --out-dir runs/tail/tune
  python -m clustering.eval.nature.tail_bench nulls --seeds 600000-600099 --out-dir runs/tail/null_tune
  python -m clustering.eval.nature.tail_bench merge --out-dir ... --out ../results/tail_x.json

Paths are relative to the working directory (tiny_groups/prototype), so the same command runs on any host.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from . import tail_cts as T
from .tiny_bench import TIERS, make

CONTRASTS = ("high", "mid", "low")
DIMS = (2, 5, 10)
# v5's published grid plus three PFER points (Bonferroni with alpha > 1 bounds the expected number of false rejections)
ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3, 1.0, 3.0, 10.0)
Q6 = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
GAP_THR = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
NULL_ALPHAS = (0.05, 0.3, 1.0)


def hits(lab, gid):
    """Per planted group: recovered (some predicted cluster with Jaccard >= 0.5), as in tiny_bench.score."""
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
    s["hits"] = hits(lab, gid)
    return s


def scene(p, c, d, s, baselines=True):
    from .tiny_methods import hdb, scored
    X, kind, gid = make(p, d, s, contrast=c)
    base = hdb(X)
    t = time.time()
    P = T.tail_parts(X)
    out, secs = {}, {"tail_parts": round(time.time() - t, 2)}
    for v in T.VARIANTS:
        out["tail_" + v] = [_sc(T.tail_labels(P, a, v, base=base), kind, gid) for a in ALPHAS]
    if baselines:
        from .tiny_baselines import gapclust_sweep
        t = time.time()
        out["lof+group"] = [_sc(scored("lof", q)(X), kind, gid) for q in Q6]
        secs["lof+group"] = round(time.time() - t, 2)
        t = time.time()
        out["gapclust_port_k30"] = [_sc(l, kind, gid) for l in gapclust_sweep(X, 30)]
        secs["gapclust_port_k30"] = round(time.time() - t, 2)
        t = time.time()
        from .tiny_oracle import oracle_labels
        out["oracle"] = [_sc(l, kind, gid) for l in oracle_labels(X, p, d, s, c, ALPHAS[:6])]
        secs["oracle"] = round(time.time() - t, 2)
    sizes = [int((gid == g).sum()) for g in np.unique(gid[gid >= 0])]
    return {"placement": p, "contrast": c, "d": d, "seed": s, "sizes": sizes, "scores": out, "secs": secs}


def grids():
    g = {"tail_" + v: list(ALPHAS) for v in T.VARIANTS}
    g.update({"lof+group": list(Q6), "gapclust_port_k30": list(GAP_THR), "oracle": list(ALPHAS[:6])})
    return g


def null_sample(kind, d, seed, n=1500):
    """kind: torus (flat T^d embedded in 20 dims, no edges), square (uniform cube, rotated into 20 dims), bench
    (tiny_bench background with no groups: mass + 8 Gaussian clusters + 2% noise)."""
    rng = np.random.default_rng(seed)
    if kind == "torus":
        from .run_theory import embed_torus
        return embed_torus(rng.uniform(0, 1, (n, d)))
    if kind == "square":
        Q = np.linalg.qr(rng.normal(size=(20, 20)))[0][:, :d]
        return rng.uniform(0, 1, (n, d)) @ Q.T
    return make("isolated", d, seed, n_groups=0)[0]


def null_scene(kind, d, seed):
    X = null_sample(kind, d, seed)
    P = T.tail_parts(X)
    out = {}
    for v in T.VARIANTS:
        out[v] = [len(T.tail_groups(P, a, v)) for a in NULL_ALPHAS]
    return {"kind": kind, "d": d, "seed": seed, "n": len(X), "groups": out}


def _status(out_dir, done, total, note=""):
    with open(os.path.join(out_dir, "STATUS.txt"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} done {done} / {total} {note}\n")


def _run(tasks, fn, out_dir, shard):
    os.makedirs(out_dir, exist_ok=True)
    i, n = shard
    mine = [t for k, t in enumerate(tasks) if k % n == i]
    for t in mine:
        name = os.path.join(out_dir, "_".join(str(x) for x in t) + ".json")
        if os.path.exists(name):
            continue
        row = fn(*t)
        tmp = name + ".tmp"
        with open(tmp, "w") as f:
            json.dump(row, f)
        os.replace(tmp, name)
        os.chmod(name, 0o600)
        _status(out_dir, len(glob.glob(os.path.join(out_dir, "*.json"))), len(tasks), f"(shard {i}/{n})")
    _status(out_dir, len(glob.glob(os.path.join(out_dir, "*.json"))), len(tasks), f"(shard {i}/{n} finished)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("scenes", "nulls", "merge"))
    ap.add_argument("--seeds")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--kinds", default="torus,square,bench")
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
