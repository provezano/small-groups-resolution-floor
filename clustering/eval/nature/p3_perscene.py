"""
Per-scene scores for the frozen tiny-bench run (the stage-8 method list), so frontier numbers can carry bootstrap CIs
and paired tests (p3_stats.py). The stage-8 JSON files keep only pooled means per grid point, so the run is repeated
with the same code, grid and seeds, and every scene's score dict is stored.

  python -m clustering.eval.nature.p3_perscene --seeds 500-504 --out results/p3_perscene_s500.json

Output: {"grids": {method: [param, ...]}, "rows": [{"placement", "contrast", "d", "seed", "scores": {method: [score
dict per grid point]}}]}. Methods and grids are exactly those of run_tiny_frontier.SWEEP / run_tiny_official.MULTI /
tinybench_cli ceiling.
"""
import argparse
import json
import os
import time

from joblib import Parallel, delayed

from .tiny_bench import TIERS, make, score
from .tiny_methods import cts3, cts3_hybrid, scored
from .tiny_baselines import gapclust_sweep, mccatch_sweep, sccad_sweep

CONTRASTS = ("high", "mid", "low")
DIMS = (2, 5, 10)
A6 = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3)
B6 = (1e-6, 1e-4, 1e-2, 0.05, 0.1, 0.3)
Q6 = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
ORACLE_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3)

SINGLE = {   # method -> (grid, f(Z, param))
    "cts3_v5": (A6, lambda Z, a: cts3(Z, alpha=a, dim_est="local", scale_min=True, border=True)),
    "cts3_v5_bh": (B6, lambda Z, a: cts3(Z, alpha=a, dim_est="local", scale_min=True, border=True, respect_base=True,
                                         select="bh")),
    "cts3_v5_hybrid": (B6, lambda Z, a: cts3_hybrid(Z, alpha=a)),
    "lof+group": (Q6, lambda Z, q: scored("lof", q)(Z)),
}
MULTI = {    # method -> (grid labels, f(Z) -> list of labels)
    "mccatch_reimpl": (("f1.0", "f0.75", "f0.5", "f0.25", "b0.05", "b0.2"), mccatch_sweep),
    "sccad_official": ((0.5, 0.6, 0.7, 0.8, 0.9), lambda Z: sccad_sweep(Z, 10)),
    "gapclust_port": ((1.0, 1.5, 2.0, 3.0, 5.0, 8.0), gapclust_sweep),
    "gapclust_port_k30": ((1.0, 1.5, 2.0, 3.0, 5.0, 8.0), lambda Z: gapclust_sweep(Z, 30)),
}


def _gap_sq_sweep(k, thrs=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0)):
    """GapClust core with squared kNN distances, as the official rflann::Neighbour returns them (p3_gapclust_full)."""
    def f(Z):
        from .p3_gapclust_full import gapclust_raw_knn
        from .tiny_baselines import gapclust_groups, groups_over_base
        from .tiny_methods import hdb
        raw = gapclust_raw_knn(Z, k, squared=True)
        base = hdb(Z)
        return [groups_over_base(Z, gapclust_groups(*raw, thr=t), base) for t in thrs]
    return f


MULTI["gapclust_official_sq"] = ((1.0, 1.5, 2.0, 3.0, 5.0, 8.0), _gap_sq_sweep(200))
MULTI["gapclust_official_sq_k30"] = ((1.0, 1.5, 2.0, 3.0, 5.0, 8.0), _gap_sq_sweep(30))


# ---- stage 15 (integration): pre-declared additions for the final rerun. They are appended after the stage-8 entries,
# so main()'s default method list (SINGLE + the first four MULTI + oracle) is unchanged.
A9 = A6 + (1.0, 3.0, 10.0)     # alpha grid extended into the PFER range (alpha > 1 = expected false groups)
# BY / BH rows keep B6: an FDR level above 1 has no meaning, so alpha > 1 applies to Bonferroni rows only
HDB_MCS = (2, 3, 4, 5, 8)


def _hdbscan_mcs(Z):
    from . import methods as M
    return [M.hdbscan_labels(Z, c, c) for c in HDB_MCS]


def _mccatch_official(Z):
    """The authors' McCatch (Java/ELKI, p3c_orig_baselines.mccatch_official_sweep). Needs the tools dir built by the
    p3c branch in the environment variable P3C_TOOLS (jdk/bin/java, mccatch_build/, jars/)."""
    tools = os.environ.get("P3C_TOOLS")
    if not tools or not os.path.exists(os.path.join(tools, "jdk", "bin", "java")):
        raise RuntimeError("mccatch_official needs P3C_TOOLS pointing at the p3c tools dir (jdk/bin/java)")
    from .p3c_orig_baselines import mccatch_official_sweep
    return mccatch_official_sweep(Z, tools)[0]


MULTI["hdbscan_mcs"] = (HDB_MCS, _hdbscan_mcs)
MULTI["mccatch_official"] = (("f1.0", "f0.75", "f0.5", "f0.25", "b0.05", "b0.2"), _mccatch_official)


def grids(methods):
    g = {m: list(SINGLE[m][0]) for m in methods if m in SINGLE}
    g.update({m: list(MULTI[m][0]) for m in methods if m in MULTI})
    if "oracle" in methods:
        g["oracle"] = list(ORACLE_ALPHAS)
    return g


def one(p, c, d, s, methods):
    X, kind, gid = make(p, d, s, contrast=c)
    out, secs = {}, {}
    for m in methods:
        t = time.time()
        if m in SINGLE:
            grid, f = SINGLE[m]
            out[m] = [score(f(X, a), kind, gid) for a in grid]
        elif m in MULTI:
            out[m] = [score(l, kind, gid) for l in MULTI[m][1](X)]
        elif m == "oracle":
            from .tiny_oracle import oracle_labels
            out[m] = [score(l, kind, gid) for l in oracle_labels(X, p, d, s, c, ORACLE_ALPHAS)]
        else:
            raise KeyError(f"unknown method {m!r} (not in SINGLE, MULTI or 'oracle')")
        secs[m] = round(time.time() - t, 2)
    return {"placement": p, "contrast": c, "d": d, "seed": s, "scores": out, "secs": secs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="500-504")
    ap.add_argument("--methods", default=",".join(list(SINGLE) + list(MULTI)[:4] + ["oracle"]))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    methods = a.methods.split(",")
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(p, c, d, s, methods) for p in TIERS for c in CONTRASTS
                                                for d in DIMS for s in range(lo, hi + 1))
    json.dump({"seeds": [lo, hi], "grids": grids(methods), "rows": rows}, open(a.out, "w"))
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
