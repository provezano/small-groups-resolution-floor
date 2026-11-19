"""
Second-pass tiny-bench frontier with per-scene scores: the frozen design, fresh seeds, calibrated selection rules.

Methods (grids as in stage 8 / p3_perscene):
- cts3_v5      CTS v5, Bonferroni (the default; group-level error proved);
- cts3_v5_by   CTS v5 + respect-base + Benjamini-Yekutieli (FDR under any dependence, candidate level);
- cts3_v5_bh   CTS v5 + respect-base + Benjamini-Hochberg (exploratory: no group-level guarantee);
- gapclust_port_k30 (plain kNN distances, the stage-8 variant) and gapclust_official_sq_k30 (squared, as official);
- mccatch_reimpl, lof+group, hdbscan_mcs; oracle (reference ceiling, not a contestant).

  python -m clustering.eval.nature.pass2_frontier --seeds 800-809 --out results/pass2_perscene_s800.json

Output format is p3_perscene's, so p3_stats / p3_leaderboard read it directly.
"""
import argparse
import json
import os

from joblib import Parallel, delayed

from . import methods as M
from . import p3_perscene as P
from .tiny_bench import TIERS
from .tiny_baselines import gapclust_sweep
from .tiny_methods import cts3

B6 = P.B6
P.SINGLE["cts3_v5_by"] = (B6, lambda Z, a: cts3(Z, alpha=a, dim_est="local", scale_min=True, border=True,
                                                  respect_base=True, select="by"))
P.MULTI["gapclust_official_sq_k30"] = ((1.0, 1.5, 2.0, 3.0, 5.0, 8.0), lambda Z: gapclust_sweep(Z, 30, squared=True))
P.MULTI["hdbscan_mcs"] = ((2, 3, 4, 5, 8), lambda Z: [M.hdbscan_labels(Z, c, c) for c in (2, 3, 4, 5, 8)])

METHODS = ("cts3_v5", "cts3_v5_by", "cts3_v5_bh", "gapclust_port_k30", "gapclust_official_sq_k30", "mccatch_reimpl",
           "lof+group", "hdbscan_mcs", "oracle")


def one(p, c, d, s, methods):
    """Runs in joblib workers; importing this module there applies the method patches above."""
    return P.one(p, c, d, s, methods)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    methods = a.methods.split(",")
    from clustering.eval.nature.pass2_frontier import one as one_  # by reference, not as __main__
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_)(p, c, d, s, methods) for s in range(lo, hi + 1)
                                                for p in TIERS for c in P.CONTRASTS for d in P.DIMS)
    json.dump({"seeds": [lo, hi], "grids": P.grids(methods), "rows": rows}, open(a.out, "w"))
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
