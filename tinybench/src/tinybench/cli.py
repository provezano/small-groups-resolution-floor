"""
Command line:
  tinybench generate --placement touching --contrast mid --d 5 --seed 500 [--shape iso] --out scene.npz
  tinybench score --scene scene.npz --labels labels.npy
  tinybench frontier --method mypkg.mymod:fit --param-grid "0.01,0.05,0.3" [--seeds 500-504] [--shape iso] [--out f.json]
  tinybench frontier --baseline gapclust_k30 [--seeds 500-504]
  tinybench ceiling [--seeds 500-504] [--out c.json]
  tinybench leaderboard results/leaderboard_v1.0_s500.json [--out LEADERBOARD.md]
"""
import argparse
import importlib
import json

import numpy as np

from .core import ceiling, frontier
from .generator import CONTRASTS, SHAPES, TIERS, make
from .scoring import score


def _seeds(s):
    a, b = (int(v) for v in s.split("-")) if "-" in s else (int(s), int(s))
    return range(a, b + 1)


def _param(v):
    try:
        return float(v)
    except ValueError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tinybench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--placement", choices=TIERS, required=True)
    g.add_argument("--contrast", choices=CONTRASTS, default="mid")
    g.add_argument("--d", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--shape", choices=SHAPES, default="iso")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--scene", required=True)
    s.add_argument("--labels", required=True)
    f = sub.add_parser("frontier")
    f.add_argument("--method")
    f.add_argument("--baseline")
    f.add_argument("--param-grid")
    f.add_argument("--seeds", default="500-504")
    f.add_argument("--shape", choices=SHAPES, default="iso")
    f.add_argument("--n-jobs", type=int, default=2)
    f.add_argument("--out")
    c = sub.add_parser("ceiling")
    c.add_argument("--seeds", default="500-504")
    c.add_argument("--n-jobs", type=int, default=2)
    c.add_argument("--out")
    lb = sub.add_parser("leaderboard")
    lb.add_argument("results")
    lb.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        X, kind, gid = make(a.placement, a.d, a.seed, contrast=a.contrast, shape=a.shape)
        np.savez(a.out, X=X, kind=kind, gid=gid)
        print(f"wrote {a.out}: {len(X)} points, {len(set(gid[gid >= 0]))} planted groups")
        return
    if a.cmd == "score":
        z = np.load(a.scene)
        print(json.dumps(score(np.load(a.labels), z["kind"], z["gid"])))
        return
    if a.cmd == "leaderboard":
        from .leaderboard import main as lbm
        lbm([a.results] + (["--out", a.out] if a.out else []))
        return
    if a.cmd == "frontier":
        if a.baseline:
            from .baselines import get
            fit, grid = get(a.baseline)
            if a.param_grid:
                grid = [_param(v) for v in a.param_grid.split(",")]
        else:
            if not (a.method and a.param_grid):
                ap.error("frontier needs --baseline, or --method and --param-grid")
            mod, fn = a.method.split(":")
            fit = getattr(importlib.import_module(mod), fn)
            grid = [_param(v) for v in a.param_grid.split(",")]
        res = frontier(fit, grid, _seeds(a.seeds), shape=a.shape, n_jobs=a.n_jobs)
    else:
        res = ceiling(_seeds(a.seeds), n_jobs=a.n_jobs)
    for k, v in res.items():
        print(f"{k:16s} {v['recall@0.25']:.2f} {v['ci@0.25']} / {v['recall@0.5']:.2f} {v['ci@0.5']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
