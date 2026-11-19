"""
Tiny-group benchmark, command line. Everything is synthetic and public.

  python -m clustering.eval.nature.tinybench_cli generate --placement touching --contrast mid --d 5 --seed 400 --out scene.npz
  python -m clustering.eval.nature.tinybench_cli score --scene scene.npz --labels labels.npy
  python -m clustering.eval.nature.tinybench_cli frontier --method mypkg.mymod:fit --param-grid "0.01,0.05,0.3" \\
      [--seeds 400-404] [--out frontier.json]
  python -m clustering.eval.nature.tinybench_cli ceiling [--seeds 400-404]

generate: writes X (n x 20), kind, gid. score: recall (Jaccard >= 0.5), false small groups per 100 points, mass
pieces, noise share. frontier: `--method` is "module:function", called as f(X, param) -> labels for each value of
--param-grid; recall is read at <= 0.25 and <= 0.5 false groups per 100 points (best grid point under the cap), per tier
(placement x contrast, pooled over d = 2, 5, 10). ceiling: the same frontier for the oracle scan (true background
density), the reference a method can be compared with.
"""
import argparse
import importlib
import json

import numpy as np
from joblib import Parallel, delayed

from .tiny_bench import TIERS, make, score

CONTRASTS = ("high", "mid", "low")
DIMS = (2, 5, 10)


def _seeds(s):
    a, b = (int(v) for v in s.split("-")) if "-" in s else (int(s), int(s))
    return range(a, b + 1)


def frontier(fit, grid, seeds, n_jobs=6):
    """fit(X, p, placement, d, seed, contrast) -> labels. Returns {tier: {recall@0.25, recall@0.5, points}}."""
    def one(p, c, d, s):
        X, kind, gid = make(p, d, s, contrast=c)
        return (p, c), [score(fit(X, g, p, d, s, c), kind, gid) for g in grid]
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(p, c, d, s) for p in TIERS for c in CONTRASTS for d in DIMS for s in seeds)
    out = {}
    for p in TIERS:
        for c in CONTRASTS:
            rs = [r for k, r in rows if k == (p, c)]
            pts = [(float(np.mean([r[i]["false_per100"] for r in rs])), float(np.mean([r[i]["recall"] for r in rs])))
                   for i in range(len(grid))]
            out[f"{p}/{c}"] = {f"recall@{cap}": round(max([rc for fp, rc in pts if fp <= cap] or [0.0]), 3)
                               for cap in (0.25, 0.5)}
            out[f"{p}/{c}"]["points"] = [(round(a, 3), round(b, 3)) for a, b in pts]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--placement", choices=TIERS, required=True)
    g.add_argument("--contrast", choices=CONTRASTS, default="mid")
    g.add_argument("--d", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--scene", required=True)
    s.add_argument("--labels", required=True)
    f = sub.add_parser("frontier")
    f.add_argument("--method", required=True)
    f.add_argument("--param-grid", required=True)
    f.add_argument("--seeds", default="400-404")
    f.add_argument("--out")
    c = sub.add_parser("ceiling")
    c.add_argument("--seeds", default="400-404")
    c.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        X, kind, gid = make(a.placement, a.d, a.seed, contrast=a.contrast)
        np.savez(a.out, X=X, kind=kind, gid=gid)
        print(f"wrote {a.out}: {len(X)} points, {len(set(gid[gid >= 0]))} planted groups")
    elif a.cmd == "score":
        z = np.load(a.scene)
        print(json.dumps(score(np.load(a.labels), z["kind"], z["gid"])))
    else:
        if a.cmd == "frontier":
            mod, fn = a.method.split(":")
            user = getattr(importlib.import_module(mod), fn)
            grid = [float(v) for v in a.param_grid.split(",")]
            res = frontier(lambda X, p, *_: user(X, p), grid, _seeds(a.seeds))
        else:
            from .tiny_oracle import oracle_labels
            alphas = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3)
            res = frontier(lambda X, al, p, d, s, c: oracle_labels(X, p, d, s, c, (al,))[0], alphas, _seeds(a.seeds))
        for k, v in res.items():
            print(f"{k:16s} {v['recall@0.25']:.2f} / {v['recall@0.5']:.2f}")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
