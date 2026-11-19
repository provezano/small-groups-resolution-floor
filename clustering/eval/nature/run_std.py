"""
Run the clusterers (baselines + any idea registered in IDEAS) on the standard benchmark tiers.
Results: results/std_<tag>.json (local); per-tier means printed.
"""
import argparse
import json
import os
import time

import numpy as np

from . import std_bench as S
from .run_idea2 import common, methods_for

IDEAS = {}            # name -> f(Z) -> labels; filled by the idea modules that want to be benchmarked


def register(name, f):
    IDEAS[name] = f


def run(names=None, tiers=None):
    res = {}
    seen = {}
    for tier, lst in S.TIERS.items():
        if tiers and tier not in tiers:
            continue
        res[tier] = {}
        for ds in lst.split():
            if ds in seen:                    # imbalanced overlaps other tiers: reuse
                res[tier][ds] = seen[ds]
                continue
            X, y = S.load(ds)
            fs = {**methods_for(len(X), False), **methods_for(len(X), True), **common(), **IDEAS}
            row = {}
            for m, f in fs.items():
                if names and m not in names:
                    continue
                t = time.time()
                try:
                    row[m] = dict(S.score(y, f(X)), sec=round(time.time() - t, 2))
                except Exception as e:
                    row[m] = {"error": repr(e)[:160]}
            res[tier][ds] = seen[ds] = row
            print(f"{tier:10s} {ds:22s} " + " ".join(f"{m}={v.get('ami', 'ERR')}" for m, v in row.items()), flush=True)
    return res


def summary(res):
    out = {}
    for tier, dss in res.items():
        ms = sorted({m for r in dss.values() for m in r})
        out[tier] = {}
        for m in ms:
            rs = [r[m] for r in dss.values() if m in r and "error" not in r[m]]
            if not rs:
                continue
            mr = [x["minority_recall"] for x in rs if x["minority_recall"] is not None]
            out[tier][m] = {"ami": round(float(np.mean([x["ami"] for x in rs])), 3),
                            "ari": round(float(np.mean([x["ari"] for x in rs])), 3),
                            "minority_recall": round(float(np.mean(mr)), 3) if mr else None,
                            "noise": round(float(np.mean([x["noise"] for x in rs])), 3),
                            "n_ok": len(rs), "n_sets": len(dss)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3)))
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--ideas", default="", help="comma list of idea modules to import (they call register)")
    ap.add_argument("--names", default="", help="only these methods (comma list)")
    a = ap.parse_args()
    for mod in filter(None, a.ideas.split(",")):
        __import__(f"clustering.eval.nature.{mod}")
    from clustering.eval.nature import run_std as registry      # ideas register on the package module, not __main__
    IDEAS.update(registry.IDEAS)
    res = run(names=set(filter(None, a.names.split(","))) or None)
    summ = summary(res)
    for tier, ms in summ.items():
        print(f"\n== {tier}")
        for m, v in sorted(ms.items(), key=lambda kv: -kv[1]["ami"]):
            print(f"  {m:24s} AMI {v['ami']:.3f} ARI {v['ari']:.3f} minority {v['minority_recall']} noise {v['noise']}"
                  f" ({v['n_ok']}/{v['n_sets']})")
    out = os.path.join(a.root, f"results/std_{a.tag}.json")
    json.dump({"per_dataset": res, "summary": summ}, open(out, "w"), indent=1)
    os.chmod(out, 0o600)
    print("wrote", out)


if __name__ == "__main__":
    main()
