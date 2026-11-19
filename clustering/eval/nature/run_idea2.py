"""
Idea 2 (per-branch persistence selection) against the baselines, on the synthetic sets (5 seeds each) and on the
real 5 h window (UMAP 20-d as in the share build, 3 seeds). Two size settings: the build's default min_cluster_size
and a small one (3) that lets groups of 3-9 exist at all. Results: results/idea2.json (local).
"""
import argparse
import json
import os
import time

import numpy as np

from . import bench as B
from . import methods as M


def methods_for(n, small):
    mcs = 3 if small else max(8, round(n / 250))
    ms = 3 if small else max(3, mcs // 2)
    tag = "small" if small else "default"
    return {
        f"hdbscan_eom[{tag}]": lambda Z: M.hdbscan_labels(Z, mcs, ms),
        f"hdbscan_leaf[{tag}]": lambda Z: M.hdbscan_labels(Z, mcs, ms, "leaf"),
        f"hdbscan_eps[{tag}]": lambda Z: M.hdbscan_labels(Z, mcs, ms, eps=_eps(Z, ms)),
        f"idea2_log[{tag}]": lambda Z: M.select(M.hdbscan_fit(Z, mcs, ms), "log"),
        f"plscan[{tag}]": lambda Z: M.plscan_labels(Z, ms),
    }


def common():
    return {"tomato": lambda Z: M.tomato_labels(Z, k=10), "leiden_knn15": lambda Z: M.leiden_labels(Z, 15)}


def _eps(Z, ms):
    """cluster_selection_epsilon: median distance to the ms-th neighbour (a data-driven guess, no labels)."""
    from sklearn.neighbors import NearestNeighbors
    return float(np.median(NearestNeighbors(n_neighbors=ms + 1).fit(Z).kneighbors(Z)[0][:, -1]))


def run_all(Z, scorer):
    out = {}
    for name, f in {**methods_for(len(Z), False), **methods_for(len(Z), True), **common()}.items():
        t = time.time()
        try:
            out[name] = dict(scorer(f(Z)), sec=round(time.time() - t, 2))
        except Exception as e:                       # a baseline failing is a result, not a crash
            out[name] = {"error": repr(e)[:200]}
    return out


def mean_table(runs):
    keys = sorted({k for r in runs for k in r})
    tab = {}
    for m in keys:
        rs = [r[m] for r in runs if m in r and "error" not in r[m]]
        if not rs:
            tab[m] = {"error": next(r[m]["error"] for r in runs if m in r)}
            continue
        tab[m] = {k: round(float(np.mean([x[k] for x in rs if x.get(k) is not None])), 3)
                  for k in rs[0] if isinstance(rs[0][k], (int, float))}
        tab[m]["sd_ari"] = round(float(np.std([x["ari"] for x in rs])), 3) if "ari" in rs[0] else None
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3)))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--real-seeds", default="7,11,23")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--out", default="results/idea2.json")
    a = ap.parse_args()
    res = {"synthetic": {}, "real": {}}
    for sc in ([] if a.skip_synth else B.SCENARIOS):
        runs = []
        for s in [int(x) for x in a.seeds.split(",")]:
            X, y, kind, tmpl = B.make_synthetic(sc, s)
            runs.append(run_all(X, lambda lab: B.score(y, lab, kind, tmpl)))
        res["synthetic"][sc] = mean_table(runs)
        print(f"\n== {sc}")
        for m, v in res["synthetic"][sc].items():
            print(f"  {m:24s} {v}")
    if not a.skip_real:
        from ...cluster import fit_umap
        d = B.load_real(a.root)
        runs = []
        for s in [int(x) for x in a.real_seeds.split(",")]:
            Z = fit_umap(d["X"], seed=s, n_components=20, n_neighbors=15, min_dist=0.0, init="pca").embedding_
            runs.append(run_all(Z, lambda lab: B.score_real(lab, d)))
        res["real"] = {m: {k: round(float(np.mean([r[m][k] for r in runs])), 3) for k in runs[0][m]
                           if isinstance(runs[0][m][k], (int, float))} if "error" not in runs[0][m] else runs[0][m]
                       for m in runs[0]}
        print("\n== real 5 h window (mean over UMAP seeds)")
        for m, v in res["real"].items():
            print(f"  {m:24s} {v}")
    out = os.path.join(a.root, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    os.chmod(out, 0o600)
    print("wrote", out)


if __name__ == "__main__":
    main()
