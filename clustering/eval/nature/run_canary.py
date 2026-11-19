"""
Planted canaries in the real 5 h window (F7a), used to test the UMAP floor (F2a) and groups inside clusters (F3a).

A canary group = 2-9 copies of one real non-job session, with Gaussian noise in the fused feature space scaled so the
median distance inside the group equals a quantile of real sessions' nearest-neighbour distances:
easy = 10th percentile, medium = 50th, hard = 90th. 15 groups per run are planted into the non-job sessions (the job is
set aside first, as proposed). Recall = share of canary groups that some predicted cluster matches at Jaccard >= 0.5.
False groups = new clusters of <= 15 sessions holding no canary, per 100 real sessions (an upper bound on the cost:
some may be real tight groups). Recall is also split by where the template sits: in a cluster of >= 30 sessions
("inside") or not ("outside"), from plain HDBSCAN on UMAP-15 of the real data.
"""
import json
import os

import numpy as np

from . import bench as B
from . import ideas_select as S
from . import methods as M

LEVELS = {"easy": 0.1, "medium": 0.5, "hard": 0.9}


def plant(X, rng, q, n_groups=15):
    from sklearn.neighbors import NearestNeighbors
    d1 = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    target = np.quantile(d1, q)
    sigma = target / np.sqrt(2 * X.shape[1])
    rows, gid, tmpl = [], [], []
    for g, t in enumerate(rng.choice(len(X), n_groups, replace=False)):
        m = int(rng.integers(2, 10))
        rows.append(X[t] + rng.normal(0, sigma, (m, X.shape[1])))
        gid += [g] * m
        tmpl.append(t)
    return np.vstack([X, *rows]), np.r_[np.full(len(X), -1), gid], np.array(tmpl)


def embed(X, kind, seed):
    import umap
    if kind == "pca20":
        from sklearn.decomposition import PCA
        return PCA(20, random_state=seed).fit_transform(X)
    nn = {"umap15": 15, "umap5": 5, "densmap15": 15}[kind]
    return umap.UMAP(n_components=20, n_neighbors=nn, min_dist=0.0, init="pca", random_state=seed, n_jobs=1,
                     densmap=kind == "densmap15").fit_transform(X)


def mcs_for(n):
    return max(8, n // 250)


CLUSTER = {
    "eom": lambda Z: M.hdbscan_labels(Z, mcs_for(len(Z)), max(3, mcs_for(len(Z)) // 2)),
    "micro": lambda Z: S.hdbscan_micro(Z, ratio=0.7),
    "micro+inner": lambda Z: S.inner_micro(Z, S.hdbscan_micro(Z, ratio=0.7)),
    "eom_mcs3": lambda Z: M.hdbscan_labels(Z, 3, 3),
}


def score(lab, gid, n_real, inside):
    rec, where = [], []
    for g in np.unique(gid[gid >= 0]):
        mg = gid == g
        best = max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0])
        rec.append(best >= 0.5)
        where.append(inside[g])
    rec, where = np.array(rec), np.array(where)
    sizes = {c: (lab == c).sum() for c in set(lab) - {-1}}
    false = sum(1 for c, s in sizes.items() if s <= 15 and not (gid[lab == c] >= 0).any())
    return {"recall": float(rec.mean()), "recall_inside": float(rec[where].mean()) if where.any() else np.nan,
            "recall_outside": float(rec[~where].mean()) if (~where).any() else np.nan,
            "false_per100": 100 * false / n_real, "groups": len(sizes)}


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
    d = B.load_real(root)
    X = d["X"][~d["auto"]].astype(float)
    base = M.hdbscan_labels(embed(X, "umap15", 7), mcs_for(len(X)), max(3, mcs_for(len(X)) // 2))
    bsize = np.array([(base == b).sum() if b >= 0 else 0 for b in base])
    res = {}
    for lvl, q in LEVELS.items():
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            Xa, gid, tmpl = plant(X, rng, q)
            inside = bsize[tmpl] >= 30
            for emb in ("umap15", "umap5", "densmap15", "pca20"):
                Z = embed(Xa, emb, seed)
                for cname, f in CLUSTER.items():
                    res.setdefault(lvl, {}).setdefault(f"{emb}+{cname}", []).append(score(f(Z), gid, len(X), inside))
    out = {lvl: {m: {k: round(float(np.nanmean([r[k] for r in rs])), 3) for k in rs[0]} for m, rs in v.items()}
           for lvl, v in res.items()}
    for lvl, v in out.items():
        print(f"\n== {lvl}")
        for m, r in v.items():
            print(f"  {m:24s} recall {r['recall']:.2f} (inside {r['recall_inside']:.2f} outside {r['recall_outside']:.2f})"
                  f"  false/100 {r['false_per100']:.2f}  groups {r['groups']:.0f}")
    p = os.path.join(root, "results/canary.json")
    json.dump(out, open(p, "w"), indent=1)
    os.chmod(p, 0o600)


if __name__ == "__main__":
    main()
