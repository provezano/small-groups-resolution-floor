"""
Public real domain: rare cell populations in single-cell RNA-seq (downloaded from official hosts into the data directory;
nothing private). Datasets:
- pbmc3k (10x, processed by scanpy/cellxgene): rare = Megakaryocytes (15 cells of 2,638). Labels are the tutorial's
  cluster annotations, so partly circular for clustering methods.
- Jurkat 1% (scCAD's example, from the authors' repo): rare = 16 Jurkat cells among 1,540 293T cells; plus a seeded
  downsample to 5 Jurkat cells (tiny regime).
- 10X PBMC (scDeepCluster repo, as cited by scCAD): no class is rare (smallest 135 cells), so classes 6, 7, 8 are
  downsampled to 4, 8 and 16 real cells (3 seeds); semi-synthetic but built from real cells (Ye et al. 2025 practice).
Preprocessing: normalise total, log1p (counts only), top 2,000 HVG, PCA 50. A rare group = an annotated population of
< 1% of cells or < 30 cells. Recovered = some predicted cluster/group matches it at Jaccard >= 0.5. False small groups =
predicted clusters of <= 30 cells holding no rare cell, per 100 cells. Groups are written over HDBSCAN default labels,
as in the tiny-group benchmark, so "hdbscan" alone is the shared base. scCAD gets the expression matrix as its authors
intend (counts with its own normalisation; log data with normalization off for pbmc3k).
"""
import json
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import methods as M
from .tiny_methods import cts, cts3, scored, hdb
from .tiny_baselines import gapclust, mccatch, groups_over_base, sccad_raw

try:
    from . import _tgpaths as _TGP
except ImportError:
    import _tgpaths as _TGP
DL = _TGP.DATA + "/dl/"


def preprocess(X, counts=True):
    import anndata
    import scanpy as sc
    a = anndata.AnnData(np.asarray(X, np.float32))
    if counts:
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1))
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, random_state=0)
    return np.ascontiguousarray(a.obsm["X_pca"])      # scanpy's PCA output can have negative strides (breaks torch)


def load_sets():
    import anndata
    import h5py
    sets = []
    a = anndata.read_h5ad(DL + "pbmc3k/pbmc3k_processed.h5ad")
    X = a.raw.X.toarray() if hasattr(a.raw.X, "toarray") else np.asarray(a.raw.X)
    y = a.obs["louvain"].astype(str).values
    sets.append(("pbmc3k", X, y, False, X))
    h = h5py.File(DL + "sccad/repo/1%Jurkat.h5", "r")
    Xj = np.array(h["X"]).astype(float); yj = np.array([v.decode() for v in h["Y"]]); h.close()
    sets.append(("jurkat1pct", Xj, yj, True, Xj))
    rng = np.random.default_rng(0)
    keep = np.r_[np.flatnonzero(yj != "jurkat"), rng.choice(np.flatnonzero(yj == "jurkat"), 5, replace=False)]
    sets.append(("jurkat5cells", Xj[keep], yj[keep], True, Xj[keep]))
    h = h5py.File(DL + "pbmc10x_scdeep/10X_PBMC.h5", "r")
    Xp = np.array(h["X"]); yp = np.array(h["Y"]).astype(str); h.close()
    for s in range(3):
        r = np.random.default_rng(s)
        idx = [np.flatnonzero(yp == c) for c in np.unique(yp) if c not in ("6", "7", "8")]
        for c, m in (("6", 4), ("7", 8), ("8", 16)):
            idx.append(r.choice(np.flatnonzero(yp == c), m, replace=False))
        idx = np.concatenate(idx)
        sets.append((f"pbmc10x_down_s{s}", Xp[idx], yp[idx], True, Xp[idx]))
    return sets


def evaluate(lab, y, rare):
    rec = {}
    for c in rare:
        mc = y == c
        best = max([(mc & (lab == q)).sum() / (mc | (lab == q)).sum() for q in set(lab[mc]) - {-1}] or [0])
        rec[c] = bool(best >= 0.5)
    rare_mask = np.isin(y, rare)
    false = sum(1 for q in set(lab) - {-1} if (lab == q).sum() <= 30 and not rare_mask[lab == q].any())
    return {"recall": float(np.mean(list(rec.values()))), "per_group": rec, "false_per100": 100 * false / len(y)}


def main(only=None):
    out = {}
    for name, X, y, counts, Xexpr in load_sets():
        n = len(y)
        cls, cnt = np.unique(y, return_counts=True)
        rare = [c for c, k in zip(cls, cnt) if k < 0.01 * n or k < 30]
        Z = preprocess(X, counts)
        base = hdb(Z)
        meths = {
            "hdbscan": lambda: base,
            "hdbscan_mcs3": lambda: M.hdbscan_labels(Z, 3, 3),
            "lof+group": lambda: scored("lof")(Z),
            "cts": lambda: cts(Z),
            "mccatch_reimpl": lambda: groups_over_base(Z, mccatch(Z), base, max_size=30),
            "gapclust_k30": lambda: gapclust(Z, k=30, base=base),
            "gapclust_k200": lambda: gapclust(Z, k=200, base=base),
            "cts3_v4": lambda: cts3(Z, dim_est="local", scale_min=True, max_m=30),
            "cts3_v4_rb": lambda: cts3(Z, dim_est="local", scale_min=True, respect_base=True, max_m=30),
        }
        for fl in (10, 2):
            def f(fl=fl):
                sub, sc_ = sccad_raw(Xexpr, floor=fl, counts=counts)
                return groups_over_base(Z, [np.flatnonzero(sub == i) for i in np.flatnonzero(sc_ >= 0.7)], base)
            meths["sccad_official" if fl == 10 else "sccad_floor2"] = f
        out[name] = {"n": n, "rare": {c: int(k) for c, k in zip(cls, cnt) if c in rare}}
        _, info = cts(Z, return_scores=True)
        out[name]["cts_collapse"] = {"representatives": info["reps"], "dim": round(info["dim"], 2)}
        if only:
            meths = {m: f for m, f in meths.items() if m in only}
            prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
            out[name] = {**prev.get(name, {}), **{k: v for k, v in out[name].items()}}
        for m, f in meths.items():
            try:
                out[name][m] = evaluate(np.asarray(f()), y, rare)
            except Exception as e:
                out[name][m] = {"error": repr(e)[:200]}
        print(f"\n== {name} (n={n}, rare={out[name]['rare']}, cts reps={info['reps']})")
        for m in meths:
            r = out[name][m]
            print(f"  {m:16s} " + (f"recall {r['recall']:.2f} false/100 {r['false_per100']:.2f}  {r['per_group']}" if "recall" in r else r["error"]))
    json.dump(out, open(OUT, "w"), indent=1, default=str); os.chmod(OUT, 0o600)


OUT = os.path.join(os.path.dirname(__file__), *[".."] * 3, "results/sc_rare.json")

if __name__ == "__main__":
    import sys
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
