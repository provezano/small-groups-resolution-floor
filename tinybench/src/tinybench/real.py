"""
Real public data tracks (vendored scene builders from clustering.eval.nature.run_kdd / run_sc / p3_tracks).

load_real(name, seed=0, **kw) -> dict(X, y, groups, name, seed, meta):
- "kdd99": KDD Cup 99, 10% file, through scikit-learn's fetch_kddcup99 (downloads once into data_home). Scene per
  seed: all records of the 12 rare attack types (2-53 records each) + normal 1500, smurf 600, neptune 600 + 300
  mid-size attacks in proportion. One-hot categoricals, signed log1p, standardised, PCA 20 (fit on the scene),
  dithered with 1e-3 x the median non-zero 1-NN distance (exact duplicates break continuous nulls). groups = rare types
  with 2-60 records in the scene.
- "pbmc10x": 10X PBMC (4,271 cells, 8 classes; the scDeepCluster file 10X_PBMC.h5 with datasets X (counts) and Y).
  One class `cls` is downsampled to `m` real cells (default class "8", m = 8; seeds 0-2 in the paper); normalise
  total 1e4, log1p, 2,000 HVG (scanpy "seurat" flavour), scale (max 10), PCA 50. Needs the optional extra
  `tinybench[singlecell]` and the file path (argument `path` or env TINYBENCH_PBMC10X).
- "pbmc10x_counts": the same scene as raw counts (cells x genes), for methods that want counts.

score_real(labels, groups, n) -> recall (Jaccard >= 0.5 per real group) and false small groups (<= 30 points holding no
group member) per 100 points.
"""
import os

import numpy as np

RARE_MAX = 60
SAMPLE = {"normal.": 1500, "smurf.": 600, "neptune.": 600}
MID_TOTAL = 300
NAMES = ("kdd99", "pbmc10x", "pbmc10x_counts")


def _kdd_frame(data_home=None):
    from sklearn.datasets import fetch_kddcup99
    b = fetch_kddcup99(percent10=True, data_home=data_home or os.environ.get("TINYBENCH_KDD_HOME"), as_frame=True)
    df = b.frame
    y = df.pop("labels").map(lambda t: t.decode() if isinstance(t, bytes) else t).to_numpy()
    for c in ("protocol_type", "service", "flag"):
        df[c] = df[c].map(lambda t: t.decode() if isinstance(t, bytes) else t)
    return df, y


def kdd_scene(df, y, seed):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    types, counts = np.unique(y, return_counts=True)
    rare = set(types[counts <= RARE_MAX])
    take = [np.flatnonzero(np.isin(y, list(rare)))]
    for t, n in SAMPLE.items():
        take.append(rng.choice(np.flatnonzero(y == t), n, replace=False))
    mid = [t for t, c in zip(types, counts) if t not in rare and t not in SAMPLE]
    mid_n = np.array([counts[types == t][0] for t in mid], float)
    for t, n in zip(mid, np.maximum(1, np.round(MID_TOTAL * mid_n / mid_n.sum())).astype(int)):
        take.append(rng.choice(np.flatnonzero(y == t), n, replace=False))
    idx = np.concatenate(take)
    X = pd.get_dummies(df.iloc[idx], columns=["protocol_type", "service", "flag"]).astype(float).to_numpy()
    X = StandardScaler().fit_transform(np.log1p(np.abs(X)) * np.sign(X))
    Z = PCA(20, random_state=seed).fit_transform(X)
    d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    Z = Z + rng.normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
    return Z, y[idx], rare


def preprocess_counts(X):
    import anndata
    import scanpy as sc
    a = anndata.AnnData(np.asarray(X, np.float32))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1))
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, random_state=0)
    return a.obsm["X_pca"]


def _pbmc(path=None):
    import h5py
    path = path or os.environ.get("TINYBENCH_PBMC10X")
    if not path or not os.path.exists(path):
        raise FileNotFoundError("pbmc10x needs the 10X_PBMC.h5 file (scDeepCluster repository); pass path= or set "
                                "TINYBENCH_PBMC10X")
    with h5py.File(path, "r") as h:
        return np.array(h["X"]), np.array(h["Y"]).astype(str)


def pbmc_index(y, cls, m, seed):
    r = np.random.default_rng(1000 * seed + m)
    return np.r_[np.flatnonzero(y != cls), r.choice(np.flatnonzero(y == cls), m, replace=False)]


def load_real(name, seed=0, cls="8", m=8, path=None, data_home=None):
    if name == "kdd99":
        df, y = _kdd_frame(data_home)
        Z, yy, rare = kdd_scene(df, y, seed)
        groups = [np.flatnonzero(yy == t) for t in sorted(rare) if 2 <= (yy == t).sum() <= RARE_MAX]
        return {"X": Z, "y": yy, "groups": groups, "name": name, "seed": seed, "meta": {"rare_types": sorted(rare)}}
    if name in ("pbmc10x", "pbmc10x_counts"):
        X, y = _pbmc(path)
        idx = pbmc_index(y, cls, m, seed)
        Xs = X[idx] if name.endswith("counts") else preprocess_counts(X[idx])
        return {"X": Xs, "y": y[idx], "groups": [np.flatnonzero(y[idx] == cls)], "name": name, "seed": seed,
                "meta": {"cls": cls, "m": m}}
    raise KeyError(f"unknown real data set {name!r}; available: {NAMES}")


def score_real(labels, groups, n=None, small=30):
    lab = np.asarray(labels)
    n = len(lab) if n is None else n
    members = np.zeros(len(lab), bool)
    rec = []
    for g in groups:
        mg = np.zeros(len(lab), bool); mg[g] = True; members |= mg
        best = max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0])
        rec.append(best >= 0.5)
    false = sum(1 for c in set(lab) - {-1} if (lab == c).sum() <= small and not members[lab == c].any())
    return {"recall": float(np.mean(rec)) if rec else float("nan"), "found": [bool(v) for v in rec],
            "false_per100": 100 * false / n}
