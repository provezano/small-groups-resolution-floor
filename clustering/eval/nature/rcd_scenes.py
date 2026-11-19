"""
Public scenes for rare-category discovery (stage 10). Each scene returns (Z, y, rare, info): features, labels (str),
the set of rare classes the analyst must find, and notes. Points are shuffled with the seed, so no method gains from
the storage order (several loaders append the rare points at the end).

- kdd: KDD Cup 99 scene of run_kdd (all 12 natural rare attack types + a sample of the rest). Rare = those 12 types.
- shuttle: Statlog Shuttle (UCI, OpenML), a 4,500-point sample in proportion to class sizes, with a floor of 6 points
  per class (the He 2010 thesis used a 4,515-point sample whose smallest class has 0.13% = 6 points).
  Rare = classes with share < 2% (2, 3, 6, 7).
- pageblocks, ecoli, yeast (UCI, OpenML): whole data, the non-rare classes subsampled to 90% per seed so that seeds
  differ. Rare = share <= 2.5% (page blocks 3, 4, 5; ecoli imL, imS, omL; yeast ERL, POX, VAC, EXC).
- pbmc3k: 10x pbmc3k, natural rare populations Megakaryocytes (15) and Dendritic (37); 90% of the other cells.
- pbmc10x: 10X PBMC (8 classes); 3 classes picked by the seed are downsampled to 4, 8 and 16 real cells; rare = them.
- digits_<kind>_m<m>: the explore scene (5 whole digit classes, 5 rare classes of m points), kind = loose (real class,
  downsampled) or tight<s> (the m nearest same-class points of a random point, pulled to their mean by s).
UCI sets are standardised; exact duplicates are dithered by 1e-3 x the median non-zero 1-NN distance (as run_kdd).
"""
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .run_sc import DL as _DL

SCR = os.environ.get("RCD_SCRATCH", _DL.rsplit("dl/", 1)[0])   # the data folder
OPENML = SCR + "rcd_openml"
_cache = {}


def _dither(Z, rng):
    d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    return Z + rng.normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)


def _openml(name):
    if name not in _cache:
        from sklearn.datasets import fetch_openml
        b = fetch_openml(name, version=1, data_home=OPENML, as_frame=True, parser="auto")
        X = b.data.apply(lambda c: c.astype(float) if c.dtype.kind in "fiub" else c.cat.codes.astype(float)
                         if hasattr(c, "cat") else c.astype(float)).to_numpy(float)
        _cache[name] = (X, b.target.astype(str).to_numpy())
    return _cache[name]


def _finish(X, y, rare, seed, info, scale=True, dither=True):
    rng = np.random.default_rng(seed + 7)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    Z = StandardScaler().fit_transform(X) if scale else X
    if dither:
        Z = _dither(Z, rng)
    return Z, y, set(rare), info


def _uci(name, seed, share):
    X, y = _openml(name)
    rng = np.random.default_rng(seed)
    v, c = np.unique(y, return_counts=True)
    rare = [a for a, b in zip(v, c) if b / len(y) <= share]
    keep = []
    for a, b in zip(v, c):
        ix = np.flatnonzero(y == a)
        keep.append(ix if a in rare else rng.choice(ix, int(round(0.9 * b)), replace=False))
    k = np.concatenate(keep)
    return _finish(X[k], y[k], rare, seed, {"name": name})


def scene(name, seed):
    if name == "kdd":
        from .run_kdd import load, scene as kscene
        if "kdd" not in _cache:
            _cache["kdd"] = load()
        df, y = _cache["kdd"]
        Z, yy, rare = kscene(df, y, seed)
        return _finish(Z, yy.astype(str), rare, seed, {"name": "kdd"}, scale=False, dither=False)
    if name == "shuttle":
        X, y = _openml("shuttle")
        rng = np.random.default_rng(seed)
        v, c = np.unique(y, return_counts=True)
        keep = []
        for a, b in zip(v, c):
            k = min(b, max(6, int(round(4500 * b / len(y)))))
            keep.append(rng.choice(np.flatnonzero(y == a), k, replace=False))
        k = np.concatenate(keep)
        yy = y[k]; v2, c2 = np.unique(yy, return_counts=True)
        rare = [a for a, b in zip(v2, c2) if b / len(yy) < 0.02]
        return _finish(X[k], yy, rare, seed, {"name": "shuttle"})
    if name == "pageblocks":
        return _uci("page-blocks", seed, 0.025)
    if name == "ecoli":
        return _uci("ecoli", seed, 0.025)
    if name == "yeast":
        return _uci("yeast", seed, 0.025)
    if name == "pbmc3k":
        import anndata
        from .run_sc import DL, preprocess
        if "pbmc3k" not in _cache:
            a = anndata.read_h5ad(DL + "pbmc3k/pbmc3k_processed.h5ad")
            X = a.raw.X.toarray() if hasattr(a.raw.X, "toarray") else np.asarray(a.raw.X)
            _cache["pbmc3k"] = (X, a.obs["louvain"].astype(str).values)
        X, y = _cache["pbmc3k"]
        rare = ["Megakaryocytes", "Dendritic cells"]
        rng = np.random.default_rng(seed)
        k = np.r_[np.flatnonzero(np.isin(y, rare)),
                  rng.choice(np.flatnonzero(~np.isin(y, rare)), int(0.9 * (~np.isin(y, rare)).sum()), replace=False)]
        Z = preprocess(X[k], False)
        return _finish(Z, y[k], rare, seed, {"name": "pbmc3k"}, scale=False, dither=False)
    if name == "pbmc10x":
        import h5py
        from .run_sc import DL, preprocess
        if "pbmc10x" not in _cache:
            h = h5py.File(DL + "pbmc10x_scdeep/10X_PBMC.h5", "r")
            _cache["pbmc10x"] = (np.array(h["X"]), np.array(h["Y"]).astype(str)); h.close()
        X, y = _cache["pbmc10x"]
        rng = np.random.default_rng(seed)
        cls = rng.permutation(np.unique(y))[:3]
        k = [np.flatnonzero(~np.isin(y, cls))]
        for c, m in zip(cls, (4, 8, 16)):
            k.append(rng.choice(np.flatnonzero(y == c), m, replace=False))
        k = np.concatenate(k)
        Z = preprocess(X[k], True)
        return _finish(Z, y[k], list(cls), seed, {"name": "pbmc10x", "sizes": dict(zip(cls.tolist(), (4, 8, 16)))},
                       scale=False, dither=False)
    if name == "covtype":
        if "covtype" not in _cache:
            from sklearn.datasets import fetch_covtype
            b = fetch_covtype(data_home=OPENML)
            _cache["covtype"] = (b.data.astype(float), b.target.astype(str))
        X, y = _cache["covtype"]
        rng = np.random.default_rng(seed)
        k = rng.choice(len(y), 5000, replace=False)
        yy = y[k]; v2, c2 = np.unique(yy, return_counts=True)
        rare = [a for a, b in zip(v2, c2) if b / len(yy) < 0.02]
        return _finish(X[k], yy, rare, seed, {"name": "covtype"})
    if name == "glass":
        X, y = _openml("glass")
        rng = np.random.default_rng(seed)
        v, c = np.unique(y, return_counts=True)
        rare = [a for a, b in zip(v, c) if b / len(y) <= 0.07]
        k = np.concatenate([np.flatnonzero(y == a) if a in rare else
                            rng.choice(np.flatnonzero(y == a), int(round(0.9 * b)), replace=False) for a, b in zip(v, c)])
        return _finish(X[k], y[k], rare, seed, {"name": "glass"})
    if name == "abalone":
        X, y = _openml("abalone")
        rng = np.random.default_rng(seed)
        v, c = np.unique(y, return_counts=True)
        rare = [a for a, b in zip(v, c) if 2 <= b <= 20]
        k = np.concatenate([np.flatnonzero(y == a) if b <= 20 else
                            rng.choice(np.flatnonzero(y == a), int(round(0.9 * b)), replace=False) for a, b in zip(v, c)])
        return _finish(X[k], y[k], rare, seed, {"name": "abalone"})
    if name.startswith("letter_"):
        _, kind, mm = name.split("_")
        X0, y0 = _openml("letter")
        rng = np.random.RandomState(seed)
        cls = rng.permutation(np.unique(y0)); big, rare = cls[:5], cls[5:10]
        idx = [rng.choice(np.flatnonzero(y0 == c), 180, replace=False) for c in big]
        idx += [rng.choice(np.flatnonzero(y0 == c), int(mm[1:]), replace=False) for c in rare]
        idx = np.concatenate(idx)
        return _finish(X0[idx], y0[idx], list(rare), seed, {"name": name})
    if name.startswith("digits_"):
        _, kind, mm = name.split("_")
        return digits(seed, int(mm[1:]), kind)
    raise ValueError(name)


def digits(seed, m, kind):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    if "digits" not in _cache:
        X0, y0 = load_digits(return_X_y=True)
        _cache["digits"] = (PCA(30, random_state=0).fit_transform(X0 / 16.0), y0)
    X0, y0 = _cache["digits"]
    rng = np.random.RandomState(seed)
    cls = rng.permutation(10); big, rare = cls[:5], cls[5:]
    idx = list(np.flatnonzero(np.isin(y0, big))); X = list(X0[idx]); y = list(y0[idx])
    for c in rare:
        pool = np.flatnonzero(y0 == c)
        if kind == "loose":
            pts = X0[rng.choice(pool, m, replace=False)]
        else:
            s = float(kind[5:])
            ctr = rng.choice(pool); pick = pool[np.argsort(np.linalg.norm(X0[pool] - X0[ctr], axis=1))[:m]]
            mu = X0[pick].mean(0); pts = mu + s * (X0[pick] - mu)
        X += list(pts); y += [c] * m
    X, y = np.array(X), np.array(y).astype(str)
    return _finish(X, y, [str(c) for c in rare], seed, {"name": f"digits_{kind}_m{m}"}, scale=False, dither=False)


SCENES = ["kdd", "shuttle", "pageblocks", "ecoli", "yeast", "pbmc3k", "pbmc10x",
          "digits_tight0.3_m4", "digits_tight0.6_m4", "digits_tight1.0_m4", "digits_loose_m4",
          "digits_tight0.3_m8", "digits_tight0.6_m8", "digits_tight1.0_m8", "digits_loose_m8"]
