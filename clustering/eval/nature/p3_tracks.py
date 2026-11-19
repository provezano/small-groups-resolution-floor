"""
Realistic plant tracks for the tiny-group benchmark.

Problem: isotropic plants inflate recall 5-7x on single-cell (stage 7, realism_pub). Here every planting recipe is a
function plant(Z, t, m, rng, ctx) -> (m x D) points around a target point t of a real point cloud Z, so each recipe can
be compared with real rare groups in the same scenes by the W1 distance between descriptor distributions
(run_realism.describe: spread, anisotropy, kNN purity, isolation; standardised on the real groups).

Recipes
- iso10 / iso50: isotropic Gaussian, sd = 10% / 50% quantile of the 1-NN distance / sqrt(2D) (stage-7 recipes).
- loc{s}: Gaussian with covariance s^2 x the covariance of the target's 30 nearest neighbours (anisotropic, follows the
  local manifold); s = 0.25, 0.5, 1.0.
- t3loc{s}: multivariate Student t (3 df) with the same scale matrix (heavy-tailed), variance-matched (x 1/sqrt(3)).
- boot: offset bootstrap, absolute offsets of a real group of another class (stage-7 recipe).
- bootrel: the same offsets, rescaled by (target's 10-NN radius / donor centroid's 10-NN radius in the donor scene), so
  a donor group keeps its size relative to its neighbourhood (scale-free transfer between scenes).
- {recipe}@{k}: the same recipe, centred not on the target cell but shifted by k x the target's 10-NN radius along a
  random direction drawn from the local 30-NN covariance (stays on the local manifold); k = 0.5, 1, 2. This is the
  placement knob: plants centred on an existing point have a centroid that sits on top of it (isolation ~0.2 on
  single-cell vs 0.77 for real groups).
Real-group tracks need no recipe: the rare classes themselves (single-cell downsampled classes, KDD rare types).

  python -m clustering.eval.nature.p3_tracks --mode w1 --out ...json      (descriptors only, cheap)
  python -m clustering.eval.nature.p3_tracks --mode rank --out ...json    (descriptors + methods at default settings)
"""
import argparse
import json
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from .run_realism import describe

N_PLANT = 10
RECIPES = ("iso10", "iso50", "loc0.25", "loc0.5", "loc1.0", "t3loc0.25", "t3loc0.5", "t3loc1.0", "boot", "bootrel")
SHIFTED = tuple(f"{r}@{k}" for r in ("iso50", "t3loc1.0", "bootrel") for k in (0.5, 1, 2))
ALL_RECIPES = RECIPES + SHIFTED
METHODS = ("hdbscan", "gapclust_k30", "lof+group", "cts3_v5_rb", "cts3_v5_rb_bh")
try:
    from . import _tgpaths as _TGP
except ImportError:
    import _tgpaths as _TGP
DL = os.environ.get("P3_DL", _TGP.DATA + "/dl/")


def _ctx(Z):
    nn = NearestNeighbors(n_neighbors=31).fit(Z)
    dist, idx = nn.kneighbors(Z)
    return {"d1": dist[:, 1], "r10": dist[:, 10], "idx30": idx[:, 1:]}


def _loc_cov(Z, t, ctx):
    P = Z[ctx["idx30"][t]]
    return np.cov(P.T) + 1e-12 * np.eye(Z.shape[1])


def plant(Z, t, m, rng, ctx, recipe, pool=None):
    D = Z.shape[1]
    if "@" in recipe:
        base, k = recipe.split("@")
        u = np.linalg.cholesky(_loc_cov(Z, t, ctx)) @ rng.normal(size=D)
        shift = float(k) * ctx["r10"][t] * u / max(np.linalg.norm(u), 1e-12)
        return plant(Z, t, m, rng, ctx, base, pool) + shift
    if recipe in ("iso10", "iso50"):
        q = 0.1 if recipe == "iso10" else 0.5
        return Z[t] + rng.normal(0, np.quantile(ctx["d1"], q) / np.sqrt(2 * D), (m, D))
    if recipe.startswith("loc") or recipe.startswith("t3loc"):
        s = float(recipe.split("loc")[1])
        L = np.linalg.cholesky(_loc_cov(Z, t, ctx))
        E = rng.normal(size=(m, D))
        if recipe.startswith("t3"):
            E = E / np.sqrt(rng.chisquare(3, size=(m, 1)) / 3) / np.sqrt(3)
        return Z[t] + s * E @ L.T
    if recipe in ("boot", "bootrel"):
        off, r_donor = pool[rng.integers(len(pool))]
        scale = 1.0 if recipe == "boot" else ctx["r10"][t] / max(r_donor, 1e-12)
        return Z[t] + scale * off[rng.integers(len(off), size=m)]
    raise ValueError(recipe)


def run_methods(Z, max_m):
    from .tiny_baselines import gapclust
    from .tiny_methods import cts3, hdb, scored
    base = hdb(Z)
    return {"hdbscan": base, "gapclust_k30": gapclust(Z, k=30, base=base), "lof+group": scored("lof", 0.02)(Z),
            "cts3_v5_rb": cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, max_m=max_m),
            "cts3_v5_rb_bh": cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, max_m=max_m,
                                  select="bh")}


def found(lab, mem, Z=None, margin_alpha=0.05):
    """Is the group recovered (best Jaccard >= 0.5)? With Z given, returns (found, margin) where margin is the group's
    boundary margin (rgt_theory.boundary_margin; > 0 means above the detection boundary), so tables can be split
    above and below the boundary. Z=None: unchanged (a bool)."""
    mg = np.zeros(len(lab), bool); mg[mem] = True
    ok = bool(max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0]) >= 0.5)
    if Z is None:
        return ok
    from .rgt_theory import boundary_margin
    return ok, float(boundary_margin(Z, np.asarray(mem), alpha=margin_alpha))


def run_scene(Z, real_groups, pool, sizes, seed, max_m, recipes, with_methods, tag):
    rng = np.random.default_rng(seed)
    ctx = _ctx(Z)
    out = []
    labs = run_methods(Z, max_m) if with_methods else {}
    for g in real_groups:
        out.append(dict(scene=tag, kind="real", phi=describe(Z, g).tolist(), **{k: found(l, g) for k, l in labs.items()}))
    targets = rng.choice(len(Z), N_PLANT, replace=False)
    ms = [int(rng.choice(sizes)) for _ in targets]
    for recipe in recipes:
        r2 = np.random.default_rng([seed, ALL_RECIPES.index(recipe)])
        rows, groups, nxt = [], [], len(Z)
        for t, m in zip(targets, ms):                      # same targets and sizes for every recipe (paired)
            rows.append(plant(Z, t, m, r2, ctx, recipe, pool)); groups.append(np.arange(nxt, nxt + m)); nxt += m
        Zp = np.vstack([Z] + rows)
        lp = run_methods(Zp, max_m) if with_methods else {}
        for g in groups:
            out.append(dict(scene=tag, kind=recipe, phi=describe(Zp, g).tolist(),
                            **{k: found(l, g) for k, l in lp.items()}))
    return out


# ---------------- real scenes ----------------
def load_pbmc():
    import h5py
    h = h5py.File(DL + "pbmc10x_scdeep/10X_PBMC.h5", "r")
    X = np.array(h["X"]); y = np.array(h["Y"]).astype(str); h.close()
    return X, y


def sc_index(y, c, m, s):
    r = np.random.default_rng(1000 * s + m)
    return np.r_[np.flatnonzero(y != c), r.choice(np.flatnonzero(y == c), m, replace=False)]


def sc_donor(X, y, c):
    """Offsets (member - centroid) of a real 16-cell group of class c in its own scene, and the 10-NN radius of the
    centroid among the other cells (as in realism_pub, seed 99)."""
    from .run_sc import preprocess
    r = np.random.default_rng(99)
    idx = np.r_[np.flatnonzero(y != c), r.choice(np.flatnonzero(y == c), 16, replace=False)]
    Z = preprocess(X[idx], True); g = np.flatnonzero(y[idx] == c)
    out = np.setdiff1d(np.arange(len(Z)), g)
    rad = NearestNeighbors(n_neighbors=10).fit(Z[out]).kneighbors(Z[g].mean(0, keepdims=True))[0][0, -1]
    return c, (Z[g] - Z[g].mean(0), float(rad))


def sc_scene(X, y, c, m, s, donors, recipes, with_methods):
    from .run_sc import preprocess
    idx = sc_index(y, c, m, s)
    Z = preprocess(X[idx], True)
    g = np.flatnonzero(y[idx] == c)
    pool = [v for cc, v in donors.items() if cc != c]
    return run_scene(Z, [g], pool, (8, 16), 50 + 1000 * s + m, 30, recipes, with_methods, f"sc/{c}/{m}/{s}")


def kdd_scene(df, y, s, recipes, with_methods):
    from . import run_kdd as K
    Z, yy, rare = K.scene(df, y, s)
    groups = [np.flatnonzero(yy == t) for t in sorted(rare) if 2 <= (yy == t).sum() <= 60]
    pool = []
    for g in groups:
        out = np.setdiff1d(np.arange(len(Z)), g)
        rad = NearestNeighbors(n_neighbors=10).fit(Z[out]).kneighbors(Z[g].mean(0, keepdims=True))[0][0, -1]
        pool.append((Z[g] - Z[g].mean(0), float(rad)))
    return run_scene(Z, groups, pool, [len(g) for g in groups], 70 + s, 15, recipes, with_methods, f"kdd/{s}")


# ---------------- analysis ----------------
def w1(A, B):
    """Exact W1 between two empirical distributions (uniform weights), Euclidean cost, transport LP."""
    C = np.linalg.norm(A[:, None] - B[None], axis=2)
    na, nb = len(A), len(B)
    Aeq = np.zeros((na + nb, na * nb))
    for i in range(na):
        Aeq[i, i * nb:(i + 1) * nb] = 1
    for j in range(nb):
        Aeq[na + j, j::nb] = 1
    r = linprog(C.ravel(), A_eq=Aeq, b_eq=np.r_[np.full(na, 1 / na), np.full(nb, 1 / nb)], bounds=(0, None),
                method="highs")
    return float(r.fun)


def w1_table(rows, recipes, real_rows=None, n_sub=120, seed=0):
    """W1 per recipe vs the real groups, descriptors standardised on the real groups. Planted sets are subsampled to
    n_sub rows (same subsample size for every recipe) to keep the LP small."""
    real = real_rows if real_rows is not None else [r for r in rows if r["kind"] == "real"]
    P0 = np.array([r["phi"] for r in real]); mu, sd = P0.mean(0), P0.std(0) + 1e-9
    B = (P0 - mu) / sd
    rng = np.random.default_rng(seed)
    out = {}
    for rc in recipes:
        A = np.array([r["phi"] for r in rows if r["kind"] == rc])
        if len(A) > n_sub:
            A = A[rng.choice(len(A), n_sub, replace=False)]
        out[rc] = round(w1((A - mu) / sd, B), 3)
    return out


def ranking_table(rows, recipes, methods=METHODS):
    real = [r for r in rows if r["kind"] == "real"]
    rq = [float(np.mean([r[m] for r in real])) for m in methods]
    out = {"real": dict(zip(methods, [round(v, 3) for v in rq]))}
    for rc in recipes:
        pl = [r for r in rows if r["kind"] == rc]
        rp = [float(np.mean([r[m] for r in pl])) for m in methods]
        out[rc] = dict(zip(methods, [round(v, 3) for v in rp]))
        out[rc]["spearman_vs_real"] = round(float(spearmanr(rp, rq).correlation), 3)
        out[rc]["mean_abs_gap"] = round(float(np.mean(np.abs(np.array(rp) - np.array(rq)))), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("w1", "rank"), default="w1")
    ap.add_argument("--recipes", default=",".join(ALL_RECIPES))
    ap.add_argument("--sc-seeds", default="0,1,2")
    ap.add_argument("--kdd-seeds", default="0,1,2,3,4")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recipes = a.recipes.split(",")
    wm = a.mode == "rank"
    X, y = load_pbmc()
    donors = dict(Parallel(n_jobs=a.n_jobs)(delayed(sc_donor)(X, y, c) for c in np.unique(y)))
    sc_rows = [r for rs in Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(sc_scene)(X, y, c, m, int(s), donors, recipes, wm) for c in np.unique(y) for m in (8, 16)
        for s in a.sc_seeds.split(",")) for r in rs]
    from . import run_kdd as K
    df, yk = K.load()
    kdd_rows = [r for rs in Parallel(n_jobs=a.n_jobs)(delayed(kdd_scene)(df, yk, int(s), recipes, wm)
                                                       for s in a.kdd_seeds.split(",")) for r in rs]
    res = {"single_cell": {"w1": w1_table(sc_rows, recipes)}, "kdd": {"w1": w1_table(kdd_rows, recipes)}}
    if wm:
        res["single_cell"]["ranking"] = ranking_table(sc_rows, recipes)
        res["kdd"]["ranking"] = ranking_table(kdd_rows, recipes)
    print(json.dumps(res, indent=1))
    json.dump({"results": res, "rows": {"single_cell": sc_rows, "kdd": kdd_rows}}, open(a.out, "w"))
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
