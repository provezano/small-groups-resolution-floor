"""
Realistic plant tracks: plant groups into a REAL point cloud (load_real) with a recipe, keep ground truth.
Vendored from clustering.eval.nature.p3_tracks (same project); see that module and the data cards for the W1 study
that picked the old recommended recipe (RECOMMENDED, deprecated; use calibrate_recipe).

Recipes (target point t, size m): iso10/iso50 (isotropic, sd from the 1-NN distance quantile), loc{s} (Gaussian with
s^2 x the target's 30-NN covariance), t3loc{s} (Student t, 3 df, same scale), boot (absolute offsets of a real group of
another class), bootrel (the same, rescaled by the ratio of 10-NN radii), and any recipe with "@k": centre shifted by
k x the 10-NN radius along a direction drawn from the local covariance.

Which recipe? No single recipe matches real groups in every domain (the stage-9 recommendation bootrel@0.5 gave a
planted/real recall ratio of 0.54 on single-cell data and 0.45 on KDD on fresh seeds). Calibrate per domain instead:
`calibrate_recipe` picks, by K-fold cross-fitting over the domain's real classes, the recipe whose planted recall
matches the real recall of the classes not being evaluated (cross-fitted held-out ratio 1.00
[0.90, 1.07] on KDD, 1.00 [0.91, 1.08] on single-cell). Fit it only on classes that are not used for evaluation.
"""
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors

_OLD_RECOMMENDED = "bootrel@0.5"   # smallest W1 to real groups in the stage-9 study; see datacards/realistic_plants.md


def __getattr__(name):
    if name == "RECOMMENDED":
        warnings.warn("tracks.RECOMMENDED is deprecated: no single recipe transfers across domains; use "
                      "calibrate_recipe (per domain, cross-fitted)", DeprecationWarning, stacklevel=2)
        return _OLD_RECOMMENDED
    raise AttributeError(name)


def calibrate_recipe(real, planted, k_folds=5, seed=0, eps=1e-3):
    """Per-domain, cross-fitted choice of a plant recipe.

    real: {class: (n_groups, rates)}: for each real class of the domain, its number of real groups and the per-method
          found rates on them (1-D array, one entry per method of the panel; the same methods everywhere).
    planted: {recipe: rates}: per-method found rates of groups planted with that recipe in the same scenes.
    For each of k_folds folds of classes: the target is the panel recall (mean over methods, groups weighted) of the
    other folds' classes; the chosen recipe minimises |log((planted + eps) / (target + eps))|; it is compared with
    the held-out fold. Returns {"recipe": choice on all classes (use it for new classes of this domain),
    "folds": per-fold choices, "held_out_ratio": pooled planted / real panel recall over the held-out folds}."""
    cls = sorted(real)
    if len(cls) < 2:
        raise ValueError("need at least 2 real classes to cross-fit")
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(cls)) % min(k_folds, len(cls))
    panel = {rc: float(np.mean(v)) for rc, v in planted.items()}

    def target(cs):
        w = np.array([real[c][0] for c in cs], float)
        r = np.array([np.mean(real[c][1]) for c in cs])
        return float((w * r).sum() / w.sum())

    def choose(t):
        return min(panel, key=lambda rc: abs(np.log((panel[rc] + eps) / (t + eps))))

    folds, P, H, W = [], [], [], []
    for f in np.unique(fold):
        tr = [c for c, g in zip(cls, fold) if g != f]
        te = [c for c, g in zip(cls, fold) if g == f]
        best = choose(target(tr))
        folds.append(best)
        w = sum(real[c][0] for c in te)
        P.append(panel[best] * w); H.append(target(te) * w); W.append(w)
    return {"recipe": choose(target(cls)), "folds": folds,
            "held_out_ratio": float((sum(P) + 1e-9) / (sum(H) + 1e-9))}


def context(Z):
    dist, idx = NearestNeighbors(n_neighbors=31).fit(Z).kneighbors(Z)
    return {"d1": dist[:, 1], "r10": dist[:, 10], "idx30": idx[:, 1:]}


def _loc_cov(Z, t, ctx):
    return np.cov(Z[ctx["idx30"][t]].T) + 1e-12 * np.eye(Z.shape[1])


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
        if not pool:
            raise ValueError("boot recipes need a donor pool: list of (offsets m x D, donor 10-NN radius)")
        off, r_donor = pool[rng.integers(len(pool))]
        scale = 1.0 if recipe == "boot" else ctx["r10"][t] / max(r_donor, 1e-12)
        return Z[t] + scale * off[rng.integers(len(off), size=m)]
    raise ValueError(recipe)


def donor_pool(Z, groups):
    """(offsets, 10-NN radius of the centroid among non-members) for each real group of a scene."""
    out = []
    for g in groups:
        rest = np.setdiff1d(np.arange(len(Z)), g)
        rad = NearestNeighbors(n_neighbors=10).fit(Z[rest]).kneighbors(Z[g].mean(0, keepdims=True))[0][0, -1]
        out.append((Z[g] - Z[g].mean(0), float(rad)))
    return out


def plant_scene(Z, recipe, seed=0, n_plant=10, sizes=(2, 3, 4, 5, 6, 7, 8, 9), pool=None):
    """Returns (Zp, gid): Z with n_plant groups appended; gid = -1 for original points, group id for planted ones."""
    rng = np.random.default_rng(seed)
    ctx = context(Z)
    rows, gid = [], [np.full(len(Z), -1)]
    for g, t in enumerate(rng.choice(len(Z), n_plant, replace=False)):
        m = int(rng.choice(sizes))
        rows.append(plant(Z, t, m, rng, ctx, recipe, pool)); gid.append(np.full(m, g))
    return np.vstack([Z] + rows), np.concatenate(gid)
