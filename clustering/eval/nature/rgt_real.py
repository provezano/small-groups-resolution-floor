"""
Real-group theory branch: rebuild every public real-group scene used so far, run the methods at their defaults, and
compute the theory-predicted detectability of each real group from ground truth (rgt_theory). One row per group.

Domains (scene recipes copied from the modules that defined them, so groups are the same):
  sc      10X PBMC, each of 8 classes downsampled to 4 / 8 / 16 real cells, seeds 0-2 (run_sc_sweep / p3_tracks)
  kdd     KDD Cup 99 rare attack types, seeds 0-4 (run_kdd.scene)
  unsw    UNSW-NB15 rare categories, seeds 700-704 (scale_real.unsw_scene)
  baron   Baron pancreas rare cell types, donors 1-4 (scale_real)
  mnist / fashion   one class downsampled to 2 / 5 / 15 images among 3,000 (scale_real embed), and the "tight"
          variant (a random image plus its m - 1 nearest same-class images)
  python -m clustering.eval.nature.rgt_real --domains sc,kdd --out ...json
"""
import argparse
import json
import math
import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from . import rgt_theory as T

RCD_SEEDS = (600, 601)
OUT = os.path.join(os.path.dirname(__file__), *[".."] * 3, "results")


def jaccard_found(lab, g):
    lab = np.asarray(lab)
    mg = np.zeros(len(lab), bool); mg[g] = True
    return bool(max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0]) >= 0.5)


def run_methods(Z, max_m, with_umap=True, seed=0):
    from . import methods as M
    from .tiny_baselines import gapclust
    from .tiny_methods import cts3, hdb, scored
    base = hdb(Z)
    out = {"hdbscan": base}
    t = time.time()
    out["cts_v5"] = cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, max_m=max_m)
    out["cts_v5_bh"] = cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, max_m=max_m,
                            select="bh")
    out["gapclust_k30"] = gapclust(Z, k=min(30, len(Z) - 2), base=base)
    try:
        out["lof_group"] = scored("lof", 0.02)(Z)
    except ValueError:                       # _group_outliers needs >= 4 flagged points (small sets)
        out["lof_group"] = scored("lof", 0.05)(Z)
    if with_umap:
        import umap
        E = umap.UMAP(n_components=min(20, Z.shape[1]), n_neighbors=15, min_dist=0.0, init="pca", random_state=seed,
                      n_jobs=1).fit_transform(Z)
        out["umap15_mcs3"] = M.hdbscan_labels(E, 3, 3)
    return out


def plug_d(Z):
    from .tiny_methods import twonn_dim
    return min(twonn_dim(Z), float(np.linalg.matrix_rank(Z - Z.mean(0))))


def theory_rows(Z, groups, max_m, graph_seed=0):
    from .run_realism import describe
    from .tiny_methods import _mcs
    N = len(Z)
    ms = max(3, _mcs(N) // 2)
    dp = plug_d(Z)
    from umap.umap_ import fuzzy_simplicial_set
    graph = fuzzy_simplicial_set(Z, 15, np.random.RandomState(graph_seed), "euclidean")[0].tocsr()
    rows = []
    for name, g in groups:
        g = np.asarray(g)
        dbg = T.bg_dim(Z, g)
        mg, det, dloc = T.boundary_margin(Z, g, M=max_m - 1, d_plug=dbg, detail=True)
        mg_grp = T.boundary_margin(Z, g, M=max_m - 1)
        mg_plug = T.boundary_margin(Z, g, M=max_m - 1, d_plug=dp)
        cond, masked = T.masking(Z, g, ms)
        peak, pers, pur = T.mode_persistence(Z, g, dbg)
        sh, bound, _ = T.umap_share(Z, g, graph=graph)
        phi = describe(Z, g)
        rows.append(dict(group=name, m=int(len(g)), N=N, margin=round(mg, 3), margin_grpd=round(mg_grp, 3),
                         margin_plug=round(mg_plug, 3), d_bg=round(dbg, 2), d_plug=round(dp, 2), best=det, mask_cond=cond, masked=masked,
                         has_peak=peak, persistence=None if not np.isfinite(pers) else round(pers, 3),
                         peak_purity=round(pur, 3), umap_share=round(sh, 3), budget=round(bound, 3),
                         spread=round(float(phi[0]), 3), aniso=round(float(phi[1]), 3), knn_purity=round(float(phi[2]), 3),
                         isolation=round(float(phi[3]), 3)))
    return rows


def scene_rows(tag, Z, groups, max_m, with_methods=True, seed=0):
    t = time.time()
    rows = theory_rows(Z, groups, max_m, seed)
    if with_methods:
        labs = run_methods(Z, max_m, seed=seed)
        for r, (_, g) in zip(rows, groups):
            r["found"] = {k: jaccard_found(l, g) for k, l in labs.items()}
    for r in rows:
        r["scene"] = tag
    print(tag, f"{time.time() - t:.0f}s", [(r["group"], r["m"], round(r["margin"], 1), r.get("found", {}).get("cts_v5"))
                                          for r in rows], flush=True)
    return rows


# ---------------------------------------------------------------- scenes
def sc_jobs(sizes=(4, 8, 16), seeds=(0, 1, 2)):
    from .p3_tracks import load_pbmc
    X, y = load_pbmc()
    for c in np.unique(y):
        for m in sizes:
            for s in seeds:
                yield ("sc", c, m, s), (X, y)


def sc_scene(X, y, c, m, s):
    from .p3_tracks import sc_index
    from .run_sc import preprocess
    idx = sc_index(y, c, m, s)
    Z = preprocess(X[idx], True)
    return Z, [(c, np.flatnonzero(y[idx] == c))], idx


def kdd_groups(seed, df, y):
    from . import run_kdd as K
    Z, yy, rare = K.scene(df, y, seed)
    return Z, [(t, np.flatnonzero(yy == t)) for t in sorted(rare) if 2 <= (yy == t).sum() <= 60]


def run_domain(dom, n_jobs=2, with_methods=True, sc_seeds=(0, 1, 2), sc_sizes=(4, 8, 16)):
    rows = []
    if dom == "sc":
        from .p3_tracks import load_pbmc
        X, y = load_pbmc()

        def job(c, m, s):
            Z, groups, _ = sc_scene(X, y, c, m, s)
            return scene_rows(f"sc/{c}/{m}/{s}", Z, groups, 30, with_methods)
        res = Parallel(n_jobs=n_jobs)(delayed(job)(c, m, s) for c in np.unique(y) for m in sc_sizes for s in sc_seeds)
    elif dom == "kdd":
        from . import run_kdd as K
        df, y = K.load()

        def job(s):
            Z, groups = kdd_groups(s, df, y)
            return scene_rows(f"kdd/{s}", Z, groups, 15, with_methods)
        res = Parallel(n_jobs=n_jobs)(delayed(job)(s) for s in range(5))
    elif dom == "unsw":
        from . import scale_data as SD
        from .scale_real import SEEDS, unsw_scene
        df, y = SD.unsw()

        def job(s):
            Z, yy, rare = unsw_scene(df, y, s)
            return scene_rows(f"unsw/{s}", Z, [(t, np.flatnonzero(yy == t)) for t in rare], 15, with_methods)
        res = Parallel(n_jobs=n_jobs)(delayed(job)(s) for s in SEEDS)
    elif dom == "baron":
        from . import scale_data as SD
        from .scale_real import sc_prep

        def job(donor):
            X, y = SD.baron(donor)
            types, counts = np.unique(y, return_counts=True)
            rare = list(types[(counts >= 2) & (counts <= 30)])
            Z = sc_prep(X)
            return scene_rows(f"baron/{donor}", Z, [(t, np.flatnonzero(y == t)) for t in rare], 30, with_methods)
        res = Parallel(n_jobs=n_jobs)(delayed(job)(d) for d in (1, 2, 3, 4))
    elif dom in ("mnist", "fashion", "mnist_tight", "fashion_tight"):
        from . import scale_data as SD
        from .scale_real import SEEDS, embed_scene, img_prep
        nm = "mnist_784" if dom.startswith("mnist") else "Fashion-MNIST"
        X, yy = SD.openml_images(nm)
        tight = dom.endswith("tight")
        P = None
        if tight:
            from sklearn.decomposition import PCA
            P = PCA(50, random_state=0).fit_transform(X / 255.0)
        specs = []
        classes = np.unique(yy)
        for s in SEEDS:
            rng = np.random.default_rng(s)
            for c in rng.choice(classes, 2, replace=False):
                for m in ((5, 15) if tight else (2, 5, 15)):
                    if tight:
                        cm = np.flatnonzero(yy == c)
                        a = rng.choice(cm)
                        grp = cm[np.argsort(np.linalg.norm(P[cm] - P[a], axis=1))[:m]]
                        idx = np.r_[rng.choice(np.flatnonzero(yy != c), 3000, replace=False), grp]
                    else:
                        idx = embed_scene(None, yy, c, m, s * 10 + m)
                    specs.append((c, m, s, idx))

        def job(c, m, s, idx):
            Z = img_prep(X[idx], s)
            d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
            Z = Z + np.random.default_rng(s).normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
            return scene_rows(f"{dom}/{c}/{m}/{s}", Z, [(c, np.flatnonzero(yy[idx] == c))], 15, with_methods)
        res = Parallel(n_jobs=n_jobs)(delayed(job)(*sp) for sp in specs)
    elif dom == "rcd":
        from . import rcd_scenes as RS
        names = ["shuttle", "pageblocks", "ecoli", "yeast", "pbmc3k", "covtype", "glass", "abalone",
                 "letter_loose_m4", "letter_loose_m8"]

        def job(nm, s):
            Z, yy, rare, _ = RS.scene(nm, s)
            groups = [(t, np.flatnonzero(yy == t)) for t in sorted(rare) if 2 <= (yy == t).sum() <= 60]
            return scene_rows(f"rcd/{nm}/{s}", np.asarray(Z, float), groups, 30, with_methods)
        res = Parallel(n_jobs=n_jobs)(delayed(job)(nm, s) for nm in names for s in RCD_SEEDS)
    else:
        raise ValueError(dom)
    for r in res:
        for row in r:
            row["domain"] = dom
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="sc,kdd")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for dom in a.domains.split(","):
        rows += run_domain(dom, a.n_jobs)
        json.dump(rows, open(a.out, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
