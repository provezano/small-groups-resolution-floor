"""
Stage-9 real-data runs.
  large     scalable CTS on large public sets: full KDD Cup 99 10% file (494k), 10X PBMC 68k with one lineage downsampled
            to 4 / 8 / 16 cells, MNIST and Fashion-MNIST (PCA-50) with one class downsampled to 4 / 8 / 15 images;
            baselines GapClust k30 and LOF + group at full size (HDBSCAN where it finishes).
  domains   new domains with known tiny groups at benchmark size (full method set incl. original cts3 and HDBSCAN base):
            UNSW-NB15 rare attack categories, Baron pancreas rare cell types (4 donors, natural sizes, count splitting),
            20 Newsgroups (LSA-50) and MNIST / Fashion-MNIST (PCA-50) with one class downsampled to 2-15.
Metrics: run_kdd.evaluate (recovered at Jaccard >= 0.5, types hit, small groups, rare-label precision, small-group
purity) and run_sc.evaluate (false small groups per 100 points). Every method at its default setting. Seeds 700-704.
Results: results/scale_<name>.json (chmod 600).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.neighbors import KDTree

from . import scale_cts as S
from . import scale_data as SD
from .run_kdd import evaluate as kdd_eval
from .run_sc import evaluate as sc_eval
from .scale_run import save
from .tiny_baselines import gapclust, groups_over_base

SEEDS = (700, 701, 702, 703, 704)


def metrics(lab, y, rare):
    lab = np.asarray(lab)
    e = kdd_eval(lab, y, set(rare))
    f = sc_eval(lab, y, list(rare))
    return {"recovered": e["recovered"], "hit": e["hit"], "small_groups": e["small_groups"],
            "small_precision": e["small_precision"], "small_purity": e["small_purity"],
            "false_per100": round(f["false_per100"], 4), "per_type": {t: v["hit"] for t, v in e["per_type"].items()}}


def lof_group(Z, q=0.02, base=None):
    from sklearn.neighbors import LocalOutlierFactor
    from .tiny_methods import _group_outliers
    s = -LocalOutlierFactor(n_neighbors=20).fit(Z).negative_outlier_factor_
    base = -np.ones(len(Z), np.int64) if base is None else base
    return _group_outliers(Z, np.flatnonzero(s >= np.quantile(s, 1 - q)), base)


def gap_k30(Z, base=None):
    return gapclust(Z, k=30, base=-np.ones(len(Z), np.int64) if base is None else base)


# ---------------------------------------------------------------- count splitting at scale
def confirm_tree(Z2, tree, S_, d, k=10):
    """sc_split.confirm with a kd-tree: ball around S's medoid reaching all of S, M points inside; p from
    Beta(M - 1, k) on (r / r_(M-1+k))^d; a ball with more than 2|S| points is not a tight group (p = 1)."""
    Dm = np.linalg.norm(Z2[S_][:, None] - Z2[S_][None], axis=2)
    i = S_[np.argmin(Dm.sum(1))]
    r = np.linalg.norm(Z2[S_] - Z2[i], axis=1).max()
    kq = min(len(Z2), 2 * len(S_) + k + 1)
    di = tree.query(Z2[i:i + 1], k=kq)[0][0]
    M = int(np.searchsorted(di, r, side="right"))
    if M > 2 * len(S_) or M - 1 + k >= len(di):
        return 0.0
    return float(-beta_dist.logcdf((r / di[M - 1 + k]) ** d, M - 1, k))


def count_split(X, prep, seed, alpha=0.05, prop_alpha=0.3, max_m=30, prop_rule="by"):
    """Cross-fitted count splitting with the scalable scan: propose on one half (CTS + BY, alpha 0.3), confirm on
    the other (Bonferroni over proposals), both directions at alpha / 2. The guarantee comes from the confirmation
    step, so any proposal rule is valid; BY is the headline (stage-11 pass2: same numbers as BH or Bonferroni
    proposals), prop_rule="bh" reproduces the stage-9 runs (exploratory)."""
    import scipy.sparse as sp
    r = np.random.default_rng(seed)
    if sp.issparse(X):
        X1 = X.copy(); X1.data = r.binomial(X.data.astype(np.int64), 0.5).astype(np.float32)
        X2 = X.copy(); X2.data = X.data - X1.data
    else:
        X1 = r.binomial(X.astype(np.int64), 0.5).astype(np.float32); X2 = X - X1
    Z1, Z2 = prep(X1), prep(X2)
    kept, n_prop = [], 0
    for Za, Zb in ((Z1, Z2), (Z2, Z1)):
        lab, info = S.cts_scale(Za, rule=prop_rule, alpha=prop_alpha, max_m=max_m, return_info=True, knn=knn_mode(Za))
        props = info["groups"]
        n_prop += len(props)
        rank = int(np.linalg.matrix_rank(Zb[:5000] - Zb.mean(0)))
        from .tiny_methods import twonn_dim
        d = min(max(1.0, twonn_dim(Zb)), rank)
        tree = KDTree(Zb)
        thr = -np.log(alpha / 2 / max(1, len(props)))
        for g in props:
            if confirm_tree(Zb, tree, np.asarray(g), d) >= thr and all(not (set(g) & set(h)) for h in kept):
                kept.append(np.asarray(g))
    return S.groups_to_labels(kept, len(Z1)), n_prop, len(kept)


def knn_mode(Z):
    """kd-trees are exact and fastest at low dimension (KDD PCA-20: 201k points in 10 s) but slow in 50-D
    (PBMC 68k: 380 s); NN-descent is used for >= 30 dimensions when n > 20,000 (its effect is measured in ann_effect)."""
    return "ann" if Z.shape[1] >= 30 and len(Z) > 20000 else "exact"


def run_methods(Z, y, rare, extra=None, hdb_ok=False, max_m=15, timeout_hdb=1500):
    out, times = {}, {}
    def run(name, f):
        t = time.time(); lab = f(); times[name] = round(time.time() - t, 1)
        out[name] = metrics(lab, y, rare); out[name]["time_s"] = times[name]
        print(name, {k: v for k, v in out[name].items() if k != "per_type"}, flush=True)
        return lab
    for rule in ("bonferroni", "bh"):
        run(f"cts_scale_{rule}", lambda: S.cts_scale(Z, rule=rule, max_m=max_m, knn=knn_mode(Z)))
    run("gapclust_k30", lambda: gap_k30(Z))
    run("lof+group", lambda: lof_group(Z))
    if hdb_ok:
        from .tiny_methods import hdb
        run("hdbscan", lambda: hdb(Z))
    for name, f in (extra or {}).items():
        run(name, f)
    return out


# ---------------------------------------------------------------- large
def large_kdd():
    """Full 10% file. Identical-record sets of > 20 are replaced by one representative before dithering
    (scale_cts.dedup_dither); every method then runs on the same 201k-point set, and is scored on it."""
    t = time.time()
    Z, y, key = SD.kdd_full(seed=700, dither=False)
    Zk, keep, rep = S.dedup_dither(Z, key, seed=700)
    yk = y[keep]
    types, counts = np.unique(y, return_counts=True)
    rare = list(types[counts <= 60])
    print("kdd", Z.shape, "->", Zk.shape, "prep", round(time.time() - t, 1), flush=True)
    res = {"n_records": len(Z), "n_points": len(Zk), "rare": rare}
    out = {}
    for rule in ("bonferroni", "bh"):
        t = time.time()
        lab, info = S.cts_scale(Zk, rule=rule, force_rep=rep, return_info=True)
        out[f"cts_scale_{rule}"] = metrics(lab, yk, rare); out[f"cts_scale_{rule}"]["time_s"] = round(time.time() - t, 1)
        out[f"cts_scale_{rule}"].update({k: info[k] for k in ("n_tests", "thr", "n_sig", "dim", "reduced")})
        print(rule, {k: v for k, v in out[f"cts_scale_{rule}"].items() if k != "per_type"}, flush=True)
    for name, f in (("gapclust_k30", lambda: gap_k30(Zk)), ("lof+group", lambda: lof_group(Zk))):
        t = time.time(); lab = f()
        out[name] = metrics(lab, yk, rare); out[name]["time_s"] = round(time.time() - t, 1)
        print(name, {k: v for k, v in out[name].items() if k != "per_type"}, flush=True)
    res["methods"] = out
    np.save(os.path.join(SD.ROOT, "..", "scale", "kdd_Zk.npy"), Zk)
    np.save(os.path.join(SD.ROOT, "..", "scale", "kdd_yk.npy"), yk)
    save("large_kdd", res)
    res["hdbscan_full"] = _hdb_subprocess("kdd")
    print("hdbscan", res["hdbscan_full"], flush=True)
    save("large_kdd", res)
    from .scale_run import ann_effect
    res["ann"] = ann_effect(Zk, name="kdd dedup")
    return res


def _hdb_subprocess(which, timeout=1800):
    cmd = [sys.executable, "-m", "clustering.eval.nature.scale_real", "hdb_one", "--which", which]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=os.path.join(os.path.dirname(__file__), *[".."] * 3))
        return json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"status": f"did not finish in {timeout} s"}
    except Exception as e:
        return {"status": f"error {e}"}


def hdb_one(which):
    from .tiny_methods import hdb
    if which == "kdd":
        base = os.path.join(SD.ROOT, "..", "scale")
        Z, y = np.load(os.path.join(base, "kdd_Zk.npy")), np.load(os.path.join(base, "kdd_yk.npy"), allow_pickle=True)
        types, counts = np.unique(y, return_counts=True)
        rare = [t for t in types if t.rstrip(".") in ("buffer_overflow", "ftp_write", "guess_passwd", "imap", "land",
                "loadmodule", "multihop", "perl", "phf", "rootkit", "spy", "warezmaster")]
    t = time.time(); lab = hdb(Z)
    m = metrics(lab, y, rare); m["time_s"] = time.time() - t
    print(json.dumps(m, default=float))


def sc_prep(X):
    """run_sc.preprocess (normalise to 1e4, log1p, 2,000 HVG, scale at 10, PCA 50); sparse input stays sparse until
    the HVG subset (68k x 32,738 dense would be 9 GB)."""
    import scipy.sparse as sp
    if not sp.issparse(X):
        from .run_sc import preprocess
        return preprocess(X, True)
    import anndata
    import scanpy as sc
    a = anndata.AnnData(X.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1))
    a = a[:, a.var.highly_variable].copy()
    a.X = np.asarray(a.X.todense())
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, random_state=0)
    return a.obsm["X_pca"]


def large_pbmc(classes=("CD34+", "Dendritic", "CD14+ Monocyte", "CD19+ B"), sizes=(4, 8, 16), seed=700):
    X, y = SD.pbmc68k()
    keep = y != "NA"
    X, y = X[keep], y[keep]
    rows = []
    for c in classes:
        for m in sizes:
            r = np.random.default_rng(seed + m)
            idx = np.r_[np.flatnonzero(y != c), r.choice(np.flatnonzero(y == c), m, replace=False)]
            Xs, ys = X[idx], y[idx]
            t = time.time(); Z = sc_prep(Xs); tp = time.time() - t
            print("pbmc68k", c, m, Z.shape, "prep", round(tp, 1), flush=True)
            extra = {"count_split_cross": lambda: count_split(Xs, sc_prep, seed)[0]}
            out = run_methods(Z, ys, [c], extra=extra, max_m=30)
            rows.append({"class": c, "size": m, "n": len(Z), "methods": out})
            save("large_pbmc", rows)
    return rows


def img_prep(X, seed, dims=50):
    from sklearn.decomposition import PCA
    return PCA(dims, random_state=seed).fit_transform(X / 255.0)


def large_images(names=("mnist_784", "Fashion-MNIST"), sizes=(4, 8, 15), classes=("0", "3", "7"), seed=700):
    rows = []
    for nm in names:
        X, y = SD.openml_images(nm)
        for c in classes:
            for m in sizes:
                r = np.random.default_rng(seed + m)
                idx = np.r_[np.flatnonzero(y != c), r.choice(np.flatnonzero(y == c), m, replace=False)]
                Z = img_prep(X[idx], seed); ys = y[idx]
                print(nm, c, m, Z.shape, flush=True)
                out = run_methods(Z, ys, [c])
                rows.append({"set": nm, "class": c, "size": m, "n": len(Z), "methods": out})
                save("large_images", rows)
    return rows


# ---------------------------------------------------------------- new domains at benchmark size
def domain_methods(Z, base):
    from .tiny_methods import cts3
    return {
        "hdbscan": lambda: base,
        "lof+group": lambda: lof_group(Z, base=base),
        "gapclust_k30": lambda: gapclust(Z, k=30, base=base),
        "cts3_v5": lambda: cts3(Z, dim_est="local", scale_min=True, border=True),
        "cts3_v5_bh": lambda: cts3(Z, dim_est="local", scale_min=True, border=True, select="bh"),
        "cts3_v5_rb": lambda: cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True),
        "cts3_v5_rb_bh": lambda: cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, select="bh"),
    }


def _domain_scene(Z, y, rare, extra=None):
    from .tiny_methods import hdb
    base = hdb(Z)
    fs = domain_methods(Z, base)
    fs.update(extra or {})
    out = {}
    for k, f in fs.items():
        t = time.time(); lab = f()
        out[k] = metrics(lab, y, rare); out[k]["time_s"] = round(time.time() - t, 2)
    return out


def unsw_scene(df, y, seed, rare_size=15):
    """All Worms (24), Backdoor / Analysis / Shellcode downsampled to 15 each; Normal 1500, Generic 600, Exploits 600,
    the other categories 300 in proportion (the run_kdd recipe). Features: one-hot proto / service / state, signed
    log1p, standardised, PCA 20, dithered at 1e-3 x the median non-zero 1-NN distance (exact duplicates)."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    take = [np.flatnonzero(y == "Worms")]
    for t in ("Backdoor", "Analysis", "Shellcode"):
        take.append(rng.choice(np.flatnonzero(y == t), rare_size, replace=False))
    for t, n in (("Normal", 1500), ("Generic", 600), ("Exploits", 600)):
        take.append(rng.choice(np.flatnonzero(y == t), n, replace=False))
    mid = ["DoS", "Fuzzers", "Reconnaissance"]
    cnt = np.array([(y == t).sum() for t in mid], float)
    for t, n in zip(mid, np.round(300 * cnt / cnt.sum()).astype(int)):
        take.append(rng.choice(np.flatnonzero(y == t), n, replace=False))
    idx = np.concatenate(take)
    Z = SD.unsw_features(df.iloc[idx], seed)
    d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    Z = Z + rng.normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
    return Z, y[idx], ["Worms", "Backdoor", "Analysis", "Shellcode"]


def embed_scene(Z_all, y, c, m, seed, n_bg=3000):
    rng = np.random.default_rng(seed)
    others = rng.choice(np.flatnonzero(y != c), n_bg, replace=False)
    idx = np.r_[others, rng.choice(np.flatnonzero(y == c), m, replace=False)]
    return idx


def domains(which="all"):
    res = {}
    if which in ("all", "unsw"):
        df, y = SD.unsw()
        rows = []
        for s in SEEDS:
            Z, yy, rare = unsw_scene(df, y, s)
            rows.append(_domain_scene(Z, yy, rare)); print("unsw", s, _brief(rows[-1]), flush=True)
        res["unsw"] = _agg(rows)
        save("domains_unsw", res["unsw"])
    if which in ("all", "baron"):
        rows = []
        for donor in (1, 2, 3, 4):
            X, y = SD.baron(donor)
            types, counts = np.unique(y, return_counts=True)
            rare = list(types[(counts >= 2) & (counts <= 30)])
            Z = sc_prep(X)
            extra = {"count_split_cross": lambda: count_split(X, sc_prep, 700 + donor, max_m=30)[0]}
            out = _domain_scene(Z, y, rare, extra)
            out["_rare"] = {t: int((y == t).sum()) for t in rare}
            rows.append(out); print("baron", donor, out["_rare"], _brief(out), flush=True)
        res["baron"] = _agg(rows, keep_rows=True)
        save("domains_baron", res["baron"])
    if which in ("all", "embed"):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        sets = {}
        txt, yt = SD.news20()
        sets["news20_lsa50"] = (txt, yt, "text")
        for nm in ("mnist_784", "Fashion-MNIST"):
            X, yi = SD.openml_images(nm)
            sets[nm + "_pca50"] = (X, yi, "img")
        for nm, (X, yy, kind) in sets.items():
            classes = np.unique(yy)
            rows = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                for c in rng.choice(classes, 2, replace=False):
                    for m in (2, 5, 15):
                        idx = embed_scene(None, yy, c, m, s * 10 + m)
                        if kind == "text":
                            T = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2).fit_transform([X[i] for i in idx])
                            Z = normalize(TruncatedSVD(50, random_state=s).fit_transform(T))
                        else:
                            Z = img_prep(X[idx], s)
                        from sklearn.neighbors import NearestNeighbors
                        d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
                        Z = Z + np.random.default_rng(s).normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
                        out = _domain_scene(Z, yy[idx], [c])
                        out["_scene"] = f"{c}/m{m}/s{s}"
                        rows.append(out); print(nm, out["_scene"], _brief(out), flush=True)
            res[nm] = _agg(rows, keep_rows=True)
            save(f"domains_{nm}", res[nm])
    if which == "embed_tight":
        # rare sub-style: one random image of class c plus its m - 1 nearest images of the same class (PCA-50 of the
        # whole set), the rest of class c removed; 3,000 images of the other classes. Real images, a coherent group.
        from sklearn.decomposition import PCA
        for nm in ("mnist_784", "Fashion-MNIST"):
            X, yy = SD.openml_images(nm)
            P = PCA(50, random_state=0).fit_transform(X / 255.0)
            classes = np.unique(yy)
            rows = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                for c in rng.choice(classes, 2, replace=False):
                    for m in (5, 15):
                        cm = np.flatnonzero(yy == c)
                        a = rng.choice(cm)
                        grp = cm[np.argsort(np.linalg.norm(P[cm] - P[a], axis=1))[:m]]
                        idx = np.r_[rng.choice(np.flatnonzero(yy != c), 3000, replace=False), grp]
                        Z = img_prep(X[idx], s)
                        from sklearn.neighbors import NearestNeighbors
                        d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
                        Z = Z + np.random.default_rng(s).normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
                        out = _domain_scene(Z, yy[idx], [c])
                        out["_scene"] = f"{c}/m{m}/s{s}"
                        rows.append(out); print(nm, "tight", out["_scene"], _brief(out), flush=True)
            res[nm + "_tight"] = _agg(rows, keep_rows=True)
            save(f"domains_{nm}_tight", res[nm + "_tight"])
    return res


def _brief(out):
    return {k: (round(v["recovered"], 2), round(v["hit"], 2), v["small_groups"], round(v["false_per100"], 2))
            for k, v in out.items() if not k.startswith("_")}


def _agg(rows, keep_rows=False):
    names = [k for k in rows[0] if not k.startswith("_")]
    agg = {k: {f: round(float(np.mean([r[k][f] for r in rows])), 3)
               for f in ("recovered", "hit", "small_groups", "small_precision", "small_purity", "false_per100", "time_s")}
           for k in names}
    out = {"mean": agg, "n_scenes": len(rows)}
    if keep_rows:
        out["rows"] = rows
    for k, v in agg.items():
        print(f"  {k:18s} rec {v['recovered']:.2f} hit {v['hit']:.2f} small {v['small_groups']:.1f} "
              f"prec {v['small_precision']:.2f} pur {v['small_purity']:.2f} false/100 {v['false_per100']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    if a.cmd == "hdb_one":
        hdb_one(a.which)
    elif a.cmd == "large_kdd":
        save("large_kdd", large_kdd())
    elif a.cmd == "large_pbmc":
        large_pbmc()
    elif a.cmd == "large_images":
        large_images()
    elif a.cmd == "domains":
        domains(a.which)
