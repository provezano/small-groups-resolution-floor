"""CTS v5 on the scikit-learn clustering-comparison datasets, plain and with planted tiny groups."""
import json, os, sys, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from clustering.eval.nature.tiny_methods import cts3

warnings.filterwarnings("ignore")
n = 1500; seed = 170
def make():
    rs = np.random.RandomState(seed)
    X, y = datasets.make_blobs(n_samples=n, random_state=seed)
    aniso = (np.dot(X, [[0.6, -0.6], [-0.4, 0.8]]), y)
    return [
        ("noisy circles", datasets.make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed), 2),
        ("noisy moons", datasets.make_moons(n_samples=n, noise=0.05, random_state=seed), 2),
        ("varied blobs", datasets.make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=seed), 3),
        ("anisotropic", aniso, 3),
        ("blobs", datasets.make_blobs(n_samples=n, random_state=8), 3),
        ("no structure", (rs.rand(n, 2), np.zeros(n, int)), 3),
    ]

def plant(X, y, rs, n_groups=5):
    """Plant tiny groups (3-8 points, tight) at random positions inside the data's bounding box."""
    lo, hi = X.min(0), X.max(0)
    sd = 0.004 * (hi - lo).mean()
    Xs, ys, gl = [X], [y], []
    nxt = y.max() + 1; start = len(X)
    for g in range(n_groups):
        m = rs.randint(3, 9)
        c = lo + rs.rand(2) * (hi - lo)
        Xs.append(c + sd * rs.randn(m, 2)); ys.append(np.full(m, nxt + g))
        gl.append(np.arange(start, start + m)); start += m
    return np.vstack(Xs), np.concatenate(ys), gl

def methods(k, X):
    conn = kneighbors_graph(X, n_neighbors=10, include_self=False)
    return {
        "k-means": cluster.MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3),
        "spectral": cluster.SpectralClustering(n_clusters=k, eigen_solver="arpack", affinity="nearest_neighbors", random_state=0),
        "ward": cluster.AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=conn),
        "DBSCAN": cluster.DBSCAN(eps=0.3),
        "OPTICS": cluster.OPTICS(min_samples=7, xi=0.05, min_cluster_size=0.1),
        "HDBSCAN": cluster.HDBSCAN(min_samples=3, min_cluster_size=15),
        "GMM": mixture.GaussianMixture(n_components=k, covariance_type="full", random_state=0),
        "CTS v5": None,
    }

def fit(name, est, X):
    if name == "CTS v5":
        return np.asarray(cts3(X, alpha=0.05, dim_est="local", scale_min=True, border=True))
    est.fit(X)
    return est.labels_.astype(int) if hasattr(est, "labels_") else est.predict(X)

def tiny_hits(lab, groups):
    hit = 0
    for g in groups:
        best = 0
        for c in set(lab[g]) - {-1}:
            mem = np.flatnonzero(lab == c)
            best = max(best, len(set(mem) & set(g)) / len(set(mem) | set(g)))
        hit += best >= 0.5
    return hit

def small_clusters(lab, maxs=15):
    u, c = np.unique(lab[lab >= 0], return_counts=True)
    return int((c <= maxs).sum())

out, figs = {}, {}
for planted in (False, True):
    rs = np.random.RandomState(0)
    rows = make()
    fig, axes = plt.subplots(len(rows), 8, figsize=(19, 14))
    res = {}
    for i, (dname, (X, y), k) in enumerate(rows):
        X = StandardScaler().fit_transform(X)
        groups = []
        if planted:
            X, y, groups = plant(X, y, rs)
        ms = methods(k, X)
        for j, (mname, est) in enumerate(ms.items()):
            t = time.time(); lab = fit(mname, est, X); dt = time.time() - t
            r = {"ari": round(float(adjusted_rand_score(y, lab)), 3) if dname != "no structure" else None,
                 "clusters": int(len(set(lab) - {-1})), "small_clusters": small_clusters(lab),
                 "tiny_found": tiny_hits(lab, groups) if planted else None, "n_tiny": len(groups), "sec": round(dt, 2)}
            res[f"{dname}|{mname}"] = r
            ax = axes[i, j]
            cols = np.array(plt.cm.tab20(np.arange(20) % 20))
            c = np.where(lab[:, None] >= 0, cols[np.maximum(lab, 0) % 20], [[0.75, 0.75, 0.75, 1]])
            ax.scatter(X[:, 0], X[:, 1], s=3, c=c, linewidths=0)
            for g in groups:
                ax.scatter(X[g, 0], X[g, 1], s=14, facecolors="none", edgecolors="k", linewidths=0.6)
            txt = f"{r['clusters']} cl"
            if planted: txt += f", tiny {r['tiny_found']}/{len(groups)}"
            elif dname == "no structure": txt = f"{r['clusters']} cl"
            if r["ari"] is not None and not planted: txt += f", ARI {r['ari']:.2f}"
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(mname, fontsize=11)
            if j == 0: ax.set_ylabel(dname, fontsize=10)
    fig.suptitle(("scikit-learn comparison datasets + 5 planted tiny groups (3-8 points, 5x tighter than background, circled)" if planted
                  else "scikit-learn comparison datasets (grey = noise / unassigned)"), fontsize=13)
    fig.tight_layout()
    key = "planted" if planted else "plain"
    os.makedirs("out", exist_ok=True)
    path = f"out/fig4_sklearn_{key}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    out[key] = res
p = "out/sklearn_compare.json"
json.dump(out, open(p, "w"), indent=1)
