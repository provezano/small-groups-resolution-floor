"""
Fixes for the cluster-count step, from the error analysis (error_analysis.py). The near-misses lose mostly by choosing
the wrong number of clusters, not by assigning points badly: ToMATo with the true K gains 0.04-0.14 AMI per tier, and
Fermat assignment on top of the true K is the best result in every tier. So each method here only changes how many
peaks ToMATo keeps (or how HDBSCAN's leftovers are placed); assignment is ToMATo's, or Fermat p = 4 where named.

- tomato_sym:   ToMATo on the symmetrised kNN graph plus the minimum spanning tree's edges, so the graph is
                connected and every peak can merge (our window's plain kNN graph falls into 43 pieces).
- tomato_loggap: K at the widest gap of log prominence. The plain gap is usually the one under the single most
                prominent peak, which gives K = 2 whenever peaks differ in height.
- tomato_boot:  keep peaks whose prominence is above twice a bootstrap noise band of the density (the band is the
                95th percentile of |density on an 80% subsample - density|, median over 20 subsamples); the
                confidence-band rule of Chazal et al. / Fasy et al.
- tomato_kvote: K = median of the gap-rule K over k in {5, 7, 10, 14, 20}; assignment at k = 10.
- *_fermat:     the same peaks, Fermat p = 4 assignment.
- hdbscan_fill: HDBSCAN EOM, then noise points join the cluster they reach at the lowest Fermat p = 4 cost.
- hdbscan_micro: HDBSCAN EOM, then tiny groups inside its noise: noise points that are mutual 3-nearest neighbours
                and much closer to each other than to any clustered point (ratio < 0.5) are linked; pieces of 2-15
                points become clusters. Groups of 2-9 points are never density peaks at k = 10, so no count rule finds
                them; this targets them directly.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from .ideas34 import _knn
from . import methods as M
from .run_std import register

K = 10


def _density(d, dim):
    return -dim * np.log(np.sqrt((d ** 2).mean(1)) + 1e-12)            # logDTM up to a constant


def _sym_graph(Z, k=K):
    """Symmetrised kNN edges plus MST edges between the graph's pieces. Returns neighbour lists, distances, CSR."""
    _, d, idx = _knn(Z, k)
    n = len(Z)
    rows = np.repeat(np.arange(n), idx.shape[1])
    G = coo_matrix((d.ravel() + 1e-12, (rows, idx.ravel())), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    nc, comp = connected_components(G, directed=False)
    if nc > 1:                                                         # link the pieces through their closest pairs
        cent = np.array([Z[comp == c].mean(0) for c in range(nc)])
        reps = [np.flatnonzero(comp == c) for c in range(nc)]
        from scipy.spatial.distance import cdist
        T = minimum_spanning_tree(cdist(cent, cent)).tocoo()
        add_r, add_c, add_v = [], [], []
        for a, b in zip(T.row, T.col):
            D = cdist(Z[reps[a]], Z[reps[b]])
            i, j = np.unravel_index(np.argmin(D), D.shape)
            add_r += [reps[a][i], reps[b][j]]; add_c += [reps[b][j], reps[a][i]]; add_v += [D[i, j] + 1e-12] * 2
        G = (G + coo_matrix((add_v, (add_r, add_c)), shape=(n, n)).tocsr()).tocsr()
    nbrs = [G.indices[G.indptr[i]:G.indptr[i + 1]].tolist() for i in range(n)]
    return nbrs, d, G


def _tomato(nbrs, dens):
    from gudhi.clustering.tomato import Tomato
    t = Tomato(graph_type="manual", density_type="manual")
    t.fit(nbrs, weights=dens)
    return t


def _set_k(t, k):
    t.n_clusters_ = int(max(1, k))
    return np.asarray(t.labels_)


def _prom(t):
    dg = t.diagram_
    return np.sort(dg[:, 0] - dg[:, 1])[::-1] if len(dg) else np.array([])


def _gap_k(prom, log=False):
    if len(prom) < 2:
        return len(prom) + 1
    v = np.log(np.maximum(prom, 1e-12)) if log else prom
    return int(np.argmax(v[:-1] - v[1:]) + 1) + 1


def _boot_band(Z, dens, k=K, B=20, frac=0.8, seed=0):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n, band = len(Z), []
    for _ in range(B):
        sub = rng.choice(n, int(frac * n), replace=False)
        d = NearestNeighbors(n_neighbors=k).fit(Z[sub]).kneighbors(Z)[0]
        band.append(np.quantile(np.abs(_density(d, Z.shape[1]) - dens), 0.95))
    return float(np.median(band))


def _fermat(Z, lab, G, dens, p=4):
    """Re-assign every point to the cluster whose peak it reaches at the lowest sum of edge length ** p."""
    seeds = np.array([np.flatnonzero(lab == c)[np.argmax(dens[lab == c])] for c in np.unique(lab)])
    H = G.copy(); H.data = H.data ** p
    _, _, src = dijkstra(H, indices=seeds, min_only=True, return_predecessors=True)
    return np.unique(src, return_inverse=True)[1]


def _setup(Z):
    nbrs, d, G = _sym_graph(Z)
    dens = _density(d, Z.shape[1])
    return nbrs, d, G, dens, _tomato(nbrs, dens)


def tomato_sym(Z):
    nbrs, d, G, dens, t = _setup(Z)
    return _set_k(t, _gap_k(_prom(t)))


def tomato_loggap(Z, fermat=False):
    nbrs, d, G, dens, t = _setup(Z)
    lab = _set_k(t, _gap_k(_prom(t), log=True))
    return _fermat(Z, lab, G, dens) if fermat else lab


def tomato_boot(Z, fermat=False):
    nbrs, d, G, dens, t = _setup(Z)
    c = _boot_band(Z, dens)
    lab = _set_k(t, 1 + int((_prom(t) > 2 * c).sum()))
    return _fermat(Z, lab, G, dens) if fermat else lab


def tomato_kvote(Z, fermat=False):
    ks = []
    for k in (5, 7, 10, 14, 20):
        nb, d, _ = _sym_graph(Z, k)
        ks.append(_gap_k(_prom(_tomato(nb, _density(d, Z.shape[1])))))
    nbrs, d, G, dens, t = _setup(Z)
    lab = _set_k(t, int(np.median(ks)))
    return _fermat(Z, lab, G, dens) if fermat else lab


def hdbscan_fill(Z, p=4):
    n = len(Z)
    mcs = max(8, n // 250)
    lab = M.hdbscan_labels(Z, mcs, max(3, mcs // 2))
    if (lab < 0).all() or (lab >= 0).all():
        return lab
    _, d, G = _sym_graph(Z)
    C = lab.max() + 1                                                  # one virtual source per cluster, cost 0
    members = np.flatnonzero(lab >= 0)
    H = G.copy(); H.data = H.data ** p
    S = coo_matrix((np.full(len(members), 1e-12), (n + lab[members], members)), shape=(n + C, n + C))
    H = coo_matrix(H); H = coo_matrix((H.data, (H.row, H.col)), shape=(n + C, n + C))
    A = (H + S).tocsr()
    _, _, src = dijkstra(A, directed=True, indices=np.arange(n, n + C), min_only=True, return_predecessors=True)
    out = lab.copy()
    out[lab < 0] = src[:n][lab < 0] - n
    return out


def hdbscan_micro(Z, k=3, ratio=0.5, max_size=15):
    from sklearn.neighbors import NearestNeighbors
    n = len(Z)
    mcs = max(8, n // 250)
    lab = M.hdbscan_labels(Z, mcs, max(3, mcs // 2)).copy()
    noise = np.flatnonzero(lab < 0)
    if len(noise) < 2 or (lab < 0).all():
        return lab
    dn, nn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z[noise])
    to_clu = NearestNeighbors(n_neighbors=1).fit(Z[lab >= 0]).kneighbors(Z[noise])[0][:, 0]
    pos = {int(p): i for i, p in enumerate(noise)}
    near = [set(map(int, r[1:])) for r in nn]
    ii, jj = [], []
    for a, p in enumerate(noise):
        for b_pt, dd in zip(nn[a][1:], dn[a][1:]):
            b = pos.get(int(b_pt))
            if b is not None and int(p) in near[b] and dd < ratio * min(to_clu[a], to_clu[b]):
                ii.append(a); jj.append(b)
    if not ii:
        return lab
    m = len(noise)
    comp = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(m, m)), directed=False)[1]
    size = np.bincount(comp)
    nxt = lab.max() + 1
    for c in np.flatnonzero((size >= 2) & (size <= max_size)):
        lab[noise[comp == c]] = nxt
        nxt += 1
    return lab


def inner_micro(Z, lab, k=3, ratio=0.5, max_size=15, min_parent=10):
    """Tight small groups inside clusters (the Gaia moving-group idea): inside each cluster of >= min_parent points,
    mutual k-nearest neighbours closer than ratio x the cluster's median 10th-neighbour distance are linked; pieces of
    2..max_size points (and under half the cluster) become new clusters."""
    from sklearn.neighbors import NearestNeighbors
    lab = lab.copy()
    nxt = lab.max() + 1
    for c in [c for c in np.unique(lab) if c >= 0]:
        P = np.flatnonzero(lab == c)
        if len(P) < min_parent:
            continue
        kk = min(10, len(P) - 1)
        dk, nk = NearestNeighbors(n_neighbors=kk + 1).fit(Z[P]).kneighbors(Z[P])
        typical = np.median(dk[:, -1])
        near = [set(r[1:k + 1]) for r in nk]
        ii, jj = [], []
        for a in range(len(P)):
            for b, dd in zip(nk[a][1:k + 1], dk[a][1:k + 1]):
                if a in near[b] and dd < ratio * typical:
                    ii.append(a); jj.append(b)
        if not ii:
            continue
        comp = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(P),) * 2), directed=False)[1]
        size = np.bincount(comp)
        for g in np.flatnonzero((size >= 2) & (size <= max_size) & (size < len(P) / 2)):
            lab[P[comp == g]] = nxt
            nxt += 1
    return lab


register("tomato_sym", tomato_sym)
register("hdbscan_micro_inner", lambda Z: inner_micro(Z, hdbscan_micro(Z)))
register("hdbscan_micro", hdbscan_micro)
register("tomato_loggap", tomato_loggap)
register("tomato_loggap_fermat", lambda Z: tomato_loggap(Z, True))
register("tomato_boot", tomato_boot)
register("tomato_boot_fermat", lambda Z: tomato_boot(Z, True))
register("tomato_kvote", tomato_kvote)
register("tomato_kvote_fermat", lambda Z: tomato_kvote(Z, True))
register("hdbscan_fill", hdbscan_fill)
