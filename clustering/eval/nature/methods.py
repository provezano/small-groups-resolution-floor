"""
Clusterers compared in the method experiments. Each takes points Z and returns labels (-1 = noise).

Baselines: HDBSCAN (excess-of-mass and leaf selection, and with cluster_selection_epsilon), PLSCAN, ToMATo (number of
clusters at the widest gap of the prominence diagram), Leiden on a kNN graph.

Idea 2, per-branch persistence: same HDBSCAN tree, but each point's lifetime in a cluster is measured in log density
(log lambda) rather than lambda. A cluster's score then reads "how many times denser can you go before it splits or
empties", so a dense pile is not worth more just because its density numbers are larger. Exact duplicates (lambda =
inf) are capped. Selection is the usual excess-of-mass dynamic programme over the condensed tree.
"""
import numpy as np


def hdbscan_fit(Z, mcs, ms, method="eom", eps=0.0):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=method,
                           cluster_selection_epsilon=float(eps), allow_single_cluster=False).fit(Z.astype(np.float64))


def hdbscan_labels(Z, mcs, ms, method="eom", eps=0.0):
    return hdbscan_fit(Z, mcs, ms, method, eps).labels_


def _tree(cl):
    t = cl.condensed_tree_.to_numpy()
    lam = t["lambda_val"].astype(float)
    fin = lam[np.isfinite(lam)]
    lam = np.where(np.isfinite(lam), lam, (fin.max() if len(fin) else 1.0) * 10)
    return t["parent"].astype(int), t["child"].astype(int), lam, t["child_size"].astype(int)


def select(cl, measure="log"):
    """Excess-of-mass selection over cl's condensed tree with lifetimes in lambda ("lin") or log lambda ("log")."""
    par, chi, lam, size = _tree(cl)
    n = int(par.min())                                  # root id == number of points
    f = np.log if measure == "log" else (lambda v: v)
    floor = lam[lam > 0].min() if (lam > 0).any() else 1e-12
    lv = f(np.maximum(lam, floor))
    birth = {n: f(floor)}
    kids = {}
    for p, c, l, s in zip(par, chi, lv, size):
        if c >= n:
            birth[c] = l
            kids.setdefault(p, []).append(c)
    stab = {c: 0.0 for c in birth}
    for p, c, l, s in zip(par, chi, lv, size):
        stab[p] += (l - birth[p]) * s
    best, keep = {}, {}
    for c in sorted(birth, reverse=True):              # children have larger ids than parents
        if c == n:
            continue
        sub = sum(best[k] for k in kids.get(c, []))
        if kids.get(c) and sub > stab[c]:
            best[c], keep[c] = sub, False
        else:
            best[c], keep[c] = stab[c], True
    chosen, stack = set(), list(kids.get(n, []))
    while stack:                                        # top-down: first kept cluster on each branch
        c = stack.pop()
        if keep[c]:
            chosen.add(c)
        else:
            stack += kids.get(c, [])
    parent_of = {c: p for p, c in zip(par, chi) if c >= n}
    lab, ids = np.full(n, -1), {c: i for i, c in enumerate(sorted(chosen))}
    for p, c in zip(par, chi):
        if c < n:
            q = p
            while q != n and q not in chosen:
                q = parent_of[q]
            if q in chosen:
                lab[c] = ids[q]
    return lab


def plscan_labels(Z, ms):
    from fast_plscan import PLSCAN
    return PLSCAN(min_samples=ms).fit(Z.astype(np.float64)).labels_


def tomato_labels(Z, k=10):
    from gudhi.clustering.tomato import Tomato
    t = Tomato(k=k, density_type="logDTM")
    t.fit(Z.astype(np.float64))
    dg = t.diagram_
    if len(dg) == 0:
        return np.asarray(t.labels_)
    prom = np.sort(dg[:, 0] - dg[:, 1])[::-1]
    gaps = prom[:-1] - prom[1:] if len(prom) > 1 else np.array([0.0])
    t.n_clusters_ = int(np.argmax(gaps) + 1) + 1        # + the infinitely persistent component(s)
    return np.asarray(t.labels_)


def leiden_labels(Z, k=15, seed=0):
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors
    nb = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    edges = {(min(i, j), max(i, j)) for i, row in enumerate(nb) for j in row}
    g = ig.Graph(n=len(Z), edges=list(edges))
    return np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, seed=seed).membership)
