"""
Idea 4, particle competition with extinction and abstention (after Silva & Zhao 2012): one particle per density peak
of a kNN graph; each particle walks (preferring nodes it already holds), raises its hold on the nodes it visits, gains
energy on its own nodes and loses it on others'. A particle with no energy goes extinct for good (so the number of
clusters emerges) instead of restarting; a node whose strongest hold is weak is left unassigned (abstains). The brake
is local (energy), not a size-balancing term.

Idea 3, saddle test: start from every density peak (ToMATo without merging) and merge two neighbouring peaks when the
density at points on the segment between them (queries only, nothing added to the data) stays above a fixed share of
the lower peak. Compared with ToMATo's own prominence merge on the same density.

Baselines specific to these: label-propagation consensus (co-assignment over runs) and Leiden.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .run_std import register


def _knn(Z, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    d, idx = nn.kneighbors(Z)
    return nn, d[:, 1:], idx[:, 1:]


def _peaks(dens, idx):
    return np.where(dens >= dens[idx].max(1))[0]


def particles(Z, k=15, lam=0.6, delta=0.1, steps_per_node=200, abstain=1.5, seed=0, max_particles=300):
    """abstain: a node is labelled only if its strongest hold is >= abstain x the second strongest."""
    rng = np.random.default_rng(seed)
    n = len(Z)
    _, d, idx = _knn(Z, k)
    dens = 1.0 / (d.mean(1) + 1e-12)
    seeds = _peaks(dens, idx)
    if len(seeds) > max_particles:
        seeds = seeds[np.argsort(-dens[seeds])[:max_particles]]
    P = len(seeds)
    V = np.full((n, P), 1.0 / P)
    V[seeds] = 0
    V[seeds, np.arange(P)] = 1.0
    fixed = np.zeros(n, bool)
    fixed[seeds] = True
    pos, home = seeds.copy(), seeds.copy()
    E = np.full(P, 0.5)
    alive = np.ones(P, bool)
    for _ in range(int(steps_per_node * n / max(P, 1))):
        a = np.where(alive)[0]
        if len(a) <= 1:
            break
        nb = idx[pos[a]]                                       # (|a|, k)
        pref = V[nb, a[:, None]]
        w = lam * pref / (pref.sum(1, keepdims=True) + 1e-12) + (1 - lam) / k
        r = rng.random(len(a))[:, None]
        j = nb[np.arange(len(a)), (np.cumsum(w, 1) < r).sum(1).clip(0, k - 1)]
        nf = ~fixed[j]
        if nf.any():                                           # raise p's hold on j, lower the others'
            jj, pp = j[nf], a[nf]
            dec = np.minimum(V[jj], (delta * E[pp] / max(len(a) - 1, 1))[:, None])
            dec[np.arange(len(jj)), pp] = 0
            dec[:, ~alive] = V[jj][:, ~alive]                  # extinct particles release their hold
            V[jj] -= dec
            V[jj, pp] += dec.sum(1)
        own = V[j].argmax(1) == a
        E[a] = np.clip(E[a] + np.where(own, delta, -delta), 0, 1)
        pos[a] = j
        dead = a[E[a] <= 0]
        if len(dead):
            alive[dead] = False
            V[:, dead] = 0
            row = V.sum(1, keepdims=True)
            V = np.where(row > 0, V / np.maximum(row, 1e-12), V)
    top2 = np.sort(V, 1)[:, -2:] if V.shape[1] > 1 else np.c_[np.zeros(n), V[:, 0]]
    lab = V.argmax(1)
    lab[top2[:, 1] < abstain * top2[:, 0]] = -1
    _, lab2 = np.unique(lab, return_inverse=True)
    return np.where(lab < 0, -1, lab2 - (1 if (lab < 0).any() else 0))


def saddle_merge(Z, k=15, share=0.5, n_query=8):
    """Merge neighbouring density peaks when density on the segment between them stays >= share x the lower peak."""
    nn, d, idx = _knn(Z, k)
    dens = 1.0 / (d.mean(1) + 1e-12)
    up = np.array([i if dens[i] >= dens[idx[i]].max() else idx[i][np.argmax(dens[idx[i]])] for i in range(len(Z))])
    root = up.copy()                                          # hill-climb to a peak
    for _ in range(64):
        nxt = root[root]
        if (nxt == root).all():
            break
        root = nxt
    peaks = np.unique(root)
    pid = {p: i for i, p in enumerate(peaks)}
    parent = list(range(len(peaks)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = {(min(root[i], root[j]), max(root[i], root[j])) for i in range(len(Z)) for j in idx[i] if root[i] != root[j]}
    t = np.linspace(0, 1, n_query + 2)[1:-1]
    for a, b in sorted(pairs, key=lambda ab: -min(dens[ab[0]], dens[ab[1]])):
        q = Z[a] + t[:, None] * (Z[b] - Z[a])
        dq = 1.0 / (nn.kneighbors(q, n_neighbors=k)[0].mean(1) + 1e-12)
        if dq.min() >= share * min(dens[a], dens[b]):
            parent[find(pid[a])] = find(pid[b])
    lab = np.array([find(pid[r]) for r in root])
    return np.unique(lab, return_inverse=True)[1]


def tomato_unmerged_then_gap(Z, k=15):
    from .methods import tomato_labels
    return tomato_labels(Z, k=k)


def lpa_consensus(Z, k=15, runs=10, seed=0):
    import igraph as ig
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    _, _, idx = _knn(Z, k)
    edges = list({(min(i, j), max(i, j)) for i, row in enumerate(idx) for j in row})
    g = ig.Graph(n=len(Z), edges=edges)
    labs = []
    for r in range(runs):
        import random
        ig.set_random_number_generator(random.Random(seed + r))
        labs.append(np.array(g.community_label_propagation().membership))
    e = np.array(edges)
    agree = np.mean([l[e[:, 0]] == l[e[:, 1]] for l in labs], 0) > 0.5
    m = coo_matrix((np.ones(agree.sum()), (e[agree, 0], e[agree, 1])), shape=(len(Z), len(Z)))
    return connected_components(m, directed=False)[1]


register("idea4_particles", lambda Z: particles(Z))
register("idea3_saddle", lambda Z: saddle_merge(Z))
register("lpa_consensus", lambda Z: lpa_consensus(Z))
