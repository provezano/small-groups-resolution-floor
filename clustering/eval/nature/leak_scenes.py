"""
Stage 15 (leak): the tiny-bench generator of tiny_bench.make, replayed with the same random calls so that every point
also carries its background-cluster id, the cluster's centre and spread. `make_labeled` returns exactly the points of
tiny_bench.make (tested), plus `cl` (cluster id, -1 for mass / noise / group points) and `info` (centres, spreads,
sizes, the orthonormal map Q). Public synthetic data only.
"""
import numpy as np

from . import tiny_bench as B


def make_labeled(placement, d=5, seed=0, n_groups=12, tpl=30, copies=30, copy_sd=0.02, contrast="mid"):
    rng = np.random.default_rng(seed)
    X, kind, gid, cl = [], [], [], []
    tc = rng.normal(0, 1.0, (tpl, d))
    for c in tc:
        X.append(c + rng.normal(0, copy_sd, (copies, d))); kind += ["mass"] * copies; gid += [-1] * copies
        cl += [-1] * copies
    bc, sizes = [], []
    for j in range(8):
        while True:
            c = rng.normal(0, 6.0, d)
            if np.linalg.norm(c) > 4 and all(np.linalg.norm(c - b[0]) > 3 for b in bc):
                break
        bc.append(c)
        m, sd = int(rng.integers(30, 151)), float(rng.uniform(0.3, 0.8))
        X.append(c + rng.normal(0, sd, (m, d))); kind += ["background"] * m; gid += [-1] * m; cl += [j] * m
        bc[-1] = (c, sd)
        sizes.append(m)
    allpts = np.vstack(X)
    lo, hi = allpts.min(0) - 2, allpts.max(0) + 2
    nn = int(0.02 * len(allpts))
    X.append(rng.uniform(lo, hi, (nn, d))); kind += ["noise"] * nn; gid += [-1] * nn; cl += [-1] * nn
    for g in range(n_groups):
        m = int(rng.integers(2, 10))
        if placement == "isolated":
            while True:
                c = rng.uniform(lo, hi)
                if np.min(np.linalg.norm(allpts - c, axis=1)) > 2.0:
                    break
            sd = 0.5 * B.CONTRAST[contrast]
        elif placement == "touching":
            b, bsd = bc[rng.integers(len(bc))]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = b + u * bsd * (np.sqrt(d) + 1.5), bsd * B.CONTRAST[contrast]
        else:
            a = tc[rng.integers(tpl)]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = a + u * 6 * copy_sd, copy_sd * B.CONTRAST[contrast] * 2
        X.append(c + rng.normal(0, sd, (m, d))); kind += ["group"] * m; gid += [g] * m; cl += [-1] * m
    Z = np.vstack(X)
    Q = np.linalg.qr(rng.normal(size=(B.D_AMBIENT, B.D_AMBIENT)))[0][:, :d]
    A = Z @ Q.T + rng.normal(0, 0.002, (len(Z), B.D_AMBIENT))
    info = {"centres": np.array([b[0] for b in bc]), "sds": np.array([b[1] for b in bc]), "sizes": np.array(sizes),
            "Q": Q, "latent": Z, "lo": lo, "hi": hi, "n_noise": nn}
    return A, np.array(kind), np.array(gid), np.array(cl), info
