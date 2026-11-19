"""
Benchmark generator for tiny groups (2-9 points) beside a near-duplicate mass, with the group's placement as the tier:
- isolated (easy): the group sits far from everything;
- touching (medium): the group sits just outside the edge of a background cluster;
- inside (hard): the group sits inside the near-duplicate mass, 6 copy spreads from a template.
Contrast sets how tight a group is against its neighbour: high = 1/4 of its spread, mid = 1/2, low = the same (for the
inside tier: 1/2, 1x and 2x the near-copy spread).
Points live in an intrinsic dimension d (2, 5, 10) mapped into 20 ambient dimensions by a random orthonormal map, plus
a little ambient noise. Scene: a mass of 30 templates x 30 near-copies (the automated job), 8 background clusters of
30-150 points with different spreads, 2% uniform noise, 12 tiny groups. Everything is public and synthetic.

Scoring: a planted group is recovered when a predicted cluster matches it at Jaccard >= 0.5. False groups = predicted
clusters of <= 15 points that hold no planted point, per 100 points. Mass pieces = predicted clusters holding >= 5 mass
points.
"""
import numpy as np

D_AMBIENT = 20
TIERS = ("isolated", "touching", "inside")


CONTRAST = {"high": 0.25, "mid": 0.5, "low": 1.0}      # group spread as a share of its neighbour's spread


def make(placement, d=5, seed=0, n_groups=12, tpl=30, copies=30, copy_sd=0.02, contrast="mid"):
    rng = np.random.default_rng(seed)
    X, kind, gid = [], [], []
    # mass: templates spread over the latent space, near-copies around each
    tc = rng.normal(0, 1.0, (tpl, d))
    for c in tc:
        X.append(c + rng.normal(0, copy_sd, (copies, d))); kind += ["mass"] * copies; gid += [-1] * copies
    # background clusters, away from the mass
    bc = []
    for _ in range(8):
        while True:
            c = rng.normal(0, 6.0, d)
            if np.linalg.norm(c) > 4 and all(np.linalg.norm(c - b[0]) > 3 for b in bc):
                break
        bc.append(c)
        m, sd = int(rng.integers(30, 151)), float(rng.uniform(0.3, 0.8))
        X.append(c + rng.normal(0, sd, (m, d))); kind += ["background"] * m; gid += [-1] * m
        bc[-1] = (c, sd)
    allpts = np.vstack(X)
    lo, hi = allpts.min(0) - 2, allpts.max(0) + 2
    nn = int(0.02 * len(allpts))
    X.append(rng.uniform(lo, hi, (nn, d))); kind += ["noise"] * nn; gid += [-1] * nn
    # tiny groups
    for g in range(n_groups):
        m = int(rng.integers(2, 10))
        if placement == "isolated":
            while True:
                c = rng.uniform(lo, hi)
                if np.min(np.linalg.norm(allpts - c, axis=1)) > 2.0:
                    break
            sd = 0.5 * CONTRAST[contrast]
        elif placement == "touching":
            b, bsd = bc[rng.integers(len(bc))]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = b + u * bsd * (np.sqrt(d) + 1.5), bsd * CONTRAST[contrast]   # just past the edge (radius ~ sd*sqrt(d))
        else:
            a = tc[rng.integers(tpl)]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = a + u * 6 * copy_sd, copy_sd * CONTRAST[contrast] * 2
        X.append(c + rng.normal(0, sd, (m, d))); kind += ["group"] * m; gid += [g] * m
    Z = np.vstack(X)
    Q = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_AMBIENT)))[0][:, :d]
    A = Z @ Q.T + rng.normal(0, 0.002, (len(Z), D_AMBIENT))
    return A, np.array(kind), np.array(gid)


def score(lab, kind, gid):
    lab = np.asarray(lab)
    rec = []
    for g in np.unique(gid[gid >= 0]):
        mg = gid == g
        best = max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0])
        rec.append(best >= 0.5)
    sizes = {c: int((lab == c).sum()) for c in set(lab) - {-1}}
    false = sum(1 for c, s in sizes.items() if s <= 15 and not (gid[lab == c] >= 0).any())
    mass = kind == "mass"
    pieces = sum(1 for c in set(lab[mass]) - {-1} if ((lab == c) & mass).sum() >= 5)
    return {"recall": float(np.mean(rec)), "false_per100": 100 * false / len(lab), "mass_pieces": pieces,
            "noise": float((lab < 0).mean())}

