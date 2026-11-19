"""
Scene generator for tiny groups (2-9 points).

Vendored from clustering.eval.nature.tiny_bench.make (same project, version used for all published numbers). With
shape="iso" the output is bit-identical to that function (a test checks this when the source package is importable).

A scene: 30 templates x 30 near-copies (sd 0.02, the "near-duplicate mass"), 8 Gaussian background clusters of 30-150
points (sd 0.3-0.8), 2% uniform noise, and 12 planted groups of 2-9 points. The tier is the group's placement:
- isolated: at least 2.0 away from every point;
- touching: just outside a background cluster's shell (distance sd * (sqrt(d) + 1.5) from its centre);
- inside: next to a near-duplicate template (6 copy-sd away).
Contrast = group spread as a share of its neighbour's spread: high 0.25, mid 0.5, low 1.0 (inside: x2 the copy sd).
Points live in intrinsic dimension d (2, 5, 10), mapped to 20 ambient dimensions by a random orthonormal map, plus
ambient noise of sd 0.002.

Group shapes (new in v1.1, the "realistic plant" tracks; the scene skeleton, centres, sizes and spreads are identical
across shapes for a given seed, so tracks are paired):
- iso: isotropic Gaussian (v1.0 behaviour).
- aniso: Gaussian with a random orientation and a power-law spectrum (eigenvalues proportional to i^-2), rescaled so
  the total variance equals the isotropic one (trace = d sd^2).
- t3: multivariate Student t with 3 degrees of freedom, variance-matched to the isotropic one (heavy tails).
- boot: offsets resampled from real public rare groups (KDD Cup 99 rare attack types and 10X PBMC downsampled
  classes; library in data/offsets_v1.npz), each donor group projected on its top-d principal axes, scaled to the
  isotropic group's RMS radius (sd sqrt(d)) and randomly rotated.
"""
import os

import numpy as np

D_AMBIENT = 20
TIERS = ("isolated", "touching", "inside")
CONTRASTS = ("high", "mid", "low")
DIMS = (2, 5, 10)
CONTRAST = {"high": 0.25, "mid": 0.5, "low": 1.0}
SHAPES = ("iso", "aniso", "t3", "boot")
_OFFSETS = None


def _offset_library():
    global _OFFSETS
    if _OFFSETS is None:
        z = np.load(os.path.join(os.path.dirname(__file__), "data", "offsets_v1.npz"), allow_pickle=False)
        _OFFSETS = [z[k] for k in sorted(z.files)]
    return _OFFSETS


def _shape_points(shape, m, d, sd, rng):
    """m points with zero-ish centre, total variance about d * sd^2, drawn from rng (a stream separate from the scene)."""
    if shape == "aniso":
        lam = 1.0 / np.arange(1, d + 1) ** 2
        lam = lam / lam.sum() * d
        R = np.linalg.qr(rng.normal(size=(d, d)))[0]
        return (rng.normal(size=(m, d)) * np.sqrt(lam)) @ R.T * sd
    if shape == "t3":
        E = rng.normal(size=(m, d)) / np.sqrt(rng.chisquare(3, size=(m, 1)) / 3)
        return E / np.sqrt(3) * sd
    if shape == "boot":
        lib = _offset_library()
        off = lib[rng.integers(len(lib))]
        off = off - off.mean(0)
        _, _, Vt = np.linalg.svd(off, full_matrices=False)
        P = off @ Vt[:d].T                                       # top-d axes (fewer if the donor has < d points)
        if P.shape[1] < d:
            P = np.c_[P, np.zeros((len(P), d - P.shape[1]))]
        rms = np.sqrt((P ** 2).sum(1).mean())
        P = P / max(rms, 1e-12) * sd * np.sqrt(d)
        R = np.linalg.qr(rng.normal(size=(d, d)))[0]
        return P[rng.integers(len(P), size=m)] @ R.T
    raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")


def make(placement, d=5, seed=0, n_groups=12, tpl=30, copies=30, copy_sd=0.02, contrast="mid", shape="iso"):
    """Returns (X, kind, gid): X (n x 20) float, kind in {mass, background, noise, group}, gid = planted group id or -1."""
    if placement not in TIERS:
        raise ValueError(f"placement must be one of {TIERS}")
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng([seed, 7919, SHAPES.index(shape)])
    X, kind, gid = [], [], []
    tc = rng.normal(0, 1.0, (tpl, d))
    for c in tc:
        X.append(c + rng.normal(0, copy_sd, (copies, d))); kind += ["mass"] * copies; gid += [-1] * copies
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
            c, sd = b + u * bsd * (np.sqrt(d) + 1.5), bsd * CONTRAST[contrast]
        else:
            a = tc[rng.integers(tpl)]
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            c, sd = a + u * 6 * copy_sd, copy_sd * CONTRAST[contrast] * 2
        pts = c + rng.normal(0, sd, (m, d))                      # always drawn, so the main stream is shape-independent
        if shape != "iso":
            pts = c + _shape_points(shape, m, d, sd, srng)
        X.append(pts); kind += ["group"] * m; gid += [g] * m
    Z = np.vstack(X)
    Q = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_AMBIENT)))[0][:, :d]
    A = Z @ Q.T + rng.normal(0, 0.002, (len(Z), D_AMBIENT))
    return A, np.array(kind), np.array(gid)


def generate(placement, contrast="mid", d=5, seed=0, shape="iso"):
    """Public API: one scene. Returns a dict with X, kind, gid and the scene's parameters."""
    X, kind, gid = make(placement, d, seed, contrast=contrast, shape=shape)
    return {"X": X, "kind": kind, "gid": gid, "placement": placement, "contrast": contrast, "d": d, "seed": seed,
            "shape": shape}
