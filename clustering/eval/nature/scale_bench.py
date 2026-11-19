"""
Scaled tiny-bench for the stage-9 scaling runs. A scene of n points is a set of independent tiny-bench tiles
(tiny_bench.make, each about 1,600 points with its own random 20-D embedding) placed far apart: each tile is shifted
by a random 20-D vector, at least 150 away from every other tile's shift. Local geometry (density, intrinsic d, group contrast) is the same as in the 1,600-point benchmark; only N,
and so the Bonferroni count, grows. Planted groups: 12 per tile in the first ceil(G / 12) tiles (G = 12..96), the
other tiles have none.
"""
import numpy as np

from .tiny_bench import D_AMBIENT, make


def make_scaled(placement, n_target, d=5, seed=0, n_groups=48, contrast="mid", spacing=150.0):
    rng = np.random.default_rng(seed)
    tiles = max(1, int(round(n_target / 1600)))
    g_tiles = min(tiles, int(np.ceil(n_groups / 12)))
    A, kinds, gids = [], [], []
    centres = []
    goff = 0
    for t in range(tiles):
        a, kind, gid = make(placement, d=d, seed=seed * 100003 + t, n_groups=12 if t < g_tiles else 0,
                            contrast=contrast)
        while True:
            c = rng.normal(size=D_AMBIENT)
            c *= spacing * (1 + tiles) ** (1 / D_AMBIENT) * 3 / np.linalg.norm(c) * rng.uniform(0.5, 1.0) ** (1 / D_AMBIENT)
            if not centres or np.min(np.linalg.norm(np.array(centres) - c, axis=1)) > spacing:
                break
        centres.append(c)
        A.append(a + c); kinds.append(kind)
        gids.append(np.where(gid >= 0, gid + goff, -1)); goff += 12 if t < g_tiles else 0
    return np.vstack(A), np.concatenate(kinds), np.concatenate(gids)
