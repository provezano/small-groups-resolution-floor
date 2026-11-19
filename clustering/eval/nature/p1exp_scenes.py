"""
Scene variants of the tiny-group benchmark for robustness checks. `make_variant` follows `tiny_bench.make` line by
line (same random draws in the same order) and adds knobs; with every knob at its default it returns exactly the
scene `tiny_bench.make` returns (checked in the tests). Knobs:
- noise: share of uniform noise points (benchmark: 0.02);
- bg_df: background clusters drawn from a Student-t with this many degrees of freedom (same scale), None = Gaussian;
- aniso: group spread along one random direction is multiplied by aniso and the other directions shrunk so the
  volume is unchanged (elongated groups), 1 = round;
- outlier_member: each group gets one extra member at 3 group radii (sd x sqrt(d)) from its centre, in a random
  direction; it carries the group's id, so recall still needs Jaccard >= 0.5 against the whole group;
- scale: number of templates, background-cluster sizes (and so the noise count) multiplied by scale; the 12 groups
  stay 12 (n ~ 960, 1620, 3290, 7430 at scale 0.5, 1, 2, 4).
Extra randomness (t mixing, directions, outlier members) comes from a second generator seeded from the scene seed, so
the base draws match the benchmark's.
"""
import numpy as np

from .tiny_bench import CONTRAST, D_AMBIENT


def make_variant(placement, d=5, seed=0, n_groups=12, tpl=30, copies=30, copy_sd=0.02, contrast="mid",
                 noise=0.02, bg_df=None, aniso=1.0, outlier_member=False, scale=1.0):
    rng = np.random.default_rng(seed)
    rx = np.random.default_rng([seed, 9173])
    tpl = int(round(tpl * scale))
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
        pts = rng.normal(0, sd, (m, d))
        if scale != 1.0:
            m2 = int(round(m * scale))
            pts = pts[:m2] if m2 <= m else np.vstack([pts, rx.normal(0, sd, (m2 - m, d))])
            m = m2
        if bg_df is not None:
            pts = pts / np.sqrt(rx.chisquare(bg_df, (m, 1)) / bg_df)          # multivariate t, scale sd
        X.append(c + pts); kind += ["background"] * m; gid += [-1] * m
        bc[-1] = (c, sd)
    allpts = np.vstack(X)
    lo, hi = allpts.min(0) - 2, allpts.max(0) + 2
    if bg_df is not None:            # the noise box follows the bulk (0.5-99.5% per axis), not the t tails
        lo, hi = np.quantile(allpts, 0.005, 0) - 2, np.quantile(allpts, 0.995, 0) + 2
    nn = int(noise * len(allpts))
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
        pts = rng.normal(0, sd, (m, d))
        if aniso != 1.0:
            Qg = np.linalg.qr(rx.normal(size=(d, d)))[0]
            s = np.full(d, aniso ** (-1.0 / (d - 1)))
            s[0] = aniso
            pts = (pts @ Qg) * s @ Qg.T
        X.append(c + pts); kind += ["group"] * m; gid += [g] * m
        if outlier_member:
            v = rx.normal(size=d); v /= np.linalg.norm(v)
            X.append((c + v * 3 * sd * np.sqrt(d))[None]); kind += ["group"]; gid += [g]
    Z = np.vstack(X)
    Q = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_AMBIENT)))[0][:, :d]
    A = Z @ Q.T + rng.normal(0, 0.002, (len(Z), D_AMBIENT))
    return A, np.array(kind), np.array(gid)


VARIANTS = {
    "base": {},
    "noise10": {"noise": 0.10},
    "noise25": {"noise": 0.25},
    "t3_background": {"bg_df": 3},
    "elongated": {"aniso": 4.0},
    "outlier_member": {"outlier_member": True},
    "scale0.5": {"scale": 0.5},
    "scale2": {"scale": 2.0},
    "scale4": {"scale": 4.0},
}


def frontier_variant(fit, grid, seeds, variant, n_jobs=2, tiers=None, contrasts=("high", "mid", "low"), dims=(2, 5, 10)):
    """`tinybench_cli.frontier` with the scene maker swapped for make_variant(**variant); same pooling and caps.
    fit(X, g) -> labels. (With variant = {} it returns exactly what the CLI frontier returns; see the tests.)"""
    from joblib import Parallel, delayed
    from .tiny_bench import TIERS, score
    tiers = tiers or TIERS

    def one(p, c, d, s):
        X, kind, gid = make_variant(p, d, s, contrast=c, **variant)
        return (p, c), [score(fit(X, g), kind, gid) for g in grid]
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(p, c, d, s) for p in tiers for c in contrasts for d in dims for s in seeds)
    out = {}
    for p in tiers:
        for c in contrasts:
            rs = [r for k, r in rows if k == (p, c)]
            pts = [(float(np.mean([r[i]["false_per100"] for r in rs])), float(np.mean([r[i]["recall"] for r in rs])))
                   for i in range(len(grid))]
            out[f"{p}/{c}"] = {f"recall@{cap}": round(max([rc for fp, rc in pts if fp <= cap] or [0.0]), 3)
                               for cap in (0.25, 0.5)}
            out[f"{p}/{c}"]["points"] = [(round(a, 3), round(b, 3)) for a, b in pts]
    return out
