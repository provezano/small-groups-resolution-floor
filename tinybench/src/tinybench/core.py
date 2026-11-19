"""Frontier and ceiling: recall at matched false-group rates, per tier (placement x contrast, pooled over d)."""
import numpy as np
from joblib import Parallel, delayed

from .generator import CONTRASTS, DIMS, TIERS, make
from .scoring import score
from .seeds import seeds as _seeds
from . import stats as S

CAPS = (0.25, 0.5)
ORACLE_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3)


def _run(fit, grid, seeds, shape, n_jobs, tiers=TIERS, contrasts=CONTRASTS, dims=DIMS, scene_aware=False):
    def one(p, c, d, s):
        X, kind, gid = make(p, d, s, contrast=c, shape=shape)
        if scene_aware:
            sc = [score(fit(X, g, p, d, s, c), kind, gid) for g in grid]
        else:
            sc = [score(fit(X, g), kind, gid) for g in grid]
        return {"placement": p, "contrast": c, "d": d, "seed": s, "scores": {"m": sc}}
    return Parallel(n_jobs=n_jobs)(delayed(one)(p, c, d, s) for p in tiers for c in contrasts for d in dims for s in seeds)


def summarise(rows, method="m", B=2000, ci=True, seed=0):
    """Per tier: recall at each cap (+ 95% bootstrap CI over scenes, stratified by d) and the pooled (false, recall)
    points of the grid."""
    out = {}
    for key in sorted({(r["placement"], r["contrast"]) for r in rows}, key=lambda k: (TIERS.index(k[0]), CONTRASTS.index(k[1]))):
        rs = [r for r in rows if (r["placement"], r["contrast"]) == key]
        R, F = S.arrays(rs, method)
        t = {}
        W = S.boot_weights(rs, B, seed) if ci else None
        for cap in CAPS:
            if ci:
                v = S.frontier_ci(rs, method, cap, W)
                t[f"recall@{cap}"] = v["value"]; t[f"ci@{cap}"] = v["ci"]
            else:
                t[f"recall@{cap}"] = round(S.frontier_from(R, F, cap), 3)
        t["points"] = [(round(float(a), 3), round(float(b), 3)) for a, b in zip(F.mean(0), R.mean(0))]
        out[f"{key[0]}/{key[1]}"] = t
    return out


def frontier(fit, grid, seeds=None, shape="iso", n_jobs=2, ci=True, return_rows=False, **kw):
    """fit(X, param) -> labels. Returns {tier: {recall@0.25, ci@0.25, recall@0.5, ci@0.5, points}}."""
    seeds = _seeds("test") if seeds is None else seeds
    rows = _run(fit, list(grid), seeds, shape, n_jobs, **kw)
    res = summarise(rows, ci=ci)
    return (res, rows) if return_rows else res


def ceiling(seeds=None, n_jobs=2, ci=True, return_rows=False, **kw):
    """The oracle scan's frontier (true background density): the per-tier reference."""
    from .oracle import oracle_labels
    seeds = _seeds("test") if seeds is None else seeds

    def fit(X, al, p, d, s, c):
        return oracle_labels(X, p, d, s, c, (al,))[0]
    rows = _run(fit, list(ORACLE_ALPHAS), seeds, "iso", n_jobs, scene_aware=True, **kw)
    res = summarise(rows, ci=ci)
    return (res, rows) if return_rows else res
