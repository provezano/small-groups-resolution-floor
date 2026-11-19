"""
tinybench: a benchmark for finding tiny groups (2-15 points) in unlabelled data.

Stable API (v1.x):
    generate(placement, contrast="mid", d=5, seed=0, shape="iso") -> dict(X, kind, gid, ...)
    score(labels, kind, gid) -> dict(recall, false_per100, mass_pieces, noise)
    frontier(fit, grid, seeds=None, shape="iso") -> {tier: {recall@0.25, ci@0.25, recall@0.5, ci@0.5, points}}
    ceiling(seeds=None) -> the oracle scan's frontier (per-tier reference)
    load_real(name, seed=0, ...) -> dict(X, y, groups, ...)   name in {"kdd99", "pbmc10x", "pbmc10x_counts"}
    score_real(labels, groups) -> dict(recall, found, false_per100)
    baselines.get(name) -> (fit(X, param) -> labels, default grid)
    seeds.seeds(split="test", version="1.1")
"""
from .core import ceiling, frontier
from .generator import CONTRASTS, DIMS, SHAPES, TIERS, generate, make
from .real import load_real, score_real
from .scoring import score
from . import baselines, seeds, stats, tracks

__version__ = "1.1.0"
__all__ = ["generate", "make", "score", "frontier", "ceiling", "load_real", "score_real", "baselines", "seeds",
           "stats", "tracks", "TIERS", "CONTRASTS", "DIMS", "SHAPES"]
