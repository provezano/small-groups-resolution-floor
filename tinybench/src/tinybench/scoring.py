"""
Scoring, vendored from clustering.eval.nature.tiny_bench.score (same project).

- recall: share of planted groups matched by some predicted cluster at Jaccard >= 0.5;
- false_per100: predicted clusters of <= 15 points holding no planted point, per 100 points;
- mass_pieces: predicted clusters holding >= 5 near-duplicate mass points;
- noise: share of points labelled -1.
"""
import numpy as np


def score(labels, kind, gid):
    lab = np.asarray(labels)
    kind, gid = np.asarray(kind), np.asarray(gid)
    if len(lab) != len(gid):
        raise ValueError(f"labels has {len(lab)} entries, scene has {len(gid)} points")
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
