"""
Standard clustering benchmarks (Gagolewski et al., clustering-data-v1 1.1.0) in pre-set tiers, so a method is judged on
instances other papers use, not only on our data. Tiers follow the dataset authors' descriptions (overlap, shape,
noise) and were fixed before any run; "imbalanced" = largest/smallest true cluster >= 8 (our regime, overlaps the
others). Points are z-scored per dataset. Scores on non-noise points (label 0 = noise in the suite): AMI, ARI with
predicted noise as singletons, K found, and minority recall = share of true clusters smaller than the median size that
come back with Jaccard >= 0.5 ("small" = at most a quarter of the largest true cluster).
"""
import os

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

DATA = os.environ.get("CLUSTER_BENCH_DATA", "")
TIERS = {
    "easy": "sipu/r15 sipu/d31 sipu/a1 sipu/s1 fcps/hepta fcps/tetra g2mg/g2mg_2_10 g2mg/g2mg_16_10 wut/z1",
    "medium": "sipu/s2 sipu/s3 sipu/a3 sipu/aggregation sipu/compound sipu/jain sipu/flame fcps/lsun fcps/twodiamonds "
              "fcps/wingnut uci/wine other/iris g2mg/g2mg_2_40 g2mg/g2mg_16_40 wut/mk3 wut/x1",
    "hard": "sipu/s4 sipu/pathbased sipu/spiral fcps/chainlink fcps/atom fcps/engytime graves/ring_noisy "
            "graves/zigzag_outliers wut/x2 wut/x3 wut/trapped_lovers uci/glass g2mg/g2mg_2_70 g2mg/g2mg_16_70 "
            "g2mg/g2mg_128_40 wut/mk4",
    "imbalanced": "fcps/target uci/yeast uci/ecoli sipu/unbalance sipu/compound wut/z2 sipu/aggregation uci/glass",
}


def load(name, root=None):
    root = root or DATA
    X = np.loadtxt(os.path.join(root, f"{name}.data.gz"), ndmin=2)
    y = np.loadtxt(os.path.join(root, f"{name}.labels0.gz"), dtype=int)
    sd = X.std(0)
    X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    return X, y


def score(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    m = y > 0
    p = pred.copy()
    noise = p < 0
    p[noise] = p.max() + 1 + np.arange(noise.sum())
    sizes = {c: int((y == c).sum()) for c in np.unique(y[m])}
    big = max(sizes.values())
    rec = []
    for c, s in sizes.items():
        if s > big / 4:
            continue
        mc = y == c
        best = max([(mc & (pred == q)).sum() / (mc | (pred == q)).sum() for q in set(pred[mc]) - {-1}] or [0])
        rec.append(best >= 0.5)
    return {"ami": round(float(adjusted_mutual_info_score(y[m], p[m])), 3),
            "ari": round(float(adjusted_rand_score(y[m], p[m])), 3),
            "k_found": int(len(set(pred) - {-1})), "k_true": len(sizes),
            "minority_recall": round(float(np.mean(rec)), 3) if rec else None,
            "noise": round(float((pred < 0).mean()), 3)}
