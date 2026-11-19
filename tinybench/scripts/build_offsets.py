"""Builds src/tinybench/data/offsets_v1.npz, the library of real rare-group shapes used by shape="boot".

Donors (public data only): 10X PBMC, each of the 8 classes downsampled to 16 real cells (seed 99) in its own scene
(normalise, log1p, 2,000 HVG, scale, PCA 50); KDD Cup 99 rare attack types with 2-60 records in the seed-0 scene
(PCA 20). Each donor is stored as member-minus-centroid offsets (m x D). Shapes only: no labels, no positions.

  TINYBENCH_PBMC10X=/path/10X_PBMC.h5 python scripts/build_offsets.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinybench.real import _kdd_frame, _pbmc, kdd_scene, preprocess_counts  # noqa: E402


def main():
    out = {}
    X, y = _pbmc()
    for c in np.unique(y):
        r = np.random.default_rng(99)
        idx = np.r_[np.flatnonzero(y != c), r.choice(np.flatnonzero(y == c), 16, replace=False)]
        Z = preprocess_counts(X[idx]); g = np.flatnonzero(y[idx] == c)
        out[f"pbmc10x_class{c}"] = (Z[g] - Z[g].mean(0)).astype(np.float32)
    df, yk = _kdd_frame()
    Z, yy, rare = kdd_scene(df, yk, 0)
    for t in sorted(rare):
        g = np.flatnonzero(yy == t)
        if 2 <= len(g) <= 60:
            out[f"kdd99_{t.strip('.')}"] = (Z[g] - Z[g].mean(0)).astype(np.float32)
    p = os.path.join(os.path.dirname(__file__), "..", "src", "tinybench", "data", "offsets_v1.npz")
    np.savez_compressed(p, **out)
    print(p, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
