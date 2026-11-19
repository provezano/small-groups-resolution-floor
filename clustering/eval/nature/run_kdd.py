"""
Real public data: KDD Cup 99 (10% file, via scikit-learn's fetcher). The rare attack types (2-53 records each: spy,
perl, phf, multihop, ftp_write, loadmodule, rootkit, imap, warezmaster, land, buffer_overflow, guess_passwd) are the
small groups. They sit beside large near-duplicate masses (smurf, neptune) and normal traffic, which is the "inside /
touching" setting of the tiny-group benchmark, on real records.

Scene (per seed): all rare-type records + a random sample of the large classes (normal 1500, smurf 600, neptune 600,
the mid-size attacks 300 in proportion). Features: one-hot protocol / service / flag, log1p of the numeric columns,
standardised, then PCA to 20 dims (fit on the scene). Exact duplicates break the continuous null of the scan, so the
points are dithered with noise of 1e-3 x the median non-zero 1-NN distance (stated, not hidden).

Every method runs at its default setting; nothing is tuned on the labels. Metrics per rare type:
- recovered: some predicted cluster matches the type at Jaccard >= 0.5 (as in the synthetic benchmark);
- hit: some predicted cluster of 2-60 points is >= 80% that type (finds part of it, cleanly).
Small-group precision: share of predicted clusters of 2-60 points that are >= 80% one rare type. Results:
results/kdd.json (local).
"""
import json
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import methods as M
from .tiny_baselines import gapclust
from .tiny_methods import cts3, cts3_hybrid, hdb, scored

try:
    from . import _tgpaths as _TGP
except ImportError:
    import _tgpaths as _TGP
DATA_HOME = os.environ.get("KDD_HOME", _TGP.DATA + "/dl/kdd")
RARE_MAX = 60
SAMPLE = {"normal.": 1500, "smurf.": 600, "neptune.": 600}
MID_TOTAL = 300


def load():
    from sklearn.datasets import fetch_kddcup99
    b = fetch_kddcup99(percent10=True, data_home=DATA_HOME, as_frame=True)
    df = b.frame
    y = df.pop("labels").map(lambda t: t.decode() if isinstance(t, bytes) else t).to_numpy()
    for c in ("protocol_type", "service", "flag"):
        df[c] = df[c].map(lambda t: t.decode() if isinstance(t, bytes) else t)
    return df, y


def scene(df, y, seed, return_idx=False):
    """One KDD scene: all rare-type records, fixed samples of the big types, a proportional sample of the rest.
    return_idx=True also returns the selected row indices into (df, y), in the order of the scene's rows."""
    rng = np.random.default_rng(seed)
    types, counts = np.unique(y, return_counts=True)
    rare = set(types[counts <= RARE_MAX])
    take = [np.flatnonzero(np.isin(y, list(rare)))]
    for t, n in SAMPLE.items():
        take.append(rng.choice(np.flatnonzero(y == t), n, replace=False))
    mid = [t for t, c in zip(types, counts) if t not in rare and t not in SAMPLE]
    mid_n = np.array([counts[types == t][0] for t in mid], float)
    for t, n in zip(mid, np.maximum(1, np.round(MID_TOTAL * mid_n / mid_n.sum())).astype(int)):
        take.append(rng.choice(np.flatnonzero(y == t), n, replace=False))
    idx = np.concatenate(take)
    import pandas as pd
    X = pd.get_dummies(df.iloc[idx], columns=["protocol_type", "service", "flag"]).astype(float).to_numpy()
    X = StandardScaler().fit_transform(np.log1p(np.abs(X)) * np.sign(X))
    Z = PCA(20, random_state=seed).fit_transform(X)
    d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    Z = Z + rng.normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
    return (Z, y[idx], rare, idx) if return_idx else (Z, y[idx], rare)


def evaluate(lab, y, rare, Z=None, margin_alpha=0.05):
    """Recovery of each rare type. With Z given, each rare type also gets its boundary margin
    (rgt_theory.boundary_margin: log(alpha / (N M)) - log p_oracle of its best recoverable piece; > 0 means above
    the detection boundary), so real-data tables can be split above and below the boundary. Z=None: unchanged."""
    lab = np.asarray(lab)
    out = {}
    small = [c for c in set(lab) - {-1} if 2 <= (lab == c).sum() <= RARE_MAX]
    pure = coherent = 0
    for c in small:
        vals, cnt = np.unique(y[lab == c], return_counts=True)
        if vals[np.argmax(cnt)] in rare and cnt.max() / cnt.sum() >= 0.8:
            pure += 1
        coherent += cnt.max() / cnt.sum() >= 0.8
    for t in sorted(rare):
        mt = y == t
        jac = max([(mt & (lab == c)).sum() / (mt | (lab == c)).sum() for c in set(lab[mt]) - {-1}] or [0])
        hit = any((lab == c).sum() <= RARE_MAX and (mt & (lab == c)).sum() / (lab == c).sum() >= 0.8
                  for c in set(lab[mt]) - {-1} if (lab == c).sum() >= 2)
        out[t] = {"n": int(mt.sum()), "recovered": bool(jac >= 0.5), "hit": bool(hit)}
        if Z is not None and 2 <= mt.sum():
            from .rgt_theory import boundary_margin
            out[t]["margin"] = float(boundary_margin(Z, np.flatnonzero(mt), alpha=margin_alpha))
    return {"per_type": out, "recovered": float(np.mean([v["recovered"] for v in out.values()])),
            "hit": float(np.mean([v["hit"] for v in out.values()])), "small_groups": len(small),
            "small_precision": round(pure / max(1, len(small)), 3),
            # group purity, any class: a flagged group that is >= 80% one class is a coherent group even when the
            # class is not a rare type (e.g. a subtype of normal traffic); rare-label precision counts it as false
            "small_purity": round(coherent / max(1, len(small)), 3)}


HEADLINE_CTS = "cts3_v5_rb_by"
METHODS = {
    "hdbscan": hdb,
    "hdbscan_mcs3": lambda Z: M.hdbscan_labels(Z, 3, 3),
    "lof+group": scored("lof", 0.02),
    "gapclust_k30": lambda Z: gapclust(Z, k=30),
    "gapclust_k200": lambda Z: gapclust(Z, k=200),
    "cts3_v4_rb": lambda Z: cts3(Z, dim_est="local", scale_min=True, respect_base=True),
    "cts3_v5": lambda Z: cts3(Z, dim_est="local", scale_min=True, border=True),
    "cts3_v5_rb": lambda Z: cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True),
    # headline CTS row: BY (valid under any dependence); BH is kept for continuity, exploratory only (its FDR over
    # reported groups runs about 5x the target, stage-11 pass2 doc)
    "cts3_v5_rb_by": lambda Z: cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, select="by"),
    "cts3_v5_rb_bh": lambda Z: cts3(Z, dim_est="local", scale_min=True, border=True, respect_base=True, select="bh"),
    "cts3_v5_hybrid": lambda Z: cts3_hybrid(Z),
}


def one(seed, df, y):
    Z, yy, rare = scene(df, y, seed)
    return {m: evaluate(f(Z), yy, rare) for m, f in METHODS.items()}


def main():
    df, y = load()
    rows = Parallel(n_jobs=4)(delayed(one)(s, df, y) for s in range(5))
    res = {}
    for m in METHODS:
        rs = [r[m] for r in rows]
        res[m] = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in ("recovered", "hit", "small_groups",
                                                                               "small_precision", "small_purity")}
        res[m]["per_type_hit"] = {t: round(float(np.mean([r["per_type"][t]["hit"] for r in rs])), 2)
                                  for t in rs[0]["per_type"]}
        print(f"{m:15s} recovered {res[m]['recovered']:.2f}  hit {res[m]['hit']:.2f}  small groups "
              f"{res[m]['small_groups']:.1f}  precision {res[m]['small_precision']:.2f}  purity {res[m]['small_purity']:.2f}", flush=True)
    print("per-type hit:", {m: res[m]["per_type_hit"] for m in ("hdbscan", HEADLINE_CTS, "cts3_v5_rb_bh", "gapclust_k30")})
    out = os.path.join(os.path.dirname(__file__), *[".."] * 3, "results/kdd.json")
    json.dump(res, open(out, "w"), indent=1); os.chmod(out, 0o600)


if __name__ == "__main__":
    main()
