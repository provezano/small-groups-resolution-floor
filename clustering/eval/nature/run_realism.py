"""
Planted-group realism (gap D), on the real window (private; nothing released).

1. Describe groups in the fused feature space by four scale-free numbers: spread (median within-group distance over
   the members' median distance to the nearest outsider), anisotropy (top-eigenvalue share of the members' offsets),
   kNN purity (share of each member's 5 nearest neighbours inside the group, capped by group size) and isolation
   (centroid to nearest outsider over the outsiders' median 10-NN distance there).
2. Compare real weak families (sessions named together by one finding; job removed) with planted groups made two ways:
   isotropic copies (the usual recipe, at the easy/medium/hard spreads used before) and offset bootstrap (copies of a
   real session plus member-minus-centroid offsets resampled from real families). Realism = mean |standardised
   difference| of the four numbers against real families.
3. The decisive check: do planted groups rank embedding + clustering choices the way real families do? Measure =
   share of within-group pairs placed in the same cluster. To avoid circularity, offsets come from a random half of
   the families and the ranking is compared with the other half (20 splits).
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from . import bench as B
from . import methods as M
from .run_canary import embed, mcs_for
from .run_idea1_real import UI, weak_families

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
CONFIGS = [(e, c) for e in ("umap15", "umap5", "pca20") for c in ("eom", "mcs3")]


def describe(X, members):
    out = np.setdiff1d(np.arange(len(X)), members)
    G = X[members]
    nn_out = NearestNeighbors(n_neighbors=10).fit(X[out])
    d_out = nn_out.kneighbors(G)[0]
    intra = np.linalg.norm(G[:, None] - G[None], axis=2)[np.triu_indices(len(G), 1)]
    spread = np.median(intra) / max(np.median(d_out[:, 0]), 1e-12)
    off = G - G.mean(0)
    ev = np.linalg.eigvalsh(off.T @ off) if len(G) > 2 else np.array([1.0])
    aniso = float(ev[-1] / max(ev.sum(), 1e-12))
    k = 5
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(G, return_distance=False)[:, 1:]
    purity = np.mean([np.isin(r, members).sum() / min(k, len(G) - 1) for r in nn])
    cd = nn_out.kneighbors(G.mean(0, keepdims=True))[0][0]
    local = np.median(NearestNeighbors(n_neighbors=10).fit(X[out]).kneighbors(X[out][nn_out.kneighbors(
        G.mean(0, keepdims=True), return_distance=False)[0]])[0][:, -1])
    return np.array([spread, aniso, purity, cd[0] / max(local, 1e-12)])


def plant_iso(X, rng, q, n=15):
    d1 = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    sig = np.quantile(d1, q) / np.sqrt(2 * X.shape[1])
    groups, rows = [], []
    for t in rng.choice(len(X), n, replace=False):
        m = int(rng.integers(2, 10)); rows.append(X[t] + rng.normal(0, sig, (m, X.shape[1]))); groups.append(m)
    return rows


def plant_boot(X, rng, offsets, n=15):
    rows = []
    for t in rng.choice(len(X), n, replace=False):
        m = int(rng.integers(2, 10)); pool = offsets[rng.integers(len(offsets))]
        rows.append(X[t] + pool[rng.integers(len(pool), size=m)])
    return rows


def augment(X, rows):
    gid = np.r_[np.full(len(X), -1), np.concatenate([[g] * len(r) for g, r in enumerate(rows)])]
    return np.vstack([X, *rows]), gid


def pair_rate(lab, groups):
    tot = hit = 0
    for mem in groups:
        l = lab[mem]
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                tot += 1; hit += int(l[a] >= 0 and l[a] == l[b])
    return hit / max(tot, 1)


def cluster(Z, c):
    return M.hdbscan_labels(Z, mcs_for(len(Z)), max(3, mcs_for(len(Z)) // 2)) if c == "eom" else M.hdbscan_labels(Z, 3, 3)


def main():
    d = B.load_real(ROOT)
    keep = ~d["auto"]
    X = d["X"][keep].astype(float)
    ui = {s["session"]: s for s in json.load(open(os.path.join(ROOT, UI)))["sessions"]}
    truth, _ = weak_families(d["session"][keep], ui)
    fams = [np.flatnonzero(truth == f) for f in np.unique(truth[truth >= 0])]
    real_desc = np.array([describe(X, f) for f in fams])
    offsets = [X[f] - X[f].mean(0) for f in fams]
    rng = np.random.default_rng(0)
    res = {"n_families": len(fams), "real": real_desc.mean(0).round(3).tolist()}
    names = ["spread", "anisotropy", "knn_purity", "isolation"]
    sd = real_desc.std(0) + 1e-9
    recipes = {f"iso_q{q}": (lambda r, q=q: plant_iso(X, r, q)) for q in (0.1, 0.5, 0.9)}
    recipes["offset_boot"] = lambda r: plant_boot(X, r, offsets)
    for name, f in recipes.items():
        rows = f(np.random.default_rng(1))
        Xa, gid = augment(X, rows)
        desc = np.array([describe(Xa, np.flatnonzero(gid == g)) for g in range(len(rows))])
        smd = np.abs(desc.mean(0) - real_desc.mean(0)) / sd
        res[name] = {"means": desc.mean(0).round(3).tolist(), "abs_smd": smd.round(2).tolist(), "realism_gap": round(float(smd.mean()), 2)}
        print(f"  {name:12s} realism gap {smd.mean():.2f}  |SMD| " + " ".join(f"{n}={v:.2f}" for n, v in zip(names, smd)))
    print("  real means   " + " ".join(f"{n}={v:.3f}" for n, v in zip(names, real_desc.mean(0))))
    # ranking agreement with held-out real families
    embs = {}
    agree = {"iso_q0.5": [], "iso_q0.1": [], "offset_boot": []}
    real_rates_all = {}
    for e, c in CONFIGS:
        Z = embed(X, e, 7); lab = cluster(Z, c)
        real_rates_all[(e, c)] = lab
    for split in range(20):
        r = np.random.default_rng(100 + split)
        perm = r.permutation(len(fams)); A, Bh = perm[: len(fams) // 2], perm[len(fams) // 2:]
        real_rank = [pair_rate(real_rates_all[cfg], [fams[i] for i in Bh]) for cfg in CONFIGS]
        for name in agree:
            rows = plant_boot(X, r, [offsets[i] for i in A]) if name == "offset_boot" else plant_iso(X, r, float(name[5:]))
            Xa, gid = augment(X, rows)
            groups = [np.flatnonzero(gid == g) for g in range(len(rows))]
            pr = []
            for e, c in CONFIGS:
                key = (name, split, e)
                if key not in embs:
                    embs[key] = embed(Xa, e, 7)
                pr.append(pair_rate(cluster(embs[key], c), groups))
            agree[name].append(spearmanr(pr, real_rank).correlation)
        embs = {}
    res["ranking_agreement_spearman"] = {k: [round(float(np.nanmean(v)), 2), round(float(np.nanstd(v)), 2)] for k, v in agree.items()}
    res["real_pair_rate_all_families"] = {f"{e}+{c}": round(pair_rate(real_rates_all[(e, c)], fams), 3) for e, c in CONFIGS}
    print("  real pair rate (all families):", res["real_pair_rate_all_families"])
    print("  ranking agreement with held-out real families (Spearman mean, sd over 20 splits):", res["ranking_agreement_spearman"])
    p = os.path.join(ROOT, "results/realism.json")
    json.dump(res, open(p, "w"), indent=1); os.chmod(p, 0o600)


if __name__ == "__main__":
    main()
