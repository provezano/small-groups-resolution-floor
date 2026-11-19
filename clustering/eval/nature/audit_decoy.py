"""
Stage 14 (audit): decoy test for the touching/low "tail clump" reading (register item 1.1, Corollary T5).

Stage 13 read the touching/low groups that GapClust and LOF find (and CTS does not) at d = 2 and 5 as clumps of tail
points beyond a cluster's edge, not as density excess. This module builds matched scenes in which each planted group is
replaced by a decoy at the same place, and scores every method's hits on the decoys exactly as on planted groups.

Arms (same seed => same mass, background clusters, noise and group placements, by replaying tiny_bench.make's stream):
- planted: tiny_bench.make("touching", d, seed, contrast="low") unchanged.
- taillaw: each group's m points are replaced by m points drawn from its background cluster's own law N(b, s^2 I),
  restricted to the region the planted points occupy as seen from b: radii in [min, max] of the planted points'
  distances to b, and directions within the smallest cone around u = (c - b)/|c - b| that holds them all. Inside that
  region the decoy has the background's shape (radial density r^(d-1) exp(-r^2 / 2 s^2), uniform over the cone), so the
  ratio (background + decoy) / background is constant there: no bump at c. Count, place, radial range (how far out the
  points are) and angular extent match the planted group.
- radial (primary decoy): each planted point keeps its distance to b, and gets a direction drawn uniformly in the same
  cone around u. This is the background's own conditional law given the radius (an isotropic cluster is uniform over
  directions at every radius), restricted to the cone. Outlyingness (every point's distance from b), count and place
  match the planted group exactly; only the clump (the angular concentration around u) is removed.
- radial2: as radial with the cone's half-angle doubled (capped at 90 degrees): same radii, a looser spread, so the
  clump is removed more fully (added on tuning seeds, before the freeze, because radial's spread at d = 2 and 5 is
  close to the planted group's).
- empty: no group points at all (the scene is pure background); the decoy of group g is the m_g scene points nearest
  its centre c_g (each point in at most one decoy, groups in order). This is "planted centres, no planted points".
Hits: Jaccard >= 0.5 with some predicted cluster, as tiny_bench.score.

  python -m clustering.eval.nature.audit_decoy run --seeds 620-639 --out-dir DIR [--shard i/n]
  python -m clustering.eval.nature.audit_decoy merge --out-dir DIR --out OUT.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.stats import chi

from .tiny_bench import CONTRAST, D_AMBIENT, score

ARMS = ("planted", "radial", "radial2", "taillaw", "empty")
DIMS = (2, 5, 10)
ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3, 1.0, 3.0, 10.0)
Q6 = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


def _cone_dirs(u, theta_max, n, rng, d):
    """n unit vectors uniform on the sphere cap {angle to u <= theta_max} in R^d (inverse CDF of sin^(d-2))."""
    if d == 1:
        return np.tile(u, (n, 1))
    th = np.linspace(0, theta_max, 2001)
    dens = np.sin(th) ** (d - 2) if d > 2 else np.ones_like(th)
    cdf = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(th))])
    cdf /= cdf[-1]
    t = np.interp(rng.uniform(size=n), cdf, th)
    v = rng.normal(size=(n, d))
    v -= (v @ u)[:, None] * u[None, :]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v


def taillaw_points(G, b, s, u, rng):
    """m points from N(b, s^2 I) restricted to the radial range and cone that the planted points G occupy."""
    m, d = G.shape
    Y = G - b
    r = np.linalg.norm(Y, axis=1)
    ang = np.arccos(np.clip((Y @ u) / r, -1, 1))
    lo, hi = chi.cdf(r.min() / s, d), chi.cdf(r.max() / s, d)
    rr = s * chi.ppf(rng.uniform(lo, hi, size=m), d)
    return b + rr[:, None] * _cone_dirs(u, float(ang.max()) + 1e-9, m, rng, d)


def radial_points(G, b, u, rng, widen=1.0):
    """Each point of G keeps |y - b|; directions uniform in the smallest cone around u that holds G (half-angle times
    `widen`, capped at 90 degrees)."""
    m, d = G.shape
    Y = G - b
    r = np.linalg.norm(Y, axis=1)
    ang = np.arccos(np.clip((Y @ u) / r, -1, 1))
    return b + r[:, None] * _cone_dirs(u, min(np.pi / 2, widen * float(ang.max())) + 1e-9, m, rng, d)


def make_arm(arm, d=5, seed=0, contrast="low", n_groups=12, tpl=30, copies=30, copy_sd=0.02, decoy_seed=None):
    """Replays tiny_bench.make('touching', ...) exactly for the planted arm (checked in the tests)."""
    rng = np.random.default_rng(seed)
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
    drng = np.random.default_rng(10_000_000 + seed if decoy_seed is None else decoy_seed)
    centres, sizes = [], []
    for g in range(n_groups):
        m = int(rng.integers(2, 10))
        b, bsd = bc[rng.integers(len(bc))]
        u = rng.normal(size=d); u /= np.linalg.norm(u)
        c, sd = b + u * bsd * (np.sqrt(d) + 1.5), bsd * CONTRAST[contrast]
        G = c + rng.normal(0, sd, (m, d))          # always drawn, so the stream stays in step
        centres.append(c); sizes.append(m)
        if arm == "planted":
            X.append(G); kind += ["group"] * m; gid += [g] * m
        elif arm == "radial":
            X.append(radial_points(G, b, u, drng)); kind += ["group"] * m; gid += [g] * m
        elif arm == "radial2":
            X.append(radial_points(G, b, u, drng, widen=2.0)); kind += ["group"] * m; gid += [g] * m
        elif arm == "taillaw":
            X.append(taillaw_points(G, b, bsd, u, drng)); kind += ["group"] * m; gid += [g] * m
        elif arm != "empty":
            raise ValueError(arm)
    Z = np.vstack(X)
    kind, gid = np.array(kind), np.array(gid)
    Q = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_AMBIENT)))[0][:, :d]
    A = Z @ Q.T + rng.normal(0, 0.002, (len(Z), D_AMBIENT))
    if arm == "empty":
        taken = np.zeros(len(Z), bool)
        for g, (c, m) in enumerate(zip(centres, sizes)):
            dist = np.linalg.norm(Z - c, axis=1)
            dist[taken] = np.inf
            idx = np.argsort(dist)[:m]
            gid[idx] = g; taken[idx] = True
    return A, kind, gid, {"centres": [c.tolist() for c in centres], "sizes": sizes}


def hits(lab, gid):
    lab = np.asarray(lab)
    out = []
    for g in np.unique(gid[gid >= 0]):
        mg = gid == g
        best = max([(mg & (lab == c)).sum() / (mg | (lab == c)).sum() for c in set(lab[mg]) - {-1}] or [0])
        out.append(int(best >= 0.5))
    return out


def _sc(lab, kind, gid):
    s = score(lab, kind, gid)
    s["hits"] = hits(lab, gid)
    return s


def scene(arm, d, s):
    from . import tail_cts as T
    from .tiny_baselines import gapclust_sweep
    from .tiny_methods import hdb, scored
    from .tiny_oracle import oracle_labels
    X, kind, gid, info = make_arm(arm, d, s)
    base = hdb(X)
    out, secs = {}, {}
    t = time.time()
    P = T.tail_parts(X)
    out["cts_full"] = [_sc(T.tail_labels(P, a, "full", base=base), kind, gid) for a in ALPHAS]
    secs["cts"] = round(time.time() - t, 2)
    t = time.time()
    out["lof+group"] = [_sc(scored("lof", q)(X), kind, gid) for q in Q6]
    secs["lof"] = round(time.time() - t, 2)
    t = time.time()
    out["gapclust_port_k30"] = [_sc(l, kind, gid) for l in gapclust_sweep(X, 30)]
    secs["gapclust"] = round(time.time() - t, 2)
    t = time.time()
    # the oracle is told the true background (no group, no decoy): params() replays the same stream
    out["oracle"] = [_sc(l, kind, gid) for l in oracle_labels(X, "touching", d, s, "low", ALPHAS[:6], base=base)]
    secs["oracle"] = round(time.time() - t, 2)
    return {"arm": arm, "placement": "touching", "contrast": "low", "d": d, "seed": s, "n": len(X),
            "sizes": info["sizes"], "scores": out, "secs": secs}


def grids():
    return {"cts_full": list(ALPHAS), "lof+group": list(Q6), "gapclust_port_k30": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
            "oracle": list(ALPHAS[:6])}


# ------------------------------------------------------------------------------------------------------- analysis
READING = ("gapclust_port_k30", "lof+group")      # the two methods the stage-13 reading is about
REF_METHODS = ("cts_full", "cts_full_A6", "oracle")


def _with_a6(rows):
    for r in rows:
        r["scores"]["cts_full_A6"] = r["scores"]["cts_full"][:6]
    return rows


def op_points(rows, cap=0.25):
    """Each method's operating point, chosen on the PLANTED arm pooled over d (the bench's frontier rule)."""
    from .p3_stats import arrays
    pl = [r for r in rows if r["arm"] == "planted"]
    out = {}
    for m in pl[0]["scores"]:
        R, F = arrays(pl, m)
        r, f = R.mean(0), F.mean(0)
        ok = np.flatnonzero(f <= cap + 1e-12)
        out[m] = int(ok[np.argmax(r[ok])]) if len(ok) else None
    return out


def _hits(r, m, g):
    return np.array(r["scores"][m][g]["hits"]) if g is not None else np.zeros(len(r["sizes"]), int)


def analyse(rows, cap=0.25, B=4000, n_perm=20000, seed=0, pool=(2, 5)):
    """Hit rates per arm, method and d at the planted-arm operating point; decoy / planted ratios with a scene
    bootstrap CI (seeds resampled within d, arms kept paired); a paired sign-flip test of planted - decoy on the
    per-scene hit counts (pooled d in `pool`), Holm over methods x decoy arms; and the frozen verdict rule."""
    from .p3_stats import holm
    rows = _with_a6(rows)
    ops = op_points(rows, cap)
    methods = list(ops)
    key = {(r["arm"], r["d"], r["seed"]): r for r in rows}
    scenes = sorted({(r["d"], r["seed"]) for r in rows if r["arm"] == "planted"})
    arms = [a for a in ARMS if any((a, d, s) in key for d, s in scenes)]
    H = {}                                   # (arm, method) -> scenes x groups hit matrix (aligned with `scenes`)
    for a in arms:
        for m in methods:
            H[(a, m)] = np.array([_hits(key[(a, d, s)], m, ops[m]) for d, s in scenes])
    dv = np.array([d for d, _ in scenes])
    out = {"cap": cap, "op_points": ops, "n_scenes_per_arm": len(scenes), "rates": {}, "false_per100": {},
           "ratio": {}, "tests": {}}
    for a in arms:
        for m in methods:
            for d in DIMS:
                out["rates"][f"{a}|{m}|d{d}"] = round(float(H[(a, m)][dv == d].mean()), 4)
            sel = np.isin(dv, pool)
            out["rates"][f"{a}|{m}|pooled"] = round(float(H[(a, m)][sel].mean()), 4)
            fr = [key[(a, d, s)]["scores"][m][ops[m]]["false_per100"] if ops[m] is not None else 0.0
                  for d, s in scenes]
            out["false_per100"][f"{a}|{m}"] = round(float(np.mean(fr)), 4)
    rng = np.random.default_rng(seed)
    sel = np.flatnonzero(np.isin(dv, pool))
    strata = [sel[dv[sel] == d] for d in pool]
    boots = [np.concatenate([st[rng.integers(len(st), size=len(st))] for st in strata]) for _ in range(B)]
    pv = {}
    for a in arms:
        if a == "planted":
            continue
        for m in methods:
            P, Dc = H[("planted", m)], H[(a, m)]
            pr, dr = P[sel].mean(), Dc[sel].mean()
            ratio = dr / pr if pr > 0 else float("nan")
            bs = []
            for b in boots:
                pb = P[b].mean()
                bs.append(Dc[b].mean() / pb if pb > 0 else np.nan)
            bs = np.array(bs)
            lo, hi = np.nanquantile(bs, [0.025, 0.975]) if np.isfinite(bs).any() else (np.nan, np.nan)
            out["ratio"][f"{a}|{m}"] = {"planted": round(float(pr), 4), "decoy": round(float(dr), 4),
                                         "ratio": round(float(ratio), 3), "ci": [round(float(lo), 3), round(float(hi), 3)]}
            diff = (P[sel].sum(1) - Dc[sel].sum(1)).astype(float)       # per-scene hit-count difference
            signs = np.random.default_rng(seed + 1).choice([-1, 1], size=(n_perm, len(diff)))
            null = (signs * diff).mean(1)
            p = float((1 + np.sum(np.abs(null) >= abs(diff.mean()) - 1e-12)) / (1 + n_perm))
            both = int(((P[sel] == 1) & (Dc[sel] == 1)).sum())
            out["tests"][f"{a}|{m}"] = {"mean_diff_per_scene": round(float(diff.mean()), 3), "p": round(p, 5),
                                         "planted_hits": int(P[sel].sum()), "decoy_hits": int(Dc[sel].sum()),
                                         "same_group_both": both}
            if m in READING:
                pv[f"{a}|{m}"] = p
    adj = holm(pv)
    for k, v in adj.items():
        out["tests"][k]["p_holm"] = v
    out["verdict"] = verdict(out)
    return out


def _rule(out, m, arm):
    e = out["ratio"].get(f"empty|{m}")
    r = out["ratio"].get(f"{arm}|{m}")
    t = out["tests"].get(f"{arm}|{m}")
    if e is None or r is None:
        return "missing"
    if e["planted"] > 0 and e["decoy"] >= 0.5 * e["planted"]:
        return "artefact"
    if r["ratio"] >= 0.5 and t.get("p_holm", 1.0) >= 0.05:
        return "tail-points"
    if r["ci"][1] < 0.5 and t.get("p_holm", 1.0) < 0.05:
        return "clump-driven"
    return "partial"


def verdict(out):
    """Frozen rule . Per reading method, pooled d = 2 and 5, decoy arm A:
    artefact      empty-arm rate >= 0.5 x planted rate;
    tail-points   A's ratio >= 0.5 and planted - A not significant (Holm >= 0.05): the method finds radius-matched
                  scatter as it finds the group, so it reacts to outlying points, not to a clump;
    clump-driven  A's ratio CI upper < 0.5 and planted - A significant (Holm < 0.05): it needs the clump (a density
                  excess that need not be a mode);
    partial       anything else.
    Primary A = radial2 (rule v2, set on tuning seeds 620-639 after rule v1 = radial proved weakly discriminating);
    secondary A = radial (rule v1). Stage 13's reading ("tail clumps, not density excess") holds for a method iff its
    v2 verdict is tail-points."""
    res = {"v2_radial2": {}, "v1_radial": {}}
    for m in READING:
        res["v2_radial2"][m] = _rule(out, m, "radial2")
        res["v1_radial"][m] = _rule(out, m, "radial")
    return res


def _status(out_dir, total, note=""):
    with open(os.path.join(out_dir, "STATUS.txt"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} done {len(glob.glob(os.path.join(out_dir, '*.json')))} / "
                f"{total} {note}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "merge", "analyse"))
    ap.add_argument("--inp")
    ap.add_argument("--seeds")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        lo, hi = (int(v) for v in a.seeds.split("-"))
        tasks = [(arm, d, s) for s in range(lo, hi + 1) for d in DIMS for arm in ARMS]
        i, n = (int(v) for v in a.shard.split("/"))
        os.makedirs(a.out_dir, exist_ok=True)
        for k, t in enumerate(tasks):
            if k % n != i:
                continue
            name = os.path.join(a.out_dir, "_".join(str(x) for x in t) + ".json")
            if os.path.exists(name):
                continue
            row = scene(*t)
            with open(name + ".tmp", "w") as f:
                json.dump(row, f)
            os.replace(name + ".tmp", name)
            os.chmod(name, 0o600)
            _status(a.out_dir, len(tasks), f"(shard {i}/{n})")
        _status(a.out_dir, len(tasks), f"(shard {i}/{n} finished)")
    elif a.cmd == "analyse":
        rows = json.load(open(a.inp))["rows"]
        res = {"cap0.25": analyse(rows, 0.25), "cap0.5": analyse(rows, 0.5)}
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        os.chmod(a.out, 0o600)
    else:
        rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out_dir, "*.json")))]
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "grids": grids()}, f)
        os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
