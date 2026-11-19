"""
Vendored from clustering.eval.nature.p3_stats (same project).
Uncertainty for tiny-bench leaderboard claims.

Input: per-scene score dicts (p3_perscene.py). A tier holds S scenes (d = 2/5/10 x seeds). The frontier of a method on
a set of scenes is: for each grid point, mean false groups per 100 points and mean recall over the scenes; the
frontier at cap c is the best mean recall among grid points with mean false rate <= c (0 if none). This is exactly the
stage-8 definition (run_tiny_frontier / tinybench_cli.frontier).

- frontier_ci: bootstrap over scenes, stratified by d (seeds resampled with replacement inside each d), B resamples;
  the operating point is re-selected inside every resample, so the CI includes the selection step. Also a seed-block
  bootstrap (resample seeds, keep all three d of a seed together) as a check.
- paired_test(A, B): H0 "A and B are exchangeable on this tier". Exact sign-flip permutation test: for every subset of
  scenes, swap A's and B's whole grid outputs on those scenes, recompute both frontiers, and compare the observed
  difference with the 2^S permutation distribution (S = 15 -> 32,768, enumerated exactly). Two-sided p, plus the
  paired bootstrap CI of the difference (same resample for both methods).
- holm: Holm-Bonferroni adjustment over a family of p-values.
"""
import itertools

import numpy as np

CAPS = (0.25, 0.5)


def arrays(rows, method):
    """(S x G) arrays of recall and false rate for one method over a list of scene rows."""
    R = np.array([[s["recall"] for s in r["scores"][method]] for r in rows])
    F = np.array([[s["false_per100"] for s in r["scores"][method]] for r in rows])
    return R, F


def frontier_from(R, F, cap, w=None):
    """R, F: (S x G). w: optional (B x S) weights (counts per resample). Returns scalar or (B,) frontier."""
    if w is None:
        r, f = R.mean(0), F.mean(0)
        ok = f <= cap + 1e-12
        return float(r[ok].max()) if ok.any() else 0.0
    n = w.sum(1, keepdims=True)
    r, f = (w @ R) / n, (w @ F) / n                      # (B x G)
    r = np.where(f <= cap + 1e-12, r, -np.inf)
    out = r.max(1)
    return np.where(np.isfinite(out), out, 0.0)


def boot_weights(rows, B=2000, seed=0, scheme="strat_d"):
    rng = np.random.default_rng(seed)
    S = len(rows)
    W = np.zeros((B, S))
    if scheme == "strat_d":
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(r["d"], []).append(i)
        for g in groups.values():
            g = np.array(g)
            pick = g[rng.integers(len(g), size=(B, len(g)))]
            for b in range(B):
                np.add.at(W[b], pick[b], 1)
    elif scheme == "seed_block":
        seeds = sorted({r["seed"] for r in rows})
        by = {s: [i for i, r in enumerate(rows) if r["seed"] == s] for s in seeds}
        pick = rng.integers(len(seeds), size=(B, len(seeds)))
        for b in range(B):
            for k in pick[b]:
                W[b, by[seeds[k]]] += 1
    else:
        pick = rng.integers(S, size=(B, S))
        for b in range(B):
            np.add.at(W[b], pick[b], 1)
    return W


def frontier_ci(rows, method, cap, W, level=0.95):
    R, F = arrays(rows, method)
    point = frontier_from(R, F, cap)
    bs = frontier_from(R, F, cap, W)
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"value": round(point, 3), "ci": [round(float(lo), 3), round(float(hi), 3)], "boot_mean": round(float(bs.mean()), 3)}


def paired_test(rows, a, b, cap, W=None, exact_max=16, n_perm=20000, seed=0):
    Ra, Fa = arrays(rows, a)
    Rb, Fb = arrays(rows, b)
    obs = frontier_from(Ra, Fa, cap) - frontier_from(Rb, Fb, cap)
    S = len(rows)
    # sign-flip permutation: swap whole per-scene outputs of A and B. Grids can differ in length, so pad the shorter
    # grid with copies of a point that is never selectable (false rate = +inf) before swapping.
    G = max(Ra.shape[1], Rb.shape[1])

    def pad(R, F):
        k = G - R.shape[1]
        return np.c_[R, np.zeros((S, k))], np.c_[F, np.full((S, k), np.inf)]
    Ra, Fa = pad(Ra, Fa); Rb, Fb = pad(Rb, Fb)
    if S <= exact_max:
        masks = np.array(list(itertools.product([0, 1], repeat=S)), bool)
    else:
        masks = np.random.default_rng(seed).integers(0, 2, size=(n_perm, S)).astype(bool)
    diffs = np.empty(len(masks))
    for i, m in enumerate(masks):
        mm = m[:, None]
        RA, FA = np.where(mm, Rb, Ra), np.where(mm, Fb, Fa)
        RB, FB = np.where(mm, Ra, Rb), np.where(mm, Fa, Fb)
        diffs[i] = frontier_from(RA, FA, cap) - frontier_from(RB, FB, cap)
    p = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    out = {"diff": round(obs, 3), "p_perm": round(p, 4), "n_perm": int(len(masks))}
    if W is not None:
        Ra0, Fa0 = arrays(rows, a)
        Rb0, Fb0 = arrays(rows, b)
        bd = frontier_from(Ra0, Fa0, cap, W) - frontier_from(Rb0, Fb0, cap, W)
        out["diff_ci"] = [round(float(v), 3) for v in np.quantile(bd, [0.025, 0.975])]
        out["p_boot_le0"] = round(float(np.mean(bd <= 0)), 4)
    return out


def holm(pvals):
    """pvals: dict key -> p. Returns dict key -> Holm-adjusted p."""
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = round(run, 4)
    return adj


def paired_fixed(rows, a, b, cap):
    """Check that ignores re-selection: fix each method's operating point at its pooled frontier choice, take the
    per-scene recall difference d_i, and run the exact sign-flip test on mean(d) (2^S flips). Usually more powerful
    than paired_test, but optimistic because the operating points were chosen on the same scenes."""
    out = {}
    d = []
    for m in (a, b):
        R, F = arrays(rows, m)
        r, f = R.mean(0), F.mean(0)
        ok = np.flatnonzero(f <= cap + 1e-12)
        g = ok[np.argmax(r[ok])] if len(ok) else None
        d.append(R[:, g] if g is not None else np.zeros(len(rows)))
    diff = d[0] - d[1]
    S = len(diff)
    signs = np.array(list(itertools.product([-1, 1], repeat=S))) if S <= 16 else \
        np.random.default_rng(0).choice([-1, 1], size=(20000, S))
    null = (signs * diff).mean(1)
    out["diff"] = round(float(diff.mean()), 3)
    out["p_signflip"] = round(float(np.mean(np.abs(null) >= abs(diff.mean()) - 1e-12)), 4)
    out["wins_ties_losses"] = [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())]
    return out


# ------------------------------------------------------------------ paired bootstrap (vendored from audit_stats, stage 14)
def op_index(R, F, cap):
    r, f = R.mean(0), F.mean(0)
    ok = np.flatnonzero(f <= cap + 1e-12)
    return int(ok[np.argmax(r[ok])]) if len(ok) else None


def _p_two(null_minus_obs, obs, B):
    """Two-sided basic-bootstrap p: H0 theta = 0, pivot T* = theta* - theta_hat; equal tails, +1 correction."""
    t = null_minus_obs
    hi = (1 + np.sum(t >= obs - 1e-12)) / (1 + B)
    lo = (1 + np.sum(t <= obs + 1e-12)) / (1 + B)
    return float(min(1.0, 2 * min(hi, lo)))


def paired_boot(rows, a, b, cap, W=None, B=10000, seed=0, scheme="strat_d"):
    """Paired bootstrap of frontier(a) - frontier(b) with the operating point re-selected in every resample (the
    recommended test; paired_test's sign-flip null is biased when the two grids are nested or of unequal length).
    p_boot = max(p_basic, p_pct): it rejects only when both bootstrap readings reject. Returns diff, p_boot, p_basic,
    ci_basic, p_pct, ci_pct, optimism, op, grid_len, n_boot, and the generic key p (= p_boot)."""
    Ra, Fa = arrays(rows, a)
    Rb, Fb = arrays(rows, b)
    if W is None:
        W = boot_weights(rows, B, seed, scheme)
    B = len(W)
    fa, fb = frontier_from(Ra, Fa, cap), frontier_from(Rb, Fb, cap)
    ba, bb = frontier_from(Ra, Fa, cap, W), frontier_from(Rb, Fb, cap, W)
    obs, bd = fa - fb, ba - bb
    q = np.quantile(bd, [0.025, 0.975])
    p_pct = float(min(1.0, 2 * min((1 + np.sum(bd <= 0)) / (1 + B), (1 + np.sum(bd >= 0)) / (1 + B))))
    p_basic = _p_two(bd - obs, obs, B)
    p = round(max(p_basic, p_pct), 5)
    return {"diff": round(float(obs), 4), "p_boot": p, "p": p, "p_basic": round(p_basic, 5),
            "ci_basic": [round(float(2 * obs - q[1]), 3), round(float(2 * obs - q[0]), 3)],
            "p_pct": round(p_pct, 5), "ci_pct": [round(float(q[0]), 3), round(float(q[1]), 3)],
            "optimism": [round(float(ba.mean() - fa), 4), round(float(bb.mean() - fb), 4)],
            "op": [op_index(Ra, Fa, cap), op_index(Rb, Fb, cap)], "grid_len": [Ra.shape[1], Rb.shape[1]],
            "n_boot": int(B)}
