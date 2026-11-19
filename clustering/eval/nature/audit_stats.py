"""
Stage 14 (audit): what is wrong with `p3_stats.paired_test`, and a replacement that does not depend on grid lengths.

`p3_stats.paired_test` swaps the two methods' whole grid outputs scene by scene and recomputes both pooled frontiers.
The pooled frontier mixes grid points by *index*: after a swap, grid point j of "A" is A's j-th point on some scenes
and B's j-th point on the others. Two consequences (see `old_test_anatomy` and the stage-14 audit doc):
1. Unequal lengths: the shorter grid is padded with unselectable points (false rate +inf). After any swap other than
   "none" or "all", every padded index has an infinite pooled false rate for *both* permuted methods. So the
   permutation null only ever sees the common prefix, while the observed difference uses the longer grid in full.
   The gain the longer method gets from its extra points has no counterpart in the null.
2. Nested grids (B is A's own prefix on the same scenes, e.g. CTS at alpha <= 10 against CTS at alpha <= 0.3): the
   prefixes are identical, so every swapped pair has the same frontier and the null collapses to {0}, plus +obs and
   -obs from the two trivial masks. Then p = 2 / 2^S (exact) or ~ 0 (random flips) for ANY non-zero difference.
Even with equal lengths, the null is "the two methods' grid outputs are exchangeable", which is false for any two
different methods; it is not "the two frontiers are equal".

Replacement (`paired_boot`, p = max(basic, percentile)): a paired, d-stratified bootstrap of the frontier difference with each method's operating
point re-selected inside every resample. The p-value is the basic (pivot) bootstrap one: under H0 (theta = 0) the
law of theta_hat is approximated by that of theta_hat* - theta_hat, which carries the selection optimism of each grid
(a longer grid's max over more points is more optimistic, and the pivot subtracts that). Only the frontier values
enter, never grid indices, so the grids' lengths and orders cannot matter. The percentile p-value (2 P*(theta* <= 0))
is reported too, because the stage-13 CIs used it; it adds selection optimism instead of removing it.

Also here: `crossfit_test`, a sample-split sign-flip test (operating points chosen on one half of the seeds, recall
difference tested on the other half, both ways, Bonferroni x 2). It is exact for its own null but loses power; it is a
check, not the main test.
"""
import itertools

import numpy as np

from . import p3_stats as S


# ---------------------------------------------------------------------------------------------------------- helpers
def op_index(R, F, cap):
    r, f = R.mean(0), F.mean(0)
    ok = np.flatnonzero(f <= cap + 1e-12)
    return int(ok[np.argmax(r[ok])]) if len(ok) else None


def _p_two(null_minus_obs, obs, B):
    """Two-sided basic-bootstrap p: H0 theta = 0, pivot T* = theta* - theta_hat; equal tails, +1 correction."""
    t = null_minus_obs
    hi = (1 + np.sum(t >= obs - 1e-12)) / (1 + B)       # P*(T* >= obs)
    lo = (1 + np.sum(t <= obs + 1e-12)) / (1 + B)       # P*(T* <= obs)
    return float(min(1.0, 2 * min(hi, lo)))


# ---------------------------------------------------------------------------------------------------- the new test
def paired_boot(rows, a, b, cap, W=None, B=10000, seed=0, scheme="strat_d"):
    """Paired bootstrap of frontier(a) - frontier(b) with the operating point re-selected in every resample.
    p_boot = max(p_basic, p_pct) is the test to use: it rejects only when both bootstrap readings reject, so it is
    valid whenever either is (P(max <= a) <= min of the two rates). The basic (pivot) p removes selection optimism
    (ties, nested grids); the percentile p protects at the cap boundary, where a grid point whose true false rate is
    just above the cap is selected by noise and the pivot over-rejects (synthetic case N2, audit doc).
    Returns diff, p_boot, p_basic, ci_basic, p_pct, ci_pct,
    the selection optimism of each method (bootstrap mean minus point value) and both operating points."""
    Ra, Fa = S.arrays(rows, a)
    Rb, Fb = S.arrays(rows, b)
    if W is None:
        W = S.boot_weights(rows, B, seed, scheme)
    B = len(W)
    fa, fb = S.frontier_from(Ra, Fa, cap), S.frontier_from(Rb, Fb, cap)
    ba, bb = S.frontier_from(Ra, Fa, cap, W), S.frontier_from(Rb, Fb, cap, W)
    obs, bd = fa - fb, ba - bb
    t = bd - obs
    q = np.quantile(bd, [0.025, 0.975])
    p_pct = float(min(1.0, 2 * min((1 + np.sum(bd <= 0)) / (1 + B), (1 + np.sum(bd >= 0)) / (1 + B))))
    p_basic = _p_two(t, obs, B)
    return {"diff": round(float(obs), 4), "p_boot": round(max(p_basic, p_pct), 5), "p_basic": round(p_basic, 5),
            "ci_basic": [round(float(2 * obs - q[1]), 3), round(float(2 * obs - q[0]), 3)],
            "p_pct": round(p_pct, 5), "ci_pct": [round(float(q[0]), 3), round(float(q[1]), 3)],
            "optimism": [round(float(ba.mean() - fa), 4), round(float(bb.mean() - fb), 4)],
            "op": [op_index(Ra, Fa, cap), op_index(Rb, Fb, cap)], "grid_len": [Ra.shape[1], Rb.shape[1]],
            "n_boot": int(B)}


def crossfit_test(rows, a, b, cap, n_perm=20000, seed=0):
    """Split seeds in two halves (alternate seeds, every d kept), choose each method's operating point on one half,
    sign-flip the per-scene recall difference on the other, both directions; p = min(1, 2 min(p1, p2))."""
    seeds = sorted({r["seed"] for r in rows})
    halves = [set(seeds[0::2]), set(seeds[1::2])]
    ps, diffs = [], []
    for h in (0, 1):
        sel = [r for r in rows if r["seed"] in halves[h]]
        tst = [r for r in rows if r["seed"] in halves[1 - h]]
        if not sel or not tst:
            continue
        dd = []
        for m in (a, b):
            g = op_index(*S.arrays(sel, m), cap)
            R, _ = S.arrays(tst, m)
            dd.append(R[:, g] if g is not None else np.zeros(len(tst)))
        d = dd[0] - dd[1]
        n = len(d)
        signs = np.array(list(itertools.product([-1, 1], repeat=n))) if n <= 16 else \
            np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
        null = (signs * d).mean(1)
        ps.append(float((1 + np.sum(np.abs(null) >= abs(d.mean()) - 1e-12)) / (1 + len(null))))
        diffs.append(float(d.mean()))
    return {"diff": round(float(np.mean(diffs)), 4), "p_crossfit": round(min(1.0, 2 * min(ps)), 5)}


# ------------------------------------------------------------------------------------------ anatomy of the old test
def old_test_anatomy(rows, a, b, cap, n_perm=4000, seed=0):
    """Why paired_test gives what it gives: grid lengths, whether the shorter grid is the longer one's prefix on these
    scenes, the longer method's frontier restricted to the common prefix (all the null can see), the gain from the
    extra points (delta_extra), and the share of non-trivial permutations whose difference is exactly 0."""
    Ra, Fa = S.arrays(rows, a)
    Rb, Fb = S.arrays(rows, b)
    Ga, Gb = Ra.shape[1], Rb.shape[1]
    g = min(Ga, Gb)
    out = {"grid_len": [Ga, Gb]}
    nested = bool(np.allclose(Ra[:, :g], Rb[:, :g]) and np.allclose(Fa[:, :g], Fb[:, :g]))
    out["nested_prefix"] = nested
    fa, fb = S.frontier_from(Ra, Fa, cap), S.frontier_from(Rb, Fb, cap)
    fa_p, fb_p = S.frontier_from(Ra[:, :g], Fa[:, :g], cap), S.frontier_from(Rb[:, :g], Fb[:, :g], cap)
    out["frontier"] = [round(fa, 4), round(fb, 4)]
    out["frontier_prefix"] = [round(fa_p, 4), round(fb_p, 4)]
    out["delta_extra"] = round((fa - fa_p) - (fb - fb_p), 4)          # > 0: the extra points help A (sign as diff)
    rng = np.random.default_rng(seed)
    Sn = len(rows)
    masks = rng.integers(0, 2, size=(n_perm, Sn)).astype(bool)
    masks = masks[(masks.sum(1) > 0) & (masks.sum(1) < Sn)]
    Ra_, Fa_, Rb_, Fb_ = Ra[:, :g], Fa[:, :g], Rb[:, :g], Fb[:, :g]
    zeros, null = 0, []
    for m in masks:
        mm = m[:, None]
        v = S.frontier_from(np.where(mm, Rb_, Ra_), np.where(mm, Fb_, Fa_), cap) - \
            S.frontier_from(np.where(mm, Ra_, Rb_), np.where(mm, Fa_, Fb_), cap)
        null.append(v)
        zeros += abs(v) < 1e-12
    null = np.array(null)
    out["null_zero_share"] = round(zeros / max(1, len(masks)), 4)
    out["null_sd"] = round(float(null.std()), 4)
    return out


def paired_test_padlast(rows, a, b, cap, exact_max=16, n_perm=20000, seed=0):
    """The minimal patch the tail branch suggested: pad the shorter grid by repeating its last point instead of an
    unselectable one. Kept only to show that it is not enough (index-aligned mixing stays; nested grids still give a
    degenerate null)."""
    Ra, Fa = S.arrays(rows, a)
    Rb, Fb = S.arrays(rows, b)
    obs = S.frontier_from(Ra, Fa, cap) - S.frontier_from(Rb, Fb, cap)
    G = max(Ra.shape[1], Rb.shape[1])

    def pad(R, F):
        k = G - R.shape[1]
        return np.c_[R, np.repeat(R[:, -1:], k, 1)], np.c_[F, np.repeat(F[:, -1:], k, 1)]
    Ra, Fa = pad(Ra, Fa); Rb, Fb = pad(Rb, Fb)
    Sn = len(rows)
    masks = np.array(list(itertools.product([0, 1], repeat=Sn)), bool) if Sn <= exact_max else \
        np.random.default_rng(seed).integers(0, 2, size=(n_perm, Sn)).astype(bool)
    diffs = np.empty(len(masks))
    for i, m in enumerate(masks):
        mm = m[:, None]
        diffs[i] = S.frontier_from(np.where(mm, Rb, Ra), np.where(mm, Fb, Fa), cap) - \
            S.frontier_from(np.where(mm, Ra, Rb), np.where(mm, Fa, Fb), cap)
    return {"diff": round(obs, 4), "p_perm": round(float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12)), 5)}


# --------------------------------------------------------------------------------------- synthetic tiers (for tests)
def synth_rows(specs, S_per_d=10, dims=(2, 5, 10), n_groups=12, rho=0.5, seed=0, false_noise=0.5):
    """Synthetic tier with known population frontiers. specs: {method: [(recall, false_per100), ...]} population
    values per grid point (recall increasing along the grid). Each scene has n_groups groups with latent difficulty
    z ~ U(0,1) (shared between methods with probability rho, so the comparison is paired) and a scene shift in recall
    by d; a group is hit at grid point g when z < r_g + shift. The false rate is f_g times a scene multiplier
    Gamma(1/false_noise^2) with mean 1, shared along the grid (monotone)."""
    rng = np.random.default_rng(seed)
    rows = []
    shift = {d: v for d, v in zip(dims, np.linspace(-0.1, 0.1, len(dims)))}
    k = 1.0 / false_noise ** 2
    for d in dims:
        for s in range(S_per_d):
            z0 = rng.uniform(size=n_groups)
            sc = {}
            for m, grid in specs.items():
                own = rng.uniform(size=n_groups)
                z = np.where(rng.uniform(size=n_groups) < rho, z0, own)
                mult = rng.gamma(k, 1 / k)
                sc[m] = [{"recall": float(np.mean(z < r + shift[d])), "false_per100": float(f * mult)}
                         for r, f in grid]
            rows.append({"d": d, "seed": s, "scores": sc})
    return rows


def pop_frontier(grid, cap):
    ok = [r for r, f in grid if f <= cap + 1e-12]
    return max(ok) if ok else 0.0
