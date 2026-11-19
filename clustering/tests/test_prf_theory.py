"""Stage 13 (prf): group-count step-up (Theorem 13.1) and the LCB d rule (Theorem 13.2). No files are written."""
import math

import numpy as np

from clustering.eval.nature import prf_theory as P


def _flat(d, c=1.0):
    def Pm(r):
        return np.minimum(1.0, c * np.asarray(r, float) ** d)

    def Pi(q):
        return (np.asarray(q, float) / c) ** (1 / d)
    return Pm, Pi


def test_group_stepup_is_self_consistent_and_nested():
    rng = np.random.default_rng(0)
    for _ in range(30):
        cands = [(float(rng.exponential(3)), tuple(sorted(rng.choice(60, rng.integers(2, 6), replace=False))))
                 for _ in range(200)]
        for shape in ("bonferroni", "by", ("trunc", 8), ("mix", 0.5, 16)):
            g, thr, nrej = P.group_stepup_any(cands, 500, 0.3, shape)
            lim = -math.log(0.3 * P.shape_beta_any(len(g), shape, 500) / 500) if g else np.inf
            best = {}
            for s, mem in cands:
                best[mem] = max(best.get(mem, -np.inf), s)
            assert all(best[m] >= lim - 1e-9 for m in g)
            assert all(set(a).isdisjoint(b) for i, a in enumerate(g) for b in g[i + 1:])


def test_group_stepup_controls_fdr_on_clumped_block_model():
    # one true group with 80 overlapping candidates (p ~ 0) plus 2000 null candidates in clumps of 5 sharing one
    # uniform p-value (strong positive dependence): the group FDR must stay <= alpha for the valid shapes
    rng = np.random.default_rng(1)
    alpha, T = 0.2, 2080
    fdp = {"by": [], ("trunc", 8): [], "bonferroni": []}
    for _ in range(400):
        cands = [(50.0, (0, 1, 2)) if j == 0 else (50.0 - 1e-6 * j, (0, 1, 2 + j)) for j in range(80)]
        for b in range(400):
            s = -math.log(rng.uniform())
            for j in range(5):
                cands.append((s - 1e-9 * j, (1000 + 10 * b + j, 1000 + 10 * b + j + 5000)))
        for sh in fdp:
            g, _, _ = P.group_stepup_any(cands, T, alpha, sh)
            false = sum(1 for m in g if min(m) >= 1000)
            fdp[sh].append(false / max(1, len(g)))
    for sh, v in fdp.items():
        assert np.mean(v) <= alpha * 1.15, (sh, np.mean(v))


def test_by_over_candidates_breaks_group_level_on_clumps():
    out = P.clump_example(10000, 100, 0.05, 1500, np.random.default_rng(2))
    assert out["by"] > 0.1                    # candidate-level BY: group FDR about 4x target
    assert out["gby"] <= 0.05 and out["bonferroni"] <= 0.05


def test_lcb_rate_conservative_on_flat_space():
    for d in (2, 5, 10):
        Pm, Pi = _flat(d)
        t = 1e-5
        for m in (2, 8, 15):
            r = P.lcb_rate_exact(Pm, Pi, m, 10, 5000, t, float(d), 80, 0.01)
            assert r <= t * (1 + 1e-3)
            # the theorem's bound t(1 + eta) is never below the exact rate
            assert r <= t * (1 + P.lcb_eta(m, 10, 80, 0.01, t)) * (1 + 1e-3)


def test_lcb_rate_matches_beta_when_lower_bound_is_slack():
    # if the outer-ring bound is never binding (d0 tiny), d_hat = d0 and the rate is the plain fixed-d formula
    Pm, Pi = _flat(5)
    t = 1e-3
    r = P.lcb_rate_exact(Pm, Pi, 5, 10, 5000, t, 5.0 * 1e-3, 80, 0.01)
    assert r < t * 1e-2
