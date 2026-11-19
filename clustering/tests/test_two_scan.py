"""Tests for the two-sample tiny-group scan (stage 14, branch two). Nothing here writes to disk."""
import numpy as np
from scipy.stats import hypergeom

from clustering.eval.nature import two_scan as W


def test_hyp_logsf_matches_scipy_and_far_tail():
    M, n = 500, 120
    for j, t in [(1, 1), (3, 10), (5, 40), (8, 9), (14, 300)]:
        ref = hypergeom.logsf(j - 1, M, n, t)
        assert abs(W.hyp_logsf(np.array([j]), M, n, np.array([t]))[0] - ref) < 1e-8
    # far tail: all 14 draws are successes with a rare success label
    v = W.hyp_logsf(np.array([14]), 40000, 1000, np.array([14]))[0]
    exact = sum(np.log((1000 - i) / (40000 - i)) for i in range(14))
    assert abs(v - exact) < 1e-8


def test_single_candidate_pvalue_is_exact_under_exchangeability():
    """Fixed centre (a random B point), fixed j: P(p <= t) <= t, and about equal to t at the achievable levels."""
    rng = np.random.default_rng(0)
    nB, nA, d = 60, 120, 3
    rej = {0.05: 0, 0.2: 0}
    reps = 3000
    for _ in range(reps):
        Z = rng.normal(size=(nB + nA, d)) * np.r_[np.ones(60), 0.3 * np.ones(120)][:, None]  # any common law
        Z = Z[rng.permutation(len(Z))]           # i.i.d. mixture law shared by A and B
        B, A = Z[:nB], Z[nB:]
        C = W.two_cands(A, B, max_m=5)
        lp = -C["score"].reshape(nB, 4)[0, 2]    # centre 0, j = 3
        for t in rej:
            rej[t] += np.exp(lp) <= t
    for t, k in rej.items():
        assert k / reps <= t + 3 * np.sqrt(t * (1 - t) / reps)
        assert k / reps >= 0.4 * t                # discrete, but not wildly conservative


def test_scan_fwer_on_iid_null():
    rng = np.random.default_rng(1)
    hits = 0
    reps = 150
    for _ in range(reps):
        B, A = rng.uniform(size=(300, 2)), rng.uniform(size=(900, 2))
        hits += len(W.two_groups(W.two_cands(A, B), 0.1)) > 0
    assert hits / reps <= 0.1 + 3 * np.sqrt(0.09 / reps)


def test_tight_group_found_and_boundary_ratio():
    """A tight m = 6 group: found with r = 40, impossible with r = 1 (r_min(6) = exp(L / 5) - 1 > 1)."""
    rng = np.random.default_rng(2)
    bg = rng.uniform(size=(400, 2))
    grp = np.array([0.5, 0.5]) + rng.normal(0, 1e-3, (6, 2))
    B = np.vstack([bg, grp])
    A40 = rng.uniform(size=(16000, 2))
    g40 = W.two_groups(W.two_cands(A40, B), 0.05)
    assert any(set(range(400, 406)) <= set(g.tolist()) for g in g40)
    C1 = W.two_cands(A40[:406], B)
    L = np.log(C1["n_tests"] / 0.05)
    assert C1["score"].max() < L                 # nothing reachable: (m - 1) log 2 < L
    assert 5 * np.log(2) < L


def test_background_draw_matches_make_geometry():
    """A drawn from the replayed parameters lies in the same latent subspace as make()'s B."""
    A, B, kind, gid, n_bg = W.make_pair("touching", 5, 3, "low", r_max=2)
    assert A.shape == (2 * n_bg, B.shape[1])
    Q = np.linalg.svd(B - B.mean(0), full_matrices=False)[2][:5]
    resid = lambda X: np.linalg.norm((X - B.mean(0)) - (X - B.mean(0)) @ Q.T @ Q, axis=1).mean()
    assert resid(A) < 0.02 and resid(B) < 0.02


def test_baselines_run():
    rng = np.random.default_rng(4)
    B, A = rng.normal(size=(200, 3)), rng.normal(size=(400, 3))
    gs = W.milo_simple(A, B, k=10)
    assert len(gs) == len(W.MILO_Q)
    s = W.c2st_scores(A, B, n_trees=10, folds=3)
    assert s.shape == (200,) and (0 <= s).all() and (s <= 1).all()
