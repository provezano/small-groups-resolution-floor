import numpy as np
from scipy.special import betainc

from clustering.eval.nature import tail_cts as T


def _ball(rng, n_rep, N, d):
    """n_rep independent sets of N i.i.d. uniform points in the unit d-ball, sorted by radius (the interior of the
    stopping ball, Result T1 step 1). Returns normalised volumes t (sorted) and unit directions."""
    t = np.sort(rng.uniform(size=(n_rep, N)), 1)
    u = rng.normal(size=(n_rep, N, d))
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    return t, u


def _cone_stat(t, u, m, k, side=1.0, dens_ratio=1.0, rng=None):
    """S = v / (v + omega (w - v)) for the half-space {side * y_1 >= 0}; nan where fewer than k half-space points."""
    v = t[:, m - 2]
    shell_t, shell_c = t[:, m - 1:], side * u[:, m - 1:, 0] >= 0
    pos = np.where(shell_c, np.arange(shell_t.shape[1])[None, :], 10 ** 6)
    kth = np.sort(pos, 1)[:, k - 1]
    ok = kth < 10 ** 6
    w = np.where(ok, shell_t[np.arange(len(t)), np.minimum(kth, shell_t.shape[1] - 1)], np.nan)
    return np.where(ok, v / (v + 0.5 * (w - v)), np.nan)


def test_cone_omega_inverse_and_half():
    for d in (2, 5, 10):
        assert abs(float(T.cone_omega(90.0, d)) - 0.5) < 1e-12
        for om in (0.05, 0.25, 0.4):
            assert abs(float(T.cone_omega(T.cone_theta(om, d), d)) - om) < 1e-9


def test_half_space_ring_is_exact_or_conservative_under_homogeneous_null():
    """Result T1: P(S <= s, J >= k) <= I_s(m - 1, k), with equality in the limit of a large ball."""
    rng = np.random.default_rng(1)
    for d, m, k in ((2, 3, 10), (5, 5, 5), (10, 4, 10)):
        t, u = _ball(rng, 40000, m - 1 + 60, d)
        S = _cone_stat(t, u, m, k)
        p = np.where(np.isnan(S), 1.0, betainc(m - 1, k, np.nan_to_num(S, nan=1.0)))
        for a in (0.01, 0.05, 0.2):
            rate = float((p <= a).mean())
            assert rate <= a * 1.15 + 0.002, (d, m, k, a, rate)
            assert rate >= a * 0.8, (d, m, k, a, rate)                     # not grossly conservative either


def test_conditional_cone_pvalue_is_uniform():
    """The conditional p-value given J (cone_logp) is exactly uniform under the null."""
    rng = np.random.default_rng(2)
    d, m, k, N = 3, 4, 10, 4 - 1 + 40 - 1
    t, u = _ball(rng, 6000, N, d)
    S = _cone_stat(t, u, m, k)
    J = (u[:, m - 1:, 0] >= 0).sum(1)
    ok = ~np.isnan(S)
    p = np.exp(T.cone_logp(S[ok], m, N, k, J[ok], 0.5))
    for a in (0.05, 0.2, 0.5):
        assert abs(float((p <= a).mean()) - a) < 0.03, (a, float((p <= a).mean()))


def test_toward_side_is_conservative_when_the_cone_side_is_denser():
    """Result T3: if the half-space shell is denser than the group's ball (one-sided tail null), the half-space p-value
    is super-uniform. Simulated by thinning the far side: keep a point with y_1 < 0 with probability 1/3."""
    rng = np.random.default_rng(3)
    d, m, k = 2, 4, 5
    t, u = _ball(rng, 30000, 300, d)
    keep = (u[:, :, 0] >= 0) | (rng.uniform(size=u.shape[:2]) < 1 / 3)
    rows = []
    for i in range(len(t)):
        ti, ui = t[i][keep[i]], u[i][keep[i]]
        rows.append((ti[:120], ui[:120]))
    tt = np.stack([r[0] for r in rows]); uu = np.stack([r[1] for r in rows])
    S = _cone_stat(tt, uu, m, k)
    p = np.where(np.isnan(S), 1.0, betainc(m - 1, k, np.nan_to_num(S, nan=1.0)))
    for a in (0.01, 0.05, 0.2):
        assert float((p <= a).mean()) <= a, a
    # and the away side (sparser) is anti-conservative: the first-order gradient bias of Result T2
    S2 = _cone_stat(tt, uu, m, k, side=-1.0)
    p2 = np.where(np.isnan(S2), 1.0, betainc(m - 1, k, np.nan_to_num(S2, nan=1.0)))
    assert float((p2 <= 0.05).mean()) > 0.1


def test_full_variant_equals_fringe_cts():
    from clustering.eval.nature import fringe_cts as F
    from clustering.eval.nature.tiny_bench import make
    X, kind, gid = make("touching", 5, 7, contrast="mid")
    P = T.tail_parts(X)
    for a in (0.05, 0.3):
        lab = T.tail_labels(P, a, "full", "joint")
        ref = F.cts_fringe(X, fix="joint", alpha=a)
        assert np.array_equal(lab >= 0, np.asarray(ref) >= 0)


def test_variants_return_scores_of_the_right_shape():
    from clustering.eval.nature.tiny_bench import make
    X, _, _ = make("isolated", 2, 3, contrast="high")
    P = T.tail_parts(X)
    for v in T.VARIANTS:
        sc = T.tail_scores(P, v)
        assert sc.shape == P["full"].shape and np.all(np.isfinite(sc))
