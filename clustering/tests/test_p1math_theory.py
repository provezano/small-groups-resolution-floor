"""Checks for the stage-9 theory helpers (exact null formula, MDL identity, v5 wrapper)."""
import math

import numpy as np
from scipy.stats import beta as beta_dist

from clustering.eval.nature import p1math_theory as T


def test_pred_rate_exact_on_flat_space():
    d = 5
    vd = T.vball(d)
    P = lambda r: vd * np.asarray(r) ** d
    Pinv = lambda q: (np.asarray(q) / vd) ** (1 / d)
    for t in (1e-2, 1e-6):
        assert abs(T.pred_rate(P, Pinv, 3, 10, 1600, t, d) / t - 1) < 1e-6
        # overstating d is anti-conservative, understating is conservative
        assert T.pred_rate(P, Pinv, 3, 10, 1600, t, 6.0) > t > T.pred_rate(P, Pinv, 3, 10, 1600, t, 4.0)


def test_mdl_identity_exact():
    for m, k, u in ((2, 10, 1e-3), (9, 10, 1e-4), (5, 40, 2e-3)):
        lam, c, lh = T.mdl_identity_terms(u, m, k)
        assert abs(-beta_dist.logcdf(u, m - 1, k) - (lam + c - lh)) < 1e-8
        assert abs(c - 0.5 * math.log(2 * math.pi * (m - 1) * (m - 1 + k) / k)) < 1 / (12 * (m - 1)) + 1 / (12 * k)


def test_alpha_mdl_matches_poisson_limit():
    # large k: alpha -> 1 / (sqrt(2 pi) (m - 1)), the Poisson-code level of Theorem 5 at N = m - 1, up to the
    # Stirling (Robbins) term of (m - 1)!
    for m in (3, 9):
        assert abs(T.alpha_mdl(m, 100000) * math.sqrt(2 * math.pi) * (m - 1) - 1) < math.expm1(1 / (12 * (m - 1))) + 0.005


def test_sphere_mass_karp_pinsky():
    d = 10
    P, _ = T.sphere_mass(d)
    area = 2 * math.pi ** ((d + 1) / 2) / math.gamma((d + 1) / 2)
    r = 0.05
    g = float(P(r)) * area / (T.vball(d) * r ** d)
    assert abs(g - (1 + (2 * d - d * d) / (8 * (d + 2)) * r * r)) < 1e-4


def test_v5_wrapper_reproduces_cts3():
    from clustering.eval.nature import p1math_scan as S
    from clustering.eval.nature.tiny_bench import make
    from clustering.eval.nature.tiny_methods import cts3
    X, kind, gid = make("touching", 5, 600, contrast="mid")
    a = cts3(X, dim_est="local", scale_min=True, border=True, fix=None)     # the wrapper is v5 (no fringe fix)
    R, isr = S.reduced_set(X)
    c, nt = S.v5_candidates(X, R, isr)
    g, _, _ = S.select(c, nt, 0.05)
    b = S.write_labels(X, R, g)
    assert np.array_equal(a, b)
