"""Stage 15 (leak): rate engine, the min-over-annuli rule, the labelled generator. Writes nothing to disk."""
import numpy as np
from scipy.special import betainc

from clustering.eval.nature import core_cands as CC
from clustering.eval.nature import core_theory as CT
from clustering.eval.nature import leak_scenes as LSC
from clustering.eval.nature import leak_theory as LT
from clustering.eval.nature import tiny_bench as B


def test_labelled_generator_matches_tiny_bench():
    for p, d, s in (("touching", 2, 3), ("inside", 5, 4), ("isolated", 10, 5)):
        A, k, g = B.make(p, d=d, seed=s)
        A2, k2, g2, cl, info = LSC.make_labeled(p, d=d, seed=s)
        assert np.array_equal(A, A2) and (k == k2).all() and (g == g2).all()
        assert ((cl >= 0) == (k == "background")).all()
        assert np.array_equal(np.bincount(cl[cl >= 0]), info["sizes"])


def test_rate_engine_matches_quadrature():
    P, Pi = CT.cluster_mass(5, 150, 2000)
    t, nt = 2.5e-6, 20000
    for m in (5, 15):
        a = CT.cc_rate_exact(P, Pi, m, 10, 2000, t, 5, 160, 0.01, tmax=CC.tmax_default(nt))
        b = LT.rate(P, Pi, m, 10, 2000, t, ("cc", 160, 0.01, "auto"), 5, nt, n_samp=40000)
        assert abs(b / a - 1) < 0.02


def test_power_law_fixed_d_is_exact():
    P, Pi = LT.power_mass(2)
    t = 5e-6
    for m in (2, 8, 15):
        assert abs(LT.rate(P, Pi, m, 10, 2000, t, ("v5",), 2, 10000, n_samp=4000) / t - 1) < 1e-6


def _tab(n=500, seed=1):
    rng = np.random.default_rng(seed)
    P, Pi = LT.gaussian_mass(2, 1.0, 0.5, noise=0.02, box=36.0, Nc_frac=0.05)
    _, _, tab = LT.rate(P, Pi, 12, 10, 2000, 5e-6, ("v5",), 2, 10000, n_samp=n, seed=seed, return_cond=True)
    return tab


def test_ccq2_is_cc_and_min_rules_are_lower():
    tab = _tab()
    for L in (20, 40, 80, 160):
        a = LT.rule_dhat(tab, ("ccq", L, 0.01, "auto", 2.0), 10000)
        b = CC.rule_dhat(tab, ("cc", L, 0.01, "auto"), 10000)
        assert np.allclose(a, b)
    mn = LT.rule_dhat(tab, ("ccmin", (20, 40, 80, 160), 0.01), 10000)
    for L in (20, 40, 80, 160):
        assert np.all(mn <= CC.rule_dhat(tab, ("cc", L, 0.01, "auto"), 10000) + 1e-12)
    lg = LT.rule_dhat(tab, ("ccmin_lg", (20, 40, 80, 160), 0.01, 20, 0.5), 10000)
    assert np.all(lg <= mn + 1e-12)
    q1 = LT.rule_dhat(tab, ("ccmin", (20, 40, 80, 160), 0.01, 1.0), 10000)
    assert np.all(q1 <= mn + 1e-12)          # a smaller q gives a larger kappa, so a lower exponent


def test_beta_cdf_ratio_lemma():
    # F(v) / v^(m-1) is nonincreasing (Proposition 15.2's step), so F(u^(1-e)) <= F(u) u^(-(m-1) e)
    v = np.linspace(1e-6, 0.999, 3000)
    for m in (2, 5, 15):
        g = betainc(m - 1, 10, v) / v ** (m - 1)
        assert np.all(np.diff(g) <= 1e-15)
        u, e = 0.05, 0.1
        assert betainc(m - 1, 10, u ** (1 - e)) <= betainc(m - 1, 10, u) * u ** (-(m - 1) * e) * (1 + 1e-12)
