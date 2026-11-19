"""Stage 14 (core): CC d rule (Theorem 14.1), candidate-table reproduction of v5, certified-base wrapper,
same-data selection bias (Proposition 14.3). No files are written."""
import math

import numpy as np

from clustering.eval.nature import core_cands as CC
from clustering.eval.nature import core_certified as CE
from clustering.eval.nature import core_theory as C
from clustering.eval.nature import p1math_scan as S
from clustering.eval.nature import tiny_bench as B


def _flat(d, c=1.0):
    def Pm(r):
        return np.minimum(1.0, c * np.asarray(r, float) ** d)

    def Pi(q):
        return (np.asarray(q, float) / c) ** (1 / d)
    return Pm, Pi


def test_kappa_and_cc_dhat_shape():
    # kappa grows with tmax and d0, shrinks with lambda; d_hat is a convex combination of d0 and D_lo
    k1 = C.kappa(5, 10, 1e-6, 5.0, 1.5)
    assert C.kappa(5, 10, 1e-4, 5.0, 1.5) >= k1 >= C.kappa(5, 10, 1e-6, 5.0, 3.0)
    assert C.kappa(5, 10, 1e-6, 10.0, 1.5) >= k1
    assert 0 < k1 < 1
    d = C.cc_dhat(5.0, 2.0, 0.3)
    assert abs(d - (0.7 * 5 + 0.3 * 2)) < 1e-12
    assert C.cc_dhat(5.0, 7.0, 0.3) == 5.0


def test_cc_rate_valid_on_flat_space_and_eta_small():
    for d in (2, 5, 10):
        Pm, Pi = _flat(d)
        t = 1e-5
        for m in (2, 8, 15):
            r = C.cc_rate_exact(Pm, Pi, m, 10, 5000, t, float(d), 40, 0.01)
            eta = C.cc_eta(m, 10, 40, 0.01, t, float(C.a_in(C.u_level(m, 10, t) ** (1 / d))))
            assert r <= t * (1 + eta) * (1 + 1e-3)
            assert 0 <= eta < 0.1


def test_cc_fixes_gaussian_core_where_fixed_d_fails():
    # Gaussian centre at d = 5: the plain test with the true d is anti-conservative, the CC rule is not
    from clustering.eval.nature import p1math_theory as T
    Pg, Pgi = T.gm_mass(np.zeros(5), [np.zeros(5)], [1.0], [1.0])
    t = 0.05 / (14 * 1600)
    fixed = C.cc_rate_exact(Pg, Pgi, 10, 10, 1600, t, 5.0, 40, 0.01, rule="fixed")
    cc = C.cc_rate_exact(Pg, Pgi, 10, 10, 1600, t, 5.0, 40, 0.01, rule="cc")
    assert fixed > 2 * t and cc < t


def test_scan_table_reproduces_v5_scores():
    Z, kind, gid = B.make("touching", d=5, seed=620, contrast="mid")
    tb = CC.scan_table(Z)
    R, isr = S.reduced_set(Z)
    c, nt = S.v5_candidates(Z, R, isr)
    assert nt == tb["n_tests"]
    ref = sorted(round(s, 9) for s, *_ in c if s >= CC.KEEP_SCORE)
    mine = sorted(round(s, 9) for s, _ in CC.candidates(tb, ("v5",)))
    assert ref == mine
    # every rule only lowers d, so it never adds a candidate above the v5 score
    for rule in (("cc", 160, 0.01, "auto"), ("adapt", 0.3, (20, 40, 80, 160), 0.01), ("lcb", 40, 0.01)):
        dh = CC.rule_dhat(tb["tab"], rule, tb["n_tests"])
        assert np.all(dh <= tb["tab"]["d_v5"] + 1e-12)


def test_certified_wrapper_default_is_cts_v5():
    from clustering.eval.nature.tiny_methods import cts3
    X, kind, gid = B.make("isolated", d=2, seed=621)
    assert np.array_equal(cts3(X, alpha=0.05, dim_est="local", scale_min=True, border=True), CE.cts_v5(X))
    lab, inf = CE.cts_v5(X, base="certified", seed=1, return_info=True)
    assert len(lab) == len(X) and inf["base_clusters"] >= 1


def test_same_data_selection_is_anticonservative():
    rng = np.random.default_rng(5)
    few = CE.min_count_selection(3, 30, rng, reps=1500)
    many = CE.min_count_selection(200, 30, rng, reps=500)
    assert few["0.05"] <= 0.05 + 0.02
    assert many["0.01"] > 0.8
