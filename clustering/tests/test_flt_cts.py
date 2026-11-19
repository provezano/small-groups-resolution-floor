import numpy as np
from scipy.stats import beta as B, spearmanr

from clustering.eval.nature import flt_cts as F
from clustering.eval.nature import hdbvs_cts as H
from clustering.eval.nature.run_theory import embed_torus
from clustering.eval.nature.tiny_bench import make


def _poisson(n=1500, d=2, seed=0):
    """Homogeneous Poisson-like sample on a flat torus (no edges; edges break the stopping-set argument)."""
    return embed_torus(np.random.default_rng(seed).uniform(0, 1, (n, d)))


def test_no_filter_reproduces_v5():
    X, _, _ = make("isolated", 2, 401, contrast="mid")
    P = F.parts(X)
    for a in (0.05, 0.3):
        assert np.array_equal(F.flt_labels(P, a, stat=None), H.v5_labels(P, a))
        # eta = 0 puts all weight on the full family: same as v5
        assert np.array_equal(F.flt_labels(P, a, stat="ring", eta=0.0), H.v5_labels(P, a))


def test_weights_sum_to_family_size():
    X, _, _ = make("touching", 5, 402, contrast="mid")
    P = F.parts(X)
    T, n_scored = P["n_tests"], len(P["m"])
    for stat in ("ring", "far", "sfhi"):
        w = F.weights(P, q=0.1, eta=0.5, stat=stat)
        # (1 - eta) on every scored test, eta * T spread over the passing ones
        assert abs(w.sum() - (0.5 * n_scored + 0.5 * T)) < 1e-6 * T
        assert w.sum() <= T + 1e-6


def test_ring_radius_independent_of_beta_ratio_under_poisson():
    """Result F1: U = (r_(m-1)/r_n)^d is independent of r_n under a homogeneous Poisson null."""
    P = F.parts(_poisson(seed=1))
    s = P["m"] == 5
    u = (P["r"][s] / P["Rk"][s]) ** 2
    rho = spearmanr(u, P["Rk"][s])[0]
    assert abs(rho) < 0.08
    # and U keeps its Beta(4, 10) law (mean 4 / 14)
    assert abs(u.mean() - 4 / 14) < 0.03
    # the Beta p-values are roughly uniform
    p = B.cdf(u, 4, 10)
    assert 0.4 < p.mean() < 0.6


def test_scale_free_outer_statistic_is_uniform_under_poisson():
    P = F.parts(_poisson(seed=2))
    s = P["m"] == 4
    assert 0.4 < P["p_sf"][s].mean() < 0.6


def test_filter_is_calibrated_on_a_few_null_scenes():
    hits = 0
    for seed in range(10):
        P = F.parts(_poisson(n=900, seed=100 + seed), base=np.full(900, -1))
        hits += len(F.flt_labels(P, 0.05, stat="ring", q=0.1, return_groups=True)) > 0
    assert hits <= 3
