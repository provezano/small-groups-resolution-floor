import numpy as np
from sklearn.neighbors import KDTree

from clustering.eval.nature import gclu_core as G


def test_dip_pvalue_is_valid_under_equal_intensities():
    rng = np.random.default_rng(600)
    lam = 12.0
    p = np.array([G.dip_pvalue(*rng.poisson(lam, 3)) for _ in range(4000)])
    assert (p <= 0.05).mean() <= 0.05 + 0.01
    # a real dip is detected
    assert G.dip_pvalue(0, 20, 25) < 1e-5


def test_lune_test_is_valid_for_overlapping_balls():
    rng = np.random.default_rng(601)
    hits = 0
    for _ in range(400):
        X2 = rng.random((rng.poisson(2000), 2)) * 10
        t2 = KDTree(X2)
        s, a, b = np.array([5.0, 5.0]), np.array([5.6, 5.0]), np.array([4.4, 5.0])
        ns_a, na, ns_b, nb = G.lune_counts(t2, X2, s, a, b, 0.5)
        from scipy.stats import binom
        p = max(binom.cdf(ns_a, ns_a + na, 0.5), binom.cdf(ns_b, ns_b + nb, 0.5))
        hits += p <= 0.05
    assert hits / 400 <= 0.05 + 0.025


def test_gclu_reports_one_cluster_on_a_torus_null():
    for seed in (602, 603):
        X = G.null_scene("torus", 2, 800, seed)
        assert G.n_clusters(G.gclu(X, seed=seed, tiny=False)) <= 1


def test_gclu_certifies_two_separated_blobs():
    rng = np.random.default_rng(604)
    X = np.vstack([rng.normal(0, 1, (400, 2)), rng.normal(0, 1, (400, 2)) + [10, 0]])
    lab = G.gclu(X, seed=604, tiny=False)
    assert G.n_clusters(lab) == 2
    assert len(set(lab[:400])) == 1 and len(set(lab[400:])) == 1


def test_multiscale_beta_scan_cuts_a_bounded_uniform_into_pieces():
    # the pruned direction: at large m the local-homogeneity null is false near the support edge
    X = G.null_scene("cube", 2, 1200, 605)
    _, info = G.mscan(X, return_info=True)
    assert info["groups"] >= 2 and max(info["sizes"]) > 100


def test_rho_star_matches_the_boundary_table():
    assert abs(G.rho_star(5, 2, 1600) - 0.29) < 0.02
    assert abs(G.rho_star(9, 10, 1600) - 0.98) < 0.02
