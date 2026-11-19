import numpy as np
import pytest
from scipy.stats import kstest
from sklearn.neighbors import NearestNeighbors

from clustering.eval.nature import hdbvs_cts as H
from clustering.eval.nature.tiny_bench import make
from clustering.eval.nature.tiny_methods import cts3


@pytest.mark.parametrize("p,d,s,c", [("isolated", 2, 500, "mid"), ("inside", 5, 401, "low")])
def test_v5_copy_matches_cts3(p, d, s, c):
    X, _, _ = make(p, d, s, contrast=c)
    P = H.v5_parts(X)
    for a in (1e-6, 0.3):
        ref = cts3(X, alpha=a, dim_est="local", scale_min=True, border=True, fix=None)     # the copy is v5
        assert (H.v5_labels(P, a) == ref).all()


def test_ratio_only_family_equals_v5():
    X, _, _ = make("isolated", 2, 500, contrast="mid")
    P = H.v5_parts(X)
    assert (H.sep_labels(P, 0.3, families=("ratio",)) == H.v5_labels(P, 0.3)).all()


@pytest.mark.parametrize("d,m", [(2, 4), (5, 3)])
def test_separation_law_is_beta_m1_1(d, m):
    # (r_(m-1) / r_m)^d ~ Beta(m - 1, 1) around a point of a homogeneous Poisson process (any intensity)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (20000, d)) * 7.0
    c = X[np.all((X > 1.5) & (X < 5.5), 1)][:2000]
    dist = NearestNeighbors(n_neighbors=m + 1).fit(X).kneighbors(c)[0]
    u = (dist[:, m - 1] / dist[:, m]) ** d
    assert kstest(u, lambda x: np.clip(x, 0, 1) ** (m - 1)).pvalue > 0.01


def test_two_family_scan_is_calibrated_on_poisson():
    from clustering.eval.nature.run_theory import embed_torus
    hits = 0
    for rep in range(8):
        X = embed_torus(np.random.default_rng(123 + rep).uniform(0, 1, (800, 2)))
        P = H.v5_parts(X, base=np.full(len(X), -1))
        hits += (H.sep_labels(P, 0.05, base=np.full(len(X), -1)) >= 0).any()
    assert hits <= 2
