import numpy as np
import pytest

from clustering.eval.nature import fringe_cts as F, scale_cts as S
from clustering.eval.nature.tiny_bench import make


def test_none_equals_scalable_v5():
    A, kind, gid, fam = F.make_scaled_fam("isolated", 3200, d=5, seed=3, n_groups=12)
    _, i0 = S.cts_scale(A, return_info=True, fix=None)
    _, i1 = F.cts_fringe(A, fix="none", return_info=True)
    assert len(i0["groups"]) == len(i1["groups"])
    assert all(np.array_equal(a, b) for a, b in zip(i0["groups"], i1["groups"]))


def test_generators_match_shared_ones():
    from clustering.eval.nature import scale_bench as SB
    A, _, _, _ = F.make_scaled_fam("inside", 3200, d=5, seed=4, n_groups=12)
    B, _, _ = SB.make_scaled("inside", 3200, d=5, seed=4, n_groups=12)
    assert np.array_equal(A, B)
    assert np.array_equal(F.make_shape("touching", 10, 5, shape="gauss")[0], make("touching", 10, 5)[0])


@pytest.mark.parametrize("d,m", [(2, 2), (5, 3), (10, 2)])
def test_joint_pvalue_is_uniform_under_the_family_model(d, m):
    """Gaussian family with known centre and spread: p = b (1 - ln b), b = U V, is exactly Uniform(0, 1)."""
    rng = np.random.default_rng(d * 10 + m)
    M, reps, ps = 30, 1500, []
    for _ in range(reps):
        Z = rng.normal(0, 1, (M, d))
        i = rng.integers(M)
        dist = np.sort(np.linalg.norm(Z - Z[i], axis=1))
        r = dist[m - 1]                         # distance to the (m - 1)-th other member
        p = F.joint_pvalues(Z, np.arange(M), np.array([i]), np.array([m]), np.array([r]),
                            np.zeros((1, d)), np.array([1.0]), np.array([M - m]), d, k_fam=1, m_extra=True)
        ps.append(p[0])
    ps = np.array(ps)
    for t in (0.01, 0.05, 0.2, 0.5):
        assert abs((ps <= t).mean() - t) < 4 * np.sqrt(t * (1 - t) / reps) + 0.005


def test_joint_removes_fringe_leak_and_keeps_far_groups():
    A, kind, gid, fam = F.make_scaled_fam("isolated", 32000, d=10, seed=610, n_groups=0)
    rows = S.knn_exact(A, S.kk_for(len(A)))
    _, i0 = F.cts_fringe(A, fix="none", rows=rows, return_info=True)
    _, i1 = F.cts_fringe(A, fix="joint", rows=rows, return_info=True)
    m0 = sum(kind[g[0]] == "mass" for g in i0["groups"])
    m1 = sum(kind[g[0]] == "mass" for g in i1["groups"])
    assert m0 >= 3 and m1 == 0
    X, kind, gid = make("isolated", 5, 601)
    r = S.knn_exact(X, S.kk_for(len(X)))
    g0 = F.cts_fringe(X, fix="none", rows=r, return_info=True)[1]["groups"]
    g1 = F.cts_fringe(X, fix="joint", rows=r, return_info=True)[1]["groups"]
    planted0 = {tuple(g) for g in g0 if (gid[g] >= 0).all()}
    planted1 = {tuple(g) for g in g1 if (gid[g] >= 0).all()}
    assert planted0 == planted1
