import numpy as np
from scipy.special import betainc

from clustering.eval.nature import fringe_cts as F
from clustering.eval.nature import ring_cts as RC
from clustering.eval.nature.tiny_bench import make


def _same(g1, g2):
    return len(g1) == len(g2) and all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(g1, g2))


def test_one_ring_is_v5_and_fringe():
    X, kind, gid = make("inside", 5, 3, contrast="mid")
    P = RC.ring_parts(X, (3, 10, 25, 50))
    for fix in ("none", "joint"):
        _, info = F.cts_fringe(X, fix=fix, alpha=0.05, return_info=True)
        assert _same(RC.ring_groups(P, 0.05, (10,), "bonf", fix), info["groups"])


def test_c_minp_between_one_and_J():
    assert RC.c_minp(5, (10,)) == 1.0
    for m in (2, 5, 15):
        c = RC.c_minp(m, (10, 25, 50), n=50000)
        assert 1.0 < c < 3.0
    assert RC.c_minp(15, (10, 25, 50), n=50000) > RC.c_minp(2, (10, 25, 50), n=50000)


def test_minp_and_gate_null_rates():
    rng = np.random.default_rng(1)
    n, m, t = 200000, 4, 0.01
    A = rng.gamma(m - 1, size=n)
    E = np.cumsum(rng.exponential(size=(n, 50)), 1)
    p = {k: betainc(m - 1, k, A / (A + E[:, k - 1])) for k in (10, 25, 50)}
    pmin = np.minimum(np.minimum(p[10], p[25]), p[50]) * RC.c_minp(m, (10, 25, 50))
    assert (pmin <= t).mean() <= 1.1 * t
    use = betainc(10, 40, E[:, 9] / E[:, 49]) >= 0.5
    pg = np.where(use, p[50], p[10])
    assert (pg <= t).mean() <= 1.05 * t
    # the reversed gate (big ring when the ring shape is front-loaded) is anti-conservative
    pr = np.where(~use, p[50], p[10])
    assert (pr <= t).mean() > 1.05 * t


def test_entry_point_runs():
    X, kind, gid = make("isolated", 2, 4, contrast="high")
    lab = RC.cts_ring(X, rings=(10, 25), calib=("gate", 0.5), fix="joint")
    assert len(lab) == len(X) and lab.max() >= 0
