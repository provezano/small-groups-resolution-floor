import numpy as np

from clustering.eval.nature import scale_cts as S
from clustering.eval.nature.scale_bench import make_scaled
from clustering.eval.nature.tiny_bench import make


def _gset(groups):
    return sorted(tuple(int(x) for x in g) for g in groups)


def test_scalable_scan_equals_cts3_v5():
    from clustering.eval.nature.tiny_methods import cts3, hdb
    A, kind, gid = make("inside", d=5, seed=400, contrast="mid")
    hb = hdb(A)
    for rule in ("bonferroni", "bh"):
        lo = cts3(A, dim_est="local", scale_min=True, border=True, select=rule)
        go = _gset(np.flatnonzero(lo == c) for c in np.unique(lo) if c > hb.max())
        _, info = S.cts_scale(A, rule=rule, return_info=True)
        assert _gset(info["groups"]) == go


def test_partitioned_rows_are_exact():
    A, _, _ = make_scaled("isolated", 6000, d=5, seed=401, n_groups=12)
    k = S.kk_for(len(A))
    D, I = S.knn_exact(A, k)
    Dp, Ip, info = S.knn_partitioned(A, k, n_cells=4, seed=0)
    assert info["ghost_ratio"] < 3          # halos are a part of the data, not all of it
    assert np.array_equal(I, Ip) and np.array_equal(D, Dp)


def test_partitioned_equals_single_machine():
    A, _, _ = make_scaled("touching", 6000, d=5, seed=402, n_groups=24)
    for rule in ("bonferroni", "bh"):
        _, info = S.cts_scale(A, rule=rule, return_info=True, fix=None)      # the partitioned form is v5 (no fringe fix)
        gp, ip = S.cts_partitioned(A, rule=rule, n_cells=4, seed=0)
        assert ip["n_tests"] == info["n_tests"]
        assert _gset(gp) == _gset(info["groups"])
