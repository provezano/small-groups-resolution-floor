import numpy as np
import pytest

from clustering.eval.nature.tiny_bench import make
from clustering.eval.nature.tiny_methods import _group_outliers, cts3, hdb
from clustering.eval.nature.tiny_baselines import gapclust_available, gapclust_raw


def _new_groups(lab, base):
    return {g for g in np.unique(lab[lab > base.max()])}


def test_by_is_stricter_than_bh():
    X, kind, gid = make("isolated", 5, 2, contrast="mid")
    base = hdb(X)
    kw = dict(dim_est="local", scale_min=True, border=True)
    n_bh = len(_new_groups(cts3(X, select="bh", **kw), base))
    n_by = len(_new_groups(cts3(X, select="by", **kw), base))
    assert n_by <= n_bh


def test_custom_order_needs_bonferroni():
    X, _, _ = make("isolated", 2, 0, contrast="high")
    with pytest.raises(ValueError):
        cts3(X, select="by", order="gap")


def test_group_outliers_few_points():
    Z = np.random.default_rng(0).normal(size=(50, 3))
    lab = _group_outliers(Z, np.array([1, 2, 3]), np.zeros(50, int))
    assert len(lab) == 50


@pytest.mark.skipif(not gapclust_available(), reason="GapClust port not shipped (GPL-3 upstream); set GAPCLUST_PY")
def test_gapclust_squared_changes_distances_only():
    X, _, _ = make("touching", 5, 1, contrast="mid")
    a = gapclust_raw(X, 30)
    b = gapclust_raw(X, 30, squared=True)
    assert np.array_equal(a[3], b[3]) and not np.allclose(a[0], b[0])
