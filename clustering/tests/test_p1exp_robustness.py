import numpy as np
import pytest

from clustering.eval.nature.p1exp_scenes import VARIANTS, make_variant
from clustering.eval.nature.tiny_bench import make, score


@pytest.mark.parametrize("p", ["isolated", "touching", "inside"])
@pytest.mark.parametrize("d", [2, 10])
def test_variant_defaults_match_benchmark(p, d):
    A, ka, ga = make(p, d, 11, contrast="low")
    B, kb, gb = make_variant(p, d, 11, contrast="low")
    assert np.array_equal(A, B) and (ka == kb).all() and (ga == gb).all()


def test_variants_build():
    for kw in VARIANTS.values():
        X, kind, gid = make_variant("touching", 5, 3, **kw)
        assert len(X) == len(kind) == len(gid) and len(np.unique(gid[gid >= 0])) == 12


def test_baselines_run_and_score():
    from clustering.eval.nature.p1exp_baselines import GRIDS
    from clustering.eval.nature.tiny_baselines import gapclust_available
    X, kind, gid = make("isolated", 2, 5, contrast="high")
    for name, (f, grid) in GRIDS.items():
        if name in ("automato", "cts_v5", "cts_v5_bh"):
            continue
        if name.startswith("gapclust") and not gapclust_available():
            continue                                  # GapClust port not shipped (GPL-3 upstream)
        lab = f(X, grid[len(grid) // 2])
        assert len(lab) == len(X), name
        r = score(lab, kind, gid)
        assert 0 <= r["recall"] <= 1


def test_base_swap_keeps_groups():
    from clustering.eval.nature.p1exp_baselines import cts_v5_base
    X, kind, gid = make("isolated", 2, 5, contrast="high")
    a = cts_v5_base(X, 0.05)
    b = cts_v5_base(X, 0.05, base="none")
    ra, rb = score(a, kind, gid), score(b, kind, gid)
    assert rb["false_per100"] <= ra["false_per100"] + 1e-9


def test_frontier_variant_matches_cli():
    from clustering.eval.nature.p1exp_scenes import frontier_variant
    from clustering.eval.nature.tinybench_cli import frontier
    from clustering.eval.nature.p1exp_baselines import hdbscan_small
    grid = [(3, 2, "eom"), (5, 3, "leaf")]
    a = frontier(lambda X, g, *_: hdbscan_small(X, g), grid, range(1, 2), n_jobs=1)
    b = frontier_variant(lambda X, g: hdbscan_small(X, g), grid, range(1, 2), {}, n_jobs=1)
    assert a == b


def test_eps_median_rule_reproduces_cts3():
    from clustering.eval.nature.p1exp_baselines import cts_v5_eps
    from clustering.eval.nature.tiny_methods import cts3
    X, kind, gid = make("inside", 5, 3)
    v5 = cts3(X, alpha=0.05, dim_est="local", scale_min=True, border=True, fix=None)     # cts_v5_eps copies v5
    assert np.array_equal(v5, cts_v5_eps(X, 0.05, eps_rule="median"))


def test_otsu_rule_survives_large_families():
    from clustering.eval.nature.p1exp_baselines import cts_v5_otsu
    from clustering.eval.nature.tiny_methods import cts3
    X, kind, gid = make_variant("inside", 2, 600, contrast="high", copies=60)
    v5 = cts3(X, alpha=0.05, dim_est="local", scale_min=True, border=True, fix=None)     # the claim is against v5
    a, b = score(v5, kind, gid), score(cts_v5_otsu(X, 0.05), kind, gid)
    assert b["false_per100"] < a["false_per100"] and b["recall"] >= 0.5
