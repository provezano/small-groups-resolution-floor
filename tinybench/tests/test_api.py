import json

import numpy as np
import pytest

import tinybench as tb
from tinybench import baselines, stats
from tinybench.leaderboard import to_markdown
from tinybench.baselines import gapclust_available


def test_generate_deterministic_and_ground_truth():
    a = tb.generate("touching", "mid", 5, 500)
    b = tb.generate("touching", "mid", 5, 500)
    assert np.array_equal(a["X"], b["X"]) and a["X"].shape[1] == 20
    assert len(set(a["gid"][a["gid"] >= 0])) == 12
    sizes = np.bincount(a["gid"][a["gid"] >= 0])
    assert sizes.min() >= 2 and sizes.max() <= 9


@pytest.mark.parametrize("shape", ["aniso", "t3", "boot"])
def test_shapes_keep_skeleton(shape):
    """Shape tracks are paired: every non-group point, every label and group size is identical to the iso scene."""
    a = tb.generate("inside", "low", 10, 501)
    b = tb.generate("inside", "low", 10, 501, shape=shape)
    ng = a["gid"] < 0
    assert np.array_equal(a["kind"], b["kind"]) and np.array_equal(a["gid"], b["gid"])
    assert np.allclose(a["X"][ng], b["X"][ng])
    assert not np.allclose(a["X"][~ng], b["X"][~ng])


def test_iso_matches_project_generator():
    pytest.importorskip("clustering.eval.nature.tiny_bench")
    from clustering.eval.nature.tiny_bench import make as ref
    for p, c, d, s in (("isolated", "high", 2, 500), ("inside", "mid", 10, 404)):
        X, kind, gid = ref(p, d, s, contrast=c)
        Y, k2, g2 = tb.make(p, d, s, contrast=c)
        assert np.array_equal(X, Y) and np.array_equal(gid, g2)


def test_score_perfect_and_empty():
    sc = tb.generate("isolated", "high", 2, 500)
    perfect = np.where(sc["gid"] >= 0, sc["gid"], -1)
    assert tb.score(perfect, sc["kind"], sc["gid"])["recall"] == 1.0
    r = tb.score(-np.ones(len(sc["gid"]), int), sc["kind"], sc["gid"])
    assert r["recall"] == 0.0 and r["false_per100"] == 0.0


def test_baselines_uniform_signature():
    sc = tb.generate("isolated", "high", 2, 500)
    names = ["hdbscan_base", "lof_group", "mccatch_reimpl"] + (["gapclust_k30"] if gapclust_available() else [])   # GapClust port not shipped (GPL-3)
    for name in names:
        fit, grid = baselines.get(name)
        lab = fit(sc["X"], grid[0])
        assert lab.shape == (len(sc["X"]),)


def test_lof_group_small_data_guard():
    X = np.random.default_rng(0).normal(size=(60, 3))
    fit, _ = baselines.get("lof_group")
    assert fit(X, 0.02).shape == (60,)          # 2% of 60 = 1-2 outliers: must not crash


@pytest.mark.skipif(not gapclust_available(), reason="GapClust port not shipped (GPL-3 upstream); set GAPCLUST_PY")
def test_frontier_small_and_ci():
    fit, _ = baselines.get("gapclust_k30")
    res = tb.frontier(fit, [2.0, 8.0], seeds=[500], n_jobs=1, tiers=("isolated",), contrasts=("high",), dims=(2,))
    t = res["isolated/high"]
    assert 0 <= t["recall@0.25"] <= 1 and t["ci@0.25"][0] <= t["recall@0.25"] + 1e-9


def test_paired_test_identical_methods():
    rows = [{"d": d, "seed": s, "scores": {"a": [{"recall": r, "false_per100": 0.1}], "b": [{"recall": r, "false_per100": 0.1}]}}
            for d in (2, 5, 10) for s, r in zip(range(3), (0.2, 0.5, 0.9))]
    out = stats.paired_test(rows, "a", "b", 0.25)
    assert out["diff"] == 0 and out["p_perm"] == 1.0


def test_paired_test_detects_clear_gap():
    rows = [{"d": d, "seed": s, "scores": {"a": [{"recall": 0.9, "false_per100": 0.1}], "b": [{"recall": 0.1, "false_per100": 0.1}]}}
            for d in (2, 5, 10) for s in range(4)]
    out = stats.paired_test(rows, "a", "b", 0.25)
    assert out["diff"] == pytest.approx(0.8) and out["p_perm"] < 0.001


def test_holm_monotone():
    adj = stats.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == 0.03 and adj["z"] == 0.06 and adj["y"] == 0.06


def test_seeds_frozen():
    assert tuple(tb.seeds.seeds("test", "1.0")) == (500, 501, 502, 503, 504)
    assert tuple(tb.seeds.seeds("test", "1.1")) == (700, 701, 702, 703, 704)


def test_plant_scene_ground_truth():
    Z = np.random.default_rng(1).normal(size=(300, 6))
    pool = tb.tracks.donor_pool(Z, [np.arange(10), np.arange(10, 18)])
    with pytest.warns(DeprecationWarning):
        old = tb.tracks.RECOMMENDED
    for rc in ("iso50", "t3loc1.0", "bootrel", old):
        Zp, gid = tb.tracks.plant_scene(Z, rc, seed=0, pool=pool)
        assert len(Zp) == len(gid) and len(set(gid[gid >= 0])) == 10 and np.array_equal(Zp[:300], Z)


def test_leaderboard_markdown():
    res = {"track": "synthetic/iso", "seeds": "500-504", "cap": 0.25,
           "methods": {"a": {"isolated/high": {"recall": 0.9, "ci": [0.8, 0.95]}}, "b": {"isolated/high": {"recall": 0.5, "ci": [0.4, 0.6]}}},
           "paired": {"isolated/high": {"best": "a", "p_holm": 0.01}}}
    md = to_markdown(res)
    assert "**0.90 [0.80, 0.95]**" in md and "| b |" in md


def test_score_real():
    lab = np.array([0, 0, 1, 1, 1, -1, 2, 2])
    r = tb.score_real(lab, [np.array([2, 3, 4])])
    assert r["recall"] == 1.0 and r["false_per100"] == pytest.approx(100 * 2 / 8)


def test_calibrate_recipe_cross_fit():
    real = {f"c{i}": (3, np.array([0.5, 0.4, 0.6])) for i in range(6)}
    planted = {"easy": np.array([0.9, 0.9, 1.0]), "match": np.array([0.5, 0.45, 0.55]), "hard": np.array([0.1, 0.0, 0.1])}
    out = tb.tracks.calibrate_recipe(real, planted, k_folds=3)
    assert out["recipe"] == "match" and set(out["folds"]) == {"match"} and abs(out["held_out_ratio"] - 1) < 0.05
    with pytest.raises(ValueError):
        tb.tracks.calibrate_recipe({"a": (1, np.ones(3))}, planted)
