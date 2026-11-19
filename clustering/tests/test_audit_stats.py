"""Stage 14 (audit): the paired_test bias, the grid-free replacement, and the decoy scene builder."""
import numpy as np
import pytest

from clustering.eval.nature import audit_decoy as D
from clustering.eval.nature import audit_stats as A
from clustering.eval.nature import p3_stats as S
from clustering.eval.nature.tiny_bench import make


def _nested_rows(S_per_d=4, gain_scene=0):
    """A = B's 6-point grid plus a 7th point equal to B's best point except +1 hit (of 12) on one scene: a true
    frontier difference of zero in any useful sense, and an observed one of 1/(12 * scenes)."""
    rng = np.random.default_rng(3)
    rows = []
    k = 0
    for d in (2, 5, 10):
        for s in range(S_per_d):
            base = [{"recall": float(r), "false_per100": float(f)} for r, f in
                    zip(np.sort(rng.integers(0, 13, 6)) / 12, np.sort(rng.uniform(0.0, 0.3, 6)))]
            base[3]["false_per100"] = 0.1               # best feasible point: index 3 at cap 0.25 (pooled)
            for j in (4, 5):
                base[j]["false_per100"] = 0.6
            extra = dict(base[3])
            if k == gain_scene:
                extra["recall"] = min(1.0, extra["recall"] + 1 / 12)
            rows.append({"d": d, "seed": s, "scores": {"b": base, "a": base + [extra]}})
            k += 1
    return rows


def test_old_test_wrong_on_nested_grids():
    rows = _nested_rows()
    assert len(rows) == 12
    old = S.paired_test(rows, "a", "b", 0.25)
    assert old["n_perm"] == 2 ** 12
    assert old["p_perm"] <= 0.001          # 2 / 2^12 (rounded): the null collapses to {0, +obs, -obs}
    an = A.old_test_anatomy(rows, "a", "b", 0.25)
    assert an["nested_prefix"] and an["null_zero_share"] == 1.0
    new = A.paired_boot(rows, "a", "b", 0.25, B=2000)
    assert 0 < new["diff"] < 0.01
    assert new["p_boot"] > 0.2


def test_old_test_misses_shorter_grid_win():
    long9 = [(0.1, .01), (0.15, .02), (0.2, .03), (0.25, .04), (0.3, .05), (0.35, .07), (0.45, .10), (0.55, .15),
             (0.6, .20)]
    short6 = [(r + 0.2, f) for r, f in [(0.3, .02), (0.4, .05), (0.5, .10), (0.55, .15), (0.6, .20), (0.65, .40)]]
    rows = A.synth_rows({"a": long9, "b": short6}, seed=11, rho=0.8)
    old = S.paired_test(rows, "a", "b", 0.25, n_perm=4000)
    new = A.paired_boot(rows, "a", "b", 0.25, B=4000)
    assert new["diff"] < -0.12
    assert new["p_boot"] < 0.01
    assert old["p_perm"] > new["p_boot"] * 10      # the old test is far weaker when the shorter grid wins


def test_new_test_ignores_grid_order_and_padding():
    rows = A.synth_rows({"a": [(0.3, .02), (0.5, .1), (0.6, .2), (0.7, .5)],
                         "b": [(0.35, .03), (0.55, .12), (0.62, .3)]}, seed=5)
    W = S.boot_weights(rows, 3000, 1)
    ref = A.paired_boot(rows, "a", "b", 0.25, W=W)
    for r in rows:      # reverse A's grid and pad B with unselectable points
        r["scores"]["a"] = r["scores"]["a"][::-1]
        r["scores"]["b"] = r["scores"]["b"] + [{"recall": 1.0, "false_per100": 99.0}] * 5
    alt = A.paired_boot(rows, "a", "b", 0.25, W=W)
    for k in ("diff", "p_boot", "p_basic", "p_pct", "ci_basic"):
        assert alt[k] == ref[k]


def test_identical_methods_not_significant():
    rows = A.synth_rows({"a": [(0.3, .02), (0.5, .1), (0.6, .2)]}, seed=2)
    for r in rows:
        r["scores"]["b"] = [dict(x) for x in r["scores"]["a"]]
    out = A.paired_boot(rows, "a", "b", 0.25, B=1000)
    assert out["diff"] == 0 and out["p_boot"] == 1.0


def test_boundary_case_max_rule_protects():
    """Extra point whose true false rate (0.27) is just above the cap: the pivot alone over-rejects, max(basic, pct)
    does not (synthetic N2 of the audit doc, small version)."""
    base6 = [(0.3, .02), (0.4, .05), (0.5, .10), (0.55, .15), (0.6, .20), (0.65, .40)]
    rej_basic = rej_max = 0
    n = 40
    for rep in range(n):
        rows = A.synth_rows({"a": base6 + [(0.66, .27), (0.70, .35), (0.80, 1.0)]}, seed=900 + rep)
        for r in rows:
            r["scores"]["b"] = r["scores"]["a"][:6]
        o = A.paired_boot(rows, "a", "b", 0.25, B=1000, seed=rep)
        rej_basic += o["p_basic"] <= 0.05
        rej_max += o["p_boot"] <= 0.05
    assert rej_max / n <= 0.05 and rej_basic > rej_max


def test_crossfit_runs():
    rows = A.synth_rows({"a": [(0.3, .02), (0.6, .2)], "b": [(0.3, .02), (0.6, .2)]}, seed=1)
    out = A.crossfit_test(rows, "a", "b", 0.25, n_perm=500)
    assert 0 <= out["p_crossfit"] <= 1


@pytest.mark.parametrize("d", [2, 5, 10])
def test_decoy_planted_arm_replays_make(d):
    X, k, g = make("touching", d, 620, contrast="low")
    Y, k2, g2, _ = D.make_arm("planted", d, 620)
    assert np.array_equal(X, Y) and np.array_equal(g, g2) and np.array_equal(k, k2)


def test_decoy_arms_geometry():
    from clustering.eval.nature.tiny_oracle import params
    d, seed = 5, 620
    pr = params("touching", d, seed, "low")
    Q = pr["Q"]
    P, _, gp, _ = D.make_arm("planted", d, seed)
    R, _, gr, _ = D.make_arm("radial", d, seed)
    E, ke, ge, info = D.make_arm("empty", d, seed)
    assert not (ke == "group").any() and len(E) == len(P) - sum(info["sizes"])
    assert [int((ge == i).sum()) for i in range(12)] == info["sizes"]
    comps = [(w, c, s) for w, c, s in pr["comps"] if w < 200 and s > 0.1]
    for i, (m, c, sd) in enumerate(pr["groups"]):
        b = min(comps, key=lambda t: np.linalg.norm(c - t[1]) / t[2])[1]
        rp = np.sort(np.linalg.norm((P @ Q)[gp == i] - b, axis=1))
        rr = np.sort(np.linalg.norm((R @ Q)[gr == i] - b, axis=1))
        assert np.allclose(rp, rr, atol=0.02)                   # radii kept (ambient noise 0.002 per coordinate)
    # background points are identical across arms
    bgP, bgE = P[gp < 0], E[(ke != "group")]
    assert np.allclose(bgP[:len(bgE)], bgE, atol=0.05)


def test_cone_dirs_within_cap():
    rng = np.random.default_rng(0)
    u = np.zeros(10); u[0] = 1
    V = D._cone_dirs(u, 0.4, 500, rng, 10)
    assert np.allclose(np.linalg.norm(V, axis=1), 1) and np.all(V @ u >= np.cos(0.4) - 1e-9)
