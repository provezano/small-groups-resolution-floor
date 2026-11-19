import numpy as np

from clustering.eval.nature.tiny_bench import make, score
from clustering.eval.nature.tiny_oracle import log_intensity, params
from clustering.eval.nature.tiny_methods import cts3, hdb


def test_make_is_deterministic():
    a = make("touching", 5, 3, contrast="low")
    b = make("touching", 5, 3, contrast="low")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_params_replays_make():
    """The oracle's parameters come from the same random stream: group sizes and the ambient map must match."""
    for p in ("isolated", "touching", "inside"):
        X, kind, gid = make(p, 5, 7, contrast="mid")
        pr = params(p, 5, 7, "mid")
        assert [m for m, _, _ in pr["groups"]] == [int((gid == g).sum()) for g in range(12)]
        for m, c, sd in pr["groups"][:3]:
            g = np.flatnonzero(gid == pr["groups"].index((m, c, sd)))
            assert np.linalg.norm(X[g].mean(0) - c @ pr["Q"].T) < 5 * sd + 0.05
        assert sum(w for w, _, _ in pr["comps"]) + pr["box"][0] == int((gid < 0).sum())


def test_intensity_peak_and_floor():
    """At a near-copy template's centre the intensity is that component's peak (it dwarfs the rest); far outside
    every cluster but inside the box it is the uniform noise floor."""
    pr = params("isolated", 2, 1, "mid")
    w, c, sd = pr["comps"][0]
    peak = np.log(w) - 2 * np.log(sd) - np.log(2 * np.pi)
    assert abs(log_intensity(c[None], pr)[0] - peak) < 0.05
    nn, lo, hi = pr["box"]
    centres = np.array([cc for _, cc, _ in pr["comps"]])
    rng = np.random.default_rng(0)
    x = rng.uniform(lo, hi, (5000, 2))
    far = x[np.min(np.linalg.norm(x[:, None] - centres[None], axis=2), 1) > 6]
    floor = np.log(nn) - np.log(hi - lo).sum()
    assert np.allclose(log_intensity(far, pr), floor, atol=1e-3)


def test_score_perfect_and_empty():
    X, kind, gid = make("isolated", 2, 0, contrast="high")
    lab = np.where(gid >= 0, gid + 1000, -1)
    s = score(lab, kind, gid)
    assert s["recall"] == 1.0 and s["false_per100"] == 0.0
    assert score(np.full(len(gid), -1), kind, gid)["recall"] == 0.0


def test_cts3_v5_runs_and_keeps_base():
    X, kind, gid = make("isolated", 5, 2, contrast="high")
    base = hdb(X)
    lab = cts3(X, dim_est="local", scale_min=True, border=True)
    new = lab > base.max()
    assert np.array_equal(lab[~new], base[~new])          # groups are written over the base, nothing else changes
    assert score(lab, kind, gid)["recall"] > 0.5
