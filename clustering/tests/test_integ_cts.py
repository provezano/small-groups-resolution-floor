"""Integration of the stage-9 to stage-12 CTS changes: vectorised scan, fringe fix as the default, weights / flt /
base options, one shared scan for rare-category discovery."""
import hashlib

import numpy as np
import pytest

from clustering.eval.nature import fringe_cts as F
from clustering.eval.nature import flt_cts as FL
from clustering.eval.nature import hdbvs_cts as H
from clustering.eval.nature import rcd_discovery as RC
from clustering.eval.nature import scale_cts as S
from clustering.eval.nature import tiny_methods as T
from clustering.eval.nature.tiny_bench import make

V5 = dict(dim_est="local", scale_min=True, border=True)

# sha256 (first 16 hex) of the int64 labels of the pre-integration code (v5 as frozen before stage 11), for
# cts3(**V5), cts3(**V5, respect_base=True, select="bh"), cts3() and cts3(dim_est="twonn", select="by")
OLD = {
    ("inside", 10, 400, "mid"): ["5f19926d21308ae0", "96ea6e5fcecd3f74", "44f4a0089a527c73", "6eeedd696ec335fc"],
    ("touching", 5, 401, "mid"): ["e7bc7c87dd9017b3", "544545dea4587cb7", "aaa98d44f41ac8dd", "fb96bb42ea53ac99"],
    ("isolated", 2, 402, "low"): ["d15bb557e3ca1290", "6d7233084249c9a6", "6058764d44e40972", "5c64ed6cf1512a5e"],
    ("inside", 5, 403, "high"): ["6ebf94c4aef104fd", "13284138fffbce53", "ffc95b26654253c4", "7b3fe7694a118412"],
}
OLD_KW = [dict(**V5), dict(**V5, respect_base=True, select="bh"), dict(), dict(dim_est="twonn", select="by")]


def _h(lab):
    return hashlib.sha256(np.asarray(lab, np.int64).tobytes()).hexdigest()[:16]


def _groups(lab, base):
    return sorted(tuple(np.flatnonzero(lab == g).tolist()) for g in np.unique(lab) if g > base.max())


@pytest.mark.parametrize("sc", list(OLD))
def test_fix_none_reproduces_old_v5_exactly(sc):
    X, _, _ = make(sc[0], d=sc[1], seed=sc[2], contrast=sc[3])
    assert [_h(T.cts3(X, fix=None, **kw)) for kw in OLD_KW] == OLD[sc]


@pytest.mark.parametrize("kw", [dict(**V5), dict(**V5, select="bh"), dict(**V5, select="by", alpha=0.3), dict(),
                                dict(dim_est="twonn"), dict(scale_min=True), dict(dim_est="local")])
def test_vectorised_scan_equals_loop(kw):
    for sc in [("inside", 10, 401, "mid"), ("touching", 2, 400, "low")]:
        X, _, _ = make(sc[0], d=sc[1], seed=sc[2], contrast=sc[3])
        for fix in (None, "joint"):
            a = T.cts3(X, fix=fix, **kw)
            b = T.cts3(X, fix=fix, vectorised=False, **kw)
            assert np.array_equal(a, b)


def test_fix_none_equals_independent_v5_copy():
    X, _, _ = make("inside", 10, 402, contrast="mid")
    P = H.v5_parts(X)
    for a in (1e-4, 0.05, 0.3):
        assert np.array_equal(T.cts3(X, alpha=a, fix=None, **V5), H.v5_labels(P, a))


def test_joint_is_v5_when_nothing_is_collapsed():
    X = np.random.default_rng(3).uniform(0, 1, (1500, 3))
    X[:4] = 0.5 + 0.002 * np.random.default_rng(4).normal(size=(4, 3))     # one planted group, no family
    a = T.cts3(X, fix=None, **V5, return_cands=True)
    b = T.cts3(X, **V5, return_cands=True)
    assert not a["fams"] and a["cands"] == b["cands"] and np.array_equal(a["labels"], b["labels"])
    assert len(a["groups"]) >= 1


def test_joint_removes_mass_false_groups_and_matches_scalable():
    X, kind, gid = make("inside", 10, 400, contrast="mid")
    base = T.hdb(X)
    g0, g1 = _groups(T.cts3(X, fix=None, **V5), base), _groups(T.cts3(X, **V5), base)
    false = lambda gs: sum(1 for g in gs if not (gid[list(g)] >= 0).any())
    assert false(g1) <= false(g0)
    assert set(g1) <= set(g0) or len(g1) <= len(g0)      # the fix only lowers scores
    _, info = S.cts_scale(X, return_info=True)
    assert sorted(tuple(g.tolist()) for g in info["groups"]) == g1
    _, i2 = F.cts_fringe(X, fix="joint", return_info=True)
    assert sorted(tuple(g.tolist()) for g in i2["groups"]) == g1


def test_cts_scale_fix_none_is_v5():
    X, _, _ = make("touching", 5, 404, contrast="mid")
    _, a = S.cts_scale(X, fix=None, return_info=True)
    _, b = F.cts_fringe(X, fix="none", return_info=True)
    assert [g.tolist() for g in a["groups"]] == [g.tolist() for g in b["groups"]]
    with pytest.raises(ValueError):
        S.cts_scale(X, fix="palm")
    with pytest.raises(ValueError):
        T.cts3(X, fix="palm")


def test_flt_option_reproduces_flt_cts():
    for sc in [("isolated", 2, 401), ("inside", 10, 400)]:
        X, _, _ = make(sc[0], sc[1], sc[2], contrast="mid")
        P = FL.parts(X)
        for a in (0.05, 0.3):
            ref = FL.flt_labels(P, a, stat="ring", q=0.1, eta=0.5)
            assert np.array_equal(T.cts3(X, alpha=a, fix=None, flt=True, **V5), ref)


def test_weights_hook():
    X, _, _ = make("touching", 5, 403, contrast="mid")
    ref = T.cts3(X, **V5)
    assert np.array_equal(T.cts3(X, weights=lambda info: np.ones(len(info["m"])), **V5), ref)
    n = len(T.cts3(X, weights=lambda info: np.ones(len(info["m"])), **V5, return_cands=True)["cands"])
    # all weight on nothing: no scan group can pass
    lab = T.cts3(X, weights=lambda info: np.zeros(len(info["m"])), **V5)
    assert np.array_equal(lab, T.hdb(X)) and n > 0
    with pytest.raises(ValueError):
        T.cts3(X, weights=np.ones(3), **V5)


def test_base_option():
    X, _, _ = make("isolated", 2, 400, contrast="mid")
    b3 = T.hdb3(X)
    lab = T.cts3(X, base="hdb3", **V5)
    new = lab > b3.max()
    assert np.array_equal(lab[~new], b3[~new])
    assert np.array_equal(T.cts3(X, base=None, **V5), T.cts3(X, base=T.hdb(X), **V5))
    with pytest.raises(ValueError):
        T.cts3(X, base="kmeans")


def test_rcd_scan_is_the_shared_scan():
    X, _, _ = make("inside", 10, 400, contrast="mid")
    for fix in (None, "joint"):
        sc = RC.scan(X, fix=fix)
        out = T.cts3(X, fix=fix, **V5, return_cands=True)
        assert sc["cands"] == out["cands"] and sc["groups"] == out["groups"] and sc["n_tests"] == out["n_tests"]
        base = T.hdb(X)
        assert sorted(g[2] for g in sc["groups"]) == _groups(T.cts3(X, fix=fix, **V5), base)
