import numpy as np
import pytest

from clustering.eval.nature import rcd_discovery as R
from clustering.eval.nature.tiny_methods import cts3, hdb


def _scene(seed=0, n=600, d=4, m=6, tight=0.02):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = np.array(["bg"] * n, dtype=object)
    for g, c in enumerate(rng.uniform(0.2, 0.8, (3, d))):
        X = np.r_[X, c + tight * rng.normal(size=(m, d))]
        y = np.r_[y, [f"r{g}"] * m]
    return X, y.astype(str)


def test_scan_matches_cts3_groups():
    X, _ = _scene()
    sc = R.scan(X)
    lab = cts3(X, dim_est="local", scale_min=True, border=True)
    base = hdb(X)
    g1 = sorted(tuple(sorted(np.flatnonzero(lab == g).tolist())) for g in np.unique(lab) if g > base.max())
    g2 = sorted(tuple(sorted(m)) for s, c, m in sc["groups"])
    assert g1 == g2 and len(g2) >= 3


def test_stopping_rule_completeness_any_order():
    """Theorem 1: at the stopping time, every class that contains a significant candidate has been seen, whatever
    the order in which the candidates are hit."""
    X, y = _scene(1)
    sc = R.scan(X)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        cands = [sc["cands"][i] for i in rng.permutation(len(sc["cands"]))]
        labelled = set()
        for s, c, mem in cands:
            if labelled.isdisjoint(mem):
                labelled.add(int(rng.choice(mem)))       # any member, not only the centre
        seen = {y[i] for i in labelled}
        for cl in set(y):
            if any(all(y[m] == cl for m in mem) for s, c, mem in sc["cands"]):
                assert cl in seen


def test_policy_finds_tight_classes_and_stops():
    X, y = _scene(2)
    log = {}
    o = R.Oracle(y, 10 ** 6, {"__never__"})
    try:
        R.pol_scan(X, o, cover=True, fallback=None, stop_log=log)
    except R.Done:
        pass
    seen = set(y[o.asked])
    assert {"r0", "r1", "r2"} <= seen
    assert log["tau"] == len(o.asked) <= 15


def test_null_rarely_stops_late():
    """Under a homogeneous null the scan flags nothing in most runs (FWER <= alpha up to the d estimate)."""
    flagged = 0
    for s in range(10):
        X = np.random.default_rng(100 + s).uniform(0, 1, (500, 3))
        flagged += len(R.scan(X)["cands"]) > 0
    assert flagged <= 3


@pytest.mark.parametrize("pol", ["malice", "grade", "seder", "interleave", "hms", "nndb", "lof_excl"])
def test_baselines_run(pol):
    X, y = _scene(3, n=300)
    v, c = np.unique(y, return_counts=True)
    pri = dict(zip(v, c / len(y)))
    f = {"malice": lambda Z, a: R.pol_malice(Z, a, pri), "grade": lambda Z, a: R.pol_grade(Z, a, pri),
         "seder": R.pol_seder, "interleave": lambda Z, a: R.pol_interleave(Z, a, len(v)), "hms": R.pol_hms,
         "nndb": lambda Z, a: R.pol_nndb(Z, a, 6 / len(y)), "lof_excl": R.pol_lof_excl}[pol]
    asked = R.run_policy(f, X, y, 60, {"r0", "r1", "r2"})
    assert 0 < len(asked) <= 60 and len(set(asked)) == len(asked)
