import numpy as np

from clustering.eval.nature import pass2_analysis as A
from clustering.eval.nature import pass2_frontier as F
from clustering.eval.nature.p3_leaderboard import TIERS


def _rows(n_seeds=6, gap=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in TIERS:
        p, c = t.split("/")
        for d in (2, 5, 10):
            for s in range(800, 800 + n_seeds):
                sc = {}
                for m in A.OURS + A.PUBLISHED + ("oracle",):
                    base = 0.5 + (gap if m in A.OURS or m == "oracle" else 0.0)
                    sc[m] = [{"recall": float(np.clip(base + rng.normal(0, 0.05), 0, 1)), "false_per100": f}
                             for f in (0.1, 0.2, 0.4)]
                rows.append({"placement": p, "contrast": c, "d": d, "seed": s, "scores": sc})
    return rows


def test_compare_detects_clear_gap_and_holm_monotone():
    rows = _rows()
    res = A.compare(rows, "cts3_v5", A.PUBLISHED, 0.25, B=200, n_perm=2000)
    for t in TIERS:
        assert res[t]["diff"] > 0.2
        assert res[t]["p_perm"] < 0.01
        assert res[t]["p_holm"] >= res[t]["p_perm"]
        assert res[t]["b"] in A.PUBLISHED


def test_compare_no_gap_not_significant():
    rows = _rows(gap=0.0, seed=1)
    res = A.compare(rows, "cts3_v5", "gapclust_port_k30", 0.25, B=200, n_perm=2000)
    assert sum(res[t]["p_holm"] < 0.05 for t in TIERS) <= 1


def test_frontier_registry_has_calibrated_rules():
    assert "cts3_v5_by" in F.P.SINGLE and "hdbscan_mcs" in F.P.MULTI and "gapclust_official_sq_k30" in F.P.MULTI
    assert set(F.METHODS) <= set(F.P.SINGLE) | set(F.P.MULTI) | {"oracle"}
