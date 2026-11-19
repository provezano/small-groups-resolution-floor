"""paired_boot (vendored, stage 15): the recommended paired test."""
import numpy as np

from tinybench import stats


def _rows(gap):
    rng = np.random.default_rng(0)
    rows = []
    for d in (2, 5, 10):
        for s in range(5):
            base = [{"recall": float(r), "false_per100": float(f)}
                    for r, f in zip(np.sort(rng.uniform(0, 0.5, 4)), (0.0, 0.1, 0.2, 0.6))]
            rows.append({"d": d, "seed": s, "scores": {"a": [dict(x, recall=min(1, x["recall"] + gap)) for x in base],
                                                        "b": base}})
    return rows


def test_identical_methods_not_rejected():
    out = stats.paired_boot(_rows(0.0), "a", "b", 0.25, B=500)
    assert out["diff"] == 0 and out["p"] == out["p_boot"] == 1.0


def test_clear_gap_rejected():
    out = stats.paired_boot(_rows(0.3), "a", "b", 0.25, B=1000)
    assert out["diff"] > 0.25 and out["p"] < 0.01
