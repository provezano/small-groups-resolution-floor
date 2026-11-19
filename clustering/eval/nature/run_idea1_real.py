"""Idea 1 on the real 5 h window: session families (parent-offspring model) vs simple link rules.

Weak truth: sessions named together by one finding form a family. Only pairs of mentioned sessions are scored, since
unmentioned sessions are unknown, not negatives. The job (w1) is removed first. Settings are fixed, not tuned on labels.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

from .bench import load_real
from .idea1 import baseline_gap, baseline_simlink, fit_families, pair_scores

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
UI = "data/processed_share/2026-10-14_5h/ui_export/sessions.json"
META = "data/interim_share/2026-10-14_5h/sessions_meta.parquet"
OUT = "results/idea1_real.json"


def weak_families(sessions, ui):
    """Union of sessions that share a finding (findings naming 2-13 sessions); -1 for sessions in no such finding."""
    by = {}
    for i, s in enumerate(sessions):
        for f in (ui.get(s) or {}).get("mentioned_by") or []:
            by.setdefault(f["id"], []).append(i)
    ii, jj = [], []
    for members in by.values():
        if 2 <= len(members) <= 13:
            ii += members[:-1]
            jj += members[1:]
    n = len(sessions)
    lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n)), directed=False)[1]
    size = np.bincount(lab)
    return np.where(size[lab] > 1, lab, -1), sum(2 <= len(m) <= 13 for m in by.values())


def summarise(truth, pred, mask):
    sc = pair_scores(truth, pred, mask)
    size = np.bincount(pred)
    sc["families_2plus"] = int((size >= 2).sum())
    sc["largest"] = int(size.max())
    sc["mentioned_linked"] = round(float((size[pred[mask]] >= 2).mean()), 3)
    return sc


def main():
    d = load_real(ROOT)
    ui = {s["session"]: s for s in json.load(open(os.path.join(ROOT, UI)))["sessions"]}
    meta = pd.read_parquet(os.path.join(ROOT, META), columns=["session_hash", "actor_hash"])
    actor = dict(zip(meta["session_hash"].astype(str), meta["actor_hash"].astype(str)))
    keep = ~d["auto"]
    sess = d["session"][keep]
    ts = d["ts"][keep].astype(float)
    o = np.argsort(ts, kind="stable")
    sess, ts = sess[o], ts[o]
    X = d["X"][keep][o].astype(float)
    X = X - X.mean(0)
    E = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    user = pd.factorize(np.array([actor.get(s, s) for s in sess]))[0]
    truth, n_find = weak_families(sess, ui)
    mask = truth >= 0
    print(f"sessions {len(sess)}, users {user.max() + 1}, findings with 2-13 sessions {n_find}, "
          f"sessions in weak families {mask.sum()}, families {len(set(truth[mask]))}, "
          f"cross-user families {sum(len(set(user[truth == f])) > 1 for f in set(truth[mask]))}")
    res = {"n": int(len(sess)), "in_families": int(mask.sum())}
    # fixed settings, plus a small range to show sensitivity (none chosen on labels)
    for g in (10, 30, 60):
        res[f"gap_user_{g}m"] = summarise(truth, baseline_gap(ts, user, g), mask)
        for c in (0.7, 0.85):
            res[f"gap_user_{g}m_cos{c}"] = summarise(truth, baseline_gap(ts, user, g, E, c), mask)
            res[f"simlink_{g}m_cos{c}"] = summarise(truth, baseline_simlink(ts, E, g, c), mask)
    fam_map, fam_cons, params, _ = fit_families(ts, user, E)
    res["model_map"] = summarise(truth, fam_map, mask)
    res["model_consensus"] = summarise(truth, fam_cons, mask)
    res["model_params"] = {k: float(np.round(v, 4)) for k, v in params.items()}
    res["singletons"] = summarise(truth, np.arange(len(sess)), mask)
    for k, v in res.items():
        if isinstance(v, dict) and "f1" in v:
            print(f"  {k:24s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} ARI {v['ari']:.3f} "
                  f"fam {v['families_2plus']:4d} largest {v['largest']:4d} mentioned_linked {v['mentioned_linked']}")
    print("  model params", res["model_params"])
    p = os.path.join(ROOT, OUT)
    json.dump(res, open(p, "w"), indent=1)
    os.chmod(p, 0o600)


if __name__ == "__main__":
    main()
