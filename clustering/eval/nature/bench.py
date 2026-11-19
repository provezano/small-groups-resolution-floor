"""
Shared benchmark for the clustering-method experiments (2026-11-09).

Synthetic sets with planted truth for the regime the build struggles with: one big job of near-duplicate sessions
(several templates, each a tight pile) next to groups of 3-9 sessions, people's mid-size groups and scattered single
sessions. Label 0 is the job, -1 the scattered singles. Metrics count how many groups the job breaks into, how many
small groups come back (best Jaccard >= 0.5), ARI on the planted structure (predicted noise counts as singletons) and
K. The real window is loaded once from the share build (same features as cluster_dedup) and cached locally.
"""
import json
import os

import numpy as np
from sklearn.metrics import adjusted_rand_score

SCENARIOS = {
    # job: n points, templates, template spread, pile jitter; small: groups, size range, spread, distance from job
    "separated": dict(job=900, templates=15, t_sd=0.30, jit=0.01, small=10, small_size=(3, 9), s_sd=0.05,
                      s_dist=3.0, mid=4, mid_size=(30, 80), m_sd=0.25, singles=300, single_sd=4.0),
    "touching": dict(job=900, templates=15, t_sd=0.30, jit=0.01, small=10, small_size=(3, 9), s_sd=0.05,
                     s_dist=1.2, mid=4, mid_size=(30, 80), m_sd=0.25, singles=300, single_sd=4.0),
    "density_10x": dict(job=900, templates=15, t_sd=0.30, jit=0.01, small=10, small_size=(3, 9), s_sd=0.25,
                        s_dist=3.0, mid=4, mid_size=(30, 80), m_sd=0.80, singles=300, single_sd=4.0),
    "no_templates": dict(job=900, templates=1, t_sd=0.0, jit=0.05, small=10, small_size=(3, 9), s_sd=0.05,
                         s_dist=3.0, mid=4, mid_size=(30, 80), m_sd=0.25, singles=300, single_sd=4.0),
}


def _unit(rng, d):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def make_synthetic(name, seed, dim=20):
    """-> X (n, dim), y (n,) with 0 = job, 1..k = planted groups, -1 = scattered singles; kind per label;
    tmpl (n,) = the job template of each job point (-1 elsewhere)."""
    p, rng = SCENARIOS[name], np.random.default_rng(seed)
    X, y, kind, tmpl = [], [], {0: "job"}, []
    centres = rng.normal(scale=p["t_sd"], size=(p["templates"], dim))
    sizes = rng.multinomial(p["job"], np.ones(p["templates"]) / p["templates"])
    for t, (c, n) in enumerate(zip(centres, sizes)):
        X.append(c + rng.normal(scale=p["jit"], size=(n, dim)))
        y += [0] * n
        tmpl += [t] * n
    lab = 1
    for _ in range(p["small"]):
        n = int(rng.integers(p["small_size"][0], p["small_size"][1] + 1))
        X.append(_unit(rng, dim) * p["s_dist"] + rng.normal(scale=p["s_sd"], size=(n, dim)))
        y += [lab] * n
        kind[lab] = "small"
        lab += 1
    for _ in range(p["mid"]):
        n = int(rng.integers(p["mid_size"][0], p["mid_size"][1] + 1))
        X.append(_unit(rng, dim) * rng.uniform(5, 8) + rng.normal(scale=p["m_sd"], size=(n, dim)))
        y += [lab] * n
        kind[lab] = "mid"
        lab += 1
    X.append(rng.normal(scale=p["single_sd"], size=(p["singles"], dim)))
    y += [-1] * p["singles"]
    tmpl = np.r_[tmpl, -np.ones(len(y) - len(tmpl), int)]
    return np.vstack(X), np.array(y), kind, tmpl


def _as_singletons(pred):
    pred = np.asarray(pred).copy()
    noise = pred < 0
    pred[noise] = pred.max() + 1 + np.arange(noise.sum())
    return pred


def score(y, pred, kind, tmpl=None):
    """tmpl given: also ARI against the finer truth where each job template is a group of its own."""
    y, pred = np.asarray(y), np.asarray(pred)
    planted = y >= 0
    job = y == 0
    jl, jc = np.unique(pred[job & (pred >= 0)], return_counts=True)
    pieces = int((jc >= 0.01 * job.sum()).sum())
    rec = []
    for g, k in kind.items():
        if k != "small":
            continue
        m = y == g
        best = 0.0
        for c in set(pred[m]) - {-1}:
            pc = pred == c
            best = max(best, (m & pc).sum() / (m | pc).sum())
        rec.append(best >= 0.5)
    out = {}
    if tmpl is not None:
        yf = np.where(job, 10_000 + tmpl, y)
        out["ari_fine"] = round(float(adjusted_rand_score(yf[planted], _as_singletons(pred)[planted])), 3)
    return {**out, "ari": round(float(adjusted_rand_score(y[planted], _as_singletons(pred)[planted])), 3),
            "job_pieces": pieces, "job_noise": round(float((pred[job] < 0).mean()), 3),
            "small_recall": round(float(np.mean(rec)), 3) if rec else None,
            "k_found": int(len(set(pred) - {-1})), "k_true": int(len(set(y) - {-1})),
            "noise": round(float((pred < 0).mean()), 3)}


REAL_CACHE = "data/processed_share/2026-10-14_5h/nature/real_cache.npz"


def load_real(root):
    """Real 5 h window as the share build featurizes it. Builds the cache on first use (needs CLUSTER_HASH_SALT)."""
    path = os.path.join(root, REAL_CACHE)
    if not os.path.exists(path):
        _build_real(root, path)
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _build_real(root, path):
    from ... import dq as DQ
    from ...linkage import build_linkage
    from ...loaders import read_real_export
    from ...pipeline import _abs, _fz_fit, _fz_transform, load_config, make_featurizer, session_text, text_embedder
    from ...preprocess import build_sessions_meta
    from ..run_ae_overseer_canaries import LABELS
    cfg = load_config(_abs("prototype/clustering/configs/share_e6_v02.yaml"))
    snap = _abs(cfg["snapshot"])
    a, r, info = read_real_export(snap, keep_text=True)
    rows = DQ.run_dq(a, r, info, cfg).rows
    sess = build_sessions_meta(rows).sort_values(["ts_start", "session_hash"], kind="stable").reset_index(drop=True)
    lk = build_linkage(rows, sess, **cfg["linkage"])["features"]
    _, E, _ = session_text(rows, sess, text_embedder(cfg, os.path.basename(os.path.normpath(snap))))
    fz = make_featurizer(cfg, cfg["seed"])
    _fz_fit(fz, sess, lk, E)
    X = _fz_transform(fz, sess, lk, E)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    ui = {s["session"]: s for s in json.load(open(_abs("data/processed_share/2026-10-14_5h/ui_export/sessions.json")))
          ["sessions"]}
    hashes = sess["session_hash"].astype(str).values.astype("U")
    auto = np.array([(ui.get(s) or {}).get("workload") == "w1" for s in hashes])
    mentioned = np.array([bool((ui.get(s) or {}).get("mentioned_by")) for s in hashes])
    idx = {s: i for i, s in enumerate(hashes)}
    confirmed = np.array([idx[s] for s in json.load(open(_abs(LABELS)))["positives_v2"].values() if s in idx], int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), auto=auto, mentioned=mentioned, confirmed=confirmed,
                        session=hashes, ts=sess["ts_start"].values.astype("datetime64[s]").astype(np.int64))
    os.chmod(path, 0o600)


def score_real(lab, d):
    """Real window: groups the job's sessions fall in, groups/noise overall, and per confirmed incident the size of its
    group and the share of that group a finding mentions."""
    auto, mentioned, conf = d["auto"], d["mentioned"], d["confirmed"]
    groups = sorted(set(lab) - {-1})
    size = {g: int((lab == g).sum()) for g in groups}
    jl, jc = np.unique(lab[auto & (lab >= 0)], return_counts=True)
    inc_size = [size.get(lab[i], 0) for i in conf]
    inc_share = [0.0 if lab[i] < 0 else float(mentioned[lab == lab[i]].mean()) for i in conf]
    return {"groups": len(groups), "noise": round(float((lab < 0).mean()), 3),
            "job_pieces": int((jc >= 0.01 * auto.sum()).sum()), "job_noise": round(float((lab[auto] < 0).mean()), 3),
            "incidents_in_noise": int(sum(lab[i] < 0 for i in conf)), "incident_median_group": float(np.median(inc_size)),
            "incident_median_mentioned_share": round(float(np.median(inc_share)), 3)}
