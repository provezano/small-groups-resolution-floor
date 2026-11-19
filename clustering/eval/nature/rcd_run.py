"""
Runner for the rare-category discovery study (stage 10). For each (scene, seed) it runs every policy with a label
budget and records the query number at which each class was first seen, plus a stopping-rule audit of the scan.

Usage (from a wrapper that puts `prototype` on sys.path; run with python -I because it reads downloaded files):
    run(scenes, seeds, out_json, budget=150, n_jobs=2, methods=None)
"""
import json
import os
import time

import numpy as np

from . import rcd_discovery as R
from .rcd_scenes import scene


def policies(Z, y, rare, seed, sc):
    v, c = np.unique(y, return_counts=True)
    pri = dict(zip(v.tolist(), (c / len(y)).tolist()))
    rshare = [pri[r] for r in rare if r in pri]
    p_up = max(rshare)
    return {
        "random": lambda Z, a: R.pol_random(Z, a, seed),
        "lof": R.pol_lof,
        "iforest": lambda Z, a: R.pol_iforest(Z, a, seed),
        "lof_excl": R.pol_lof_excl,
        # He & Carbonell family. The first runs (tuning, and the first fresh-seed run) counted the point itself
        # among its K nearest neighbours ("_incl"); the default now counts K other points, which is much better on
        # tight classes. The summary gives each baseline the better of the variants, per run.
        "nndb": lambda Z, a: R.pol_nndb(Z, a, float(np.mean(rshare))),
        "malice_dec": lambda Z, a: R.pol_malice(Z, a, pri, "dec"),
        "malice_inc": lambda Z, a: R.pol_malice(Z, a, pri, "inc"),
        "grade": lambda Z, a: R.pol_grade(Z, a, pri, 0.99),
        "grade09": lambda Z, a: R.pol_grade(Z, a, pri, 0.9),
        "grade_li09": lambda Z, a: R.pol_grade_li(Z, a, p_up, 0.9),
        "grade_li": lambda Z, a: R.pol_grade_li(Z, a, p_up, 0.99),
        "nndb_incl": lambda Z, a: R.pol_nndb(Z, a, float(np.mean(rshare)), self_excl=False),
        "malice_dec_incl": lambda Z, a: R.pol_malice(Z, a, pri, "dec", self_excl=False),
        "malice_inc_incl": lambda Z, a: R.pol_malice(Z, a, pri, "inc", self_excl=False),
        "grade_incl": lambda Z, a: R.pol_grade(Z, a, pri, 0.99, self_excl=False),
        "grade09_incl": lambda Z, a: R.pol_grade(Z, a, pri, 0.9, self_excl=False),
        "grade_li_incl": lambda Z, a: R.pol_grade_li(Z, a, p_up, 0.99, self_excl=False),
        "grade_li09_incl": lambda Z, a: R.pol_grade_li(Z, a, p_up, 0.9, self_excl=False),
        "seder": R.pol_seder,
        "interleave": lambda Z, a: R.pol_interleave(Z, a, len(v), seed),
        "hms": R.pol_hms,
        # ours
        "scan_groups_lof": lambda Z, a: R.pol_scan(Z, a, sc=sc, cover=False, fallback="lof", medoid="mean"),
        "scan_cover_lofx": lambda Z, a: R.pol_scan(Z, a, sc=sc, cover=True, fallback="lof_excl"),
        "scan_mix2": lambda Z, a: R.pol_scan(Z, a, sc=sc, cover=True, fallback="lof_excl", mix=2),
        "scan_mix1": lambda Z, a: R.pol_scan(Z, a, sc=sc, cover=True, fallback="lof_excl", mix=1),
        "adaptive": lambda Z, a: R.pol_adaptive(Z, a, sc=sc, patience=3),
        "adaptive1": lambda Z, a: R.pol_adaptive(Z, a, sc=sc, patience=1),
        "scan_lofkey_seq": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof"), rule="seq"),
        "pf_ts3": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof", "iforest"), rule="ts", seed=seed),
        "pf_ts4": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof", "iforest", "hms"), rule="ts", seed=seed),
        "pf_ucb3": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof", "iforest"), rule="ucb", c=0.5),
        "pf_ts_rank": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "rankavg"), rule="ts", seed=seed),
        "pf_ts_lof": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof"), rule="ts", seed=seed),
        "pf_ts3k3": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof3", "iforest3"), rule="ts", seed=seed),
        "scan_seq_k3": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof3"), rule="seq"),
        "lof_excl3": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("lof3",), rule="seq"),
        "pf_warm1": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof3", "iforest3"), rule="ts", seed=seed, warm=1),
        "pf_warm2": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof3", "iforest3"), rule="ts", seed=seed, warm=2),
        "pf_warm3": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("scan", "lof3", "iforest3"), rule="ts", seed=seed, warm=3),
        "iforest_excl": lambda Z, a: R.pol_portfolio(Z, a, sc=sc, arms=("iforest",), rule="seq", seed=seed),
    }


def one(name, seed, budget, methods, alpha=0.05):
    Z, y, rare, info = scene(name, seed)
    y = np.asarray(y).astype(str)
    rare = sorted(str(r) for r in rare)
    t0 = time.time()
    sc = R.scan(Z, alpha)
    res = {"scene": name, "seed": seed, "n": len(y), "rare": {r: int((y == r).sum()) for r in rare},
           "n_classes": int(len(np.unique(y))), "scan_time": round(time.time() - t0, 2),
           "n_cands": len(sc["cands"]), "n_groups": len(sc["groups"]), "n_fams": len(sc["fams"]), "dim": sc["dim"]}
    # stopping-rule audit: the scan alone, no target, no fallback
    for cover in (False, True):
        log = {}
        o = R.Oracle(y, 10 ** 9, {"__never__"})
        try:
            R.pol_scan(Z, o, sc=sc, cover=cover, fallback=None, medoid="centre", stop_log=log)
        except R.Done:
            pass
        key = "cover" if cover else "groups"
        res[f"tau_{key}"] = log.get("tau", len(o.asked))
        res[f"audit_{key}"] = R.audit(sc, y, rare, o.asked)
        res[f"classes_at_tau_{key}"] = sorted(set(y[o.asked].tolist())) if o.asked else []
    hits, times = {}, {}
    pols = policies(Z, y, rare, seed, sc)
    for k in (methods or pols):
        t = time.time()
        try:
            asked = R.run_policy(pols[k], Z, y, budget, set(rare))
            fh = R.first_hits(asked, y)
            hits[k] = {r: fh.get(r) for r in rare}
        except Exception as e:                       # a baseline that fails is recorded, not hidden
            hits[k] = {"error": repr(e)[:200]}
        times[k] = round(time.time() - t, 2)
    res["hits"], res["time"] = hits, times
    return res


def run(scenes, seeds, out, budget=150, n_jobs=2, methods=None):
    from joblib import Parallel, delayed
    jobs = [(s, sd) for s in scenes for sd in seeds]
    rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(s, sd, budget, methods) for s, sd in jobs)
    json.dump({"budget": budget, "rows": rows}, open(out, "w"), indent=1, default=str)
    os.chmod(out, 0o600)
    return rows


def summarise(rows, budget):
    """Per scene and method: mean queries until every rare class is seen (budget + 1 when not all are seen), share of
    runs with all found, mean share of rare classes found at the budget, mean area under the discovery curve."""
    out = {}
    for r in rows:
        sc = out.setdefault(r["scene"], {})
        rare = list(r["rare"])
        for k, h in r["hits"].items():
            if "error" in h:
                continue
            t = [h[c] if h[c] is not None else None for c in rare]
            allq = max(t) if all(x is not None for x in t) else budget + 1
            curve = np.array([np.mean([x is not None and x <= q for x in t]) for q in range(1, budget + 1)])
            d = sc.setdefault(k, {"q_all": [], "all_found": [], "found": [], "auc": []})
            d["q_all"].append(allq); d["all_found"].append(allq <= budget)
            d["found"].append(float(np.mean([x is not None for x in t]))); d["auc"].append(float(curve.mean()))
    return {s: {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in sc.items()} for s, sc in out.items()}
