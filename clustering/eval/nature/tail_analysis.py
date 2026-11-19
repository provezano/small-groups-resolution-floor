"""
Summaries for the directional-ring branch (stage 13): frontier per tier, paired tests with Holm over every variant,
null false-scene rates, and the touching/low error analysis.

  python -m clustering.eval.nature.tail_analysis bench --inp scenes.json --out summary.json [--err err.json]
  python -m clustering.eval.nature.tail_analysis nulls --inp nulls.json --out null_summary.json
"""
import argparse
import json
import os

import numpy as np

from . import p3_stats as S
from .tail_bench import ALPHAS, NULL_ALPHAS

REF = "tail_full_A6"          # v5 + fringe on v5's published grid (alpha <= 0.3)
BASELINES = ("lof+group", "gapclust_port_k30")
TIERS9 = [(p, c) for p in ("isolated", "touching", "inside") for c in ("high", "mid", "low")]


def add_ref(rows):
    for r in rows:
        r["scores"][REF] = r["scores"]["tail_full"][:6]
    return rows


def op_point(rows, m, cap):
    R, F = S.arrays(rows, m)
    r, f = R.mean(0), F.mean(0)
    ok = np.flatnonzero(f <= cap + 1e-12)
    return int(ok[np.argmax(r[ok])]) if len(ok) else None


def compare(rows, a, b, cap, W, test="fixed"):
    """Paired comparison that is fair when the two grids differ in length.
    WHICH TEST TO PASS  pass test="boot". It is what `bench` here and
    `two_analysis.bench` now use, and the only audited choice: "fixed" is invalid on the three counts in
    audit_stats.py:1-15 whenever the two grids are nested or of unequal length. The default is left at
    "fixed" ONLY so that this shared helper does not silently change the statistics of callers the frozen audit rule did
    not cover (union_analysis.py:54, which also reads the "p_signflip" key this function only returns under
    "fixed"). Those callers pass it explicitly; see UNION-SIGNFLIP in the backlog.
    test="fixed": exact-style sign-flip test on the
    per-scene recall difference at each method's pooled operating point (p3_stats.paired_fixed; optimistic about the
    choice of operating point, symmetric between the two methods). CI: stratified bootstrap of the frontier difference
    with the operating point re-selected in every resample.
    test="boot" (stage 15): p from p3_stats.paired_boot instead (generic key "p"; the fixed-point test is kept under
    "fixed_op"). test="fixed" (default) is unchanged."""
    if test == "boot":
        t = S.paired(rows, a, b, cap, test="boot")
        t["fixed_op"] = S.paired_fixed(rows, a, b, cap)
        t["frontier_diff"] = t["diff"]
        return t
    if test != "fixed":
        raise ValueError("test must be 'fixed' or 'boot'")
    t = S.paired_fixed(rows, a, b, cap)
    Ra, Fa = S.arrays(rows, a)
    Rb, Fb = S.arrays(rows, b)
    bd = S.frontier_from(Ra, Fa, cap, W) - S.frontier_from(Rb, Fb, cap, W)
    t["frontier_diff"] = round(S.frontier_from(Ra, Fa, cap) - S.frontier_from(Rb, Fb, cap), 3)
    t["diff_ci"] = [round(float(v), 3) for v in np.quantile(bd, [0.025, 0.975])]
    t["p_perm"] = t["p_signflip"]
    return t


def bench(rows, cap=0.25, n_perm=20000, test="boot"):
    rows = add_ref(rows)
    methods = sorted(rows[0]["scores"])
    variants = [m for m in methods if m.startswith("tail_") and m != REF]
    out = {"cap": cap, "n_scenes": len(rows), "frontier": {}, "vs_ref": {}, "vs_base": {}, "per_d": {}}
    pv = {}
    for p, c in TIERS9:
        tr = [r for r in rows if r["placement"] == p and r["contrast"] == c]
        key = f"{p}/{c}"
        W = S.boot_weights(tr, B=2000, seed=0)
        out["frontier"][key] = {m: round(S.frontier_from(*S.arrays(tr, m), cap), 3) for m in methods}
        for m in methods:
            g = op_point(tr, m, cap)
            out["per_d"].setdefault(key, {})[m] = {
                "grid_index": g, **{str(d): (round(float(np.mean([r["scores"][m][g]["recall"] for r in tr
                                                                    if r["d"] == d])), 3) if g is not None else 0.0)
                                     for d in (2, 5, 10)}}
        for v in variants:
            t = compare(tr, v, REF, cap, W, test=test)
            out["vs_ref"].setdefault(key, {})[v] = t
            pv[(key, v)] = S.p_of(t)
    adj = S.holm({f"{k}|{v}": p for (k, v), p in pv.items()})
    for (k, v) in pv:
        out["vs_ref"][k][v]["p_holm"] = adj[f"{k}|{v}"]
    # every variant (and the reference) against the better baseline of each tier, Holm over all of them
    pb = {}
    for p, c in TIERS9:
        tr = [r for r in rows if r["placement"] == p and r["contrast"] == c]
        key = f"{p}/{c}"
        W = S.boot_weights(tr, B=2000, seed=0)
        fb = {b: out["frontier"][key][b] for b in BASELINES}
        b = max(fb, key=fb.get)
        for v in variants + [REF]:
            t = compare(tr, v, b, cap, W, test=test)
            t["baseline"] = b
            out["vs_base"].setdefault(key, {})[v] = t
            pb[(key, v)] = S.p_of(t)
    adj = S.holm({f"{k}|{v}": p for (k, v), p in pb.items()})
    for (k, v) in pb:
        out["vs_base"][k][v]["p_holm"] = adj[f"{k}|{v}"]
    out["n_tests_holm_vs_ref"] = len(pv)
    out["n_tests_holm_vs_base"] = len(pb)
    out["mean_frontier"] = {m: round(float(np.mean([out["frontier"][k][m] for k in out["frontier"]])), 3)
                            for m in methods}
    return out


def error_analysis(rows, err, cap=0.25, methods=("tail_full_A6", "tail_full", "tail_tilt", "tail_tiltp",
                                                 "tail_half", "lof+group", "gapclust_port_k30", "oracle")):
    """touching/low: per group hit at each method's pooled frontier point, joined with the group features."""
    rows = add_ref(rows)
    tr = [r for r in rows if r["placement"] == "touching" and r["contrast"] == "low"]
    gi = {m: op_point(tr, m, cap) for m in methods}
    feat = {(e["d"], e["seed"]): e for e in err["rows"]}
    recs = []
    for r in tr:
        e = feat.get((r["d"], r["seed"]))
        if e is None:
            continue
        for g, ge in enumerate(e["groups"]):
            h = {m: (r["scores"][m][gi[m]]["hits"][g] if gi[m] is not None else 0) for m in methods}
            thr05 = e["log_n_tests"] - np.log(0.05)
            recs.append({"d": r["d"], "seed": r["seed"], "m": ge["m"], "mode": ge["mode"], "n_match": ge["n_match"],
                         "margin_full_a005": round(ge["best"]["full"] - thr05, 2) if ge["n_match"] else None,
                         "best": ge["best"], "dd": ge["dd"], "hits": h})
    summ = {}
    for d in (2, 5, 10):
        R = [x for x in recs if x["d"] == d]
        if not R:
            continue
        s = {"groups": len(R), "mode_share": round(np.mean([x["mode"] for x in R]), 3),
             "no_matching_ball": round(np.mean([x["n_match"] == 0 for x in R]), 3)}
        for m in methods:
            s[f"hit_{m}"] = round(np.mean([x["hits"][m] for x in R]), 3)
        for b in ("lof+group", "gapclust_port_k30"):
            sel = [x for x in R if x["hits"][b] and not x["hits"]["tail_full_A6"]]
            s[f"{b}_only_vs_ref"] = len(sel)
            s[f"{b}_only_mode_share"] = round(np.mean([x["mode"] for x in sel]), 3) if sel else None
            s[f"{b}_only_no_ball"] = round(np.mean([x["n_match"] == 0 for x in sel]), 3) if sel else None
            mg = [x["margin_full_a005"] for x in sel if x["margin_full_a005"] is not None]
            s[f"{b}_only_margin_median"] = round(float(np.median(mg)), 2) if mg else None
            s[f"{b}_only_m_median"] = float(np.median([x["m"] for x in sel])) if sel else None
        md = [x for x in R if x["mode"]]
        s["mode_groups_hit"] = {m: round(np.mean([x["hits"][m] for x in md]), 3) for m in methods} if md else None
        summ[str(d)] = s
    return {"op_points": gi, "summary": summ, "groups": recs}


def nulls(rows):
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        for d in (2, 5, 10):
            R = [r for r in rows if r["kind"] == kind and r["d"] == d]
            if not R:
                continue
            cell = {"scenes": len(R)}
            for v in R[0]["groups"]:
                for j, a in enumerate(NULL_ALPHAS):
                    cell[f"{v}|{a}"] = round(float(np.mean([r["groups"][v][j] > 0 for r in R])), 3)
                    cell[f"{v}|{a}|mean_groups"] = round(float(np.mean([r["groups"][v][j] for r in R])), 3)
            out[f"{kind}|d{d}"] = cell
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("bench", "nulls"))
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--err")
    a = ap.parse_args(argv)
    rows = json.load(open(a.inp))["rows"]
    if a.cmd == "bench":
        res = {"cap0.25": bench(rows, 0.25), "cap0.5": bench(rows, 0.5), "alphas": list(ALPHAS)}
        if a.err:
            res["error_analysis"] = error_analysis(rows, json.load(open(a.err)))
    else:
        res = nulls(rows)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
