"""
Leaderboard with uncertainty from per-scene scores (p3_perscene.py output).

For each tier and cap: frontier value + 95% bootstrap CI (scenes stratified by d; and a seed-block bootstrap as a
check), best method, exact paired permutation test best vs runner-up (Holm over the 9 tiers), and named claims from
stage 8 tested the same way. Also checks that the rerun reproduces the stage-8 pooled numbers.

  python -m clustering.eval.nature.p3_leaderboard results/p3_perscene_s500.json \
      results/p3_leaderboard_s500.json
"""
import json
import os
import sys

import numpy as np

from . import p3_stats as S

TIERS = [f"{p}/{c}" for p in ("isolated", "touching", "inside") for c in ("high", "mid", "low")]
PUBLISHED = ("mccatch_reimpl", "sccad_official", "gapclust_port", "gapclust_port_k30", "gapclust_official_sq",
             "gapclust_official_sq_k30", "lof+group", "hdbscan_mcs")     # hdbscan_mcs: strongest published on isolated
OURS = ("cts3_v5", "cts3_v5_bh", "cts3_v5_hybrid")
NOTES = {"mccatch_reimpl": "our reimplementation", "sccad_official": "authors' code, floor 10",
         "gapclust_port": "core port, Euclidean kNN, k=200 (stage-8 variant)",
         "gapclust_port_k30": "core port, Euclidean kNN, k=30 (stage-8 variant)",
         "gapclust_official_sq": "core port, squared kNN as official, k=200",
         "gapclust_official_sq_k30": "core port, squared kNN as official, k=30",
         "oracle": "true background density; reference, not a contestant"}
CLAIMS = [  # (name, A, B, tiers)
    ("v5_or_bh_vs_best_published", "best_ours", "best_published", TIERS),
    ("bh_beats_gapclust_k30_touching", "cts3_v5_bh", "gapclust_port_k30", ["touching/high", "touching/mid"]),
    ("gapclust_k30_beats_bh_touching_low", "gapclust_port_k30", "cts3_v5_bh", ["touching/low"]),
    ("bonferroni_vs_bh_isolated", "cts3_v5", "cts3_v5_bh", ["isolated/high", "isolated/mid", "isolated/low"]),
    ("hybrid_vs_bh_touching_low", "cts3_v5_hybrid", "cts3_v5_bh", ["touching/low"]),
    ("lof_vs_bh_touching_low", "lof+group", "cts3_v5_bh", ["touching/low"]),
]


def family(m):
    return "cts" if m.startswith("cts") else "gapclust" if m.startswith("gapclust") else m


def tier_rows(rows, t):
    p, c = t.split("/")
    return [r for r in rows if r["placement"] == p and r["contrast"] == c]


def analyse(data, B=2000, test="perm", B_boot=10000):
    """test="perm" (default, unchanged): p3_stats.paired_test. test="boot": p3_stats.paired_boot (stage-14 audit;
    the test the final rerun uses), B_boot resamples; results then carry the generic key "p"."""
    rows = data["rows"]
    methods = list(rows[0]["scores"])
    contestants = [m for m in methods if m != "oracle"]
    out = {"caps": {}}
    for cap in S.CAPS:
        res = {"methods": {m: {} for m in methods}, "paired": {}, "claims": {}}
        pv = {}
        for t in TIERS:
            rs = tier_rows(rows, t)
            W = S.boot_weights(rs, B, 0, "strat_d")
            Wb = S.boot_weights(rs, B, 1, "seed_block")
            for m in methods:
                v = S.frontier_ci(rs, m, cap, W)
                vb = S.frontier_ci(rs, m, cap, Wb)
                res["methods"][m][t] = {"recall": v["value"], "ci": v["ci"], "ci_seed_block": vb["ci"]}
            vals = {m: res["methods"][m][t]["recall"] for m in contestants}
            order = sorted(contestants, key=lambda m: -vals[m])
            best = order[0]
            ru = next(m for m in order[1:] if family(m) != family(best))   # runner-up from another method family
            pt = S.paired(rs, best, ru, cap, W, test=test, B_boot=B_boot)
            pt["fixed_op"] = S.paired_fixed(rs, best, ru, cap)
            pt.update(best=best, runner_up=ru, tie_at_top=[m for m in contestants if vals[m] == vals[best]])
            res["paired"][t] = pt
            pv[t] = S.p_of(pt)
        for t, p in S.holm(pv).items():
            res["paired"][t]["p_holm"] = p
        # named claims
        for name, a, b, tiers in CLAIMS:
            cl, cp = {}, {}
            for t in tiers:
                rs = tier_rows(rows, t)
                vals = {m: res["methods"][m][t]["recall"] for m in contestants}
                aa = max(OURS, key=lambda m: vals[m]) if a == "best_ours" else a
                bb = max(PUBLISHED, key=lambda m: vals[m]) if b == "best_published" else b
                if aa not in methods or bb not in methods:
                    continue
                W = S.boot_weights(rs, B, 0, "strat_d")
                r = S.paired(rs, aa, bb, cap, W, test=test, B_boot=B_boot)
                r.update(a=aa, b=bb, fixed_op=S.paired_fixed(rs, aa, bb, cap))
                cl[t] = r; cp[t] = S.p_of(r)
            for t, p in S.holm(cp).items():
                cl[t]["p_holm"] = p
            res["claims"][name] = cl
        out["caps"][str(cap)] = res
    return out


def reproduce(rows, ref_paths):
    """Max abs difference between pooled frontier from per-scene rows and the stage-8 JSONs."""
    diffs = {}
    for path in ref_paths:
        if not os.path.exists(path):
            continue
        ref = json.load(open(path))
        for t in TIERS:
            rs = tier_rows(rows, t)
            v = ref[t]
            items = v.items() if "recall@0.25" not in v else [("oracle", v)]
            for m, mv in items:
                if m not in rs[0]["scores"]:
                    continue
                R, F = S.arrays(rs, m)
                for cap in S.CAPS:
                    d = abs(round(S.frontier_from(R, F, cap), 3) - mv[f"recall@{cap}"])
                    diffs[m] = max(diffs.get(m, 0), d)
    return diffs


def to_release_json(res_cap, cap, seeds):
    return {"benchmark": "tinybench", "version": "1.0", "track": "synthetic/iso", "seeds": seeds, "cap": cap,
            "methods": {m: {t: {"recall": v["recall"], "ci": v["ci"]} for t, v in tv.items()}
                        for m, tv in res_cap["methods"].items()},
            "notes": NOTES,
            "paired": {t: {k: v for k, v in pt.items() if k in ("best", "runner_up", "diff", "p_perm", "p", "test", "p_holm", "diff_ci")}
                       for t, pt in res_cap["paired"].items()}}


def merge(paths):
    """Merge per-scene files with different method sets (same scenes)."""
    data = json.load(open(paths[0]))
    key = lambda r: (r["placement"], r["contrast"], r["d"], r["seed"])
    for p in paths[1:]:
        extra = {key(r): r for r in json.load(open(p))["rows"]}
        for r in data["rows"]:
            r["scores"].update(extra[key(r)]["scores"])
    return data


def main():
    srcs, dst = sys.argv[1].split(","), sys.argv[2]
    data = merge(srcs)
    out = analyse(data)
    nat = os.path.dirname(os.path.abspath(dst))
    out["reproduction_max_abs_diff"] = reproduce(data["rows"], [
        os.path.join(nat, "tiny_frontier_s500_cts3_v5_cts3_v5_bh_cts3_v5_hybrid.json"),
        os.path.join(nat, "tiny_official_s500.json"), os.path.join(nat, "ceiling_s500.json")])
    json.dump(out, open(dst, "w"), indent=1); os.chmod(dst, 0o600)
    rel = os.environ.get("P3_RELEASE_RESULTS")
    if rel:
        lo, hi = data["seeds"]
        for cap in S.CAPS:
            p = os.path.join(rel, f"leaderboard_v1.0_s{lo}_cap{cap}.json")
            json.dump(to_release_json(out["caps"][str(cap)], cap, f"{lo}-{hi}"), open(p, "w"), indent=1)
    for cap, res in out["caps"].items():
        print(f"\n== cap {cap}")
        for t in TIERS:
            pt = res["paired"][t]
            print(f"{t:14s} best {pt['best']:18s} {res['methods'][pt['best']][t]['recall']:.2f} "
                  f"{res['methods'][pt['best']][t]['ci']}  vs {pt['runner_up']:18s} "
                  f"{res['methods'][pt['runner_up']][t]['recall']:.2f}  diff {pt['diff']:+.2f} {pt.get('diff_ci')} "
                  f"p {S.p_of(pt):.4f} holm {pt['p_holm']:.4f} | fixed-op p {pt['fixed_op']['p_signflip']:.4f} "
                  f"w/t/l {pt['fixed_op']['wins_ties_losses']}")
        for name, cl in res["claims"].items():
            print(" ", name, {t: (v["a"], v["b"], v["diff"], S.p_of(v), v["p_holm"]) for t, v in cl.items()})
    print("reproduction", out["reproduction_max_abs_diff"])


if __name__ == "__main__":
    main()
