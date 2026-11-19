"""
Significance of the tiny-bench headline on 30 fresh seeds (pass2_frontier.py output), with p3_stats' tools.

Per tier and cap (0.25 and 0.5 false groups per 100 points):
- frontier + 95% bootstrap CI (stratified by d) for every method;
- headline test: CTS default (v5, Bonferroni) against the best published method on that tier (chosen on the same
  scenes, which favours the opponent), exact-in-distribution sign-flip test with re-selection of operating points
  (20,000 random flips; S = 90 scenes), Holm over the 9 tiers; paired bootstrap CI of the difference;
- the same for CTS + BY (calibrated FDR-type option) and CTS + BH (exploratory), and BY / BH against Bonferroni;
- a split-half replication check (seeds 800-814 against 815-829).

  python -m clustering.eval.nature.pass2_analysis OUT.json IN1.json,IN2.json,...
"""
import json
import os
import sys

import numpy as np

from . import p3_stats as S
from .p3_leaderboard import TIERS, tier_rows

PUBLISHED = ("gapclust_port_k30", "gapclust_official_sq_k30", "mccatch_reimpl", "lof+group", "hdbscan_mcs")
PUBLISHED_OFFICIAL = ("gapclust_official_sq_k30", "mccatch_reimpl", "lof+group", "hdbscan_mcs")
OURS = ("cts3_v5", "cts3_v5_by", "cts3_v5_bh")


def load(paths):
    rows, grids = [], None
    for p in paths:
        d = json.load(open(p))
        rows += d["rows"]
        grids = d["grids"]
    return rows, grids


def best_of(rs, methods, cap):
    vals = {m: S.frontier_from(*S.arrays(rs, m), cap) for m in methods}
    return max(vals, key=vals.get), vals


def compare(rows, a, b_set, cap, B=2000, n_perm=20000, test="perm", B_boot=10000):
    """a vs best of b_set on every tier; Holm over tiers. test="perm" (default, unchanged) or "boot"
    (p3_stats.paired_boot, stage-14 audit; generic key "p")."""
    res, pv = {}, {}
    for t in TIERS:
        rs = tier_rows(rows, t)
        W = S.boot_weights(rs, B, 0, "strat_d")
        b, vals = best_of(rs, b_set, cap) if isinstance(b_set, tuple) else (b_set, None)
        r = S.paired(rs, a, b, cap, W, test=test, n_perm=n_perm, B_boot=B_boot)
        r["fixed_op"] = S.paired_fixed(rs, a, b, cap)
        r.update(a=a, b=b)
        res[t] = r
        pv[t] = S.p_of(r)
    for t, p in S.holm(pv).items():
        res[t]["p_holm"] = p
    return res


def analyse(rows, B=2000, test="perm"):
    methods = list(rows[0]["scores"])
    out = {"n_scenes": len(rows), "seeds": sorted({r["seed"] for r in rows}), "caps": {}}
    for cap in S.CAPS:
        c = {"frontier": {}, "tests": {}}
        for t in TIERS:
            rs = tier_rows(rows, t)
            W = S.boot_weights(rs, B, 0, "strat_d")
            c["frontier"][t] = {m: S.frontier_ci(rs, m, cap, W) for m in methods}
        c["tests"]["v5_vs_best_published"] = compare(rows, "cts3_v5", PUBLISHED, cap, B, test=test)
        c["tests"]["v5_vs_best_published_official_forms"] = compare(rows, "cts3_v5", PUBLISHED_OFFICIAL, cap, B, test=test)
        c["tests"]["by_vs_best_published"] = compare(rows, "cts3_v5_by", PUBLISHED, cap, B, test=test)
        c["tests"]["bh_vs_best_published"] = compare(rows, "cts3_v5_bh", PUBLISHED, cap, B, test=test)
        c["tests"]["by_vs_v5"] = compare(rows, "cts3_v5_by", "cts3_v5", cap, B, test=test)
        c["tests"]["bh_vs_v5"] = compare(rows, "cts3_v5_bh", "cts3_v5", cap, B, test=test)
        c["tests"]["bh_vs_by"] = compare(rows, "cts3_v5_bh", "cts3_v5_by", cap, B, test=test)
        c["tests"]["v5_vs_gapclust_k30"] = compare(rows, "cts3_v5", "gapclust_port_k30", cap, B, test=test)
        c["tests"]["by_vs_gapclust_k30"] = compare(rows, "cts3_v5_by", "gapclust_port_k30", cap, B, test=test)
        c["tests"]["oracle_vs_v5"] = compare(rows, "oracle", "cts3_v5", cap, B, test=test)
        # split-half replication of the headline test
        half = {}
        for nm, lo, hi in (("800-814", 800, 814), ("815-829", 815, 829)):
            sub = [r for r in rows if lo <= r["seed"] <= hi]
            if not sub:
                continue
            half[nm] = {t: {k: v for k, v in r.items() if k in ("a", "b", "diff", "p_perm", "p", "p_holm")}
                        for t, r in compare(sub, "cts3_v5", PUBLISHED, cap, 500, 5000, test=test).items()}
        c["split_half"] = half
        out["caps"][str(cap)] = c
    return out


def show(out):
    for cap, c in out["caps"].items():
        print(f"\n===== cap {cap}  ({out['n_scenes']} scenes)")
        meth = list(next(iter(c["frontier"].values())))
        print(f"{'tier':14s} " + " ".join(f"{m[:12]:>12s}" for m in meth))
        for t in TIERS:
            print(f"{t:14s} " + " ".join(f"{c['frontier'][t][m]['value']:12.3f}" for m in meth))
        for name, res in c["tests"].items():
            print(f"-- {name}")
            for t in TIERS:
                r = res[t]
                print(f"  {t:14s} {r['a'][:12]:12s} vs {r['b'][:24]:24s} diff {r['diff']:+.3f} {r.get('diff_ci')} "
                      f"p {S.p_of(r):.4f} holm {r['p_holm']:.4f} | fixed p {r['fixed_op']['p_signflip']:.4f} "
                      f"w/t/l {r['fixed_op']['wins_ties_losses']}")
        for nm, h in c["split_half"].items():
            print(f"-- split {nm}: " + ", ".join(f"{t} {h[t]['diff']:+.2f} (holm {h[t]['p_holm']:.3f})" for t in TIERS))


def main():
    dst, srcs = sys.argv[1], sys.argv[2].split(",")
    rows, grids = load(srcs)
    out = analyse(rows)
    out["grids"] = grids
    json.dump(out, open(dst, "w"), indent=1)
    os.chmod(dst, 0o600)
    show(out)


if __name__ == "__main__":
    main()
