"""
Stage 14 (audit): re-check every stored comparison that used `p3_stats.paired_test` (or `paired_fixed` through
`tail_analysis.compare`), from the stored per-scene results, with `audit_stats.paired_boot`. Holm families are the
original ones. Nothing is re-run.

  python -m clustering.eval.nature.audit_recheck OUT.json [--B 10000]

Sources (callers found by grep, 2026-11-10):
  p3     p3_leaderboard.analyse -> p3_leaderboard_s500.json (best vs runner-up per tier; named claims)
  shape  unshipped script p3/shape_analysis.py -> p3_shape_tracks_s700_analysis.json (winner per shape and tier, raw p)
  pass2  pass2_analysis.analyse -> pass2_frontier_stats.json (10 test families x 2 caps, Holm over 9 tiers)
  hdbvs  unshipped script hdbvs/e5_final.py -> hdbvs_final_s840.json (54 tests, Holm over all; grids cut at 0.3)
  flt    unshipped script flt/final_an.py -> flt_final_s880.json (36 tests, Holm over all)
  ring   unshipped script ring/final_an.py -> ring_final_s960.json (180 tests, Holm over all)
  tail   tail_analysis.bench -> tail_final_summary.json (vs_ref 108 and vs_base 117 per cap; p = paired_fixed)
Not callers (per-scene or per-group Wilcoxon / sign tests on a fixed quantity, no grids): gclu, lpb2, conf, prf, rep,
and one more family. p1exp stores pooled frontiers only and ran no paired test.
"""
import argparse
import json
import os

import numpy as np

from . import _tgpaths as TGP
from . import audit_stats as A
from . import p3_stats as S

NAT = os.path.join(TGP.ROOT, "results")
SCR = os.path.join(TGP.ROOT, "results")
TIERS = [f"{p}/{c}" for p in ("isolated", "touching", "inside") for c in ("high", "mid", "low")]


def _j(*p):
    return json.load(open(os.path.join(*p)))


def tier_rows(rows, t):
    p, c = t.split("/")
    return [r for r in rows if r["placement"] == p and r["contrast"] == c]


def verdict(diff, p, level=0.05):
    if p is None or p >= level:
        return "n.s."
    return "A>B" if diff > 0 else "A<B"


class Recheck:
    def __init__(self, B=10000):
        self.B = B
        self._W = {}

    def W(self, key, rs):
        if key not in self._W:
            self._W[key] = S.boot_weights(rs, self.B, 14, "strat_d")
        return self._W[key]

    def one(self, src, fam, t, cap, rows, a, b, old_diff, old_p, wkey):
        rs = tier_rows(rows, t)
        nb = A.paired_boot(rs, a, b, cap, W=self.W((wkey, t), rs))
        an = A.old_test_anatomy(rs, a, b, cap, n_perm=1000)
        cf = A.crossfit_test(rs, a, b, cap, n_perm=4000)
        return {"src": src, "family": fam, "tier": t, "cap": cap, "a": a, "b": b,
                "old_diff": old_diff, "old_p": old_p, "new_diff": nb["diff"], "new_p": nb["p_boot"],
                "p_basic": nb["p_basic"], "p_pct": nb["p_pct"], "ci_basic": nb["ci_basic"], "ci_pct": nb["ci_pct"], "optimism": nb["optimism"],
                "p_crossfit": cf["p_crossfit"], "grid_len": an["grid_len"], "nested": an["nested_prefix"],
                "delta_extra": an["delta_extra"], "null_zero_share": an["null_zero_share"], "op": nb["op"]}


def holm_fill(recs, fam_key, old_holm=None):
    fams = {}
    for i, r in enumerate(recs):
        fams.setdefault(r[fam_key], []).append(i)
    for idx in fams.values():
        adj = S.holm({i: recs[i]["new_p"] for i in idx})
        for i in idx:
            recs[i]["new_holm"] = adj[i]
    for r in recs:
        r["old_verdict"] = verdict(r["old_diff"], r.get("old_holm", r["old_p"]))
        r["new_verdict"] = verdict(r["new_diff"], r["new_holm"])
        r["changed"] = r["old_verdict"] != r["new_verdict"]
    return recs


# ------------------------------------------------------------------------------------------------------- sources
def src_p3(R):
    data = _j(NAT, "p3_perscene_s500.json")
    extra = {(r["placement"], r["contrast"], r["d"], r["seed"]): r for r in _j(NAT, "p3_perscene_s500_gapsq.json")["rows"]}
    rows = data["rows"]
    for r in rows:
        r["scores"].update(extra[(r["placement"], r["contrast"], r["d"], r["seed"])]["scores"])
    old = _j(NAT, "p3_leaderboard_s500.json")
    recs = []
    for cap_s, res in old["caps"].items():
        cap = float(cap_s)
        for t, pt in res["paired"].items():
            x = R.one("p3", f"p3|{cap_s}|best_vs_runner_up", t, cap, rows, pt["best"], pt["runner_up"], pt["diff"],
                      pt["p_perm"], "p3")
            x["old_holm"] = pt["p_holm"]; recs.append(x)
        for name, cl in res["claims"].items():
            for t, pt in cl.items():
                x = R.one("p3", f"p3|{cap_s}|{name}", t, cap, rows, pt["a"], pt["b"], pt["diff"], pt["p_perm"], "p3")
                x["old_holm"] = pt["p_holm"]; recs.append(x)
    return holm_fill(recs, "family")


def src_shape(R):
    rows = _j(NAT, "p3_shape_tracks_s700.json")["rows"]
    old = _j(NAT, "p3_shape_tracks_s700_analysis.json")
    recs = []
    for sh, tv in old["winner"].items():
        rs_sh = [r for r in rows if r["shape"] == sh]
        for t, (best, other, diff, p) in tv.items():
            x = R.one("shape", f"shape|{sh}|{t}", t, 0.25, rs_sh, best, other, diff, p, f"shape{sh}")
            recs.append(x)
    recs = holm_fill(recs, "family")          # one test per family: Holm = raw, as the original (raw p)
    return recs


def src_pass2(R):
    rows = []
    for s in (800, 810, 820):
        rows += _j(NAT, f"pass2_perscene_s{s}.json")["rows"]
    old = _j(NAT, "pass2_frontier_stats.json")
    recs = []
    for cap_s, c in old["caps"].items():
        for name, res in c["tests"].items():
            for t in TIERS:
                pt = res[t]
                x = R.one("pass2", f"pass2|{cap_s}|{name}", t, float(cap_s), rows, pt["a"], pt["b"], pt["diff"],
                          pt["p_perm"], "pass2")
                x["old_holm"] = pt["p_holm"]; x["old_fixed_p"] = pt["fixed_op"]["p_signflip"]; recs.append(x)
    return holm_fill(recs, "family")


def _cut(rows, grids, maxalpha=0.3, keep=("hdbscan_mcs",)):
    """hdbvs/an.load: alpha grids restricted to <= 0.3 (the frozen design of e5_final)."""
    for r in rows:
        for k, v in list(r["scores"].items()):
            if k in grids and isinstance(grids[k][0], float) and k not in keep:
                r["scores"][k] = [s for a, s in zip(grids[k], v) if a <= maxalpha + 1e-12]
    return rows


def _flat(src, path_rows, path_old, cut=False, cap=0.25):
    D = _j(*path_rows)
    rows = _cut(D["rows"], D["grids"]) if cut else D["rows"]
    old = _j(*path_old)
    recs = []
    for k, pt in old["tests"].items():
        t, a, b = k.split("|")
        x = R_.one(src, src, t, cap, rows, a, b, pt["diff"], pt["p_perm"], src)
        x["old_holm"] = pt["p_holm"]; x["old_fixed_p"] = pt.get("fixed", {}).get("p_signflip"); recs.append(x)
    return holm_fill(recs, "family")


def src_tail(R):
    from .tail_analysis import add_ref
    rows = add_ref(_j(NAT, "tail_scenes_final_s1040.json")["rows"])
    old = _j(NAT, "tail_final_summary.json")
    recs = []
    for ck in ("cap0.25", "cap0.5"):
        res = old[ck]
        cap = res["cap"]
        for fam in ("vs_ref", "vs_base"):
            for t, vv in res[fam].items():
                for v, pt in vv.items():
                    b = pt.get("baseline", "tail_full_A6")
                    x = R.one("tail", f"tail|{ck}|{fam}", t, cap, rows, v, b, pt["frontier_diff"], pt["p_perm"],
                              "tail")
                    x["old_holm"] = pt["p_holm"]; recs.append(x)
    return holm_fill(recs, "family")


R_ = None


def main(argv=None):
    global R_
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--only", default="p3,shape,pass2,hdbvs,flt,ring,tail")
    a = ap.parse_args(argv)
    R_ = Recheck(a.B)
    out = {"B": a.B, "test": "audit_stats.paired_boot: p = max(basic, percentile) of a d-stratified bootstrap with re-selection", "records": []}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    done = {r["src"] for r in out["records"]}
    for s in a.only.split(","):
        if s in done:
            continue
        if s == "p3":
            recs = src_p3(R_)
        elif s == "shape":
            recs = src_shape(R_)
        elif s == "pass2":
            recs = src_pass2(R_)
        elif s == "hdbvs":
            recs = _flat("hdbvs", (NAT, "hdbvs_variants_s840.json"), (NAT, "hdbvs_final_s840.json"), cut=True)
        elif s == "flt":
            recs = _flat("flt", (NAT, "flt_scenes_final_s880.json"), (NAT, "flt_final_s880.json"))
        elif s == "ring":
            recs = _flat("ring", (SCR, "ring", "final_s960.json"), (NAT, "ring_final_s960.json"))
        elif s == "tail":
            recs = src_tail(R_)
        out["records"] += recs
        tmp = a.out + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1, default=float)
        os.replace(tmp, a.out)
        os.chmod(a.out, 0o600)
        ch = sum(r["changed"] for r in recs)
        print(f"{s}: {len(recs)} comparisons, {ch} verdicts changed", flush=True)


if __name__ == "__main__":
    main()
