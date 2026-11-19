"""Predicted vs actual recovery on real groups. usage: python analyse.py rows.json [rows2.json ...] OUT"""
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

files, out = sys.argv[1:-1], sys.argv[-1]
rows = [r for f in files for r in json.load(open(f))]
rows = [r for r in rows if "found" in r]
METHODS = ["cts_v5", "cts_v5_bh", "gapclust_k30", "lof_group", "hdbscan", "umap15_mcs3"]
FLOOR = -1000.0


def val(r, key):
    v = r[key]
    return FLOOR if v is None or (isinstance(v, float) and not np.isfinite(v)) or v == "-inf" else float(v)


def boot_auc(y, s, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    a = []
    for _ in range(B):
        i = rng.integers(len(y), size=len(y))
        if y[i].min() != y[i].max():
            a.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.quantile(a, q)), 3) for q in (0.025, 0.975)]


res = {"n_groups": len(rows), "by_domain": defaultdict(int)}
for r in rows:
    res["by_domain"][r["domain"]] += 1
any_found = np.array([any(r["found"][m] for m in METHODS if m in r["found"]) for r in rows])
scores = {"margin (theory, local d)": "margin", "margin, plug-in d (Beta-scan prediction)": "margin_plug",
          "margin, group-position d": "margin_grpd", "kNN purity (descriptor, not theory)": "knn_purity",
          "UMAP internal share (Result 3)": "umap_share"}
auc = {}
for nm, key in scores.items():
    s = np.array([val(r, key) for r in rows])
    auc[nm] = {}
    for m in METHODS + ["any"]:
        y = any_found if m == "any" else np.array([r["found"].get(m, False) for r in rows])
        if y.min() == y.max():
            continue
        auc[nm][m] = {"auc": round(float(roc_auc_score(y, s)), 3), "ci": boot_auc(y, s), "n_found": int(y.sum())}
res["auc"] = auc
# calibration at the fixed theory threshold 0
cal = {}
for key in ("margin", "margin_plug"):
    s = np.array([val(r, key) for r in rows])
    cal[key] = {}
    for stratum, mask in (("above (>0)", s > 0), ("below (<=0)", s <= 0)):
        c = {"n": int(mask.sum())}
        for m in METHODS + ["any"]:
            y = any_found if m == "any" else np.array([r["found"].get(m, False) for r in rows])
            c[m] = round(float(y[mask].mean()), 3) if mask.any() else None
        cal[key][stratum] = c
    # finer bins
    bins = [(-np.inf, -50), (-50, -10), (-10, 0), (0, 10), (10, 50), (50, np.inf)]
    cal[key]["bins"] = []
    for lo, hi in bins:
        mask = (s > lo) & (s <= hi)
        cal[key]["bins"].append({"bin": f"({lo},{hi}]", "n": int(mask.sum()),
                                 **{m: (round(float(np.mean([rows[i]["found"].get(m, False) for i in np.flatnonzero(mask)])), 3)
                                        if mask.any() else None) for m in METHODS},
                                 "any": round(float(any_found[mask].mean()), 3) if mask.any() else None})
res["calibration"] = cal
# per domain, at margin > 0
pdm = {}
for dom in sorted(set(r["domain"] for r in rows)):
    idx = [i for i, r in enumerate(rows) if r["domain"] == dom]
    s = np.array([val(rows[i], "margin") for i in idx])
    pdm[dom] = {"n": len(idx), "n_above": int((s > 0).sum())}
    for m in ("cts_v5", "gapclust_k30", "lof_group", "any"):
        y = np.array([any_found[i] if m == "any" else rows[i]["found"].get(m, False) for i in idx])
        pdm[dom][m] = {"found_above": f"{int(y[s > 0].sum())}/{int((s > 0).sum())}",
                       "found_below": f"{int(y[s <= 0].sum())}/{int((s <= 0).sum())}"}
res["per_domain"] = pdm
# method ranking within strata: paired exact sign test (McNemar) per pair
rank = {}
for key in ("margin", "margin_plug"):
    s = np.array([val(r, key) for r in rows])
    rank[key] = {}
    for stratum, mask in (("above", s > 0), ("below", s <= 0)):
        sub = [rows[i] for i in np.flatnonzero(mask)]
        tab = {m: round(float(np.mean([r["found"].get(m, False) for r in sub])), 3) for m in METHODS}
        pairs = {}
        for a, b in combinations(["cts_v5", "cts_v5_bh", "gapclust_k30", "lof_group", "umap15_mcs3"], 2):
            wa = sum(r["found"].get(a, False) and not r["found"].get(b, False) for r in sub)
            wb = sum(r["found"].get(b, False) and not r["found"].get(a, False) for r in sub)
            p = binomtest(wa, wa + wb).pvalue if wa + wb else 1.0
            pairs[f"{a} vs {b}"] = {"a_only": wa, "b_only": wb, "p": round(float(p), 4)}
        rank[key][stratum] = {"n": len(sub), "recall": tab, "pairs": pairs}
res["ranking"] = rank
# theory-component checks
mk = [r for r in rows if r["masked"]]
res["masking"] = {"n_masked": len(mk), "hdbscan_found_among_masked": int(sum(r["found"]["hdbscan"] for r in mk)),
                  "n_cond_true": int(sum(r["mask_cond"] for r in rows))}
s = np.array([val(r, "margin") for r in rows])
peak = np.array([r["has_peak"] for r in rows])
res["identifiability"] = {
    "has_peak_share": round(float(peak.mean()), 3),
    "found_any|peak": round(float(any_found[peak].mean()), 3) if peak.any() else None,
    "found_any|no_peak": round(float(any_found[~peak].mean()), 3) if (~peak).any() else None,
    "above_and_no_peak": int(((s > 0) & ~peak).sum()),
    "found_any|above,no_peak": round(float(any_found[(s > 0) & ~peak].mean()), 3) if ((s > 0) & ~peak).any() else None,
    "found_any|below,peak": round(float(any_found[(s <= 0) & peak].mean()), 3) if ((s <= 0) & peak).any() else None}
um = np.array([r["umap_share"] for r in rows]); fu = np.array([r["found"].get("umap15_mcs3", False) for r in rows])
res["umap_budget"] = {"share>=0.5": f"{int(fu[um >= 0.5].sum())}/{int((um >= 0.5).sum())}",
                      "share<0.5": f"{int(fu[um < 0.5].sum())}/{int((um < 0.5).sum())}",
                      "violations_of_bound": int(sum(r["umap_share"] > r["budget"] + 1e-9 for r in rows))}
# errors: found while predicted below, missed while predicted above
res["surprises"] = {
    "found_below": [(r["scene"], r["group"], r["m"], val(r, "margin"), [m for m in METHODS if r["found"].get(m)])
                    for r in rows if val(r, "margin") <= 0 and any(r["found"].get(m) for m in METHODS)],
    "missed_above": [(r["scene"], r["group"], r["m"], val(r, "margin"), r["knn_purity"], r["has_peak"])
                     for r in rows if val(r, "margin") > 0 and not any(r["found"].get(m) for m in METHODS)]}
print(json.dumps({k: v for k, v in res.items() if k != "surprises"}, indent=1))
print("found below:", len(res["surprises"]["found_below"]), "missed above:", len(res["surprises"]["missed_above"]))
json.dump(res, open(out, "w"), indent=1, default=str)
os.chmod(out, 0o600)
