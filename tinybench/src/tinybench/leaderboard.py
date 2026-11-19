"""
Leaderboard table from a results JSON.

Results format (results/*.json):
{"benchmark": "tinybench", "version": "1.1", "seeds": "test v1.0 (500-504)", "track": "synthetic/iso",
 "cap": 0.25, "methods": {name: {"tier": {"recall": float, "ci": [lo, hi]}, ...}, ...},
 "notes": {name: "short provenance note"}, "paired": {"tier": {"best": name, "runner_up": name, "diff": float,
 "p_perm": float, "p_holm": float}}}

to_markdown(res) -> a Markdown table, one row per method, one column per tier, value [CI]; the best value per tier is
bold only when the paired test against the runner-up survives Holm correction at 0.05 (otherwise it is marked "~" =
tied within noise).
"""
import json

from .generator import CONTRASTS, TIERS

TIER_ORDER = [f"{p}/{c}" for p in TIERS for c in CONTRASTS]


def to_markdown(res, digits=2):
    tiers = [t for t in TIER_ORDER if any(t in v for v in res["methods"].values())]
    head = "| method | " + " | ".join(tiers) + " |\n|---|" + "---|" * len(tiers) + "\n"
    paired = res.get("paired", {})
    lines = []
    for m, v in res["methods"].items():
        cells = []
        for t in tiers:
            if t not in v:
                cells.append("–"); continue
            r = v[t]
            s = f"{r['recall']:.{digits}f}"
            if r.get("ci"):
                s += f" [{r['ci'][0]:.{digits}f}, {r['ci'][1]:.{digits}f}]"
            pt = paired.get(t)
            if pt and pt.get("best") == m:
                s = f"**{s}**" if pt.get("p_holm", 1) < 0.05 else f"{s} ~"
            cells.append(s)
        note = res.get("notes", {}).get(m, "")
        lines.append(f"| {m}{' (' + note + ')' if note else ''} | " + " | ".join(cells) + " |")
    title = (f"Recall at <= {res.get('cap')} false groups per 100 points, track {res.get('track')}, seeds "
             f"{res.get('seeds')}. 95% bootstrap CI over scenes (stratified by d). Bold = best and beats the runner-up "
             f"(exact paired permutation test, Holm over tiers, 0.05); ~ = best but not separable from the runner-up.\n\n")
    return title + head + "\n".join(lines) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    md = to_markdown(json.load(open(a.results)))
    if a.out:
        open(a.out, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
