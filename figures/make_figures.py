"""Re-plot existing results for the paper. No new experiments; reads stored JSONs only."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Resolve relative to this file; the old absolute path was another machine's checkout and broke the rebuild.
NAT = Path(__file__).resolve().parents[1] / "results"          # stored result files shipped with this deposit
OUT = Path(__file__).resolve().parent / "regenerated"            # never overwrites figures/paper/
OUT.mkdir(exist_ok=True)


# --- the frozen audit rule (2026-11-14): significance comes from the paired bootstrap, not the sign-flip test -----------
# The stored stage-9/stage-11 JSONs carry "p_holm" from the re-selection sign-flip test, which audit_stats.py
# documents as invalid (exchangeability of the two methods' grid outputs is not "equal frontiers", and the
# pooled frontier mixes grid points by index). The recomputed values live in audit_recheck.json.
# Fill states and labels below must come from there, or the figure disagrees with the text.
def boot_tests(family):
    """{tier: {"diff", "p_holm", "diff_ci"}} for one recheck family, from paired_boot. Fails loudly."""
    recs = [r for r in json.load(open(str(NAT) + "/audit_recheck.json"))["records"]
            if r["family"] == family]
    if not recs:
        raise SystemExit("the frozen audit rule: no recheck records for family " + family)
    out = {r["tier"]: {"diff": r["old_diff"], "p_holm": r["new_holm"], "diff_ci": r["ci_basic"]} for r in recs}
    tiers = [p + "/" + c for p in ("isolated", "touching", "inside") for c in ("high", "mid", "low")]
    missing = [t for t in tiers if t not in out]
    if missing:
        raise SystemExit("the frozen audit rule: family %s is missing tiers %s" % (family, missing))
    return out

# Okabe-Ito (colour-blind safe), fixed order
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
         sky="#56B4E9", purple="#CC79A7", grey="#7f7f7f", black="#000000")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
# Figure 1 is made by skl_compare.py (see REPRODUCE.txt); the copy used in the paper is figures/paper/fig_opening_sklearn.png.

# ---------------------------------------------------------------- Figure 2: boundary
b = json.load(open(NAT / "scale_boundary.json"))
rgt = json.load(open(NAT / "rgt_predicted_vs_actual.json"))

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.45))
ax = axes[0]
cols = [C["sky"], C["blue"], C["purple"], C["verm"]]
for col, r in zip(cols, [r for r in b["rows"] if r["d"] == 5 and r["m"] == 3]):
    g = np.array(r["grid"]) / r["rho_star"]
    ax.plot(g, r["detect"], color=col, marker="o", ms=2.5, lw=1.0,
            label=f"N = $10^{int(round(np.log10(r['N'])))}$")
ax.axvline(1.0, color=C["grey"], ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_xlim(0.25, 4)
ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_xlabel(r"group radius / boundary $\rho^*$")
ax.set_ylabel("share of planted groups detected")
ax.set_title("(a) flip at $\\rho^*$ ($d$=5, $m$=3)", loc="left")
ax.legend(frameon=False, loc="lower left", fontsize=6.8)

ax = axes[1]
series = [((2, 3), C["blue"], "o"), ((5, 3), C["orange"], "s"), ((5, 5), C["green"], "^")]
for (d, m), col, mk in series:
    rows = [r for r in b["rows"] if r["d"] == d and r["m"] == m]
    N = np.array([r["N"] for r in rows])
    ax.plot(N, [r["rho_star"] for r in rows], color=col, ls="--", lw=1.0)
    ax.plot(N, [r["rho50"] for r in rows], color=col, ls="none", marker=mk, ms=4,
            label=f"$d$={d}, $m$={m}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("background points N")
ax.set_ylabel(r"radius (units of spacing)")
ax.set_title("(b) 50% radius vs N", loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.set_ylim(0.006, 12)
pass

ax = axes[2]
bins = rgt["calibration"]["margin"]["bins"]
labels = ["≤−50", "−50..−10", "−10..0", "0..10", "10..50", ">50"]
x = np.arange(len(bins))
for key, col, mk, lab in [("cts_v5", C["blue"], "o", "CTS (ours)"),
                          ("gapclust_k30", C["orange"], "s", "GapClust k30"),
                          ("lof_group", C["green"], "^", "LOF + group")]:
    ax.plot(x, [bb[key] for bb in bins], color=col, marker=mk, ms=3.5, lw=1.0, label=lab)
ax.axvline(2.5, color=C["grey"], ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_xlabel("boundary margin of a real group (nats)")
ax.set_ylabel("share found")
ax.set_title("(c) 339 real groups", loc="left")
ax.text(2.6, 0.05, "above\nboundary", fontsize=6.5, color=C["grey"])
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_boundary")

# ---------------------------------------------------------------- Figure 3: tiers with CIs
st = json.load(open(NAT / "pass2_frontier_stats.json"))
fr = st["caps"]["0.25"]["frontier"]
tests = st["caps"]["0.25"]["tests"]["v5_vs_best_published"]      # point estimates and the "b" arm name
boot = boot_tests("pass2|0.25|v5_vs_best_published")             # the frozen audit rule: significance only from here
names = {"hdbscan_mcs": "HDBSCAN", "gapclust_port_k30": "GapClust", "lof+group": "LOF+group"}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
methods = [("cts3_v5", "CTS, Bonferroni (default)", C["blue"], "o", True),
           ("cts3_v5_by", "CTS + BY", C["sky"], "D", True),
           ("best", "best published (per tier)", C["orange"], "s", True),
           ("oracle", "oracle local scan", C["grey"], "o", False)]
for ax, place in zip(axes, ["isolated", "touching", "inside"]):
    for j, contrast in enumerate(["high", "mid", "low"]):
        t = f"{place}/{contrast}"
        for i, (key, lab, col, mk, filled) in enumerate(methods):
            k = tests[t]["b"] if key == "best" else key
            v = fr[t][k]
            xx = j + (i - 1.5) * 0.17
            ax.errorbar(xx, v["value"], yerr=[[v["value"] - v["ci"][0]], [v["ci"][1] - v["value"]]],
                        fmt=mk, color=col, ms=4, lw=0.9, capsize=1.5,
                        mfc=col if filled else "white", label=lab if (j == 0 and place == "isolated") else None)
        tag = names[tests[t]["b"]]
        ax.text(j, -0.13, tag, ha="center", fontsize=6.2, color=C["orange"])
        sig = boot[t]["p_holm"] < 0.05
        mark = ("win" if boot[t]["diff"] > 0 else "loss") if sig else "tie"
        if t == "touching/low":
            # Holm 0.30 under the bootstrap because LOF's best point sits at the cap, but the basic CI
            # excludes zero and the loss is Holm < 0.001 against GapClust k30, whose operating point is
            # stable. Reported as a loss, as the text explains.
            mark = "loss"
        ax.text(j, 1.02, mark, ha="center", fontsize=6.8,
                color={"win": C["blue"], "tie": C["grey"], "loss": C["verm"]}[mark])
    ax.set_xticks(range(3)); ax.set_xticklabels(["high", "mid", "low"])
    ax.set_title(place, pad=12)
    ax.set_ylim(-0.18, 1.08)
    ax.axhline(0, color="#cccccc", lw=0.5)
axes[0].set_ylabel("recall at ≤ 0.25 false groups / 100 pts")
fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
save(fig, "fig_tiers")

# ---------------------------------------------------------------- Figure 4: Theorem 9.1 predicted vs measured
# TRANSCRIBED values (typed in, not read from a file here). Source: results/p1math_null.json, keys torus|sphere|gm /d2|d5|d10,
# measured_mean_rejections and predicted_mean_rejections at "0.001" for "true" and "twonn", divided by 22400 tests and x 1000
# (measured, predicted). All 18 cells agree with the file to within 0.005; run figures/check_fig4_cells.py to repeat the comparison.
cells = [
    ("torus", 2, "true", 0.91, 0.95), ("torus", 2, "TwoNN", 0.94, 0.90),
    ("torus", 5, "true", 0.33, 0.35), ("torus", 5, "TwoNN", 0.51, 0.51),
    ("torus", 10, "true", 0.06, 0.06), ("torus", 10, "TwoNN", 0.65, 0.60),
    ("sphere", 2, "true", 0.92, 1.00), ("sphere", 2, "TwoNN", 0.96, 1.01),
    ("sphere", 5, "true", 1.41, 1.39), ("sphere", 5, "TwoNN", 1.20, 1.15),
    ("sphere", 10, "true", 3.87, 3.82), ("sphere", 10, "TwoNN", 1.86, 1.72),
    ("mixture", 2, "true", 0.98, 0.94), ("mixture", 2, "TwoNN", 1.12, 1.03),
    ("mixture", 5, "true", 2.02, 1.94), ("mixture", 5, "TwoNN", 2.41, 2.19),
    ("mixture", 10, "true", 9.25, 9.80), ("mixture", 10, "TwoNN", 5.04, 5.17),
]
fig, ax = plt.subplots(figsize=(3.0, 2.6))
style = {"torus": (C["blue"], "o"), "sphere": (C["orange"], "s"), "mixture": (C["green"], "^")}
for geo, (col, mk) in style.items():
    pts = [(p, mm) for g, _, _, mm, p in cells if g == geo]
    ax.plot([p for p, _ in pts], [mm for _, mm in pts], ls="none", marker=mk, color=col, ms=4.5,
            label={"torus": "flat torus", "sphere": "round sphere", "mixture": "Gaussian mixture"}[geo])
lim = [0.04, 14]
ax.plot(lim, lim, color=C["grey"], lw=0.8, ls="--")
ax.axhline(1.0, color="#bbbbbb", lw=0.6, ls=":")
ax.text(0.05, 1.12, "nominal $10^{-3}$", fontsize=6.5, color=C["grey"])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"predicted share of $p\leq10^{-3}$ ($\times10^{-3}$)")
ax.set_ylabel(r"measured share ($\times10^{-3}$)")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_thm91")
print("done")
