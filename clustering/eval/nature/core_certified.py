"""
Stage 14 (core), item 1.3: CTS v5 over a certified base, as an opt-in `base="certified"` (a wrapper; no shared file
is edited).

cts_v5(X, base=...):
- base="hdbscan" (default, the shipped CTS v5): HDBSCAN 8/4 base, groups written on top. The base is not tested.
- base="certified": the base is GCLU's large-scale stage (gclu_core.gclu with tiny=False): a random half proposes
  clusters, the other half certifies every split with the exact split dip test (GC2), failed splits are merged, every
  point is assigned. Groups from the unchanged v5 scan are written on top. Settings = GCLU v2 (h_rule="both",
  sel="eom") or v1 ("frozen": h_rule="min", sel="leaf").
Guarantees: tiny-group FWER <= alpha (the v5 scan, unchanged) and, separately,
P(any false split of the base) <= alpha_base (GC2 + fallback); jointly <= alpha + alpha_base by a union bound.

Also: the no-split (same-data) certification and the selection-bias counterexample used to settle GCLU's open
question (Proposition 14.3).
Public data only.
"""
import numpy as np

from . import gclu_core as G
from . import p1math_scan as S
from . import tiny_methods as TM

VARIANT_KW = {"v2": {"h_rule": "both", "sel": "eom"}, "v1": {}}


def certified_base(X, alpha_base=0.05, seed=0, variant="v2", same_data=False, return_info=False):
    kw = dict(VARIANT_KW[variant])
    if same_data:
        kw["same_data"] = "full"
    lab, info = G.gclu(np.asarray(X, float), alpha=alpha_base, seed=seed, tiny=False, return_info=True, **kw)
    return (lab, info) if return_info else lab


def cts_v5(X, alpha=0.05, base="hdbscan", alpha_base=0.05, seed=0, variant="v2", return_info=False):
    """CTS v5 (cts3 with dim_est="local", scale_min=True, border=True) with a selectable base."""
    X = np.asarray(X, float)
    R, is_rep = S.reduced_set(X)
    info = {}
    if base == "hdbscan":
        b = TM.hdb(X)
    elif base == "certified":
        b, inf = certified_base(X, alpha_base, seed, variant, return_info=True)
        info = {"proposed": inf["proposed"], "large": inf["large"], "n_split_tests": len(inf["tests"]),
                "min_split_p": min([t["p"] for t in inf["tests"]] or [1.0])}
    else:
        raise ValueError(base)
    lab = TM._scan_select(X, R, is_rep, 15, 10, alpha, base=b, dim_est="local", scale_min=True)
    info["base_clusters"] = int(len(set(b.tolist()) - {-1}))
    info["tiny"] = int(lab.max() - b.max()) if lab.max() > b.max() else 0
    return (lab, info) if return_info else lab


# ---------------- Proposition 14.3: the no-split certification is not valid (selection bias) ----------------
def min_count_selection(n_pos, lam, rng, reps=4000, levels=(0.05, 0.01, 0.001)):
    """Same-data saddle selection in its simplest form (Proposition 14.3). A path crosses n_pos disjoint equal balls
    of a homogeneous Poisson process (mean count lam each, so "no dip" is true). The saddle is the interior ball with
    the smallest count; the two references are the balls with the largest count on each side of it (all read from
    the same data). p = max(p_a, p_b), the exact binomial dip p-value of gclu_core.dip_pvalue. Returns
    P(p <= level). A design with saddle and references fixed in advance gives <= level."""
    from scipy.stats import binom
    hits = np.zeros(len(levels))
    for _ in range(reps):
        c = rng.poisson(lam, n_pos)
        j = 1 + int(np.argmin(c[1:-1]))
        s, a, b = int(c[j]), int(c[:j].max()), int(c[j + 1:].max())
        pa = binom.cdf(s, s + a, 0.5) if s + a > 0 else 1.0
        pb = binom.cdf(s, s + b, 0.5) if s + b > 0 else 1.0
        hits += np.array([max(pa, pb) <= x for x in levels])
    return dict(zip(map(str, levels), (hits / reps).tolist()))
