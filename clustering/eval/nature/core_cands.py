"""
Stage 14 (core): the v5 candidate scan with the outer-annulus radii kept, so that every d rule of stage 14 (v5, LCB-d,
the curvature-corrected CC rule, gated rules) can be applied to the same candidates afterwards.

`scan_table(Z)` reproduces p1math_scan.v5_candidates (CTS v5: dim_est="local", scale_min=True, border=True) and
returns, per tested (i, m): the v5 exponent d_v5, the radii r_(m-1), r_n and r_(n+L) for L in LS, and the members.
Only rows that some rule could still reject are kept: every rule here lowers d below d_v5, which lowers the score, so
rows with a v5 score below KEEP_SCORE can never be selected (KEEP_SCORE is far below every threshold used).
`apply_rule` turns the table into (score, members) candidates for a d rule; `select` reuses p1math_scan / prf_theory.
Public synthetic data only.
"""
import math

import numpy as np
from scipy.special import betaincinv
from scipy.stats import beta as beta_dist, chi2
from sklearn.neighbors import NearestNeighbors

from . import core_theory as C
from . import p1math_scan as S
from . import prf_theory as P
from . import tiny_methods as TM

LS = (10, 20, 40, 80, 160)
KEEP_SCORE = 8.0           # -log p; the loosest threshold used (trunc16 at alpha 0.1, n_tests ~ 2e4) is about 9.9


def scan_table(Z, max_m=15, k_bg=10, q=0.5):
    R, is_rep = S.reduced_set(Z)
    Zr = Z[R]
    rank = int(np.linalg.matrix_rank(Zr - Zr.mean(0), tol=1e-8 * np.abs(Zr).max()))
    dim0 = min(max(1.0, TM.twonn_dim(Zr)), rank)
    n_loc = 30
    kk0 = min(len(R) - 1, max_m + k_bg + n_loc)
    kk = min(len(R) - 1, max_m + k_bg + max(LS))
    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Zr).kneighbors(Zr)
    mu_all = np.log(np.maximum(dist[:, 2], 1e-12) / np.maximum(dist[:, 1], 1e-12))
    lo_q = chi2.ppf(0.001, 2 * n_loc) / 2
    dims = TM._self_dims(dist[:, :kk0 + 1], is_rep, max_m, k_bg, kk0, q)
    rows, mems, n_tests = [], [], 0
    for i in range(len(R)):
        if is_rep[i]:
            continue
        for m in range(2, max_m + 1):
            if m - 1 + k_bg > kk0:
                break
            members = idx[i, :m]
            if is_rep[members].any():
                break
            n_tests += 1
            n = m - 1 + k_bg
            r, Rk = dist[i, m - 1], dist[i, n]
            if Rk <= r or r <= 0:
                continue
            dd = min(dim0, dims[m])
            ring = idx[i, m:m + n_loc]
            dd = min(max(dd, lo_q / max(mu_all[ring].sum(), 1e-12)), rank)
            score = -beta_dist.logcdf(min((r / Rk) ** dd, 1.0), m - 1, k_bg)
            if score < KEEP_SCORE:
                continue
            outer = [dist[i, min(n + L, kk)] for L in LS]
            rows.append([i, m, dd, r, Rk, *outer, score, min(n + LS[-1], kk) - n])
            mems.append(tuple(sorted(int(x) for x in R[members])))
    cols = ["i", "m", "d_v5", "r", "rn"] + [f"r{L}" for L in LS] + ["score_v5", "Lmax"]
    tab = {c: np.array([row[j] for row in rows], float) for j, c in enumerate(cols)}
    return {"tab": tab, "mems": mems, "n_tests": n_tests, "dim0": dim0, "rank": rank, "n_R": len(R)}


def _score(r, rn, d, m, k):
    u = np.minimum((r / rn) ** d, 1.0)
    return -beta_dist.logcdf(u, m - 1, k)


def dlo_of(tab, L, delta, k_bg=10):
    """Outer-slope lower bound D_lo = log b / log(r_n / r_(n+L)), b = (1 - delta)-quantile of Beta(n, L)."""
    m = tab["m"]
    n = m - 1 + k_bg
    Lm = np.minimum(L, tab["Lmax"])
    b = betaincinv(n, Lm, 1 - delta)
    rL = tab[f"r{L}"]
    lam = rL / tab["rn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dlo = np.where(lam > 1, np.log(b) / np.log(1 / lam), np.inf)
    return dlo, lam


def rule_dhat(tab, rule, n_tests, k_bg=10, dim0=None, tmax_auto=None):
    """d_hat per row for a rule spec: ("v5",), ("lcb", L, delta[, "dim0"]), ("cc", L, delta, tmax),
    ("gate", L, delta, tmax, gamma): CC unless the outer slope's upper bound is below gamma * d0 (saturation), then LCB;
    also "gate2", "adapt", and (stage 15) ("ccmin", Ls, delta), ("ccmin_lg", Ls, delta, Lg, gamma).
    tmax_auto (stage 15, cost branch): the value that every "auto" tmax takes (scalar or one per row); None keeps
    tmax_default(n_tests). Theorem 14.1 needs tmax >= the level t the selection uses; cts3 passes
    max(tmax_default, alpha / n_sel) by default (see tiny_methods.TMAX_MODES)."""
    d0 = tab["d_v5"].copy()
    # recursive calls pass tmax_auto only when it is set, so wrappers with the old signature keep working
    _rd = (lambda r: rule_dhat(tab, r, n_tests, k_bg, dim0)) if tmax_auto is None else \
        (lambda r: rule_dhat(tab, r, n_tests, k_bg, dim0, tmax_auto=tmax_auto))
    kind = rule[0]
    if kind == "v5":
        return d0
    if kind == "lcb":
        L, delta = rule[1], rule[2]
        if len(rule) > 3 and rule[3] == "dim0":
            d0 = np.full_like(d0, dim0)
        dlo, _ = dlo_of(tab, L, delta, k_bg)
        return np.maximum(1e-3, np.minimum(d0, dlo))
    if kind in ("cc", "gate"):
        L, delta, tmax = rule[1], rule[2], rule[3]
        if tmax == "auto":
            tmax = tmax_default(n_tests) if tmax_auto is None else tmax_auto
        dlo, lam = dlo_of(tab, L, delta, k_bg)
        m = tab["m"].astype(int)
        u = np.array([C.u_level(mm, k_bg, tmax) for mm in m])
        kap = C.a_in(u ** (1.0 / d0)) / C.a_out(np.maximum(lam, 1 + 1e-12))
        dh = C.cc_dhat(d0, np.where(np.isfinite(dlo), dlo, d0), kap)
        if kind == "gate":
            gamma = rule[4]
            dhi, _ = dlo_of(tab, L, 1 - delta, k_bg)          # upper bound on the outer slope
            sat = dhi < gamma * d0
            dh = np.where(sat, np.minimum(d0, dlo), dh)
        return np.maximum(1e-3, dh)
    if kind == "gate2":
        # ("gate2", Lc, delta_c, Lg, q_g, gamma): CC with (Lc, delta_c), except where the gate statistic
        # D_g = log b_g / log(r_n / r_(n+Lg)), b_g the q_g-quantile of Beta(n, Lg), is below gamma * d0; there LCB-d
        # with (Lg, 0.01). q_g = 0.99 gives the lower bound D_lo, q_g = 0.01 the upper bound.
        Lc, dc, Lg, qg, gamma = rule[1:6]
        dh = _rd(("cc", Lc, dc, "auto"))
        dg, _ = dlo_of(tab, Lg, 1 - qg, k_bg)
        dl, _ = dlo_of(tab, Lg, 0.01, k_bg)
        sat = dg < gamma * d0
        return np.maximum(1e-3, np.where(sat, np.minimum(d0, dl), dh))
    if kind == "adapt":
        # ("adapt", gamma, Ls, qg) (post-hoc, after the fresh-seed error analysis): CC from the widest annulus
        # L in Ls whose slope upper bound (qg-quantile) is >= gamma * d0, i.e. the widest window not seen to
        # saturate; if every annulus looks saturated, LCB-d on the narrowest one.
        gamma, Ls, qg = rule[1], rule[2], rule[3]
        Ls = sorted(Ls)
        dh = np.minimum(d0, dlo_of(tab, Ls[0], 0.01, k_bg)[0])
        for L in Ls:                                   # widest passing annulus wins (later overwrites)
            dg, _ = dlo_of(tab, L, 1 - qg, k_bg)
            ok = dg >= gamma * d0
            cc = _rd(("cc", L, 0.01, "auto"))
            dh = np.where(ok, cc, dh)
        return np.maximum(1e-3, dh)
    if kind == "ccmin":
        # ("ccmin", Ls, delta) (stage 15 leak): the smallest CC exponent over the annuli L in Ls. Valid whenever (C2')
        # holds on at least one of them (Theorem 15.1). Same values as leak_theory.rule_dhat with q = 2.
        Ls, delta = rule[1], rule[2]
        dh = None
        for L in Ls:
            x = _rd(("cc", L, delta, "auto"))
            dh = x if dh is None else np.minimum(dh, x)
        return dh
    if kind == "ccmin_lg":
        # ("ccmin_lg", Ls, delta, Lg, gamma) (stage 15 leak, frozen as ("ccmin_lg", (20, 40, 80, 160), 0.01, 20, 0.3)):
        # ccmin, further lowered to LCB-d on the annulus Lg wherever that annulus is seen to saturate (99% upper slope
        # bound < gamma d0). Lowering d_hat is always safe, so the validity class is that of ccmin. Valid whenever
        # (C2') holds on at least one annulus (Thm 15.1), eta <= 0.18. Exact rate <= 0.81 t on Gaussian clusters of
        # >= 60 points at d = 2/5/10. Not covered for clusters of about n + 20 points or fewer (Props 14.4, 15.5).
        # Reference: leak_theory.rule_dhat.
        Ls, delta, Lg, gamma = rule[1:5]
        dh = _rd(("ccmin", Ls, delta))
        dhi, _ = dlo_of(tab, Lg, 0.99, k_bg)
        lcb = np.minimum(d0, dlo_of(tab, Lg, 0.01, k_bg)[0])
        return np.maximum(1e-3, np.where(dhi < gamma * d0, np.minimum(dh, lcb), dh))
    raise ValueError(rule)


def candidates(table, rule, k_bg=10):
    tab = table["tab"]
    if not len(tab.get("m", [])):
        return []
    dh = rule_dhat(tab, rule, table["n_tests"], k_bg, table["dim0"])
    sc = _score(tab["r"], tab["rn"], dh, tab["m"], k_bg)
    return [(float(s), mem) for s, mem in zip(sc, table["mems"])]


def select(cands, n_tests, alpha, how):
    """how = "bonferroni" (greedy over the Bonferroni line) or "g_trunc16" (Theorem 13.1's group step-up)."""
    if how == "bonferroni":
        return S.select(cands, n_tests, alpha, "bonferroni")[0]
    return P.group_stepup_any(cands, n_tests, alpha, ("trunc", 16))[0]


def tmax_default(n_tests, alpha=0.1):
    return alpha * 16 / P.harmonic(16) / max(1, n_tests)
