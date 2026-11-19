"""
Stage 15 (leak): exact false-alarm rates of any outer-annulus d rule, and the rules tried for items 1.8 / 1.9.

Rate engine (Rao-Blackwellised). At a sample point x with N - 1 other iid points of ball-mass function P, the masses of
the order statistics are q_(j) = G_j / G_N (G_j partial sums of iid Exp(1)). By Lemma 14.0, given the outer statistics
O = (r_n, r_(n+10), ..., r_(n+160)), P(p <= t | O) = F(P(c rho) / P(rho)), rho = r_n, c = u_t^(1 / d_hat(O)), F the
Beta(m - 1, k) CDF. So E over O (sampled exactly from Gamma increments) gives the exact rate with Monte Carlo error only
in O; every rule is evaluated by the shared code (core_cands.rule_dhat), so the theory and the experiments use one
implementation.

Rules added here (all functions of O only, so Lemma 14.0 applies):
- ("ccq", L, delta, tmax, q): the CC rule with the s^2 profile replaced by s^q (0 < q <= 2). kappa_q = A_q(c0) /
  A_q,out(lambda), A_q(c) = (1 - c^q) / (q log 1/c), A_q,out(l) = (l^q - 1) / (q log l). q = 2 is Theorem 14.1;
  q -> 0 gives kappa -> 1, i.e. LCB-d. Valid under (C2'_q): the slope deficit per unit of s^q, read on the outer
  window, is at least the inner one (sufficient: (d0 - D(s)) / s^q nondecreasing).
- ("adaptq", gamma, Ls, qg, q): the adaptive annulus of stage 14 with ccq in place of cc.
Public synthetic data only.
"""
import math

import numpy as np
from scipy.special import betainc, betaincinv

from . import core_cands as CC
from . import core_theory as CT

LS = (10, 20, 40, 80, 160)


def a_q_in(c, q):
    c = np.asarray(c, float)
    return np.where(c < 1 - 1e-9, (1 - c ** q) / (q * np.log(1 / np.maximum(c, 1e-300))), 1.0)


def a_q_out(lam, q):
    lam = np.asarray(lam, float)
    return np.where(lam > 1 + 1e-9, (lam ** q - 1) / (q * np.log(np.maximum(lam, 1 + 1e-12))), 1.0)


def rule_dhat(tab, rule, n_tests, k_bg=10, dim0=None):
    """core_cands.rule_dhat plus the ccq / adaptq rules."""
    kind = rule[0]
    if kind == "ccq":
        L, delta, tmax, q = rule[1:5]
        if tmax == "auto":
            tmax = CC.tmax_default(n_tests)
        d0 = tab["d_v5"]
        dlo, lam = CC.dlo_of(tab, L, delta, k_bg)
        m = tab["m"].astype(int)
        um = {mm: CT.u_level(mm, k_bg, tmax) for mm in np.unique(m)}
        u = np.array([um[mm] for mm in m])
        kap = a_q_in(u ** (1.0 / d0), q) / a_q_out(np.maximum(lam, 1 + 1e-12), q)
        return np.maximum(1e-3, CT.cc_dhat(d0, np.where(np.isfinite(dlo), dlo, d0), kap))
    if kind == "adaptq":
        gamma, Ls, qg, q = rule[1:5]
        Ls = sorted(Ls)
        d0 = tab["d_v5"]
        dh = np.minimum(d0, CC.dlo_of(tab, Ls[0], 0.01, k_bg)[0])
        for L in Ls:
            dg, _ = CC.dlo_of(tab, L, 1 - qg, k_bg)
            ok = dg >= gamma * d0
            dh = np.where(ok, rule_dhat(tab, ("ccq", L, 0.01, "auto", q), n_tests, k_bg, dim0), dh)
        return np.maximum(1e-3, dh)
    if kind == "ccmin":
        # ("ccmin", Ls, delta[, q]): the smallest CC exponent over the annuli L in Ls. Valid whenever (C2') holds on
        # at least one of them (Theorem 15.1); q != 2 uses the s^q profile of "ccq".
        Ls, delta = rule[1], rule[2]
        q = rule[3] if len(rule) > 3 else 2.0
        dh = None
        for L in Ls:
            x = rule_dhat(tab, ("ccq", L, delta, "auto", q), n_tests, k_bg, dim0)
            dh = x if dh is None else np.minimum(dh, x)
        return dh
    if kind == "ccmin_lg":
        # ("ccmin_lg", Ls, delta, Lg, gamma): ccmin, further lowered to LCB-d on the annulus Lg wherever that annulus is
        # seen to saturate (99% upper slope bound < gamma d0). Lowering d_hat is always safe, so the validity class
        # is that of ccmin (Theorem 15.1); the gate only adds protection where no rule is proved (Prop 14.4).
        Ls, delta, Lg, gamma = rule[1:5]
        d0 = tab["d_v5"]
        dh = rule_dhat(tab, ("ccmin", Ls, delta), n_tests, k_bg, dim0)
        dhi, _ = CC.dlo_of(tab, Lg, 0.99, k_bg)
        lcb = np.minimum(d0, CC.dlo_of(tab, Lg, 0.01, k_bg)[0])
        return np.maximum(1e-3, np.where(dhi < gamma * d0, np.minimum(dh, lcb), dh))
    if kind == "adaptmin":
        # ("adaptmin", gamma, Ls, qg): ccmin over the annuli not seen to saturate (slope upper bound >= gamma d0);
        # LCB-d on the narrowest annulus if all look saturated.
        gamma, Ls, qg = rule[1], sorted(rule[2]), rule[3]
        d0 = tab["d_v5"]
        dh = np.full_like(d0, np.inf)
        for L in Ls:
            dg, _ = CC.dlo_of(tab, L, 1 - qg, k_bg)
            x = rule_dhat(tab, ("ccq", L, 0.01, "auto", 2.0), n_tests, k_bg, dim0)
            dh = np.where(dg >= gamma * d0, np.minimum(dh, x), dh)
        lcb = np.minimum(d0, CC.dlo_of(tab, Ls[0], 0.01, k_bg)[0])
        return np.maximum(1e-3, np.where(np.isfinite(dh), dh, lcb))
    return CC.rule_dhat(tab, rule, n_tests, k_bg, dim0)


def sample_outer(m, k, N, n_samp, rng, Ls=LS):
    """Masses q_n, q_(n+L) (L in Ls) of the order statistics among N - 1 iid points, n = m - 1 + k."""
    n = m - 1 + k
    Ls = sorted(Ls)
    inc = [rng.gamma(n, size=n_samp)]
    prev = 0
    for L in Ls:
        inc.append(rng.gamma(L - prev, size=n_samp))
        prev = L
    rest = rng.gamma(N - 1 - n - Ls[-1] + 1, size=n_samp)          # Gamma_N = Gamma_(N-1+1): one extra term
    G = np.cumsum(np.vstack(inc), axis=0)
    tot = G[-1] + rest
    return G[0] / tot, {L: G[j + 1] / tot for j, L in enumerate(Ls)}


def rate(P, Pinv, m, k, N, t, rule, d0, n_tests, n_samp=20000, seed=0, return_cond=False):
    """Exact P(p <= t) (Monte Carlo over the outer statistics only) at a sample point, for any rule of rule_dhat."""
    rng = np.random.default_rng(seed)
    qn, qL = sample_outer(m, k, N, n_samp, rng)
    rho = Pinv(qn)
    tab = {"m": np.full(n_samp, float(m)), "d_v5": np.full(n_samp, float(d0)), "rn": rho, "r": np.zeros(n_samp),
           "Lmax": np.full(n_samp, float(max(LS)))}
    for L in LS:
        tab[f"r{L}"] = np.maximum(Pinv(qL[L]), rho * (1 + 1e-12))
    dh = rule_dhat(tab, rule, n_tests, k)
    u = CT.u_level(m, k, t)
    c = u ** (1.0 / dh)
    v = np.minimum(P(c * rho) / P(rho), 1.0)
    cond = betainc(m - 1, k, v)
    if return_cond:
        return cond, dh, tab
    return float(cond.mean())


def gaussian_mass(d, sigma=1.0, off=0.0, noise=0.0, box=40.0, Nc_frac=1.0):
    """Ball mass around a point at offset `off` (in sigma) from the centre of an isotropic Gaussian of weight Nc_frac,
    plus a uniform floor of weight `noise` over a cube of side `box` (same form as core_theory.cluster_mass)."""
    from scipy.stats import chi2, ncx2
    vb = math.pi ** (d / 2) / math.gamma(d / 2 + 1)

    def P(r):
        r = np.asarray(r, float) / sigma
        g = ncx2.cdf(r ** 2, d, off ** 2) if off > 0 else chi2.cdf(r ** 2, d)
        return np.minimum(Nc_frac * g + noise * np.minimum(vb * (r * sigma) ** d / box ** d, 1.0), 1.0) + 1e-300

    grid = np.exp(np.linspace(math.log(1e-5), math.log(box * math.sqrt(d) * 2), 4000))
    lpg = np.log(P(grid))
    ok = np.r_[True, np.diff(lpg) > 1e-14]

    def Pinv(qq):
        return np.exp(np.interp(np.log(np.clip(np.asarray(qq, float), 1e-300, 1.0)), lpg[ok], np.log(grid[ok])))
    return P, Pinv


def power_mass(d):
    """Homogeneous Poisson-like mass P(r) = r^d on the unit-mass ball (a control: every rule has rate <= t)."""
    def P(r):
        return np.minimum(np.asarray(r, float) ** d, 1.0) + 1e-300

    def Pinv(qq):
        return np.asarray(qq, float) ** (1.0 / d)
    return P, Pinv
