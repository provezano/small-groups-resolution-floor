"""
Stage 14 (core): a curvature-corrected d rule for the Beta scan, read from the outer annulus only.

Setting (as in prf_theory, Theorem 13.2). Test (x, m), ring k, n = m - 1 + k. P_x(s) = mass of B(x, s), D_x(s) its
log-slope, Dbar_x(a, b) the log-average of D_x over [a, b]. The test uses U = (r_(m-1) / r_n)^d_hat, p = F(U),
F = Beta(m - 1, k) CDF, u_t = F^-1(t).

Key fact (stopping set). Given rho = r_n, the n - 1 inner points are iid from mu restricted to B(x, rho) and the outer
points are iid from mu outside it, independently. So V = P_x(r_(m-1)) / P_x(rho) ~ Beta(m - 1, k) is independent of
rho AND of every statistic O of the outer points (radii r_(n+1), r_(n+2), ...). Any d_hat = d_hat(O) gives
    P(p <= t | O) = F( P_x(c rho) / P_x(rho) ),  c = u_t^(1 / d_hat),
so the test is valid at level t on the event {O = o} iff Dbar_x(c rho, rho) >= d_hat(o). A gate G(O) and any rule
that switches on G is valid if each branch is valid on its own event (the Beta law of V is untouched by conditioning
on G, because V is independent of O).

The curvature-corrected (CC) rule. With d0 a base exponent, lambda = r_(n+L) / r_n and the outer slope bound
D_lo = log b / log(1 / lambda) (b = (1 - delta)-quantile of Beta(n, L), as in Theorem 13.2),
    kappa = A(c0) / A_out(lambda),  A(c) = (1 - c^2) / (2 log(1/c)),  A_out(l) = (l^2 - 1) / (2 log l),
    c0 = u_tmax^(1 / d0),
    d_hat = d0 - kappa * max(0, d0 - D_lo).
A(c) and A_out are the log-scale averages of (s / rho)^2 over the inner window [c rho, rho] and the outer window
[rho, lambda rho]. Theorem 14.1 (doc) proves validity, P(p <= t) <= t (1 + eta), under
    (C2) h(s) = (d0 - D_x(s)) / s^2 satisfies h(s) <= h(rho) <= h(s') for s in [c rho, rho], s' in [rho, lambda rho]
(the deficit of the slope below d0 grows at least quadratically across rho; true to second order for any smooth
density whose fourth-order term does not oppose the second, false once the ball saturates).

Public synthetic data only.
"""
import math

import numpy as np
from scipy.special import betainc, betaincinv
from scipy.stats import beta as beta_dist


def a_in(c):
    """Log-scale average of s^2 over [c, 1]."""
    c = np.asarray(c, float)
    return np.where(c < 1 - 1e-9, (1 - c ** 2) / (2 * np.log(1 / np.maximum(c, 1e-300))), 1.0)


def a_out(lam):
    """Log-scale average of s^2 over [1, lam]."""
    lam = np.asarray(lam, float)
    return np.where(lam > 1 + 1e-9, (lam ** 2 - 1) / (2 * np.log(np.maximum(lam, 1 + 1e-12))), 1.0)


def u_level(m, k, t):
    return float(betaincinv(m - 1, k, t))


def kappa(m, k, tmax, d0, lam):
    """kappa = A(c0) / A_out(lambda), c0 = u_tmax^(1/d0). Nondecreasing in tmax and d0, so a larger tmax is safe."""
    c0 = u_level(m, k, tmax) ** (1.0 / np.asarray(d0, float))
    return a_in(c0) / a_out(lam)


def cc_dhat(d0, dlo, kap):
    """d_hat = d0 - kappa (d0 - D_lo)_+ (a convex combination of d0 and D_lo when D_lo < d0)."""
    d0 = np.asarray(d0, float)
    return d0 - kap * np.maximum(0.0, d0 - np.asarray(dlo, float))


def cc_eta(m, k, L, delta, t, kap, n_grid=40000):
    """Inflation of the CC rule from the noise of D_lo (Theorem 14.1, step 6): P(p <= t) <= t (1 + eta), with
    eta = E_B[F(u_t^(1 - kappa (1 - kappa_B)_+))] / t - 1, kappa_B = log B / log b, B ~ Beta(n, L)."""
    n = m - 1 + k
    u = u_level(m, k, t)
    b = float(betaincinv(n, L, 1 - delta))
    z = (np.arange(n_grid) + 0.5) / n_grid
    q = beta_dist.ppf(z, n, L)
    kb = np.log(q) / math.log(b)
    ex = 1 - kap * np.maximum(0.0, 1 - kb)
    return float(np.mean(betainc(m - 1, k, u ** ex)) / t - 1)


def cc_rate_exact(P, Pinv, m, k, N, t, d0, L, delta, tmax=None, n_q=96, rule="cc", gate=None):
    """Exact P(p <= t) of the CC (or LCB, rule="lcb"; or fixed d0, rule="fixed") test at a sample point with N - 1
    other iid points of mass function P (two-dimensional Gauss-Legendre over q_(n+L) and B = q_n / q_(n+L)).
    gate: optional (dlo, d0) -> bool array; where True the LCB branch is used instead of CC."""
    tmax = t if tmax is None else tmax
    n = m - 1 + k
    u = u_level(m, k, t)
    b = float(betaincinv(n, L, 1 - delta))
    x, w = np.polynomial.legendre.leggauss(n_q)
    z = 0.5 * (x + 1)
    qL = beta_dist.ppf(z, n + L, N - n - L)
    Bq = beta_dist.ppf(z, n, L)
    QL, BB = np.meshgrid(qL, Bq, indexing="ij")
    W = np.outer(w, w) * 0.25
    qn = BB * QL
    R, RL = Pinv(qn), Pinv(QL)
    lam = np.maximum(RL / np.maximum(R, 1e-300), 1 + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlo = np.where(RL > R, math.log(b) / np.log(1 / lam), d0)
    if rule == "fixed":
        dh = np.full_like(R, d0)
    elif rule == "lcb":
        dh = np.minimum(d0, dlo)
    else:
        dh = cc_dhat(d0, dlo, kappa(m, k, tmax, d0, lam))
        if gate is not None:
            g = gate(dlo, d0, lam)
            dh = np.where(g, np.minimum(d0, dlo), dh)
    dh = np.maximum(dh, 1e-3)
    v = np.minimum(P(u ** (1 / dh) * R) / qn, 1.0)
    return float(np.sum(W * betainc(m - 1, k, v)))


def log_slope(P, s, h=2e-3):
    s = np.asarray(s, float)
    return (np.log(P(s * math.exp(h))) - np.log(P(s * math.exp(-h)))) / (2 * h)


def c2_check(P, Pinv, m, k, N, tmax, d0, L, n_q=24, n_s=60):
    """Share of (rho, lambda) quadrature mass where condition (C2) fails, and the worst violation, for the windows the
    rule actually uses: inner [c0 rho, rho], outer [rho, lambda rho]."""
    n = m - 1 + k
    c0 = u_level(m, k, tmax) ** (1 / d0)
    x, w = np.polynomial.legendre.leggauss(n_q)
    z = 0.5 * (x + 1)
    qL = beta_dist.ppf(z, n + L, N - n - L)
    Bq = beta_dist.ppf(z, n, L)
    bad, tot = 0.0, 0.0
    for i, ql in enumerate(qL):
        for j, bq in enumerate(Bq):
            wt = w[i] * w[j] * 0.25
            rho, rl = float(Pinv(bq * ql)), float(Pinv(ql))
            si = rho * np.exp(np.linspace(math.log(c0), 0, n_s))
            so = np.exp(np.linspace(math.log(rho), math.log(max(rl, rho * (1 + 1e-6))), n_s))
            hi = (d0 - log_slope(P, si)) / si ** 2
            ho = (d0 - log_slope(P, so)) / so ** 2
            hr = hi[-1]
            inner_ok = hi.max() <= hr + 1e-9 * abs(hr) + 1e-12
            outer_ok = ho.min() >= hr - 1e-9 * abs(hr) - 1e-12
            ok = inner_ok and (hr <= 0 or outer_ok)       # h(rho) <= 0: no deficit inside, the outer side is not used
            tot += wt
            bad += wt * (not ok)
    return bad / tot


# ---------------- the frozen gated rule (two annuli) ----------------
def gated_rate_exact(P, Pinv, m, k, N, t, d0, Lc=160, dc=0.01, Lg=20, qg=0.01, gamma=0.3, tmax=None, n_q=40,
                     rule="gated"):
    """Exact P(p <= t) for the frozen gated rule at a sample point with N - 1 other iid points of mass function P.
    q_(n+Lc) ~ Beta(n + Lc, N - n - Lc), B' = q_(n+Lg) / q_(n+Lc) ~ Beta(n + Lg, Lc - Lg), Bg = q_n / q_(n+Lg) ~
    Beta(n, Lg), all independent, and independent of V. rule = "gated" | "cc" | "lcb" (near annulus) | "fixed"."""
    tmax = t if tmax is None else tmax
    n = m - 1 + k
    u = u_level(m, k, t)
    bc = float(betaincinv(n, Lc, 1 - dc))
    bg = float(betaincinv(n, Lg, qg))
    bl = float(betaincinv(n, Lg, 0.99))
    x, w = np.polynomial.legendre.leggauss(n_q)
    z = 0.5 * (x + 1)
    W = np.einsum("i,j,l->ijl", w, w, w) / 8
    q3 = beta_dist.ppf(z, n + Lc, N - n - Lc)[:, None, None]
    b2 = beta_dist.ppf(z, n + Lg, Lc - Lg)[None, :, None]
    b1 = beta_dist.ppf(z, n, Lg)[None, None, :]
    qLg = q3 * b2
    qn = qLg * b1
    R, RLg, RLc = Pinv(qn), Pinv(qLg), Pinv(np.broadcast_to(q3, qn.shape))
    lam_c = np.maximum(RLc / R, 1 + 1e-12)
    lam_g = np.maximum(RLg / R, 1 + 1e-12)
    dlo_c = math.log(bc) / np.log(1 / lam_c)
    dg = math.log(bg) / np.log(1 / lam_g)
    dl = math.log(bl) / np.log(1 / lam_g)
    if rule == "fixed":
        dh = np.full(qn.shape, float(d0))
    elif rule == "lcb":
        dh = np.minimum(d0, dl)
    else:
        dh = cc_dhat(d0, dlo_c, kappa(m, k, tmax, d0, lam_c))
        if rule == "gated":
            dh = np.where(dg < gamma * d0, np.minimum(d0, dl), dh)
    dh = np.maximum(dh, 1e-3)
    v = np.minimum(P(u ** (1 / dh) * R) / qn, 1.0)
    return float(np.sum(W * betainc(m - 1, k, v)))


def gated_eta(m, k, t, d0, gamma0, Lc=160, dc=0.01, Lg=20, qg=0.01, gamma=0.3, tmax=None, n_q=200):
    """Bound of Theorem 14.2 for the gated rule under the class: (Q) + (T) everywhere, and (C2') for the wide window
    wherever the true near-annulus slope is >= gamma0 * d0. Returns eta with P(p <= t) <= t (1 + eta).
    Terms (max over the two branches, since the branch taken depends on the unknown geometry):
      CC branch, C2' region:     F(u^(1 - kappa0 (1 - kB_c)_+)), kB_c = log B_c / log b_c;
      CC branch, outside region:  F(u^min(1, gamma kB_g)),        kB_g = log B_g / log b_g   (needs kB_g < gamma0/gamma);
      LCB branch:                 F(u^min(1, kB_l)),               kB_l = log B_g / log b_l."""
    tmax = t if tmax is None else tmax
    n = m - 1 + k
    u = u_level(m, k, t)
    kap0 = float(a_in(u_level(m, k, tmax) ** (1 / d0)))
    bc = float(betaincinv(n, Lc, 1 - dc))
    bg = float(betaincinv(n, Lg, qg))
    bl = float(betaincinv(n, Lg, 0.99))
    x, w = np.polynomial.legendre.leggauss(n_q)
    z = 0.5 * (x + 1)
    Bg = beta_dist.ppf(z, n, Lg)[:, None]
    Bp = beta_dist.ppf(z, n + Lg, Lc - Lg)[None, :]
    Bc = Bg * Bp
    W = np.outer(w, w) / 4
    kc = np.log(Bc) / math.log(bc)
    kg = np.log(Bg) / math.log(bg)
    kl = np.log(Bg) / math.log(bl)
    cc_in = betainc(m - 1, k, u ** (1 - kap0 * np.maximum(0, 1 - kc)))
    cc_out = betainc(m - 1, k, u ** np.minimum(1, gamma * kg))
    cc = np.where(kg >= gamma0 / gamma, cc_in, np.maximum(cc_in, cc_out))
    lcb = betainc(m - 1, k, u ** np.minimum(1, kl)) * np.ones_like(cc)
    return float(np.sum(W * np.maximum(cc, lcb)) / t - 1)


def cluster_mass(d, Nc, N, off=0.0, noise_frac=0.02, box=20.0):
    """Mass function around a point at distance `off` (in spreads) from the centre of one Gaussian cluster of Nc of
    the N points (spread 1), with a uniform noise floor of noise_frac * N points in a cube of side `box` (the tiny-bench
    background, one cluster in view)."""
    from scipy.stats import ncx2
    wc, wn = Nc / N, noise_frac
    vb = math.pi ** (d / 2) / math.gamma(d / 2 + 1)

    def P(r):
        r = np.asarray(r, float)
        g = ncx2.cdf(r ** 2, d, off ** 2) if off > 0 else __import__("scipy.stats", fromlist=["chi2"]).chi2.cdf(r ** 2, d)
        return np.minimum(wc * g + wn * np.minimum(vb * r ** d / box ** d, 1.0), 1.0) + 1e-300

    grid = np.exp(np.linspace(math.log(1e-4), math.log(box * math.sqrt(d)), 3000))
    lpg = np.log(P(grid))
    ok = np.r_[True, np.diff(lpg) > 0]

    def Pinv(q):
        return np.exp(np.interp(np.log(np.asarray(q, float)), lpg[ok], np.log(grid[ok])))
    return P, Pinv
