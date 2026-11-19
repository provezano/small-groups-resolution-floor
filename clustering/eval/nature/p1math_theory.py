"""
Exact calculations behind the stage-9 theory notes for the paper.

1. Validity of the Beta scan beyond local homogeneity (Theorem 9.1). For n = m - 1 + k and a sample of N points with
   density f, the ball masses around a sample point are uniform order statistics, so
       P(U <= u) = E_q[ BetaCDF( P_x(B(x, u^(1/d_hat) R(q))) / q ; m - 1, k ) ],   q ~ Beta(n, N - n),
   where R(q) is the radius whose ball holds mass q. `pred_rate` evaluates this one-dimensional integral for any mass
   function P_x (flat torus seen through its chord embedding, round sphere, Gaussian mixture).
2. Fast null scan: per-test counts of p <= t (no selection), to compare with the prediction.
3. MDL constant for the intensity-free (Beta) code (Theorem 9.5).
Public synthetic data only.
"""
import math
from functools import lru_cache

import numpy as np
from scipy.special import betainc, betaincinv, gammaln, hyp2f1
from scipy.stats import beta as beta_dist, ncx2
from sklearn.neighbors import NearestNeighbors


def vball(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


# ---------------- mass functions P_x(B(x, r)) ----------------
@lru_cache(maxsize=None)
def _torus_table(d, n_grid=160):
    """Flat torus [0,1)^d embedded by x -> (cos 2 pi x, sin 2 pi x) / (2 pi): the squared chord distance is
    sum_i sin^2(pi delta_i) / pi^2 with delta_i iid uniform, and sin^2(pi delta) ~ Beta(1/2, 1/2), whose Laplace
    transform is exp(-t/2) I_0(t/2). The CDF of the sum is inverted numerically (de Hoog), in log-log form."""
    import mpmath as mp
    mp.mp.dps = 40

    def L(t):
        z = t / 2
        return (mp.besseli(0, z) * mp.exp(-z)) ** d / t

    rmax = math.sqrt(d) / math.pi
    rs = np.exp(np.linspace(math.log(1e-3), math.log(rmax * 0.999), n_grid))
    ps = []
    for r in rs:
        if r < 0.02:
            g = 1 + math.pi ** 2 * r ** 2 * d / (2 * (d + 2))      # second-order expansion, exact to O(r^4)
            ps.append(vball(d) * r ** d * g)
        else:
            ps.append(float(mp.invertlaplace(L, math.pi ** 2 * r ** 2, method="dehoog")))
    ps = np.minimum(np.maximum.accumulate(np.array(ps)), 1.0)
    return np.log(rs), np.log(ps)


def torus_mass(d):
    lr, lp = _torus_table(d)

    def P(r):
        r = np.asarray(r, float)
        out = np.interp(np.log(np.maximum(r, 1e-300)), lr, lp)
        small = r < math.exp(lr[0])
        out = np.where(small, math.log(vball(d)) + d * np.log(np.maximum(r, 1e-300)), out)
        return np.exp(out)

    def Pinv(q):
        q = np.asarray(q, float)
        return np.exp(np.interp(np.log(q), lp, lr))
    return P, Pinv


def sphere_mass(d, a=1.0):
    """Unit-area-normalised mass of a Euclidean ball of radius r around a point of the round d-sphere of radius a in
    R^(d+1): the cap of angle theta, r = 2 a sin(theta / 2); fraction = I_{sin^2 theta}(d/2, 1/2) / 2 (theta <= pi/2)."""
    def P(r):
        r = np.minimum(np.asarray(r, float), 2 * a)
        th = 2 * np.arcsin(r / (2 * a))
        s2 = np.sin(th) ** 2
        half = 0.5 * betainc(d / 2, 0.5, s2)
        return np.where(th <= math.pi / 2, half, 1 - half)

    grid = np.exp(np.linspace(math.log(1e-4 * a), math.log(2 * a * 0.9999), 4000))
    lpg = np.log(np.maximum(P(grid), 1e-300))

    def Pinv(q):
        return np.exp(np.interp(np.log(np.asarray(q, float)), lpg, np.log(grid)))
    return P, Pinv


def gm_mass(x, mus, sds, ws):
    """Gaussian mixture in R^d: P(|Y - x| <= r) = sum_j w_j ncx2.cdf(r^2 / s_j^2, d, |x - mu_j|^2 / s_j^2)."""
    d = len(x)
    nc = [float(np.sum((x - mu) ** 2) / s ** 2) for mu, s in zip(mus, sds)]

    def P(r):
        r = np.asarray(r, float)
        return sum(w * ncx2.cdf(r ** 2 / s ** 2, d, c) for w, s, c in zip(ws, sds, nc))

    grid = np.exp(np.linspace(math.log(1e-4 * min(sds)), math.log(50 * max(sds)), 600))
    lpg = np.log(np.maximum(P(grid), 1e-300))
    ok = np.r_[True, np.diff(lpg) > 0]

    def Pinv(q):
        return np.exp(np.interp(np.log(np.asarray(q, float)), lpg[ok], np.log(grid[ok])))
    return P, Pinv


# ---------------- exact prediction (Theorem 9.1) ----------------
_GL = np.polynomial.legendre.leggauss(96)


def pred_rate(P, Pinv, m, k, N, t, d_hat):
    """P(p <= t) for the test (m, k) at a sample point, with N - 1 other iid points of mass function P."""
    n = m - 1 + k
    u = float(betaincinv(m - 1, k, t))
    x, w = _GL
    z = 0.5 * (x + 1)                                   # quantile levels in (0, 1)
    q = beta_dist.ppf(z, n, N - n)                      # mass of the ball reaching the n-th neighbour
    R = Pinv(q)
    v = np.minimum(P(u ** (1 / d_hat) * R) / q, 1.0)
    return float(0.5 * np.sum(w * betainc(m - 1, k, v)))


def thm_bound(P, Pinv, m, k, N, t, d_hat, n_r=200):
    """Theorem 9.1 bound t * E[(1 + eps(R))_+^(m-1)], eps(R) = sup_{r <= R} phi(r) / phi(R) - 1, phi(r) = P(r) / r^d_hat.
    Infinite when d_hat exceeds the smallest-scale dimension (phi blows up at 0); returned as inf then."""
    n = m - 1 + k
    x, w = _GL
    q = beta_dist.ppf(0.5 * (x + 1), n, N - n)
    R = Pinv(q)
    out = []
    for Ri in R:
        rr = Ri * np.exp(np.linspace(math.log(1e-3), 0, n_r))
        lphi = np.log(np.maximum(P(rr), 1e-300)) - d_hat * np.log(rr)
        out.append(math.exp(max(0.0, (lphi[:-1].max() - lphi[-1]) * (m - 1))))
    out = np.array(out)
    if lphi[0] > lphi[1] + 1e-9 and d_hat > 0:          # still decreasing at the smallest radius tried
        pass
    return float(t * 0.5 * np.sum(w * out))


# ---------------- fast null scan (no selection) ----------------
def twonn(X):
    d = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)[0]
    mu = np.log(np.maximum(d[:, 2], 1e-12) / np.maximum(d[:, 1], 1e-12))
    mu = mu[mu > 0]
    return float((len(mu) - 1) / mu.sum())


def scan_logp(X, d_hat, max_m=15, k=10, dist=None):
    """log p for every (point, m), m = 2..max_m; shape (N, max_m - 1)."""
    if dist is None:
        dist = NearestNeighbors(n_neighbors=max_m + k).fit(X).kneighbors(X)[0]
    out = np.empty((len(dist), max_m - 1))
    for m in range(2, max_m + 1):
        r, R = dist[:, m - 1], dist[:, m - 1 + k]
        out[:, m - 2] = beta_dist.logcdf(np.clip((r / R) ** d_hat, 1e-300, 1), m - 1, k)
    return out


# ---------------- MDL, intensity-free form (Theorem 9.5) ----------------
def _hyp_n1m(n, m, u, tol=1e-15):
    """2F1(n, 1; m; u) = sum_j (n)_j / (m)_j u^j, summed directly (scipy's hyp2f1 returns nan for very large n)."""
    if n * u >= m:
        return float(hyp2f1(n, 1, m, u))
    s, term, j = 1.0, 1.0, 0
    while term > tol * s and j < 100000:
        term *= (n + j) * u / (m + j)
        s += term
        j += 1
    return s


def mdl_identity_terms(u, m, k):
    """Exact decomposition -log BetaCDF(u; m-1, k) = Lambda_U + c(m, k) - log 2F1(n, 1; m; u), with
    Lambda_U = n KL((m-1)/n || u) the scale-invariant likelihood ratio and c(m, k) a constant (Stirling: about
    0.5 log(2 pi (m-1) n / k))."""
    n = m - 1 + k
    j = m - 1
    lam = j * math.log(j / (n * u)) + k * math.log(k / (n * (1 - u)))
    c = (math.log(n / k) + gammaln(m) + gammaln(k + 1) + n * math.log(n) - gammaln(n + 1)
         - j * math.log(j) - k * math.log(k))
    h = _hyp_n1m(n, m, u)
    return lam, c, math.log(h)


def alpha_mdl(m, k, u=0.0, W=math.sqrt(2 * math.pi)):
    """Per-test level at which the MDL rule (scale coded by its right-Haar mixture, the elevation log theta coded to
    optimal precision over a range of width W) coincides with the Beta scan: exact up to the Stirling terms.
    alpha = 2F1(n, 1; m; u) * exp(-c(m, k)) * sqrt(2 pi / I) / W, I = (m - 1) k / (n + 1) (Fisher information of log
    theta)."""
    n = m - 1 + k
    _, c, lh = mdl_identity_terms(max(u, 1e-12), m, k)
    I = (m - 1) * k / (n + 1)
    return math.exp(lh - c + 0.5 * math.log(2 * math.pi / I) - math.log(W))
