"""
Stage 13 (prf): group-level FDR for the CTS scan, and the validity of the plugged-in d.

Item 1. Group-count step-up (Theorem 13.1). Candidates are sorted by score; greedy disjoint selection is nested in the
threshold (a later candidate never removes an earlier group), so G(t), the groups reported at threshold t, grows with
t. The rule reports G(t*) with t* the largest candidate p-value p_(j) such that
    p_(j) <= alpha * beta(|G(p_(j))|) / T,
where T is the number of tests and beta a Blanchard-Roquain shape function. The reported set is then self-consistent
with respect to its own size, so its group-level FDR is <= alpha * T0 / T under any dependence. Shapes:
    "bonferroni": beta(r) = 1                 (nu = point mass at 1; this is the plain Bonferroni scan)
    "by":         beta(r) = r / H_T           (nu proportional to 1/k on 1..T; Benjamini-Yekutieli shape)
    ("trunc", K): beta(r) = min(r, K) / H_K   (nu proportional to 1/k on 1..K, K fixed in advance)
    "bh":         beta(r) = r                 (NOT a valid shape under arbitrary dependence; exploratory only)

Item 2. Validity of a plug-in d (Theorem 13.2): with V ~ Beta(m - 1, k) the exact mass ratio, V = U^(Dbar / d_hat)
exactly, Dbar the average log-slope of the ball mass between r_(m-1) and r_n. `slope_bound_rate` evaluates the
resulting bound under a quasi-concave slope; `lcb_candidates` is the v5 scan with a d rule whose proof needs only
quasi-concavity (a one-sided lower bound on the slope read beyond the ring, independent of V).
Public synthetic data only.
"""
import math

import numpy as np
from scipy.special import betainc, betaincinv
from scipy.stats import beta as beta_dist
from sklearn.neighbors import NearestNeighbors

from . import tiny_methods as TM


def harmonic(n):
    n = int(n)
    if n < 1000:
        return float(np.sum(1.0 / np.arange(1, n + 1)))
    return math.log(n) + np.euler_gamma + 1 / (2 * n) - 1 / (12 * n * n)


def shape_beta(r, shape, T):
    """beta(r) for the shapes above (r = number of reported groups)."""
    if r < 1:
        return 0.0
    if shape == "bonferroni":
        return 1.0
    if shape == "by":
        return r / harmonic(T)
    if shape == "bh":
        return float(r)
    if isinstance(shape, tuple) and shape[0] == "trunc":
        K = int(shape[1])
        return min(r, K) / harmonic(K)
    if isinstance(shape, tuple) and shape[0] == "pt":
        x, pp = float(shape[1]), float(shape[2])    # nu = (1 - p) delta_1 + p delta_x
        return (1 - pp) * (r >= 1) + pp * x * (r >= x)
    raise ValueError(shape)


def shape_vec(r, shape, T):
    r = np.asarray(r, float)
    if shape == "bonferroni":
        return (r >= 1).astype(float)
    if shape == "by":
        return r / harmonic(T)
    if shape == "bh":
        return r
    if isinstance(shape, tuple) and shape[0] == "pt":
        x, pp = float(shape[1]), float(shape[2])
        return (1 - pp) * (r >= 1) + pp * x * (r >= x)
    K = int(shape[1])
    return np.minimum(r, K) / harmonic(K)


def greedy_path(cands):
    """Candidates (score, members, ...) sorted by score (desc); returns the sorted list and g_j = number of greedy
    disjoint groups after the first j + 1 candidates, plus the flag 'candidate j became a group'."""
    cs = sorted(cands, key=lambda c: -c[0])
    used, g, gl, isg = set(), 0, [], []
    for c in cs:
        new = used.isdisjoint(c[1])
        if new:
            used.update(c[1]); g += 1
        gl.append(g); isg.append(new)
    return cs, np.array(gl), np.array(isg, bool)


def group_stepup(cands, n_tests, alpha, shape="by"):
    """Theorem 13.1's rule. Returns (groups, threshold score, n_rejected candidates)."""
    if not cands:
        return [], np.inf, 0
    cs, gl, isg = greedy_path(cands)
    sc = np.array([c[0] for c in cs])
    lines = np.array([-math.log(alpha * shape_beta(g, shape, n_tests) / max(1, n_tests)) if g > 0 else np.inf
                      for g in gl])
    ok = np.flatnonzero(sc >= lines)
    if not len(ok):
        return [], np.inf, 0
    j = ok[-1]
    groups = [cs[i][1] for i in range(j + 1) if isg[i]]
    return groups, float(sc[j]), int(j + 1)


def is_self_consistent(groups, cands, n_tests, alpha, shape):
    """Check: every reported group's p-value <= alpha beta(|G|) / T."""
    if not groups:
        return True
    s = {}
    for c in cands:                     # the same set can be a candidate from several seeds; the best one is reported
        s[c[1]] = max(s.get(c[1], -np.inf), c[0])
    lim = -math.log(alpha * shape_beta(len(groups), shape, n_tests) / n_tests)
    return all(s[g] >= lim - 1e-9 for g in groups)


# ---------------- the BY clumping example (Proposition 13.1c) ----------------
def clump_example(T, L, alpha, reps, rng, lam_f=1):
    """One true group always rejected with L overlapping candidates (p ~ 0); T0 = T - L null candidates with exact
    uniform p-values, independent, falling in disjoint blocks of size lam_f (one block = one possible false group;
    all members of a block share the block's minimum p-value up to a negligible jitter, so a false group brings
    lam_f rejected candidates). Returns the group-level FDR of BY over candidates and of the group-count rules."""
    nb = (T - L) // lam_f
    out = {"by": [], "gby": [], ("trunc", 32): [], "bonferroni": []}
    HT = harmonic(T)
    for _ in range(reps):
        pb = rng.uniform(size=nb)
        # BY over candidates: candidate p-values = block minima repeated lam_f times, plus L zeros
        ps = np.sort(pb)
        # number rejected at BY: largest r with p_(r) <= alpha r / (T H_T), counting the L true ones first
        # candidate list sorted: L zeros, then each block min repeated lam_f times
        j = np.arange(1, nb + 1)
        rc = L + lam_f * j                                      # candidates rejected if block j's p is the cut
        okb = np.flatnonzero(ps <= alpha * rc / (T * HT))
        V = (okb[-1] + 1) if len(okb) else 0
        out["by"].append(V / (V + 1))
        for sh in ("gby", ("trunc", 32), "bonferroni"):
            shp = "by" if sh == "gby" else sh
            g = 1 + j                                           # groups if block j is the cut
            lim = alpha * shape_vec(g, shp, T) / T
            okg = np.flatnonzero(ps <= lim)
            V = (okg[-1] + 1) if len(okg) else 0
            out[sh].append(V / (V + 1))
    return {str(k): float(np.mean(v)) for k, v in out.items()}


# ---------------- item 2: slopes ----------------
def log_slope(P, s, h=2e-3):
    """D(s) = d log P / d log s, central difference in log s."""
    s = np.asarray(s, float)
    return (np.log(P(s * math.exp(h))) - np.log(P(s * math.exp(-h)))) / (2 * h)


def quasi_concave(D, tol=1e-3):
    """True if the sequence has no interior local minimum (it rises then falls, or is monotone), up to tol."""
    D = np.asarray(D, float)
    i = int(np.argmax(D))
    up = np.all(np.diff(D[:i + 1]) >= -tol)
    dn = np.all(np.diff(D[i:]) <= tol)
    return bool(up and dn)


_GL = np.polynomial.legendre.leggauss(96)


def slope_bound_rate(P, Pinv, m, k, N, t, d_hat, D0):
    """Theorem 13.3's bound E_R[ F( u_t^min(1, min(D0, D(R)) / d_hat) ) ] for a sample point with mass function P,
    R the radius of the ball of mass q ~ Beta(n, N - n) (valid when D is quasi-concave on (0, R])."""
    n = m - 1 + k
    u = float(betaincinv(m - 1, k, t))
    x, w = _GL
    q = beta_dist.ppf(0.5 * (x + 1), n, N - n)
    R = Pinv(q)
    DR = log_slope(P, R)
    kap = np.minimum(1.0, np.minimum(D0, DR) / d_hat)
    return float(0.5 * np.sum(w * betainc(m - 1, k, u ** kap)))


def lcb_eta(m, k, L, delta, t, n_grid=40000):
    """Inflation of the LCB d rule (Theorem 13.4): P(p <= t) <= t (1 + eta). Exact one-dimensional integral over
    B = P(r_n) / P(r_(n+L)) ~ Beta(n, L), independent of V."""
    n = m - 1 + k
    u = float(betaincinv(m - 1, k, t))
    b = float(betaincinv(n, L, 1 - delta))
    z = (np.arange(n_grid) + 0.5) / n_grid
    q = beta_dist.ppf(z, n, L)
    kap = np.minimum(1.0, np.log(q) / math.log(b))
    return float(np.mean(betainc(m - 1, k, u ** kap)) / t - 1)


def lcb_candidates(Z, R, is_rep, max_m=15, k_bg=10, L=160, delta=0.05, d0="twonn", floor=1e-3):
    """v5 candidates with the LCB d rule: d_hat(i, m) = min(d0, D_lo(i, m)), D_lo = log b / log(r_n / r_(n+L)),
    b = Beta(n, L) upper delta-quantile. d0 = min(TwoNN, rank). No test-scale d_m and no local raise."""
    Zr = Z[R]
    rank = int(np.linalg.matrix_rank(Zr - Zr.mean(0), tol=1e-8 * np.abs(Zr).max()))
    dim0 = min(max(1.0, TM.twonn_dim(Zr)), rank)
    kk0 = min(len(R) - 1, max_m + k_bg + 30)            # same m-loop limit as v5
    kk = min(len(R) - 1, max_m + k_bg + L)
    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Zr).kneighbors(Zr)
    cands, n_tests = [], 0
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
            nb = min(n + L, kk)
            b = float(betaincinv(n, nb - n, 1 - delta))
            Rb = dist[i, nb]
            dlo = math.log(b) / math.log(Rk / Rb) if Rb > Rk else dim0
            dd = max(floor, min(dim0, dlo))
            score = -beta_dist.logcdf(min((r / Rk) ** dd, 1.0), m - 1, k_bg)
            cands.append((score, tuple(sorted(members)), i, m))
    return cands, n_tests


# ---------------- added on resume (added later) ----------------
def shape_beta_any(r, shape, T):
    """shape_beta plus the mixture ("mix", w, K): nu = (1 - w) delta_1 + w * (nu_trunc K), so
    beta(r) = (1 - w) 1{r >= 1} + w min(r, K) / H_K. Any mixture of shape measures is a shape measure."""
    if isinstance(shape, tuple) and shape[0] == "mix":
        w, K = float(shape[1]), int(shape[2])
        return 0.0 if r < 1 else (1 - w) + w * min(r, K) / harmonic(K)
    return shape_beta(r, shape, T)


def group_stepup_any(cands, n_tests, alpha, shape="by"):
    """group_stepup with shape_beta_any (same rule, Theorem 13.1)."""
    if not cands:
        return [], np.inf, 0
    cs, gl, isg = greedy_path(cands)
    sc = np.array([c[0] for c in cs])
    T = max(1, n_tests)
    bt = [shape_beta_any(g, shape, n_tests) if g > 0 else 0.0 for g in gl]      # beta = 0 (e.g. ("pt", x, 1.0), r < x): no line
    lines = np.array([-math.log(alpha * b_ / T) if b_ > 0 else np.inf for b_ in bt])
    ok = np.flatnonzero(sc >= lines)
    if not len(ok):
        return [], np.inf, 0
    j = ok[-1]
    return [cs[i][1] for i in range(j + 1) if isg[i]], float(sc[j]), int(j + 1)


def lcb_rate_exact(P, Pinv, m, k, N, t, d0, L, delta, n_q=96):
    """Exact P(p <= t) of the LCB-d test at a sample point with N - 1 other iid points of mass function P.
    Ball masses of the order statistics: q_(n+L) ~ Beta(n + L, N - n - L) and B = q_(n) / q_(n+L) ~ Beta(n, L),
    independent; given them, U = q_(m-1) / q_(n) ~ Beta(m - 1, k) is independent of both. The test rejects iff
    r_(m-1) <= u_t^(1 / d_hat) r_(n), i.e. U <= P(u_t^(1/d_hat) R) / q_(n). Two-dimensional Gauss-Legendre."""
    n = m - 1 + k
    u = float(betaincinv(m - 1, k, t))
    b = float(betaincinv(n, L, 1 - delta))
    x, w = np.polynomial.legendre.leggauss(n_q)
    z = 0.5 * (x + 1)
    qL = beta_dist.ppf(z, n + L, N - n - L)
    Bq = beta_dist.ppf(z, n, L)
    QL, BB = np.meshgrid(qL, Bq, indexing="ij")
    W = np.outer(w, w) * 0.25
    qn = BB * QL
    R, RL = Pinv(qn), Pinv(QL)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlo = np.where(RL > R, math.log(b) / np.log(R / RL), d0)
    dh = np.minimum(d0, dlo)
    v = np.minimum(P(u ** (1 / dh) * R) / qn, 1.0)
    return float(np.sum(W * betainc(m - 1, k, v)))


def slope_profile(P, s_lo, s_hi, n=400):
    """D(s) on a log grid between s_lo and s_hi."""
    s = np.exp(np.linspace(math.log(s_lo), math.log(s_hi), n))
    return s, log_slope(P, s)
