"""
Stage-9 wrappers around CTS v5 (no edits to tiny_methods.py):
- `reduced_set`: run cts3's collapse and capture the reduced set it would scan (R, is_rep, d settings);
- `v5_candidates`: the v5 Beta scan (dim_est="local", scale_min=True), returning every candidate with its score and
  optionally a curvature-corrected score (item 2);
- `select`: Bonferroni / Benjamini-Hochberg / Benjamini-Yekutieli thresholds, then greedy non-overlapping selection,
  exactly as in _scan_select;
- `write_labels`: groups written over the HDBSCAN base, as in _scan_select.
With curvature=None and rule="bonferroni" this reproduces cts3(Z, dim_est="local", scale_min=True, border=True).
"""
import math

import numpy as np
from scipy.stats import beta as beta_dist, chi2
from sklearn.neighbors import NearestNeighbors

from . import tiny_methods as TM


def reduced_set(Z, **kw):
    """Collapse step of cts3 (v5 settings by default); returns R, is_rep."""
    kw = {"dim_est": "local", "scale_min": True, "border": True, **kw}
    box = {}
    orig = TM._scan_select

    def grab(Z_, R, is_rep, *a, **k):
        box["R"], box["is_rep"] = R, is_rep
        return np.zeros(len(Z_), int)
    TM._scan_select = grab
    try:
        TM.cts3(Z, **kw)
    finally:
        TM._scan_select = orig
    return box["R"], box["is_rep"]


def v5_candidates(Z, R, is_rep, max_m=15, k_bg=10, curvature=None, L=20, q=0.5):
    """All v5 candidates: list of (score, members, i, m) with score = -log p; n_tests as in _scan_select.
    curvature: None, or "fit" (item 2): fit log j = a + d log r_j + beta r_j^2 over ring neighbours j = m .. n + L with
    d fixed to the test's d ("fitpos": beta_hat clipped at 0), and use V_hat = U * exp(beta_hat (r^2 - R^2)), the corrected ratio of ball masses."""
    Zr = Z[R]
    rank = int(np.linalg.matrix_rank(Zr - Zr.mean(0), tol=1e-8 * np.abs(Zr).max()))
    dim = min(max(1.0, TM.twonn_dim(Zr)), rank)
    n_loc = 30
    kk = min(len(R) - 1, max_m + k_bg + max(n_loc, L))
    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Zr).kneighbors(Zr)
    mu_all = np.log(np.maximum(dist[:, 2], 1e-12) / np.maximum(dist[:, 1], 1e-12))
    lo_q = chi2.ppf(0.001, 2 * n_loc) / 2
    kk0 = min(len(R) - 1, max_m + k_bg + n_loc)          # what _scan_select uses for the m loop / self dims
    dims = TM._self_dims(dist[:, :kk0 + 1], is_rep, max_m, k_bg, kk0, q)
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
            r, Rk = dist[i, m - 1], dist[i, m - 1 + k_bg]
            if Rk <= r or r <= 0:
                continue
            dd = min(dim, dims[m])
            ring = idx[i, m:m + n_loc]
            dd = min(max(dd, lo_q / max(mu_all[ring].sum(), 1e-12)), rank)
            lu = dd * math.log(r / Rk)
            if curvature in ("fit", "fitpos"):
                n = m - 1 + k_bg
                js = np.arange(m, min(n + L, kk) + 1)
                rj = dist[i, js]
                y = np.log(js) - dd * np.log(rj)
                x = rj ** 2
                xc = x - x.mean()
                b = float((xc * (y - y.mean())).sum() / max((xc ** 2).sum(), 1e-300))
                if curvature == "fitpos":
                    b = max(b, 0.0)          # deflate only (convex tails); never inflate
                lu = lu + b * (r ** 2 - Rk ** 2)
            score = -beta_dist.logcdf(min(math.exp(lu), 1.0), m - 1, k_bg)
            cands.append((score, tuple(sorted(members)), i, m))
    return cands, n_tests


def threshold(scores, n_tests, alpha, rule):
    """Score threshold (scores are -log p) for Bonferroni, BH or BY over n_tests hypotheses (untested count as p = 1)."""
    if rule == "bonferroni":
        return -math.log(alpha / max(1, n_tests))
    c = 1.0 if rule == "bh" else float(np.sum(1.0 / np.arange(1, n_tests + 1)))
    sc = np.sort(np.asarray(scores))[::-1]
    ok = np.flatnonzero(sc >= -np.log(alpha * np.arange(1, len(sc) + 1) / (max(1, n_tests) * c)))
    return sc[ok[-1]] if len(ok) else np.inf


def select(cands, n_tests, alpha, rule="bonferroni"):
    thr = threshold([c[0] for c in cands], n_tests, alpha, rule)
    used, groups, n_rej = set(), [], 0
    for c in sorted(cands, key=lambda t: -t[0]):
        if c[0] < thr:
            break
        n_rej += 1
        if used.isdisjoint(c[1]):
            groups.append(c[1]); used.update(c[1])
    return groups, n_rej, thr


def write_labels(Z, R, groups, base=None):
    lab = (TM.hdb(Z) if base is None else base).copy()
    nxt = lab.max() + 1
    for mem in groups:
        lab[R[list(mem)]] = nxt; nxt += 1
    return lab
