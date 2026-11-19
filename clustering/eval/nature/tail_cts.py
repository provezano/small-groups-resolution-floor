"""
CTS with a directional ring (stage 13,.

v5 compares a candidate's radius r = r_(m-1) with a full ring, the (m - 1 + 10)-th neighbour. A touching group's ring
falls partly into the dense cluster next to it. Here the ring is taken only inside a half-space (or a cone of fixed
volume share omega) pointing away from the mass, with the direction estimated from points outside a larger ball.

Exact null (Result T2 in the doc). Let rho = r_n, n = m - 1 + K_out. Under a homogeneous Poisson null (Palm at the
centre), given rho and the configuration outside B(x, rho), the n - 1 interior points are i.i.d. uniform in the ball
(stopping-set argument, F1). The direction u is a function of the outside points only (the centroid of neighbours
n + 1 .. n + L), so given (rho, u) the interior points have i.i.d. normalised volumes t ~ U(0, 1) and independent cone
marks ~ Bernoulli(omega). With N = n - 1:
  v = t_(m-1)                       ~ Beta(m - 1, N - m + 2)       (the group's volume share);
  J = number of cone points among the N - m + 1 shell points        ~ Bin(N - m + 1, omega), independent of v;
  given v and J = j >= k, Y = (w - v) / (1 - v) ~ Beta(k, j - k + 1), w = volume share of the k-th cone shell point.
The statistic is S = v / (v + omega (w - v)) (the Beta ratio with the ring volume counted in the cone only); its exact
conditional p-value P(S <= s | J = j) is a one-dimensional integral (`cone_pvalue`). J < k gives p = 1.

Also: the tilt-corrected full ring (Result T4): under an exponentially tilted Poisson null lambda(y) ~ exp(g . y) the
ball mass is Lambda_g(r) ~ r^d 0F1(; d/2 + 1; |g|^2 r^2 / 4), so U_g = Lambda_g(r) / Lambda_g(R_10) is exactly
Beta(m - 1, 10) for a g fixed by the outside; g is estimated from the outer neighbours' mean displacement.

Everything else (collapse on v5's own rows, plug-in d with the local guard, Bonferroni over n_tests, greedy selection,
the stage-11 fringe joint gate) is v5 as in `ring_cts` / `fringe_cts`.
"""
import numpy as np
from scipy.special import betainc, betaln, hyp0f1

from . import fringe_cts as F
from . import scale_cts as S

MAX_M, K_BG, N_LOC, ALPHA = S.MAX_M, S.K_BG, S.N_LOC, S.ALPHA


def cone_omega(theta_deg, d):
    """Share of the unit sphere inside a cone of half-angle theta (<= 90 degrees) in dimension d (real d >= 1)."""
    d = np.maximum(np.asarray(d, float), 1.0 + 1e-6)
    th = np.deg2rad(theta_deg)
    return 0.5 * betainc((d - 1) / 2, 0.5, np.sin(th) ** 2)


def cone_theta(omega, d):
    """Half-angle (degrees) whose cone holds a share omega of the sphere in dimension d (inverse of cone_omega)."""
    from scipy.special import betaincinv
    d = np.maximum(np.asarray(d, float), 1.0 + 1e-6)
    return np.rad2deg(np.arcsin(np.sqrt(betaincinv((d - 1) / 2, 0.5, 2 * omega))))


def cone_logp(s, m, N, k, j, omega, panels=16):
    """log P(S <= s | J = j) under the conditional null (vectorised over candidates; m, N, j, omega arrays or scalars).
    p = Int_0^{v*} f_v(v) [1 - I_{y(v)}(k, j - k + 1)] dv, f_v = Beta(m - 1, N - m + 2) density,
    y(v) = v (1 - s) / (s omega (1 - v)), v* = s omega / (1 - s + s omega) (y(v*) = 1).
    Integrated in y (v = s omega y / (1 - s + s omega y)) by composite Gauss-Legendre in log space."""
    from scipy.special import logsumexp
    s, m, N, j, omega = np.broadcast_arrays(*(np.asarray(a, float) for a in (s, m, N, j, omega)))
    out = np.zeros(s.shape)
    ok = (j >= k) & (s > 0) & (s < 1)
    if not ok.any():
        return out
    s_, m_, N_, j_, om = (a[ok][:, None] for a in (s, m, N, j, omega))
    x, w = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, 1.0, panels + 1)
    y = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * x[None, :]).ravel()
    wy = (np.repeat(np.diff(edges), len(x)) / 2 * np.tile(w, panels))[None, :]
    y = y[None, :]
    den = 1 - s_ + s_ * om * y
    v = s_ * om * y / den
    lv = np.log(s_ * om * (1 - s_)) - 2 * np.log(den)
    with np.errstate(divide="ignore"):
        lG = np.log(betainc(j_ - k + 1, k, 1 - y))
        lh = (m_ - 2) * np.log(v) + (N_ - m_ + 1) * np.log1p(-v) - betaln(m_ - 1, N_ - m_ + 2) + lv + lG + np.log(wy)
    out[ok] = np.minimum(logsumexp(lh, axis=1), 0.0)
    return out


def _half_score(inc, v, sd, rho, dd, m, k):
    """-log p of the half-space ring (marginal test, S ~ Beta(m - 1, k)); p = 1 when fewer than k half-space shell
    points lie inside the ball (censored, conservative: Result T1)."""
    n_c = len(v)
    J = inc.sum(1)
    pos = np.where(inc, np.arange(inc.shape[1])[None, :], 10 ** 6)
    kth = np.sort(pos, 1)[:, k - 1] if inc.shape[1] >= k else np.full(n_c, 10 ** 6)
    has = J >= k
    w = np.where(has, (sd[np.arange(n_c), np.minimum(kth, sd.shape[1] - 1)] / rho) ** dd, 1.0)
    sstat = v / (v + 0.5 * np.maximum(w - v, 1e-300))
    with np.errstate(divide="ignore"):
        return np.where(has, -np.log(np.maximum(betainc(m - 1, k, np.clip(sstat, 0, 1)), 1e-320)), 0.0)


def tail_parts(Z, K_out=60, L=20, k_cone=10, omegas=(0.5, 0.25), tilt=True, rows=None, family=True, max_m=MAX_M,
               screen=2.0, gate=1e-3):
    """Collapse + v5 scan + directional statistics for every v5 candidate (all candidates kept, none dropped).
    Returns per-candidate arrays: P (reduced-set row), m, r, dd, full (v5 score, -log p), cone[omega] (-log p of the
    cone ring), J[omega], tilt (-log p of the tilt-corrected full ring), g2 (|g|^2 r^2 term), pj (fringe joint p)."""
    n = len(Z)
    kk0 = S.kk_for(n, max_m, K_BG)
    need = max_m - 1 + K_out + L
    kk = min(n - 1, max(kk0, need))
    if rows is None:
        rows = S.knn_exact(Z, kk)
    D, I = rows
    dg = F.collapse_rows_diag(D[:, :kk0 + 1], I[:, :kk0 + 1], Z)
    R, is_rep, members = F._reduce(Z, dg["collapsed"], dg["comp"])
    kr0 = S.kk_for(len(R), max_m, K_BG)
    kr = min(len(R) - 1, max(kr0, need))
    if len(R) == n and not is_rep.any() and D.shape[1] - 1 >= kr:
        Dr, Ir = D[:, :kr + 1], I[:, :kr + 1]
    else:
        Dr, Ir = S.knn_exact(Z[R], kr)
    mu, lr = S.local_stats(Dr[:, :kr0 + 1], Ir[:, :kr0 + 1], is_rep, max_m, K_BG)
    dim, rank, dims = S.reduce_stats([mu], [lr], S.scatter_of(Z[R]), max_m, K_BG)
    from scipy.stats import chi2
    lo_q = chi2.ppf(0.001, 2 * N_LOC) / 2
    live = ~is_rep
    rid = np.flatnonzero(live)
    d, ii = Dr[live], Ir[live]
    rep_cols = is_rep[ii[:, :max_m]]
    first = np.where(rep_cols.any(1), np.argmax(rep_cols, 1), max_m)
    csum = np.concatenate([np.zeros((len(ii), 1)), np.cumsum(mu[ii], 1)], 1)
    Zr = Z[R]
    n_tests = 0
    acc = {k: [] for k in ("P", "m", "r", "dd", "full", "tilt", "g2", "toward", "toward5", "half5", "tiltg",
                           "gate_p", "tiltp", "gate_pt")}
    for om in omegas:
        acc[f"cone{om}"] = []; acc[f"coneJ{om}"] = []; acc[f"J{om}"] = []
    for m in range(2, max_m + 1):
        if m - 1 + K_BG > kr0:
            break
        valid = first >= m
        n_tests += int(valid.sum())
        r, R10 = d[:, m - 1], d[:, m - 1 + K_BG]
        ok = valid & (R10 > r) & (r > 0)
        if not ok.any():
            continue
        oi = np.flatnonzero(ok)
        ring = csum[oi, min(m + N_LOC, kr0 + 1)] - csum[oi, m]
        dd = min(dim, dims[m])
        dd = np.minimum(np.maximum(dd, lo_q / np.maximum(ring, 1e-12)), rank)
        rr, RR = r[oi], R10[oi]
        u10 = (rr / RR) ** dd
        with np.errstate(divide="ignore"):
            full = -np.log(np.maximum(betainc(m - 1, K_BG, u10), 1e-320))
        acc["P"].append(rid[oi]); acc["m"].append(np.full(len(oi), m, np.int16)); acc["r"].append(rr)
        acc["dd"].append(dd); acc["full"].append(full)
        n_out = m - 1 + K_out
        if n_out + L > kr:                       # not enough neighbours for the outer ball: no directional test
            for om in omegas:
                acc[f"cone{om}"].append(np.zeros(len(oi))); acc[f"coneJ{om}"].append(np.zeros(len(oi)))
                acc[f"J{om}"].append(np.zeros(len(oi), int))
            acc["tilt"].append(full.copy()); acc["g2"].append(np.zeros(len(oi)))
            acc["toward"].append(np.zeros(len(oi))); acc["toward5"].append(np.zeros(len(oi)))
            acc["half5"].append(np.zeros(len(oi))); acc["tiltg"].append(full.copy()); acc["gate_p"].append(np.ones(len(oi)))
            acc["tiltp"].append(full.copy()); acc["gate_pt"].append(np.ones(len(oi)))
            continue
        c = Zr[rid[oi]]
        outer = Zr[ii[oi, n_out + 1:n_out + L + 1]] - c[:, None, :]           # strictly outside B(x, rho)
        mdisp = outer.mean(1)
        u = -mdisp / np.maximum(np.linalg.norm(mdisp, axis=1, keepdims=True), 1e-300)   # away from the outer mass
        shell = Zr[ii[oi, m:n_out]] - c[:, None, :]                              # interior points beyond the group
        nrm = np.maximum(np.linalg.norm(shell, axis=2), 1e-300)
        cos = (shell * u[:, None, :]).sum(2) / nrm
        rho = d[oi, n_out]
        sd = d[oi, m:n_out]
        N = n_out - 1
        v = (rr / rho) ** dd
        for om in omegas:
            if om >= 0.5:
                inc = cos >= 0.0
                omv = np.full(len(oi), 0.5)
            else:
                th = cone_theta(om, dd)
                inc = cos >= np.cos(np.deg2rad(th))[:, None]
                omv = np.full(len(oi), om)
            J = inc.sum(1)
            # volume share of the k-th cone shell point
            pos = np.where(inc, np.arange(inc.shape[1])[None, :], 10 ** 6)
            kth = np.sort(pos, 1)[:, k_cone - 1] if inc.shape[1] >= k_cone else np.full(len(oi), 10 ** 6)
            has = J >= k_cone
            w = np.where(has, (sd[np.arange(len(oi)), np.minimum(kth, sd.shape[1] - 1)] / rho) ** dd, 1.0)
            sstat = v / (v + omv * np.maximum(w - v, 1e-300))
            # marginal test: S ~ Beta(m - 1, k) (Basu: S is ancillary for the intensity, so independent of the count
            # in the ball); a censored candidate (fewer than k cone points inside the ball) gets p = 1
            with np.errstate(divide="ignore"):
                marg = np.where(has, -np.log(np.maximum(betainc(m - 1, k_cone, np.clip(sstat, 0, 1)), 1e-320)), 0.0)
            # conditional test given J (exact for any J); computed where it can matter (screen on the marginal score)
            cj = np.zeros(len(oi))
            scr = has & (marg >= screen)
            if scr.any():
                cj[scr] = -cone_logp(sstat[scr], m, N, k_cone, J[scr], omv[scr])
            acc[f"cone{om}"].append(marg); acc[f"coneJ{om}"].append(cj); acc[f"J{om}"].append(J)
        # half-space TOWARD the outer mass (Result T3: conservative under the one-sided tail null, where the density
        # on the mass side is at least the density in the group's ball); k = k_cone and k = k_cone / 2 (the second
        # keeps the ring's radius equal to v5's ring of 10, so only the direction changes, not the ring size)
        acc["toward"].append(_half_score(cos <= 0.0, v, sd, rho, dd, m, k_cone))
        acc["toward5"].append(_half_score(cos <= 0.0, v, sd, rho, dd, m, max(1, k_cone // 2)))
        acc["half5"].append(_half_score(cos >= 0.0, v, sd, rho, dd, m, max(1, k_cone // 2)))
        if tilt:
            # g from the outer neighbours: E[Y] ~ (E|Y|^2 / d) g for a tilted, locally isotropic shell
            q2 = (outer ** 2).sum(2).mean(1)
            g = dd[:, None] * mdisp / q2[:, None]
            g2 = (g ** 2).sum(1)
            g2 = np.maximum(g2 - dd ** 2 / (L * q2), 0.0)                       # remove the null mean of |g_hat|^2
            b = dd / 2 + 1
            ratio = hyp0f1(b, g2 * rr ** 2 / 4) / hyp0f1(b, g2 * RR ** 2 / 4)
            ut = np.clip(u10 * ratio, 0.0, 1.0)
            with np.errstate(divide="ignore"):
                acc["tilt"].append(-np.log(np.maximum(betainc(m - 1, K_BG, ut), 1e-320)))
            acc["g2"].append(g2 * RR ** 2)
            # gated tilt (Result T5): use the tilt only where the outer displacement rejects g = 0 at level `gate`
            # (X2 = d L |mean Y|^2 / E|Y|^2 ~ chi2_d under an isotropic shell); elsewhere the plain ring of 10
            x2 = dd * L * (mdisp ** 2).sum(1) / q2
            gp = chi2.sf(x2, dd)
            acc["gate_p"].append(gp)
            acc["tiltg"].append(np.where(gp < gate, acc["tilt"][-1], full))
            # tangent-projected gated tilt: on a curved manifold the outer mean displacement has a normal component
            # (curvature) that is not a density gradient. Keep only its part in the local tangent space, estimated by
            # the top-k principal directions (k = round(d)) of the centred outer displacements (outside points only).
            oc = outer - mdisp[:, None, :]
            C = np.einsum("nli,nlj->nij", oc, oc) / L
            _, V = np.linalg.eigh(C)                                             # ascending eigenvalues
            kt = np.clip(np.rint(dd).astype(int), 1, outer.shape[2])
            coef = np.einsum("nij,ni->nj", V, mdisp)                             # coordinates in the eigenbasis
            keep = np.arange(outer.shape[2])[None, :] >= outer.shape[2] - kt[:, None]
            mt2 = (coef ** 2 * keep).sum(1)
            x2t = dd * L * mt2 / q2
            gpt = chi2.sf(x2t, kt)
            g2t = np.maximum(dd ** 2 * mt2 / q2 ** 2 - dd * kt / (L * q2), 0.0)
            ratio_t = hyp0f1(b, g2t * rr ** 2 / 4) / hyp0f1(b, g2t * RR ** 2 / 4)
            with np.errstate(divide="ignore"):
                st = -np.log(np.maximum(betainc(m - 1, K_BG, np.clip(u10 * ratio_t, 0.0, 1.0)), 1e-320))
            acc["tiltp"].append(np.where(gpt < gate, st, full)); acc["gate_pt"].append(gpt)
        else:
            acc["tilt"].append(full.copy()); acc["g2"].append(np.zeros(len(oi)))
            acc["tiltg"].append(full.copy()); acc["gate_p"].append(np.ones(len(oi)))
            acc["tiltp"].append(full.copy()); acc["gate_pt"].append(np.ones(len(oi)))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out.update(n=n, R=R, is_rep=is_rep, Ir=Ir, Dr=Dr, n_tests=n_tests, dim=dim, families=len(members),
               omegas=tuple(omegas), K_out=K_out, L=L, k_cone=k_cone, gate=gate)
    out["pj"] = None
    if family and len(members) and len(out["P"]):
        fam_c, fam_s2, fam_M = F.family_models(Z, R, is_rep, members, q_ucb=0.999, dim=dim)
        out["pj"] = F.joint_pvalues(Z, R, out["P"], out["m"].astype(np.int64), out["r"], fam_c, fam_s2, fam_M, dim)
    return out


VARIANTS = ("full", "half", "quarter", "halfJ", "quarterJ", "min_full_half", "half5", "toward", "toward5", "tilt",
            "tiltg", "tiltp")


def tail_scores(parts, variant="full", fix="joint"):
    """-log p of every candidate for one variant. min_full_half: Bonferroni over the two tests (- log 2)."""
    if variant == "full":
        sc = parts["full"]
    elif variant == "half":
        sc = parts["cone0.5"]
    elif variant == "quarter":
        sc = parts["cone0.25"]
    elif variant == "halfJ":
        sc = parts["coneJ0.5"]
    elif variant == "quarterJ":
        sc = parts["coneJ0.25"]
    elif variant == "min_full_half":
        sc = np.maximum(parts["full"], parts["cone0.5"]) - np.log(2)
    elif variant == "tilt":
        sc = parts["tilt"]
    elif variant in ("tiltg", "tiltp"):
        sc = parts[variant]
    elif variant in ("toward", "toward5", "half5"):
        sc = parts[variant]
    else:
        raise ValueError(variant)
    if fix == "joint" and parts["pj"] is not None:
        with np.errstate(divide="ignore"):
            sc = np.minimum(sc, -np.log(np.maximum(parts["pj"], 1e-300)))
    return sc


def tail_groups(parts, alpha=ALPHA, variant="full", fix="joint"):
    """Bonferroni over n_tests (alpha > 1 bounds the expected number of false rejections, PFER), greedy pass as v5."""
    sc = tail_scores(parts, variant, fix)
    thr = -np.log(alpha / max(1, parts["n_tests"]))
    sig = sc >= thr
    Sg, Pg, Mg = sc[sig], parts["P"][sig], parts["m"][sig]
    order = np.lexsort((Mg, Pg, -Sg))
    used, groups = set(), []
    for o in order:
        mem = parts["Ir"][Pg[o]][:Mg[o]]
        if used.isdisjoint(mem.tolist()):
            groups.append(parts["R"][np.sort(mem)]); used.update(mem.tolist())
    return groups


def tail_labels(parts, alpha=ALPHA, variant="full", fix="joint", base=None):
    return S.groups_to_labels(tail_groups(parts, alpha, variant, fix), parts["n"], base)


def cts_tail(Z, alpha=ALPHA, variant="half", fix="joint", base=None):
    """One-call entry point."""
    return tail_labels(tail_parts(Z, family=(fix == "joint")), alpha, variant, fix, base)
