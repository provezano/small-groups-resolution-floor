"""
Idea 1, session families: a parent-offspring (self-exciting) model over sessions, after Zhuang, Ogata & Vere-Jones
(2002), against simple sessionization.

Each session j either comes from its user's background or is the offspring of an earlier session i, with weight
kappa * exp(-dt/tau)/tau * exp(-(1 - cos_ij)/h) * (1 if same user else beta). The background rate of each user varies
in time (a kernel over that user's own sessions, weighted by how likely each is background), so bursts of one user
are not read as triggering by default. EM fits tau, h, beta and kappa; families are the trees of the most likely
parent links (and, as a check, links that persist over sampled parent trees).

Baselines: same user + start within g minutes of the previous session (chained), and the same with a cosine floor.
Synthetic truth plants families that interleave within a user and cross users, plus a bursty templated job.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score


# ---------- synthetic streams ----------

def make_stream(seed, n_users=40, hours=5.0, dim=32, fam_per_user=1.5, p_cross=0.25, job=True, delay="exp",
                child_sd=0.35, bg_per_user=8):
    """-> ts (s), user, E (unit rows), fam (true family id, -1 = background single), kind per family."""
    rng = np.random.default_rng(seed)
    T = hours * 3600
    ts, user, E, fam, kind = [], [], [], [], {}
    topics = rng.normal(size=(n_users, dim))
    for u in range(n_users):                                # background: bursty in time (on/off), user topic + noise
        n = rng.poisson(bg_per_user)
        on = rng.uniform(0, T, size=max(1, n // 4))
        for _ in range(n):
            ts.append(float(np.clip(rng.choice(on) + rng.normal(0, 600), 0, T)))
            user.append(u)
            E.append(topics[u] + rng.normal(scale=1.0, size=dim))
            fam.append(-1)
    fid = 0
    for u in range(n_users):                                # families: root + offspring chain/tree
        for _ in range(rng.poisson(fam_per_user)):
            t0, c0 = rng.uniform(0, T * 0.9), topics[u] + rng.normal(scale=1.0, size=dim)
            nodes = [(t0, u, c0)]
            crossed = False
            for _ in range(int(rng.integers(2, 9))):
                pt, pu, pc = nodes[int(rng.integers(len(nodes)))]
                cu = int(rng.integers(n_users)) if rng.random() < p_cross else pu
                crossed |= cu != pu
                dt = rng.exponential(180) if delay == "exp" else rng.lognormal(np.log(180), 1.5)
                nodes.append((pt + dt, cu, pc + rng.normal(scale=child_sd, size=dim)))
            for t, uu, c in nodes:
                ts.append(min(t, T))
                user.append(uu)
                E.append(c)
                fam.append(fid)
            kind[fid] = "cross" if crossed else "within"
            fid += 1
    if job:                                                 # templated job: one user, bursts with a pause
        ju, tmpl = n_users, rng.normal(size=(12, dim)) * 2
        for b0 in (0.3 * T, 0.45 * T, 0.8 * T):
            for k in range(300):
                ts.append(b0 + k * 3 + rng.exponential(2))
                user.append(ju)
                E.append(tmpl[int(rng.integers(12))] + rng.normal(scale=0.1, size=dim))
                fam.append(fid)
        kind[fid] = "job"
    o = np.argsort(ts, kind="stable")
    E = np.array(E)[o]
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return np.array(ts)[o], np.array(user)[o], E, np.array(fam)[o], kind


# ---------- methods ----------

def _components(n, i, j):
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def baseline_gap(ts, user, gap_min, E=None, cos_min=None):
    """Chain each session to the previous session of the same user if it started within gap_min (and, with cos_min,
    if their cosine is at least cos_min)."""
    ii, jj = [], []
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if ts[b] - ts[a] <= gap_min * 60 and (cos_min is None or E[a] @ E[b] >= cos_min):
                ii.append(a)
                jj.append(b)
    return _components(len(ts), ii, jj)


def baseline_simlink(ts, E, gap_min, cos_min):
    """Link each session to its most similar earlier session (any user) that started within gap_min, if the cosine is
    at least cos_min. The strongest rule without a model."""
    ii, jj = [], []
    for j in range(len(ts)):
        lo = np.searchsorted(ts, ts[j] - gap_min * 60)
        if lo >= j:
            continue
        cs = E[lo:j] @ E[j]
        k = int(np.argmax(cs))
        if cs[k] >= cos_min:
            ii.append(lo + k)
            jj.append(j)
    return _components(len(ts), ii, jj)


def fit_families(ts, user, E, window_s=3600, max_parents=200, iters=30, bg_bw_s=1800, seed=0, n_samples=50,
                 h_min=0.0):
    """EM for the parent-offspring model. Returns (map_family, consensus_family, params, rho_bg)."""
    n = len(ts)
    cand = []                                               # candidate parents within the window (most recent first)
    for j in range(n):
        lo = np.searchsorted(ts, ts[j] - window_s)
        p = np.arange(lo, j)[::-1][:max_parents]
        cand.append(p)
    tau, h, beta, kappa = 300.0, 0.1, 0.1, 0.5
    phi = np.full(n, 0.5)
    T = ts.max() - ts.min() + 1
    users = np.unique(user)
    for _ in range(iters):
        mu = np.zeros(n)                                    # per-user background, time-varying, weighted by phi
        for u in users:
            m = user == u
            d = (ts[m][:, None] - ts[m][None, :]) / bg_bw_s
            mu[m] = (np.exp(-0.5 * d ** 2) * phi[m][None, :]).sum(1) / (np.sqrt(2 * np.pi) * bg_bw_s)
        mu = np.maximum(mu, 1e-9)
        rho, sdt, s1c, scr, stot = [], 0.0, 0.0, 0.0, 0.0
        new_phi = np.empty(n)
        for j in range(n):
            p = cand[j]
            if len(p) == 0:
                new_phi[j] = 1.0
                rho.append(np.zeros(0))
                continue
            dt = ts[j] - ts[p]
            cs = E[p] @ E[j]
            same = user[p] == user[j]
            w = kappa * np.exp(-dt / tau) / tau * np.exp(-(1 - cs) / h) / h * np.where(same, 1.0, beta)
            z = mu[j] + w.sum()
            r = w / z
            new_phi[j] = mu[j] / z
            rho.append(r)
            sdt += (r * dt).sum()
            s1c += (r * (1 - cs)).sum()
            scr += r[~same].sum()
            stot += r.sum()
        phi = new_phi
        if stot < 1e-9:
            break
        tau = max(sdt / stot, 1.0)
        h = max(s1c / stot, 1e-3, h_min)
        beta = float(np.clip(scr / stot / max(1e-9, 1 - scr / stot), 1e-4, 1.0))
        kappa = stot / n
    # most likely parent per session
    ii, jj = [], []
    for j in range(n):
        if len(rho[j]) and rho[j].max() > phi[j]:
            ii.append(int(cand[j][int(np.argmax(rho[j]))]))
            jj.append(j)
    fam_map = _components(n, ii, jj)
    # consensus over sampled parent trees: keep links sampled in > half of the draws
    rng = np.random.default_rng(seed)
    cnt = {}
    for _ in range(n_samples):
        for j in range(n):
            if not len(rho[j]):
                continue
            u = rng.random()
            if u < phi[j]:
                continue
            k = int(np.searchsorted(np.cumsum(rho[j]), u - phi[j]))
            k = min(k, len(rho[j]) - 1)
            key = (int(cand[j][k]), j)
            cnt[key] = cnt.get(key, 0) + 1
    keep = [k for k, c in cnt.items() if c > n_samples / 2]
    fam_cons = _components(n, [a for a, _ in keep], [b for _, b in keep])
    return fam_map, fam_cons, dict(tau=tau, h=h, beta=beta, kappa=kappa), phi


# ---------- scoring ----------

def pair_scores(truth, pred, mask=None):
    """Pairwise precision/recall over co-family pairs; truth -1 = single (never a positive pair)."""
    t = np.where(truth < 0, -(np.arange(len(truth)) + 1), truth)
    if mask is not None:
        t, pred = t[mask], pred[mask]
    def pairs(lab):
        out = 0
        _, c = np.unique(lab, return_counts=True)
        return int((c * (c - 1) // 2).sum())
    tp = pairs(np.array([f"{a}|{b}" for a, b in zip(t, pred)]))
    pt, pp = pairs(t), pairs(pred)
    prec = tp / pp if pp else 1.0
    rec = tp / pt if pt else 1.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "ari": round(float(adjusted_rand_score(t, pred)), 3)}


def score_stream(fam, kind, pred):
    """Families scored without the job's sessions (900 near-identical sessions would swamp the pair counts); the job
    is scored on its own: how many families it lands in and how many other sessions join them."""
    job = [f for f, v in kind.items() if v == "job"]
    nj = ~np.isin(fam, job)
    out = {"families": pair_scores(fam, pred, nj)}
    for k in ("within", "cross"):
        ids = [f for f, v in kind.items() if v == k]
        m = np.isin(fam, ids)
        # recall on that kind's pairs; precision measured against everything those sessions were merged with
        tm = np.where(m, fam, -1)
        sc = pair_scores(tm, pred, nj)
        out[k + "_recall"] = sc["recall"]
    if job:
        jm = fam == job[0]
        out["job_families"] = int(len(set(pred[jm])))
        out["job_absorbed_others"] = int(np.isin(pred[~jm], np.unique(pred[jm])).sum())
    return out
