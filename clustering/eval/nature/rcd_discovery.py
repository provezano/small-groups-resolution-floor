"""
Rare-category discovery with the exact Beta scan (stage 10, rare-category discovery branch).

Setting. An analyst labels one point at a time and wants to see at least one point of every rare class with as few
labels as possible. A method is a query policy: it picks the next point, possibly using the labels seen so far.

Our method.
1. `scan`: the CTS scan (collapse of near-duplicate families, then the exact Beta test for every point and size
   m = 2..15, Bonferroni over all tests). It calls the shared `tiny_methods.cts3` (dim_est="local", scale_min=True,
   border=True, return_cands=True), with the stage-11 fringe fix by default (fix=None gives the stage-10 v5 scan).
   Unlike the labels it also returns every significant candidate set, not only the greedy non-overlapping groups.
2. Queries: first the greedy groups (best score first; we query the member that is the candidate's centre), then the
   "cover" phase: every significant candidate that holds no labelled point yet gets its centre queried.
3. Stopping rule: stop when every significant candidate holds a labelled point. Because the significant set is fixed
   before any label is seen, a rare class can be unfound at that moment only if no significant candidate lies inside
   it (proof in the accompanying paper). Any query order keeps this.
4. Fallback for loose classes: after (or between) scan queries, an outlier order (LOF) that skips points close to a
   labelled point.

Baselines (re-implemented from the published pseudo-code of the cited papers):
random; LOF; isolation forest; NNDB (He & Carbonell 2007, binary case, run until the budget ends); MALICE (He &
Carbonell 2007 / He 2010 thesis Alg. 3, true priors); GRADE and GRADE-LI (He, Liu & Lawrence 2008, thesis Alg. 5 and
6, true priors / true upper bound); SEDER (He & Carbonell 2009, thesis Alg. 4, prior-free); Interleave (Pelleg &
Moore 2004, mixture with one component per class, without their semi-supervised refit); an HMS-style hierarchical
mean shift (after Vatturi & Wong 2009; our reading, the paper's exact ranking rule was not available).
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

from .tiny_methods import cts3


# ----------------------------------------------------------------------------------------------------------------- scan
def scan(Z, alpha=0.05, max_m=15, k_bg=10, n_loc=30, fix="joint"):
    """The shared CTS scan (tiny_methods.cts3 with dim_est="local", scale_min=True, border=True, Bonferroni), with
    the fringe fix by default (fix=None: v5 as in stage 10). Returns a dict with
    cands: list of (score = -log p, centre (orig index), members (orig indices, sorted tuple)) for every significant
           candidate, best first; groups: the greedy non-overlapping subset (same rule as cts3);
    n_tests, thr (= -log(alpha / n_tests)), fams (collapsed families), R, dim."""
    if n_loc != 30:
        raise ValueError("the shared scan uses n_loc = 30")
    out = cts3(Z, alpha=alpha, max_m=max_m, k_bg=k_bg, dim_est="local", scale_min=True, border=True, fix=fix,
               base=-np.ones(len(Z), int), return_cands=True)
    out.pop("labels")
    return out


# ------------------------------------------------------------------------------------------------------ query harness
class Done(Exception):
    pass


class Oracle:
    """Answers label queries, records the order, ends the run at the budget or when every target class is seen."""

    def __init__(self, y, budget, targets):
        self.y, self.B, self.targets = y, budget, set(targets)
        self.asked, self.labels, self.seen = [], {}, set()

    def __call__(self, i):
        i = int(i)
        if i in self.labels:
            return self.labels[i]          # repeated questions are free and never count
        if len(self.asked) >= self.B or self.targets <= self.seen:
            raise Done
        self.asked.append(i); lab = self.y[i]; self.labels[i] = lab; self.seen.add(lab)
        if len(self.asked) >= self.B or self.targets <= self.seen:
            raise Done
        return lab


def run_policy(policy, Z, y, budget, targets, **kw):
    o = Oracle(y, budget, targets)
    try:
        policy(Z, o, **kw)
    except Done:
        pass
    return o.asked


def first_hits(asked, y):
    out = {}
    for t, i in enumerate(asked):
        out.setdefault(y[i], t + 1)
    return out


# -------------------------------------------------------------------------------------------------- simple orderings
def _lof_scores(Z, k=20):
    from sklearn.neighbors import LocalOutlierFactor
    return -LocalOutlierFactor(n_neighbors=min(k, len(Z) - 1)).fit(Z).negative_outlier_factor_


def _iforest_scores(Z, seed=0):
    from sklearn.ensemble import IsolationForest
    return -IsolationForest(random_state=seed).fit(Z).score_samples(Z)


def pol_random(Z, ask, seed=0):
    for i in np.random.default_rng(seed).permutation(len(Z)):
        ask(i)


def pol_order(Z, ask, scores):
    for i in np.argsort(-scores, kind="stable"):
        ask(i)


def pol_lof(Z, ask):
    pol_order(Z, ask, _lof_scores(Z))


def pol_iforest(Z, ask, seed=0):
    pol_order(Z, ask, _iforest_scores(Z, seed))


def excl_order(Z, scores, k=10):
    """An outlier order that skips any point within its own k-distance of a labelled point (prior-free analogue of
    MALICE's exclusion). Returns a generator factory: next index given the labelled set."""
    dk = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[0][:, -1]
    return np.argsort(-scores, kind="stable"), dk


class ExclQueue:
    def __init__(self, Z, scores, k=10, exclude=True):
        self.Z = Z
        self.order, self.dk = excl_order(Z, scores, k)
        self.pos, self.exclude = 0, exclude
        self.lab_pts = []

    def note(self, i):
        self.lab_pts.append(int(i))

    def next(self, labelled):
        while self.pos < len(self.order):
            i = int(self.order[self.pos]); self.pos += 1
            if i in labelled:
                continue
            if self.exclude and self.lab_pts:
                d = np.linalg.norm(self.Z[self.lab_pts] - self.Z[i], axis=1).min()
                if d <= self.dk[i]:
                    continue
            return i
        return None


def pol_lof_excl(Z, ask, k=10):
    q = ExclQueue(Z, _lof_scores(Z), k)
    while True:
        i = q.next(ask.labels)
        if i is None:
            break
        ask(i); q.note(i)
    pol_lof(Z, ask)       # anything left, plain order


# --------------------------------------------------------------------------------------- He & Carbonell family (priors)
def _sorted_dist(Z):
    from scipy.spatial.distance import cdist
    D = cdist(Z, Z).astype(np.float32)
    order = np.argsort(D, 1, kind="stable").astype(np.int32)
    return D, order, np.take_along_axis(D, order, 1)


def _counts_within(sortedD, rad):
    """|NN(x_i, rad)| for every i (rad scalar or per row), self included."""
    rad = np.broadcast_to(np.asarray(rad, np.float32), (sortedD.shape[0],))
    return np.array([np.searchsorted(sortedD[i], rad[i], side="right") for i in range(len(sortedD))])


def _max_drop(sortedD, order, n_c, rad):
    """s_i = max over x_k in NN(x_i, rad) of (n_i - n_k) = n_i - min_k n_k (the ball includes x_i)."""
    cnt = np.maximum(_counts_within(sortedD, rad), 1)
    pm = np.minimum.accumulate(n_c[order], axis=1)
    return n_c - pm[np.arange(len(n_c)), cnt - 1]


def pol_nndb(Z, ask, prior, self_excl=True):
    """NNDB (thesis Alg. 1), binary case with p2 = prior; after each query the loop simply continues (no break), with
    MALICE's exclusion of the queried point's r' ball so that the run does not stay in one class."""
    n = len(Z)
    D, order, sD = _sorted_dist(Z)
    K = max(2, int(round(n * prior)))
    r = float(sD[:, K if self_excl else K - 1].min())      # K-th other point (self_excl) or K-th point counting itself
    n_c = _counts_within(sD, r)
    excl = np.zeros(n, bool)
    for t in range(1, n + 1):
        s = _max_drop(sD, order, n_c, t * r).astype(float)
        s[excl] = -np.inf
        i = int(np.argmax(s))
        if not np.isfinite(s[i]):
            break
        ask(i); excl[i] = True; excl[D[i] <= r] = True


def pol_malice(Z, ask, priors, cls_order="dec", self_excl=True):
    """MALICE (thesis Alg. 3). priors: dict class -> share; the largest class is class 1 (never sought); the others
    are sought in decreasing order of prior."""
    n = len(Z)
    D, order, sD = _sorted_dist(Z)
    cls = sorted(priors, key=lambda c: -priors[c])
    if cls_order == "inc":              # the pseudo-code says c = 2..m without fixing the numbering; we run both
        cls = [cls[0]] + cls[1:][::-1]
    rr = {}
    for c in cls[1:]:
        K = max(2, int(round(n * priors[c])))
        rr[c] = float(sD[:, min(n - 1, K if self_excl else K - 1)].min())
    r1 = max(rr.values())
    rad_of = lambda c: rr.get(c, r1)
    excl = np.zeros(n, bool)
    found = set()
    for c in cls[1:]:
        if c in found:
            continue
        n_c = _counts_within(sD, rr[c])
        pm = np.minimum.accumulate(n_c[order], axis=1)
        t = 2
        while t <= n:
            cnt = np.maximum(_counts_within(sD, t * rr[c]), 1)
            s = (n_c - pm[np.arange(n), cnt - 1]).astype(float)
            s[excl] = -np.inf
            i = int(np.argmax(s))
            if not np.isfinite(s[i]):
                break
            lab = ask(i)
            excl[i] = True; excl[D[i] <= rad_of(lab)] = True
            if lab == c:
                found.add(c); break
            if lab not in found and lab != cls[0]:
                t -= 1
            found.add(lab)
            t += 1


def _global_similarity(Z, K, alpha):
    from scipy.spatial.distance import cdist
    D2 = cdist(Z, Z, "sqeuclidean")
    sd = np.sqrt(np.sort(D2, 1)[:, min(K, len(Z) - 1)])
    sigma = max(float(sd.min()), 1e-12)
    W = np.exp(-D2 / (2 * sigma ** 2)); np.fill_diagonal(W, 0.0)
    d = W.sum(1); d[d <= 0] = 1e-300
    W = W / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
    A = np.linalg.inv(np.eye(len(Z)) - alpha * W)
    return A


def _grade_core(Z, ask, A, a_of, seek, a_excl):
    """Shared loop of GRADE / GRADE-LI on 'distance' = -A (sorted rows)."""
    n = len(Z)
    order = np.argsort(-A, 1, kind="stable").astype(np.int32)
    sA = -np.take_along_axis(A, order, 1)                  # ascending 'distance'
    excl = np.zeros(n, bool)
    found = set()
    for c, a in seek:
        if c is not None and c in found:
            continue
        n_c = _counts_within(sA, -a)
        pm = np.minimum.accumulate(n_c[order], axis=1)
        t = 2
        while t <= n:
            cnt = np.maximum(_counts_within(sA, -a / t), 1)
            s = (n_c - pm[np.arange(n), cnt - 1]).astype(float)
            s[excl] = -np.inf
            i = int(np.argmax(s))
            if not np.isfinite(s[i]):
                return
            lab = ask(i)
            excl[i] = True; excl[A[i] >= a_excl(lab)] = True
            found.add(lab)
            if c is not None and lab == c:
                break
            t += 1


def pol_grade(Z, ask, priors, alpha=0.99, self_excl=True):
    """GRADE (thesis Alg. 5) with the true priors."""
    n = len(Z)
    cls = sorted(priors, key=lambda c: -priors[c])
    K = max(2, int(round(max(n * priors[c] for c in cls[1:]))))
    A = _global_similarity(Z, K, alpha)
    srt = -np.sort(-A, 1)
    a = {c: float(srt[:, min(n - 1, max(2, int(round(n * priors[c]))) - (0 if self_excl else 1))].max()) for c in cls[1:]}
    a1 = min(a.values())
    _grade_core(Z, ask, A, None, [(c, a[c]) for c in cls[1:]], lambda lab: a.get(lab, a1))


def pol_grade_li(Z, ask, p_upper, alpha=0.99, self_excl=True):
    """GRADE-LI (thesis Alg. 6): only an upper bound p on the minority-class shares."""
    n = len(Z)
    K = max(2, int(round(n * p_upper)))
    A = _global_similarity(Z, K, alpha)
    a = float((-np.sort(-A, 1))[:, min(n - 1, K if self_excl else K - 1)].max())
    _grade_core(Z, ask, A, None, [(None, a)], lambda lab: a)


# ---------------------------------------------------------------------------------------------------- SEDER (prior-free)
def _cv_bandwidth(x, grid=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0), max_n=2000, seed=0):
    """Leave-one-out likelihood cross-validation for a 1-D Gaussian kernel, grid of multiples of the normal-reference
    width. Not the default: on page blocks (discretised features, dithered ties) it picks tiny widths and SEDER then
    needs 465 queries to see all 5 classes, against 58 with the normal-reference width (thesis: 36)."""
    rng = np.random.default_rng(seed)
    xs = x if len(x) <= max_n else rng.choice(x, max_n, replace=False)
    sd = xs.std() + 1e-12
    h0 = 1.06 * sd * len(xs) ** (-0.2)
    D2 = (xs[:, None] - xs[None, :]) ** 2
    best, bh = -np.inf, h0
    for g in grid:
        h = g * h0
        K = np.exp(-D2 / (2 * h * h)); np.fill_diagonal(K, 0.0)
        f = K.sum(1) / ((len(xs) - 1) * np.sqrt(2 * np.pi) * h)
        ll = np.log(np.maximum(f, 1e-300)).sum()
        if ll > best:
            best, bh = ll, h
    return bh


def _ref_bandwidth(x):
    """Normal-reference width 1.06 sd n^(-1/5) (Scott 1992); the SEDER default here."""
    return 1.06 * (x.std() + 1e-12) * len(x) ** (-0.2)


def seder_scores(Z, chunk=None, bandwidth="ref"):
    """SEDER (thesis Alg. 4, eqs. 3.10-3.11), in log space so that it works in 50 dimensions."""
    X = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    n, d = X.shape
    bw = _ref_bandwidth if bandwidth == "ref" else _cv_bandwidth
    sig = np.array([bw(X[:, j]) for j in range(d)])
    b = np.ones(d)
    for j in range(d):
        x = X[:, j]; h = sig[j]
        Kx = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * h * h))
        Aj = np.mean((Kx * x[None, :] ** 2).sum(1) / Kx.sum(1))
        B, C = h * h, np.mean(x ** 2)
        b[j] = (-B + np.sqrt(B * B + 4 * Aj * C)) / (2 * Aj)
    var = sig ** 2 * b
    chunk = chunk or max(10, int(2e7 / (n * d)))
    logs = np.empty(n)
    for s0 in range(0, n, chunk):
        xk = X[s0:s0 + chunk]                               # c x d
        diff = xk[:, None, :] - b[None, None, :] * X[None, :, :]     # c x n x d
        logD = -0.5 * (diff ** 2 / var).sum(2) - 0.5 * np.log(2 * np.pi * var).sum() - np.log(n)
        mx = logD.max(1, keepdims=True)
        w = np.exp(logD - mx)                               # c x n
        g = np.einsum("cn,cnd->cd", w, diff) / var          # gradient / exp(mx), up to sign
        logs[s0:s0 + chunk] = mx[:, 0] + np.log(np.linalg.norm(g, axis=1) + 1e-300)
    return logs, sig


def pol_seder(Z, ask):
    X = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    logs, sig = seder_scores(Z)
    lab_pts = []
    for i in np.argsort(-logs, kind="stable"):
        if lab_pts and np.any(np.all(np.abs(X[lab_pts] - X[i]) <= 3 * sig, axis=1)):
            continue
        ask(i); lab_pts.append(int(i))
    for i in np.argsort(-logs, kind="stable"):      # budget left: plain order
        ask(i)


# ------------------------------------------------------------------------------------------- Pelleg & Moore Interleave
def pol_interleave(Z, ask, n_comp, seed=0):
    from sklearn.mixture import GaussianMixture
    cov = "full" if Z.shape[1] <= 20 else "diag"
    gm = GaussianMixture(n_components=min(n_comp, len(Z) // 2), covariance_type=cov, reg_covar=1e-3,
                         random_state=seed, max_iter=200).fit(Z)
    resp = gm.predict_proba(Z)
    own = resp.argmax(1)
    from scipy.stats import multivariate_normal
    queues = []
    for c in range(gm.n_components):
        mem = np.flatnonzero(own == c)
        if not len(mem):
            continue
        if cov == "full":
            ll = multivariate_normal(gm.means_[c], gm.covariances_[c], allow_singular=True).logpdf(Z[mem])
        else:
            v = gm.covariances_[c]; ll = -0.5 * (((Z[mem] - gm.means_[c]) ** 2) / v + np.log(2 * np.pi * v)).sum(1)
        queues.append(list(mem[np.argsort(ll)]))
    while any(queues):
        for q in queues:
            while q and q[0] in ask.labels:
                q.pop(0)
            if q:
                ask(q.pop(0))


# ------------------------------------------------------------------------------------- HMS-style hierarchical mean shift
def hms_clusters(Z, levels=30, growth=1.25, iters=15):
    """Mean shift on the previous level's modes (weighted by size), bandwidth growing by `growth` per level; modes
    closer than h/2 merge. Returns clusters as (birth level, death level, members, centre)."""
    from scipy.spatial.distance import cdist
    n = len(Z)
    d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    h = float(np.median(d1[d1 > 0])) if (d1 > 0).any() else 1.0
    C = Z.copy(); w = np.ones(n); mem = [[i] for i in range(n)]; born = [0] * n
    out = []
    for lev in range(1, levels + 1):
        h *= growth
        Y = C.copy()
        for _ in range(iters):
            D2 = cdist(Y, C, "sqeuclidean")
            K = np.exp(-D2 / (2 * h * h)) * w[None, :]
            Y = K @ C / K.sum(1, keepdims=True)
        G = NearestNeighbors(radius=h / 2).fit(Y).radius_neighbors_graph(Y, mode="connectivity")
        lab = connected_components(G, directed=False)[1]
        if lab.max() == len(C) - 1:
            continue
        newC, neww, newmem, newborn = [], [], [], []
        for g in range(lab.max() + 1):
            ix = np.flatnonzero(lab == g)
            if len(ix) == 1:
                j = ix[0]; newC.append(Y[j]); neww.append(w[j]); newmem.append(mem[j]); newborn.append(born[j])
                continue
            for j in ix:
                out.append((born[j], lev, mem[j], C[j]))
            ww = w[ix]
            newC.append((Y[ix] * ww[:, None]).sum(0) / ww.sum()); neww.append(ww.sum())
            newmem.append(sum((mem[j] for j in ix), [])); newborn.append(lev)
        C, w, mem, born = np.array(newC), np.array(neww), newmem, newborn
        if len(C) == 1:
            break
    for j in range(len(C)):
        out.append((born[j], levels + 1, mem[j], C[j]))
    return out


def pol_hms(Z, ask, max_frac=0.05):
    cl = hms_clusters(Z)
    n = len(Z)
    cl = [c for c in cl if len(c[2]) <= max(15, max_frac * n)]
    cl.sort(key=lambda c: (-(c[1] - c[0]), len(c[2])))
    for b, dth, mem, ctr in cl:
        mem = np.asarray(mem)
        if any(int(i) in ask.labels for i in mem):
            continue
        ask(mem[np.argmin(np.linalg.norm(Z[mem] - ctr, axis=1))])
    pol_lof(Z, ask)


# ---------------------------------------------------------------------------------------------------- our policies
def scan_queue(sc, Z, cover=True, medoid="centre"):
    """The scan's query list: the greedy groups (best first), then (cover=True) the centres of significant
    candidates that would still hold no labelled point. Returns a list of (kind, candidate) to be resolved lazily."""
    items = [("group", g) for g in sc["groups"]]
    if cover:
        items += [("cover", c) for c in sc["cands"]]
    return items


def _pick(Z, cand, medoid):
    s, c, mem = cand
    if medoid == "centre":
        return c
    mem = np.asarray(mem)
    return int(mem[np.argmin(np.linalg.norm(Z[mem] - Z[mem].mean(0), axis=1))])


def pol_scan(Z, ask, sc=None, alpha=0.05, cover=True, fallback="lof_excl", mix=0, medoid="centre", stop_log=None,
             fam_first=False):
    """Scan first. mix = 0: all scan queries, then the fallback. mix = r > 0: after every r scan queries, one fallback
    query (interleaving keeps the guarantee: the stopping rule only needs every significant candidate hit).
    stop_log (dict) receives 'tau': the number of queries made when the stopping rule fires."""
    if sc is None:
        sc = scan(Z, alpha)
    items = scan_queue(sc, Z, cover)
    fb = ExclQueue(Z, _lof_scores(Z) if fallback.startswith("lof") else _iforest_scores(Z),
                   exclude=fallback.endswith("excl")) if fallback else None
    if fam_first:
        for f in sc["fams"]:
            i = int(f[np.argmin(np.linalg.norm(Z[f] - Z[f].mean(0), axis=1))])
            ask(i)
            if fb: fb.note(i)
    k = 0
    for kind, cand in items:
        if any(m in ask.labels for m in cand[2]):
            continue                         # already hit: nothing to ask
        i = _pick(Z, cand, medoid)
        ask(i); k += 1
        if fb: fb.note(i)
        if mix and fb and k % mix == 0:
            j = fb.next(ask.labels)
            if j is not None:
                ask(j); fb.note(j)
    if stop_log is not None:
        stop_log["tau"] = len(ask.asked)
    if not fallback:
        return
    if fb:
        while True:
            j = fb.next(ask.labels)
            if j is None:
                break
            ask(j); fb.note(j)
    pol_lof(Z, ask)


def pol_adaptive(Z, ask, sc=None, alpha=0.05, patience=3, stop_log=None):
    """Adaptive: run scan queries while they keep paying; after `patience` scan queries in a row that found no new
    class, alternate one fallback query (LOF with exclusion) with one scan query until the scan list is done; a new
    class from the scan resets the counter. The guarantee needs only that every scan item is eventually asked."""
    if sc is None:
        sc = scan(Z, alpha)
    items = [it for it in scan_queue(sc, Z, True)]
    fb = ExclQueue(Z, _lof_scores(Z))
    seen = set(); miss = 0; pos = 0; turn_fb = False
    while pos < len(items):
        if miss >= patience and turn_fb:
            j = fb.next(ask.labels)
            turn_fb = False
            if j is not None:
                lab = ask(j); fb.note(j)
                if lab not in seen:
                    seen.add(lab)
                continue
        kind, cand = items[pos]; pos += 1
        if any(m in ask.labels for m in cand[2]):
            continue
        i = _pick(Z, cand, "centre")
        lab = ask(i); fb.note(i); turn_fb = True
        if lab not in seen:
            seen.add(lab); miss = 0
        else:
            miss += 1
    if stop_log is not None:
        stop_log["tau"] = len(ask.asked)
    while True:
        j = fb.next(ask.labels)
        if j is None:
            break
        ask(j); fb.note(j)
    pol_lof(Z, ask)


# --------------------------------------------------------------------------------------------- stopping-rule audit
def audit(sc, y, rare, asked_at_tau):
    """For the theorem: per rare class, does a significant candidate lie entirely inside it (detectable), does a greedy
    group have its centre in it, and was it found by the time the rule stopped."""
    labs = [y[i] for i in asked_at_tau]
    out = {}
    for c in rare:
        pure = any(all(y[m] == c for m in mem) for s, ctr, mem in sc["cands"])
        grp = any(y[ctr] == c for s, ctr, mem in sc["groups"])
        out[str(c)] = {"detectable": bool(pure), "owns_group": bool(grp), "found_at_stop": bool(c in labs)}
    return out


# ---------------------------------------------------------------------------------------------- portfolio (adaptive)
class ScanArm:
    """Proposes the centre of the next significant candidate that holds no labelled point: greedy groups first
    (ordered by `key`: 'score' or 'lof' = LOF of the centre, most outlying first), then every other significant
    candidate by score (the cover phase). Exhausted = the stopping rule has fired."""

    def __init__(self, Z, sc, key="lof", lof=None):
        G = list(sc["groups"])
        if key == "lof" and G:
            lof = _lof_scores(Z) if lof is None else lof
            G = [G[j] for j in np.argsort([-lof[c] for s, c, m in G], kind="stable")]
        self.items = G + list(sc["cands"])
        self.pos = 0

    def propose(self, labelled):
        while self.pos < len(self.items):
            s, c, mem = self.items[self.pos]
            if any(m in labelled for m in mem):
                self.pos += 1; continue
            return c
        return None

    def note(self, i, lab):
        pass


class OrderArm:
    """An outlier order with MALICE-style exclusion: skips points within their own k-distance of a labelled point."""

    def __init__(self, Z, scores, k=10):
        self.q = ExclQueue(Z, scores, k)
        self.buf = None

    def propose(self, labelled):
        if self.buf is not None and self.buf not in labelled:
            lp = self.q.lab_pts
            if not lp or np.linalg.norm(self.q.Z[lp] - self.q.Z[self.buf], axis=1).min() > self.q.dk[self.buf]:
                return self.buf
        self.buf = self.q.next(labelled)
        return self.buf

    def note(self, i, lab):
        self.q.note(i); self.buf = None if self.buf == i else self.buf


class HMSArm:
    def __init__(self, Z, max_frac=0.05):
        cl = hms_clusters(Z)
        n = len(Z)
        cl = [c for c in cl if len(c[2]) <= max(15, max_frac * n)]
        cl.sort(key=lambda c: (-(c[1] - c[0]), len(c[2])))
        self.items = [(np.asarray(m), int(np.asarray(m)[np.argmin(np.linalg.norm(Z[np.asarray(m)] - ctr, axis=1))]))
                      for b, d, m, ctr in cl]
        self.pos = 0

    def propose(self, labelled):
        while self.pos < len(self.items):
            mem, c = self.items[self.pos]
            if any(int(m) in labelled for m in mem):
                self.pos += 1; continue
            return c
        return None

    def note(self, i, lab):
        pass


def pol_portfolio(Z, ask, sc=None, arms=("scan", "lof", "iforest"), rule="ts", seed=0, c=1.0, scan_key="lof",
                  stop_log=None, warm=0):
    """Adaptive portfolio. Every arm proposes its next point; a bandit picks which arm asks. Reward = the label is a
    class not seen before. rule='ts': Thompson sampling with Beta(1 + wins, 1 + losses); the scan arm starts with
    Beta(2, 1) because each of its proposals is a certified density excess. rule='ucb': UCB1 with constant c.
    The stopping rule of the scan is unaffected (any order); stop_log['tau'] records when the scan arm is exhausted."""
    rng = np.random.default_rng(seed)
    if sc is None and "scan" in arms:
        sc = scan(Z)
    lof = _lof_scores(Z)
    A = {}
    for a in arms:
        if a == "scan":
            A[a] = ScanArm(Z, sc, scan_key, lof)
        elif a in ("lof", "lof3"):
            A[a] = OrderArm(Z, lof, 10 if a == "lof" else 3)
        elif a in ("iforest", "iforest3"):
            A[a] = OrderArm(Z, _iforest_scores(Z, seed), 10 if a == "iforest" else 3)
        elif a == "rankavg":
            from scipy.stats import rankdata
            A[a] = OrderArm(Z, rankdata(lof) + rankdata(_iforest_scores(Z, seed)))
        elif a == "hms":
            A[a] = HMSArm(Z)
    win = {a: (1.0 if a == "scan" else 0.0) for a in A}
    pulls = {a: 0 for a in A}
    seen = set(); t = 0; tau_done = False
    fails = 0; warm_on = warm > 0 and "scan" in A      # warm start: the scan arm alone until `warm` misses in a row
    while True:
        props = {a: arm.propose(ask.labels) for a, arm in A.items()}
        live = [a for a in A if props[a] is not None]
        if stop_log is not None and not tau_done and ("scan" not in A or props.get("scan") is None):
            stop_log["tau"] = len(ask.asked); tau_done = True
        if not live:
            break
        if warm_on and props["scan"] is not None and fails < warm:
            draw = {b: float(b == "scan") for b in live}
        elif rule == "seq":                   # fixed priority: the first live arm in `arms` asks
            draw = {b: -arms.index(b) for b in live}
        elif rule == "ts":
            draw = {a: rng.beta(1 + win[a], 1 + pulls[a] - (win[a] - (1.0 if a == "scan" else 0.0))) for a in live}
        else:
            draw = {a: (win[a] + 1) / (pulls[a] + 2) + c * np.sqrt(np.log(t + 2) / (pulls[a] + 1)) for a in live}
        a = max(live, key=lambda b: draw[b])
        i = props[a]
        lab = ask(i); t += 1; pulls[a] += 1
        if lab not in seen:
            seen.add(lab); win[a] += 1
            if a == "scan":
                fails = 0
        elif a == "scan":
            fails += 1
        for arm in A.values():
            arm.note(i, lab)
    pol_lof(Z, ask)
