"""XIII-C Workstream C1 (re-verification before new neural tests).  G = 3, C = 2, K = 3, Q = 6, alpha = 0.8, shared copying with prob. beta.
 V1  xii_bayes.posterior_shared vs an INDEPENDENT direct enumeration written from genlib.sample's generative semantics (role, Y_q, per-question
     copy event and copied value enumerated explicitly; no shared code beyond the reports X).
 V2  sufficiency of m: posterior_shared depends on X only through (per-group self-agreement pattern, values at q, m_H for the three pairs),
     m_H(q) = #{q' != q : all four sources of the pair equal}.  Proof in JOINT_RELATIONAL_FEATURE_AND_REPAIR.md; checked here on pairs of
     episodes that share those statistics but differ in raw values (constructed by value relabellings that keep m).
 V3  beta = 1 local sufficiency (XIII-B proposition) re-checked: on-support, posterior_shared is a function of (self-agreement bits, values at q).
 V4  LOCALQ / FULL accuracy gap at beta = 0.6 (XIII-B: 3.6 points) re-computed on a fresh stream.
 V5  J-pair validity: relabelled episodes have positive generator probability and keep every per-group statistic.
-> logs/c_math.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); out = {}; K, Q, al = 3, 6, 0.8
def direct(X, q, beta):
    """p(Y_q | X) by explicit enumeration of: reliable group g (the other two collude), Y_q' for every question, copy event and copied value."""
    a = X.reshape(6, 3, 2); post = np.zeros(K)
    for g in range(3):
        H = [h for h in range(3) if h != g]; tot = np.ones(K)
        for p in range(Q):
            # colluder pair H at question p: sum over copy event / copied value
            srcs = [a[p, h, s] for h in H for s in range(2)]
            col = beta * sum(np.prod([1.0 if v == c else 0.0 for v in srcs]) for c in range(K)) / K + (1 - beta) * (1 / K) ** 4
            def rel(y): return np.prod([al if a[p, g, s] == y else (1 - al) / (K - 1) for s in range(2)])
            if p == q: tot = tot * np.array([rel(y) for y in range(K)]) * col / K
            else: tot = tot * (sum(rel(y) for y in range(K)) / K) * col
        post += tot / 3
    return post / post.sum()
rng = np.random.default_rng(87_000_000)
for beta in (0.6, 1.0):
    E2 = GL.Env(3, 2, beta=beta); d = GL.sample(E2, 400, rng); X = d["X"]; q = rng.integers(0, 6, 400)
    PS = XB.posterior_shared(E2, X, q); PD = np.array([direct(X[j], q[j], beta) for j in range(400)])
    out[f"V1_beta{beta}"] = dict(max_abs_diff=float(np.nanmax(np.abs(PS - PD))), n=400)
def mstat(X, q):
    a = X.reshape(len(X), 6, 3, 2); i = np.arange(len(X)); r = []
    for g, h in ((0, 1), (0, 2), (1, 2)):
        e = (a[:, :, g, 0] == a[:, :, g, 1]) & (a[:, :, h, 0] == a[:, :, h, 1]) & (a[:, :, g, 0] == a[:, :, h, 0]); r.append(e.sum(1) - e[i, q])
    return np.stack(r, 1)
E2 = GL.Env(3, 2, beta=0.6); d = GL.sample(E2, 20000, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
# V2: random relabellings of single (group, question q' != q) value pairs that keep self-agreement; keep only those that keep m
perms = [np.array(p) for p in itertools.permutations(range(3))]; diffs = []; diffs_m = []
for j in range(4000):
    x = X[j].copy(); p = (q[j] + 1 + rng.integers(5)) % 6; g = rng.integers(3); x[p, 2 * g:2 * g + 2] = perms[rng.integers(6)][x[p, 2 * g:2 * g + 2]]
    P0 = XB.posterior_shared(E2, X[j:j + 1], q[j:j + 1]); P1 = XB.posterior_shared(E2, x[None], q[j:j + 1])
    (diffs if (mstat(x[None], q[j:j + 1]) == mstat(X[j:j + 1], q[j:j + 1])).all() else diffs_m).append(float(np.abs(P0 - P1).max()))
out["V2_sufficiency_of_m"] = dict(n_m_kept=len(diffs), max_abs_diff_m_kept=max(diffs), n_m_changed=len(diffs_m), mean_abs_diff_m_changed=float(np.mean(diffs_m)))
# V3 beta = 1: local function check -- group episodes by (self-agreement bits at all questions, values at q) and check posterior constancy
E1 = GL.Env(3, 2, beta=1.0); d1 = GL.sample(E1, 30000, rng); X1 = d1["X"]; q1 = rng.integers(0, 6, len(X1)); P1 = XB.posterior_shared(E1, X1, q1)
a1 = X1.reshape(len(X1), 6, 3, 2); keyf = lambda j: (tuple((a1[j, :, :, 0] == a1[j, :, :, 1]).ravel()), tuple(a1[j, q1[j]].ravel()))
groups = {}
for j in range(len(X1)): groups.setdefault(keyf(j), []).append(j)
spread = max((float(np.ptp(P1[v], 0).max()) for v in groups.values() if len(v) > 1), default=0.0)
out["V3_beta1_local_sufficiency"] = dict(n_cells_multi=sum(len(v) > 1 for v in groups.values()), max_within_cell_spread=spread)
# V4 LOCALQ vs FULL at beta = 0.6 (LOCALQ: posterior of the reports with other questions reduced to self-agreement bits, i.e. the exact
# posterior after randomising cross-group value identities at q' != q -- computed by averaging the shared posterior over relabellings)
d4 = GL.sample(E2, 3000, rng); X4, Y4 = d4["X"], d4["Y"]; q4 = rng.integers(0, 6, 3000); PF = XB.posterior_shared(E2, X4, q4)
acc_full = float((PF.argmax(-1) == Y4[np.arange(3000), q4]).mean())
def logmarg(Xx):
    """log p(X) under the generator (sum over the 3 roles; no noise group when G = 3, C = 2)."""
    rel, _, _ = GL._group_lik(E2, Xx); relm = rel.mean(-1); a_ = Xx.reshape(len(Xx), 6, 3, 2); terms = []
    for g in range(3):
        H = [h for h in range(3) if h != g]; srcs = a_[:, :, H, :].reshape(len(Xx), 6, 4); alleq = (srcs == srcs[..., :1]).all(-1)
        colH = E2.beta * alleq / K + (1 - E2.beta) / K ** 4; terms.append(np.log(relm[:, :, g]).sum(1) + np.log(colH).sum(1) - np.log(3))
    T = np.stack(terms); mx = T.max(0); return mx + np.log(np.exp(T - mx).sum(0))
PL = np.zeros_like(PF); R = 48; LW = []; PS_ = []
for r_ in range(R):
    Xr = X4.copy()
    for g in range(3):
        pm = np.stack(perms)[rng.integers(0, 6, (3000, 6))]                             # (3000,6,3): per episode, per question value permutation of group g
        for p in range(6):
            m_ = q4 != p; v = Xr[:, p, 2 * g:2 * g + 2]; Xr[m_, p, 2 * g:2 * g + 2] = np.take_along_axis(pm[m_, p], v[m_], 1)
    LW.append(logmarg(Xr)); PS_.append(XB.posterior_shared(E2, Xr, q4))
LW = np.stack(LW); w = np.exp(LW - LW.max(0)); w = w / w.sum(0); PL = (w[..., None] * np.stack(PS_)).sum(0)
acc_loc = float((PL.argmax(-1) == Y4[np.arange(3000), q4]).mean())
out["V4_localq_gap_beta0.6"] = dict(acc_full=acc_full, acc_localq_mc=acc_loc, gap=acc_full - acc_loc, note="LOCALQ = posterior of the orbit of X under per-group, per-question value relabellings at q != q (self-agreement and values at q kept); importance-weighted Monte-Carlo over 48 relabellings (weights p(x'))")
# V5 J-pairs
dj = GL.sample(E2, 2000, rng); Xj = dj["X"]; qj = rng.integers(0, 6, 2000); hj = rng.integers(0, 3, 2000); Xj2 = Xj.copy()
nonid = [p for p in perms if tuple(p) != (0, 1, 2)]
for j in range(2000):
    pm = nonid[rng.integers(5)]; cols = [2 * hj[j], 2 * hj[j] + 1]; others = [t for t in range(6) if t != qj[j]]; Xj2[j][np.ix_(others, cols)] = pm[Xj[j][np.ix_(others, cols)]]
aa, bb = Xj.reshape(2000, 6, 3, 2), Xj2.reshape(2000, 6, 3, 2)
same_self = bool(((aa[..., 0] == aa[..., 1]) == (bb[..., 0] == bb[..., 1])).all()); same_q = bool((aa[np.arange(2000), qj] == bb[np.arange(2000), qj]).all())
P2 = XB.posterior_shared(E2, Xj2, qj); out["V5_Jpairs"] = dict(self_agreement_kept=same_self, values_at_q_kept=same_q, finite_posterior_frac=float(np.isfinite(P2).all(1).mean()),
    frac_m_changed=float((mstat(Xj2, qj) != mstat(Xj, qj)).any(1).mean()))
json.dump(out, open(os.path.join(HERE, "..", "logs", "c_math.json"), "w"), indent=1); print(json.dumps(out, indent=1))
