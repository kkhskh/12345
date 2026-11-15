"""Review 1 (math): per-pair check of the d_das.py E5/E3/QF5/QF4 families (construction copied from d_das.make_families, sampler = genlib
C=1 beta=1, which genlib states is the T4 distribution).  Checks for EVERY pair: on-support (exact likelihood > 0), which eligibility
indicators change, and response distances between rules, including rules NOT considered in the package (A4: n>=4 ignoring q; A3: n=5 OR
(n=4 & agree at q) ... ) to show what the four families do and do not exclude."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np, genlib as GL
np.seterr(all="ignore"); env = GL.Env(3, 1)
def parts(X, q):
    rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); i = np.arange(len(X)); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]; return W, Lg, cnt, a_[..., 0] == a_[..., 1]
RULES = dict(E=lambda c, a: (c == 5) & a, A1=lambda c, a: c == 5, A2=lambda c, a: (c >= 4) & a, A4=lambda c, a: c >= 4,
             SOFT=lambda c, a: ((c == 5) & a) + 0.1 * ((c == 4) & a))
def ind(X, q, w): W, Lg, cnt, ag = parts(X, q); return RULES[w](cnt, ag)
def alg_logp(X, q, w):
    W, Lg, cnt, ag = parts(X, q); e = RULES[w](cnt, ag).astype(float); f = e.sum(1, keepdims=True) - e
    P = np.maximum((W[..., None] * f[..., None] * Lg).sum(1), 1e-30); return np.log(P / P.sum(-1, keepdims=True))
def support(X):  # C=1 beta=1: positive likelihood iff some group self-agrees at all 6 questions
    a = X.reshape(len(X), 6, 3, 2); return (a[..., 0] == a[..., 1]).all(1).any(1)
rng = np.random.default_rng(5); X = GL.sample(env, 30000, rng)["X"]; q = rng.integers(0, 6, len(X)); fam = {k: ([], [], []) for k in ("E5", "E3", "QF5", "QF4")}
a6 = X.reshape(len(X), 6, 3, 2); eq = a6[..., 0] == a6[..., 1]; nper = 600
for i in range(len(X)):
    if min(len(v[2]) for v in fam.values()) >= nper: break
    qq = q[i]; cnt = eq[i].sum(0) - eq[i, qq]; agq = eq[i, qq]; kind = ["E5", "E3", "QF5", "QF4"][rng.integers(4)]
    if len(fam[kind][2]) >= nper: continue
    x = X[i].copy()
    if kind in ("E5", "E3"):
        lo, hi = (4, 5) if kind == "E5" else (3, 4); hs = [h for h in range(3) if agq[h] and cnt[h] in (lo, hi)]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [p for p in range(6) if p != qq]
        if cnt[h] == hi: cand = [p for p in others if eq[i, p, h]]; p = cand[rng.integers(len(cand))]; x[p, 2 * h + 1] = (x[p, 2 * h] + rng.integers(1, 3)) % 3
        else: cand = [p for p in others if not eq[i, p, h]]; p = cand[rng.integers(len(cand))]; x[p, 2 * h + 1] = x[p, 2 * h]
    else:
        n_ = 5 if kind == "QF5" else 4; hs = [h for h in range(3) if cnt[h] == n_]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; x[qq, 2 * h + 1] = (x[qq, 2 * h] + rng.integers(1, 3)) % 3 if agq[h] else x[qq, 2 * h]
    e2 = x.reshape(6, 3, 2); ok_b = eq[i].all(0).any(); ok_e = (e2[..., 0] == e2[..., 1]).all(0).any()
    if not (ok_b and ok_e): continue
    fam[kind][0].append(X[i]); fam[kind][1].append(x); fam[kind][2].append(qq)
for k, (Xb, Xd, qq) in fam.items():
    Xb, Xd, qq = np.array(Xb), np.array(Xd), np.array(qq)
    ch = {w: (ind(Xb, qq, w) != ind(Xd, qq, w)).any(1) for w in RULES}
    pred = {w: alg_logp(Xd, qq, w) - alg_logp(Xb, qq, w) for w in RULES}; pred = {w: v - v.mean(-1, keepdims=True) for w, v in pred.items()}
    PB = np.log(GL.posterior_closed(env, Xd, qq)) - np.log(GL.posterior_closed(env, Xb, qq)); PB -= PB.mean(-1, keepdims=True)
    print(k, "n", len(qq), "on-support base/edit", support(Xb).mean(), support(Xd).mean(),
          "frac pairs whose indicator changes:", {w: round(float(c.mean()), 3) for w, c in ch.items()},
          "| E vs exact", float(np.abs(pred["E"] - PB).max()),
          "| mean|pred_w - pred_E|:", {w: round(float(np.abs(pred[w] - pred["E"]).mean()), 3) for w in RULES if w != "E"})
