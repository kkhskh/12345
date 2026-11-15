"""Workstream C1/C5 (XIII-B): how much of the C=2 decision needs the CROSS-QUESTION identity relation J_H, and how much is available from
(a) per-group statistics plus (b) the full joint values at the target question q (a LOCAL cross-group relation)?
Exact Bayes posteriors of CENSORED observations under the true shared-copy generator (C = 2, beta in {1, 0.6}):
  FULL     : all reports (xii_bayes.posterior_shared; the exact posterior)
  LOCALQ   : at q the full joint values; at every q' != q only each group's SELF-AGREEMENT bit (cross-group identity at q' censored)
  PERGROUP : per-group information only -- at q each group's own pair, at q' != q self-agreement bits; colluders treated through the exact
             marginal of THEIR OWN reports (no cross-group term anywhere): this is the best rule that never compares groups
Likelihood of a censored q' under role (g reliable, H colluders): reliable agrees w.p. pr = alpha^2 + (1-alpha)^2/(K-1); noise w.p. 1/K;
colluders: beta * [both agree] + (1 - beta) * prod_h (1/K)^{a_h} (1 - 1/K)^{1 - a_h}.  At q (LOCALQ): the exact joint likelihood of all six values
given y and role (shared copy event at q).  Accuracy / KL to FULL on 200,000 episodes per beta; also on the J-pair family of adapt2.py
(per-group statistics AND values at q identical by construction, so LOCALQ and PERGROUP predict exactly zero change).  -> logs/censored_bayes.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, genlib as GL, xii_bayes as XB
np.seterr(all="ignore")
K, A = 3, 0.8; pr = A ** 2 + (1 - A) ** 2 / (K - 1)
def lik_q(X, q, env, y, g, H, joint=True):
    """likelihood of the six reports at question q given Y_q = y and role; joint=False -> colluder groups' pairs treated marginally (no shared copy)."""
    n = len(X); a = X[np.arange(n), q].reshape(n, 3, 2); out = np.ones(n)
    p1 = lambda v: np.where(v == y, A, (1 - A) / (K - 1))
    out *= p1(a[:, g, 0]) * p1(a[:, g, 1])
    noise = [h for h in range(3) if h != g and h not in H]
    for h in noise: out *= (1 / K) ** 2
    if joint:
        allEq = np.all(a[:, list(H)].reshape(n, -1) == a[:, list(H)].reshape(n, -1)[:, :1], 1)
        out *= env.beta * allEq / K + (1 - env.beta) / K ** (2 * len(H))
    else:
        for h in H: out *= env.beta * (a[:, h, 0] == a[:, h, 1]) / K + (1 - env.beta) / K ** 2
    return out
def censored_post(X, q, env, mode):
    n = len(X); A6 = X.reshape(n, 6, 3, 2); ag = (A6[..., 0] == A6[..., 1])        # (n,6,3)
    P = np.zeros((n, K))
    for g in range(3):
        for H in itertools.combinations([h for h in range(3) if h != g], env.C):
            lw = np.zeros(n)
            for qq in range(6):
                m = qq != q
                lq = np.log(np.where(ag[:, qq, g], pr, 1 - pr))
                for h in range(3):
                    if h != g and h not in H: lq += np.log(np.where(ag[:, qq, h], 1 / K, 1 - 1 / K))
                if mode == "PERGROUP":
                    for h in H: lq += np.log(env.beta * ag[:, qq, h] + (1 - env.beta) * np.where(ag[:, qq, h], 1 / K, 1 - 1 / K))
                else:
                    both = ag[:, qq, H[0]] & ag[:, qq, H[1]]; ind = np.where(ag[:, qq, H[0]], 1 / K, 1 - 1 / K) * np.where(ag[:, qq, H[1]], 1 / K, 1 - 1 / K)
                    lq += np.log(env.beta * both + (1 - env.beta) * ind)
                lw += np.where(m, lq, 0.0)
            for y in range(K): P[:, y] += np.exp(lw) * lik_q(X, q, env, y, g, H, joint=(mode == "LOCALQ"))
    return P / P.sum(1, keepdims=True)
res = {}
for beta in (1.0, 0.6):
    env = GL.Env(3, 2, beta=beta); rng = np.random.default_rng(78_000_000 + int(10 * beta)); d = GL.sample(env, 200000, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); y = d["Y"][np.arange(len(X)), q]
    Pf = XB.posterior_shared(env, X, q); r = dict(FULL_acc=float((Pf.argmax(-1) == y).mean()))
    for mode in ("LOCALQ", "PERGROUP"):
        Pc = censored_post(X, q, env, mode); kl = (Pf * (np.log(Pf + 1e-12) - np.log(Pc + 1e-12))).sum(-1)
        r[f"{mode}_acc"] = float((Pc.argmax(-1) == y).mean()); r[f"{mode}_KL_to_full"] = float(np.nanmean(kl)); r[f"{mode}_decision_differs"] = float((Pc.argmax(-1) != Pf.argmax(-1)).mean())
    res[f"beta{beta}"] = r; print("beta", beta, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "censored_bayes.json"), "w"), indent=1)
