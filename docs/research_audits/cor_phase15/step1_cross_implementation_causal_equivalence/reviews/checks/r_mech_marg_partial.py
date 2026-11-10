"""Reviewer check (mechanistic): is the T-model overshoot of the marginalisation operator (drop fraction 1.14-1.37) collateral
damage from the encoder absorbing own-question content correlated with the counts?
Exactly the necessity.py streams (fit 12_300_000 x4000, eval 12_450_000 x1000, draws rng(EVAL+1000q+j), 8 draws).
Conditions per model:
  marg_orig      necessity.py's operator (E.fit_encoder; must reproduce logs/conf/necessity.json)
  marg_partial   same operator, encoder fitted jointly on [count one-hot, own-question covariates] (one-hot of the
                 question's 6 reports + its 3 within-pair agreements), only the count block used for the delta
  input_marg     no activation intervention: average of the model's softmax over the same 8 natural draws
                 X' = (X_q, X'_{-q}) (input-level on-support marginalisation; the reference the operator should match)
  input_marg_site_only  natural draws patched at the site only (full-site patch of the 8 cond draws, averaged)
Reports acc, drop fraction, KL(own||out), KL(input_marg||out), mean delta norm.  Usage: r_mech_marg_partial.py M..."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np
import eqv as E
t0 = time.time(); FIT, EVAL, NF, NE, ND = 12_300_000, 12_450_000, 4000, 1000, 8
Xf = E.sample(NF, FIT)["X"]; De = E.sample(NE, EVAL); X, Y = De["X"], De["Y"]
post = E.G.posterior(E.F4, X); own = post["own_y"]; n0 = E.loo(X)
draws = {q: [E.cond_pairs(X, q, np.random.default_rng(EVAL + 1000 * q + j)) for j in range(ND)] for q in range(E.Q)}
acc = lambda P: float((P.argmax(-1) == Y).mean())
def kl(p, r): return float((p * (np.log(p + 1e-12) - np.log(r + 1e-12))).sum(-1).mean())


def cov_feats(Xs, k):            # own-question covariates of question k: one-hot of its 6 reports + 3 agreements
    return np.c_[np.eye(3)[Xs[:, k]].reshape(len(Xs), -1), E.agree(Xs)[:, k]]


def fit_partial(kind, Xs, A):
    Fd = E.site_features(kind, Xs); W = {}
    for t, F in Fd.items():
        k = t - 36 if kind == "P_full" else t // 6
        Z = np.c_[np.ones(len(Xs)), F, cov_feats(Xs, k)]
        W[t] = np.linalg.lstsq(Z, A[:, t], rcond=None)[0][1:1 + F.shape[1]]
    return W


out = {}
for m in sys.argv[1:]:
    M = E.Model(m); k = m[0]; d = M.d; prim = "P_full" if k == "P" else "T_own"
    Af = M.acts(Xf)[0]; A0, L0, _, _ = M.acts(X); P0 = E.softmax(L0); a0 = acc(P0); aown = acc(own)
    encs = {"marg_orig": E.fit_encoder(prim, Xf, Af), "marg_partial": fit_partial(prim, Xf, Af)}
    P = {c: np.zeros_like(P0) for c in list(encs) + ["input_marg", "input_marg_site_only"]}; nrm = {c: [] for c in encs}
    for q in range(E.Q):
        pos = E.site_positions(k, q)
        for Xd in draws[q]:
            nd = E.loo(Xd); Ad, Ld, _, _ = M.acts(Xd)
            for c, enc in encs.items():
                dl = E.encoder_delta(enc, prim, X, n0, nd, d, pos); nrm[c].append(float(np.linalg.norm(dl[:, pos], axis=-1).mean()))
                P[c][:, q] += E.softmax(M.run(X, dl)[:, q]) / ND
            P["input_marg"][:, q] += E.softmax(Ld[:, q]) / ND
            dl = np.zeros_like(A0); dl[:, pos] = Ad[:, pos] - A0[:, pos]
            P["input_marg_site_only"][:, q] += E.softmax(M.run(X, dl)[:, q]) / ND
            nrm.setdefault("natural_site", []).append(float(np.linalg.norm(dl[:, pos], axis=-1).mean()))
    res = {"clean": dict(acc=a0, KL_own=kl(own, P0))}
    for c, Pq in P.items():
        res[c] = dict(acc=round(acc(Pq), 4), drop_frac=round((a0 - acc(Pq)) / (a0 - aown), 3),
                      KL_own_red=round(1 - kl(own, Pq) / kl(own, P0), 3), KL_inputmarg_to_out=round(kl(P["input_marg"], Pq), 4),
                      mean_delta_norm=round(float(np.mean(nrm[c])), 3) if c in nrm else None)
    res["natural_site_delta_norm"] = round(float(np.mean(nrm["natural_site"])), 3)
    out[m] = res; print(m, json.dumps(res), f"[{time.time() - t0:.0f}s]", flush=True)
