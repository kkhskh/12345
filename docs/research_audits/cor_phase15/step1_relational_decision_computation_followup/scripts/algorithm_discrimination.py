"""Workstream A4: functional (input-intervention) discrimination between candidate algorithms.
Candidate algorithms share the reliability/likelihood terms of the exact T4 rule and differ only in the cross-group eligibility factor f_g:
  p_alg(y) ∝ sum_g W_g L_g(y) f_g
    B   (Bayes):            f_g = S - e_g,  e_h = [n_h = 5][h agrees at q]           (Investigation VII/IX; exact)
    A1  (perfect-only):     e_h = [n_h = 5]                    (ignores agreement at q)
    A2  (threshold-4):      e_h = [n_h >= 4][h agrees at q]    (soft boundary)
    A3  (separable discount): f_g = 1 - e_g                    (no cross-group term; 'perfect & agreeing => excluded')
    A0  (no relation):      f_g = 1                            (pure reliability weighting)
Valid on-support input edits that separate them (base = natural eval episodes; edited group h != the decision's... any group):
  D1 q-disagree : h with n_h = 5 and agreeing at q, in episodes with S >= 2 (else the edit is OFF-SUPPORT: no admissible colluder)
                  -> change h's second source at q to another value   (B: e_h 1->0 ; A1: unchanged)
  D2 q-agree    : h with n_h = 5 and disagreeing at q -> set h's second source at q equal to its first   (B: e_h 0->1 ; A1: unchanged)
  F45 R1/R2/R3, F54o : Investigation VI-IX families (from the eval set)
Metric per episode: response vector D = centred-logit change at q (3 candidates); distance to algorithm a = mean_y |D_model - D_a|.
Usage: algorithm_discrimination.py EVAL(dev|path) MODEL_SPECS...   MODEL_SPEC = parent:spec:seed:suffix   -> logs/algdisc_<eval>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
evp = sys.argv[1]; specs = sys.argv[2:]
ev = NV.load_eval() if evp == "dev" else NV.load_eval(evp); X = ev["X"]; q = ev["q"]; n = len(X); idx = np.arange(n)
env = GL.Env(G=3, C=1)
def parts(X, q):
    rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); i = np.arange(len(X))
    lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi
    cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]; ag = a_[..., 0] == a_[..., 1]
    return W, Lg, cnt, ag
def alg_logp(X, q, which):
    W, Lg, cnt, ag = parts(X, q)
    if which == "B": e = (cnt == 5) & ag; f = e.sum(1, keepdims=True) - e
    elif which == "A1": e = cnt == 5; f = e.sum(1, keepdims=True) - e
    elif which == "A2": e = (cnt >= 4) & ag; f = e.sum(1, keepdims=True) - e
    elif which == "A3": e = (cnt == 5) & ag; f = 1 - e
    elif which == "A0": f = np.ones_like(cnt)
    P = (W[..., None] * f[..., None] * Lg).sum(1); P = np.maximum(P, 1e-30); P = P / P.sum(-1, keepdims=True); return np.log(P)
ALGS = ("B", "A1", "A2", "A3", "A0")
# ---- build D1 / D2 edits
rng = np.random.default_rng(55_000_001 + (0 if evp == "dev" else 7))
W0, L0, cnt0, ag0 = parts(X, q)
def make(kind):
    base_i, Xe = [], []
    for i in range(n):
        S_i = int(((cnt0[i] == 5) & ag0[i]).sum())
        if kind == "D1" and S_i < 2: continue          # removing the only eligible group would make the episode impossible (off-support)
        cand = [h for h in range(3) if cnt0[i, h] == 5 and (ag0[i, h] if kind == "D1" else not ag0[i, h])]
        if not cand: continue
        h = cand[rng.integers(len(cand))]; x = X[i].copy(); s0, s1 = 2 * h, 2 * h + 1
        if kind == "D1": x[q[i], s1] = (x[q[i], s0] + rng.integers(1, 3)) % 3
        else: x[q[i], s1] = x[q[i], s0]
        base_i.append(i); Xe.append(x)
    return np.array(base_i), np.array(Xe)
fams = {}
for k in ("D1", "D2"): bi, Xe = make(k); fams[k] = (X[bi], Xe, q[bi])
for r in ("R1", "R2", "R3"):
    s = (ev["reg_F45"] == r) & ev["orient_F45"].astype(bool); fams[f"F45_{r}"] = (X[s], ev["X_F45"][s], q[s])
fams["F54o"] = (ev["X_O"], ev["X_F54o"], ev["q_O"])
# ---- algorithm predictions
pred = {}
for f, (Xb, Xa, qq) in fams.items():
    pred[f] = {a: F.centred(alg_logp(Xa, qq, a) - alg_logp(Xb, qq, a)) for a in ALGS}
# sanity: B equals the exact posterior response
ii = np.arange(len(fams["D1"][0])); cB = lambda Xx, qq: (lambda c: c - c.mean(-1, keepdims=True))(np.log(NV.Bm.episode_posterior(NV.ENV, Xx)["post_y"][np.arange(len(Xx)), qq]))
exact = {f: cB(Xa, qq) - cB(Xb, qq) for f, (Xb, Xa, qq) in fams.items()}
out = dict(n={f: len(v[2]) for f, v in fams.items()}, B_vs_exact_maxdiff={f: float(np.abs(pred[f]["B"] - exact[f]).max()) for f in fams},
           alg_pairwise_distance={f: {a: float(np.abs(pred[f][a] - pred[f]["B"]).mean()) for a in ALGS} for f in fams}, models={})
resp = {}
for sp in specs:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]
    M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); fwd = lambda Xx: F.forward(M, Xx)
    import jax; fj = jax.jit(fwd); r = {}
    for f, (Xb, Xa, qq) in fams.items():
        i = np.arange(len(qq)); Lb = F.batched(fj, Xb)[i, qq]; La = F.batched(fj, Xa)[i, qq]
        D = F.centred(La) - F.centred(Lb); resp[(sp, f)] = D
        r[f] = {a: float(np.abs(D - pred[f][a]).mean()) for a in ALGS}; r[f]["closest"] = min(ALGS, key=lambda a: r[f][a])
    out["models"][sp] = r; print(sp, {f: (r[f]["closest"], round(r[f]["B"], 3)) for f in fams}, flush=True)
# per-episode error correlation across models (errors relative to exact Bayes), per family
corr = {}
for f in fams:
    E = {sp: (resp[(sp, f)] - exact[f]).ravel() for sp in specs}; c = {}
    for a in specs:
        for b in specs:
            if a < b: c[f"{a}|{b}"] = float(np.corrcoef(E[a], E[b])[0, 1])
    corr[f] = c
out["error_correlation"] = corr
json.dump(out, open(os.path.join(HERE, "..", "logs", f"algdisc_{'dev' if evp == 'dev' else os.path.basename(evp).split('.')[0]}" + os.environ.get("ALG_TAG", "") + ".json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n", "B_vs_exact_maxdiff", "alg_pairwise_distance")}, indent=0))
