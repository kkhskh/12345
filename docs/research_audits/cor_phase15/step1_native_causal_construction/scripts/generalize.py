"""Workstream D: mathematically motivated generalization tests (no retraining; same 43-token layout, so no new tokenization is needed).
Models were trained on T4 = Env(G=3, C=1, beta=1).  Test environments:
  base  : Env(3, C=1, beta=1)      (sanity)
  C2    : Env(3, C=2, beta=1)      two colluding groups (both copy the same value), one reliable group
  b07   : Env(3, C=1, beta=0.7)    partial copying: the colluder copies with probability 0.7, else reports independently at random
References (all at the target question q):
  TRUE   : exact Bayes posterior of the TEST environment (posterior_brute)
  C1B    : the C=1, beta=1 closed form  p ∝ sum_g W_g L_g m_g,  m_g = S - e_g  (what the trained algorithm computes), applied unchanged;
           undefined when S = 0 (all m_g = 0) -> those episodes are reported separately
  A0     : no-collusion averaging  p ∝ sum_g W_g L_g   (eligibility ignored)
  LIM    : the beta -> 1 limit of the exact Bayes posterior (beta = 1 - eps, eps -> 0): only role assignments whose colluder has the
           MINIMUM number d_h of disagreeing questions (over all Q, including q) survive:  p ∝ sum_g W_g L_g #{h != g : d_h = min_h' d_h'}.
           Equals C1B whenever S >= 1 (d* = 0 <=> perfect and agreeing at q); on S = 0 it is the unique continuous extension of the trained rule.
Distances: mean KL(ref || model) and argmax agreement; accuracy vs the true Y_q.
Representation-level check: a linear read-out of e (per group) from the Investigation X E-subspace coordinates Q^T z (q7 tokens), fit on
base-environment episodes, applied to the test environment -> accuracy of e (is the VARIABLE still computed as defined?).
Usage: generalize.py RSUF SPEC... -> logs/generalize<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das")
ENVS = {"base": GL.Env(3, 1), "C2": GL.Env(3, 2), "b07": GL.Env(3, 1, beta=0.7)}
def refs(env, X, q):
    rel, col, noi = GL._group_lik(GL.Env(3, 1), X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
    e = ((cnt == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1)
    nz = lambda P: P / np.maximum(P.sum(-1, keepdims=True), 1e-300)
    C1B = nz((W[..., None] * (S[:, None] - e)[..., None] * Lg).sum(1)); A0 = nz((W[..., None] * Lg).sum(1))
    eqa = (X.reshape(len(X), 6, 3, 2)[..., 0] == X.reshape(len(X), 6, 3, 2)[..., 1]); dh = 6 - eqa.sum(1); emin = (dh == dh.min(1, keepdims=True)).astype(float)
    LIM = nz((W[..., None] * (emin.sum(1, keepdims=True) - emin)[..., None] * Lg).sum(1))
    return dict(TRUE=GL.posterior_brute(env, X, q), C1B=C1B, LIM=LIM, A0=A0), e, S
data = {}
for k, env in ENVS.items():
    rng = np.random.default_rng({"base": 65_000_000, "C2": 65_100_000, "b07": 65_200_000}[k]); d = GL.sample(env, 6000, rng); q = rng.integers(0, 6, 6000)
    R_, e, S = refs(env, d["X"], q); data[k] = dict(X=d["X"], Y=d["Y"][np.arange(6000), q], q=q, refs=R_, e=e, S=S)
BRUTE = {b: GL.posterior_brute(GL.Env(3, 1, beta=b), data['b07']['X'], data['b07']['q']) for b in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]}
tok = lambda qq: np.concatenate([(36 + qq)[:, None], qq[:, None] * 6 + np.arange(6)[None]], 1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
out = {k: dict(S_dist={int(s): float((v["S"] == s).mean()) for s in range(4)},
               ref_dist={f"{a}|{b}": float(kl(v["refs"][a][v["S"] > 0], v["refs"][b][v["S"] > 0]).mean()) for a in v["refs"] for b in v["refs"] if a != b},
               bayes_acc=float((v["refs"]["TRUE"].argmax(-1) == v["Y"]).mean())) for k, v in data.items()}
print("envs", json.dumps(out), flush=True)
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy")))
    fwd = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"]))({}))
    res = {}; probes = None
    for k, v in data.items():
        Ls, Zs = [], []
        for i in range(0, 6000, 1000):
            L, Z = fwd(jnp.asarray(v["X"][i:i + 1000])); Ls.append(np.asarray(L)); Zs.append(np.asarray(Z))
        L = np.concatenate(Ls); Z = np.concatenate(Zs); i = np.arange(6000); Lq = L[i, v["q"]]; P = np.exp(Lq - Lq.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        zE = Z[i[:, None], tok(v["q"])].reshape(6000, -1) @ Q
        if k == "base": probes = [LogisticRegression(max_iter=2000, C=1.0).fit(zE[:4000], v["e"][:4000, g]) for g in range(3)]
        sl = slice(4000, 6000) if k == "base" else slice(0, 6000)
        e_acc = float(np.mean([(probes[g].predict(zE[sl]) == v["e"][sl, g]).mean() for g in range(3)]))
        e_acc_maj = float(np.mean([max(v["e"][sl, g].mean(), 1 - v["e"][sl, g].mean()) for g in range(3)]))
        r = dict(acc=float((P.argmax(-1) == v["Y"]).mean()), e_readout_acc=e_acc, e_readout_majority=e_acc_maj)
        for part, msk in (("S>0", v["S"] > 0), ("S=0", v["S"] == 0), ("all", v["S"] >= 0)):
            if msk.sum() < 20: continue
            r[part] = {nm: dict(KL=float(kl(Rf[msk], P[msk]).mean()), argmax=float((Rf[msk].argmax(-1) == P[msk].argmax(-1)).mean())) for nm, Rf in v["refs"].items() if not (nm == "C1B" and part != "S>0")}
            r[part]["n"] = int(msk.sum())
        if k == "b07":   # implicit copying probability: which beta's exact posterior is the model closest to, on S = 0 and overall?
            grid = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]; r["beta_eff"] = {}
            for part, msk in (("S=0", v["S"] == 0), ("all", v["S"] >= 0)):
                kls = {b: float(kl(BRUTE[b][msk], P[msk]).mean()) for b in grid}; r["beta_eff"][part] = dict(kl=kls, best=min(kls, key=kls.get))
        res[k] = r
    out[sp] = res
    print(sp, {k: (round(r["acc"], 3), round(r["e_readout_acc"], 3), {p: {nm: round(x["KL"], 3) for nm, x in r[p].items() if nm != "n"} for p in ("S>0", "S=0") if p in r}) for k, r in res.items()}, "beta_eff", {p: x["best"] for p, x in res["b07"]["beta_eff"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "generalize" + os.environ.get("GE_TAG", "") + ".json"), "w"), indent=1)
