"""Re-evaluate a trained DAS subspace (logs/das/<tag>_R.npy) on DISCRIMINATING counterfactual pairs (no retraining).
For the E variable, the high-level counterfactual is computed under two competing high-level rules sharing W, L:
   B  (Bayes):           f_g = S_src - e_src,g
   A3 (separable):       f_g = 1 - e_src,g
Subsets of held-out pairs (stream DAS_EVAL_STREAM + 2, as das.py):
   all            : pairs with e_src != e_base
   disc           : argmax(cf_B) != argmax(cf_A3)   (the rules disagree on the counterfactual decision)
Report per subset: IIA_B (patched argmax == cf_B argmax), IIA_A3, KL(cf_B || patched), and for the RANDOM subspace the same.
Usage: das_eval.py PARENT SPEC SEED SUFFIX LAYER TOK K  -> logs/das/<tag>_disc.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
parent, spec, seed, suf, LAYER, TOK, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), (sys.argv[4] if sys.argv[4] != "-" else ""), int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
EVSTREAM = int(os.environ.get("DAS_EVAL_STREAM", 61_500_000))
M = F.load_model(parent, spec, seed=seed, suffix=suf); tag = f"{M['tag']}_E_L{LAYER}_{TOK}_k{K}" + ("" if EVSTREAM == 61_500_000 else f"_ev{EVSTREAM}")
R = jnp.asarray(np.load(os.path.join(HERE, "..", "logs", "das", tag + "_R.npy")))
env = GL.Env(G=3, C=1)
def hl(X, q):
    rel, col, noi = GL._group_lik(env, X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
    return W, Lg, ((cnt == 5) & (a_[..., 0] == a_[..., 1])).astype(float)
def tokens(q, xp=np):
    if TOK == "ro": return xp.stack([36 + q], 1)
    if TOK == "q7": return xp.concatenate([(36 + q)[:, None], q[:, None] * 6 + xp.arange(6)[None]], 1)
    if TOK == "src6": return q[:, None] * 6 + xp.arange(6)[None]
    return xp.broadcast_to(xp.arange(43)[None], (q.shape[0], 43))
def patched(Rm, Xb, Xs, qb, qs):
    rs = {}; F.forward(M, Xs, rec=rs); xs = rs[f"resid{LAYER}"] if LAYER < 3 else rs["resid3"]
    tb, ts = tokens(qb, jnp), tokens(qs, jnp); i = jnp.arange(Xb.shape[0])[:, None]
    def fn(x):
        vb = x[i, tb].reshape(Xb.shape[0], -1); vs = xs[i, ts].reshape(Xb.shape[0], -1); Q, _ = jnp.linalg.qr(Rm)
        return x.at[i, tb].set((vb + ((vs - vb) @ Q) @ Q.T).reshape(Xb.shape[0], tb.shape[1], 64))
    return F.forward(M, Xb, resid_patch={LAYER: fn})[jnp.arange(Xb.shape[0]), qb]
pj = jax.jit(patched)
rng = np.random.default_rng(EVSTREAM + 2); d = NV.Bm.sample_fast(NV.ENV, 24000, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
W, Lg, e = hl(X, q); S = e.sum(1); two = np.where(S >= 2)[0]; n = 6000
bi = np.concatenate([rng.choice(two, n // 2), rng.choice(np.where(S == 1)[0], n - n // 2)])
si = np.where(rng.random(n) < 0.6, rng.choice(two, n), rng.integers(0, len(X), n))
Xb, Xs, qb, qs = X[bi], X[si], q[bi], q[si]; Wb, Lb, eb = W[bi], Lg[bi], e[bi]; es = e[si]
norm = lambda P: (np.maximum(P, 0) + 1e-12) / (np.maximum(P, 0) + 1e-12).sum(-1, keepdims=True)
cfB = norm((Wb[..., None] * (es.sum(1, keepdims=True) - es)[..., None] * Lb).sum(1)); cfA = norm((Wb[..., None] * (1 - es)[..., None] * Lb).sum(1))
keep = np.abs(es - eb).sum(1) > 0; disc = keep & (cfB.argmax(-1) != cfA.argmax(-1))
def run(Rm, sel):
    L = np.concatenate([np.asarray(pj(Rm, jnp.asarray(Xb[sel][i:i + 500]), jnp.asarray(Xs[sel][i:i + 500]), jnp.asarray(qb[sel][i:i + 500]), jnp.asarray(qs[sel][i:i + 500]))) for i in range(0, sel.sum(), 500)])
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    return dict(n=int(sel.sum()), IIA_B=float((L.argmax(-1) == cfB[sel].argmax(-1)).mean()), IIA_A3=float((L.argmax(-1) == cfA[sel].argmax(-1)).mean()),
                KL_B=float((cfB[sel] * (np.log(cfB[sel]) - np.log(P + 1e-12))).sum(-1).mean()))
Rr = jnp.asarray(np.linalg.qr(np.random.default_rng(9).standard_normal(R.shape))[0])
L0 = F.batched(jax.jit(lambda X_: F.forward(M, X_)), Xb)[np.arange(n), qb]
out = dict(tag=tag, all=run(R, keep), disc=run(R, disc), rand_all=run(Rr, keep), rand_disc=run(Rr, disc),
           base_disc_IIA_B=float((L0[disc].argmax(-1) == cfB[disc].argmax(-1)).mean()))
json.dump(out, open(os.path.join(HERE, "..", "logs", "das", tag + "_disc.json"), "w"), indent=1)
print(tag, json.dumps({k: (v if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()}) for k, v in out.items() if k != "tag"}), flush=True)
