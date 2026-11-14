"""Workstream C2 (false positives / false negatives): identification on a DISCRIMINATING pair distribution.
math_xii.py M4 shows that on the Investigation XI decoy-DAS pair distribution the E and A1 (perfect-only) counterfactual decisions differ on
only 0.35% of pairs, so no interchange metric on that distribution can separate E from A1.  Here the pair distribution is restricted to pairs
where cf_E and cf_A1 have DIFFERENT argmax (only possible when some group is perfect but disagrees at q in the base or source).
The fitted E-subspace (identity interchange, no retraining) is evaluated against both targets (all discriminating pairs, and the 'inf'
subset where the E counterfactual also differs from the base's own Bayes decision, so that matching cf_E requires the patch to act):
   IIA_E, IIA_A1 (patched argmax agreement), KL_E, KL_A1, and the base (no intervention) agreement with each target.
A genuine E mediator: IIA_E high, IIA_A1 low.  If the subspace merely reproduces 'some' eligibility counterfactual, both are mediocre.
Usage: disc_ident.py RSUF SPEC... -> logs/disc_ident<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("DI_STREAM", 66_800_000)))
d = NV.Bm.sample_fast(NV.ENV, 60000, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); ii = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(len(X), 6, 3, 2)[ii, q]; pf = (GL.counts(X, 3)[ii, q] == 5).astype(float); e = pf * (a_[..., 0] == a_[..., 1])
bi = rng.integers(0, len(X), 200000); si = rng.integers(0, len(X), 200000)
def cf(x, b): P = np.maximum((W[b][..., None] * (x.sum(1, keepdims=True) - x)[..., None] * Lg[b]).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
cE, cA = cf(e[si], bi), cf(pf[si], bi); okE = e[si].sum(1) >= 1; okA = pf[si].sum(1) >= 1
sel = np.where((cE.argmax(-1) != cA.argmax(-1)) & okE & okA)[0][:3000]
bi, si, cE, cA = bi[sel], si[sel], cE[sel], cA[sel]; Xb, Xs, qb, qs = X[bi], X[si], q[bi], q[si]
cB0 = cf(e[bi], bi); inf2 = cE.argmax(-1) != cB0.argmax(-1)      # subset where the E counterfactual ALSO changes the Bayes decision of the base
tokj = lambda q_: jnp.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + jnp.arange(6)[None]], 1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
out = dict(n_pairs=int(len(sel)))
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q = jnp.asarray(np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy")))[0], jnp.float32)
    def logits(Xb_, Xs_, qb_, qs_):
        rs = {}; F.forward(M, Xs_, rec=rs); xs = rs["resid2"]; tb, ts = tokj(qb_), tokj(qs_); i = jnp.arange(Xb_.shape[0])[:, None]
        def fn(x):
            vb = x[i, tb].reshape(Xb_.shape[0], -1); vs = xs[i, ts].reshape(Xb_.shape[0], -1); vp = vb + ((vs - vb) @ Q) @ Q.T; return x.at[i, tb].set(vp.reshape(Xb_.shape[0], 7, 64))
        return F.forward(M, Xb_, resid_patch={2: fn})[jnp.arange(Xb_.shape[0]), qb_]
    lj = jax.jit(logits)
    L = np.concatenate([np.asarray(lj(jnp.asarray(Xb[i:i + 500]), jnp.asarray(Xs[i:i + 500]), jnp.asarray(qb[i:i + 500]), jnp.asarray(qs[i:i + 500]))) for i in range(0, len(Xb), 500)])
    L0 = F.batched(jax.jit(lambda X_: F.forward(M, X_)), Xb)[np.arange(len(Xb)), qb]
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    r = dict(IIA_E=float((L.argmax(-1) == cE.argmax(-1)).mean()), IIA_A1=float((L.argmax(-1) == cA.argmax(-1)).mean()), KL_E=float(kl(cE, P).mean()), KL_A1=float(kl(cA, P).mean()),
             base_E=float((L0.argmax(-1) == cE.argmax(-1)).mean()), base_A1=float((L0.argmax(-1) == cA.argmax(-1)).mean()),
             n_inf=int(inf2.sum()), IIA_E_inf=float((L.argmax(-1) == cE.argmax(-1))[inf2].mean()), IIA_A1_inf=float((L.argmax(-1) == cA.argmax(-1))[inf2].mean()),
             base_E_inf=float((L0.argmax(-1) == cE.argmax(-1))[inf2].mean()), per_pair_inf=inf2.astype(int).tolist(),
             per_pair_hitE=(L.argmax(-1) == cE.argmax(-1)).astype(int).tolist(), per_pair_hitA1=(L.argmax(-1) == cA.argmax(-1)).astype(int).tolist())
    out[sp] = r; print(sp, {k: round(v, 3) for k, v in r.items() if not k.startswith("per_")}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "disc_ident" + os.environ.get("DI_TAG", "") + ".json"), "w"))
