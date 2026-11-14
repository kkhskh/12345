"""Workstream B1: matched-budget subspace decoys evaluated where the competing variables imply DIFFERENT decisions.
Subspaces: E, A1 (perfect-only), PERM (group-permuted eligibility) -- all trained by das_decoy.py with the IDENTICAL procedure and budget
(L2, q7 tokens, k=6, 1200 Adam steps, batch 128, same pair sampler family) -- plus a random 6-dim subspace and a source-free mean patch.
Evaluation pairs (stream 73_200_000): base/source pairs on which the three rules' counterfactual DECISIONS are not all equal.
For each subspace R and each rule V in {E, A1, PERM}: fidelity of the observed patched change of centred log-probabilities to the change
predicted by rule V (cf_V(base W,L; source value) vs base Bayes):  slope, R^2.  A subspace 'identifies' V only if its fidelity to V clearly
exceeds its fidelity to the alternatives AND exceeds the random subspace and mean patch.
Usage: decoy_disc.py DECOYDIR SPEC... -> logs/decoy_disc<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
DD = sys.argv[1]; env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("DD_STREAM", 73_200_000)))
d = NV.Bm.sample_fast(NV.ENV, 40000, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); ii = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(len(X), 6, 3, 2)[ii, q]; pf = (GL.counts(X, 3)[ii, q] == 5).astype(float); e = pf * (a_[..., 0] == a_[..., 1])
bi = rng.integers(0, len(X), 2000000); si = rng.integers(0, len(X), 2000000)
def cf(x, b): P = np.maximum((W[b][..., None] * (x.sum(1, keepdims=True) - x)[..., None] * Lg[b]).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
cfs = {"E": cf(e[si], bi), "A1": cf(pf[si], bi), "PERM": cf(np.roll(e[si], 1, 1), bi)}
ok = (e[si].sum(1) >= 1) & (pf[si].sum(1) >= 1) & (e[bi].sum(1) >= 1)
dec = {k: v.argmax(-1) for k, v in cfs.items()}; disc = ok & ((dec["E"] != dec["A1"]) | (dec["E"] != dec["PERM"]))
# two strata, reported separately: E != A1 decisions (rare; up to 1000) and E != PERM decisions (up to 1000, excluding the first stratum)
dA = np.where(ok & (dec["E"] != dec["A1"]))[0][:1000]; dP = np.setdiff1d(np.where(ok & (dec["E"] != dec["PERM"]))[0], dA)[:1000]
sel = np.concatenate([dA, dP]); STRAT = {"EvA1": np.arange(len(dA)), "EvPERM": np.arange(len(dA), len(sel))}; bi, si = bi[sel], si[sel]; cfs = {k: v[sel] for k, v in cfs.items()}; PB0 = cf(e[bi], bi)
Xb, Xs, qb, qs = X[bi], X[si], q[bi], q[si]
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
pred = {k: cen(v) - cen(PB0) for k, v in cfs.items()}
def fid(o, p, m=None):
    if m is not None: o, p = o[m], p[m]
    o, p = o.ravel(), p.ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); return dict(slope=b, R2=float(((1 - ((o - b * p) ** 2).sum() / ((o - o.mean()) ** 2).sum()) if ((o - o.mean()) ** 2).sum() > 1e-10 else 0.0)))
tokj = lambda q_: jnp.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + jnp.arange(6)[None]], 1)
out = dict(n_pairs=int(len(sel)), n_EvA1=int(len(dA)), n_EvPERM=int(len(dP)), frac_E_ne_A1=float((cfs["E"].argmax(-1) != cfs["A1"].argmax(-1)).mean()), frac_E_ne_PERM=float((cfs["E"].argmax(-1) != cfs["PERM"].argmax(-1)).mean()))
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    def run(Q, src_override=None):
        Qj = jnp.asarray(Q, jnp.float32)
        def logits(Xb_, Xs_, qb_, qs_, zsrc):
            rs = {}; F.forward(M, Xs_, rec=rs); xs = rs["resid2"]; tb, ts = tokj(qb_), tokj(qs_); i = jnp.arange(Xb_.shape[0])[:, None]
            def fn(x):
                vb = x[i, tb].reshape(Xb_.shape[0], -1); vs = xs[i, ts].reshape(Xb_.shape[0], -1) if zsrc is None else zsrc; vp = vb + ((vs - vb) @ Qj) @ Qj.T
                return x.at[i, tb].set(vp.reshape(Xb_.shape[0], 7, 64))
            return F.forward(M, Xb_, resid_patch={2: fn})[jnp.arange(Xb_.shape[0]), qb_]
        outL = []
        for i in range(0, len(Xb), 500):
            zs = None if src_override is None else jnp.asarray(np.repeat(src_override[None], min(500, len(Xb) - i), 0))
            outL.append(np.asarray(logits(jnp.asarray(Xb[i:i + 500]), jnp.asarray(Xs[i:i + 500]), jnp.asarray(qb[i:i + 500]), jnp.asarray(qs[i:i + 500]), zs)))
        L = np.concatenate(outL); P = np.exp(L - L.max(-1, keepdims=True)); return P / P.sum(-1, keepdims=True)
    L0 = F.batched(jax.jit(lambda X_: F.forward(M, X_)), Xb)[np.arange(len(Xb)), qb]; P0 = np.exp(L0 - L0.max(-1, keepdims=True)); P0 /= P0.sum(-1, keepdims=True)
    res = {}
    subs = {r: np.linalg.qr(np.load(os.path.join(DD, f"{M['tag']}_{r}_R.npy")))[0] for r in ("E", "A1", "PERM")}
    subs["random"] = np.linalg.qr(np.random.default_rng(9).standard_normal((448, 6)))[0]
    # source-free mean patch inside the E subspace: mean q7 state over natural episodes
    rr = {}; F.forward(M, jnp.asarray(X[:2000]), rec=rr); z2 = np.asarray(rr["resid2"]); zbar = np.stack([z2[np.arange(2000)[:, None], np.concatenate([(36 + q[:2000])[:, None], q[:2000, None] * 6 + np.arange(6)[None]], 1)].reshape(2000, -1)]).mean(1)[0]
    for nm, Q in subs.items():
        obs = cen(run(Q)) - cen(P0); res[nm] = {V: fid(obs, pred[V]) for V in pred}; res[nm]["mean_abs_obs"] = float(np.abs(obs).mean())
        for st, m in STRAT.items(): res[nm][st] = {V: fid(obs, pred[V], m) for V in pred}
    obs = cen(run(subs["E"], src_override=zbar)) - cen(P0); res["E_meanpatch"] = {V: fid(obs, pred[V]) for V in pred}; res["E_meanpatch"]["mean_abs_obs"] = float(np.abs(obs).mean())
    out[sp] = res
    print(sp, {nm: {V: round(x["R2"], 2) for V, x in r.items() if isinstance(x, dict) and "R2" in x} for nm, r in res.items()}, flush=True)
    print("   EvA1 stratum", {nm: {V: round(r["EvA1"][V]["R2"], 2) for V in pred} for nm, r in res.items() if "EvA1" in r}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "decoy_disc" + os.environ.get("DD_TAG", "") + ".json"), "w"), indent=1)
