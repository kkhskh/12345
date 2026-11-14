"""Workstream A2/A3: where does the CROSS-GROUP part of the weighting (S - e_g) enter the mixture head?  Natural-activation path swaps.
Base episode b, donor episode d with the SAME target question q but a DIFFERENT number of eligible groups S (natural episodes).
In every mixture head, the layer-2 QUERY of the readout token (resid2 at token 36+q -> LN -> W_Q) is taken from the donor; keys/values from the base.
Competing predictions for the resulting decision (base likelihoods and W from the base):
  QUERY-POOLED : the query carries the pooled eligibility state S, the keys carry e_g:  weights  W_g * max(S_d - e_b,g, 0) * sumL_g  (S from the donor)
                 (if the donor's S makes every weight 0 the Bayes formula is undefined -> such pairs are excluded and counted)
  KEY-PERGROUP : the keys carry m_g = S - e_g already; the query carries no eligibility:  no change (prediction 0)
  QUERY-FULL   : the query carries the donor's whole eligibility vector: weights W_g (S_d - e_d,g) sumL_g
Fidelity regression (slope, R^2) of the observed change of centred log-probabilities on each prediction, plus a shuffled-pairing null and a
same-S donor control (a donor with the SAME S: QUERY-POOLED predicts no change).
Usage: l2_query_swap.py SPEC... -> logs/l2_query_swap<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("QS_STREAM", 72_300_000))); NP = 8000
d = NV.Bm.sample_fast(NV.ENV, NP, rng); X = d["X"]; q = rng.integers(0, 6, NP); ii = np.arange(NP)
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(NP, 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1)
PB = GL.posterior_closed(env, X, q)
cand = [b for b in range(NP) if S[b] in (1, 2)][:2000]; base = np.array(cand); N_ = len(base)
dd = np.array([rng.choice(np.where((q == q[b]) & (S != S[b]))[0]) for b in base]); ds = np.array([rng.choice(np.where((q == q[b]) & (S == S[b]) & (ii != b))[0]) for b in base])
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def mix(wts, b):
    P = (wts[..., None] * Lg[b]).sum(1); ok = P.sum(-1) > 0; P = P / np.maximum(P.sum(-1, keepdims=True), 1e-300); return P, ok
def fid(obs, pred, ok):
    o, p = obs[ok].ravel(), pred[ok].ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); r2 = float(((1 - ((o - b * p) ** 2).sum() / ((o - o.mean()) ** 2).sum()) if ((o - o.mean()) ** 2).sum() > 1e-10 else 0.0))
    return dict(slope=b, R2=r2, n=int(ok.sum()))
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    def _r2(Xc):
        r = {}; F.forward(M, jnp.asarray(Xc), rec=r); return np.asarray(r["resid2"]), np.asarray(r["pattern2"])
    x2 = np.concatenate([_r2(X[i:i + 1000])[0] for i in range(0, len(X), 1000)]); A2 = _r2(X[:500])[1]
    tqk = q[:, None] * 6 + np.arange(6)[None]
    mass = np.array([A2[ii[:500], h, 36 + q[:500]][np.arange(500)[:, None], tqk[:500]].sum(1).mean() for h in range(4)]); MH = [h for h in range(4) if mass[h] >= 0.8]
    if not MH: out[sp] = dict(mixture_heads=[]); print(sp, "no mixture head"); continue
    lay = M["layers"][2]
    def run(donor):
        xb = jnp.asarray(x2[base]); z = F.N.ln(xb, lay["g1"], lay["b1"]); zd = F.N.ln(jnp.asarray(x2[donor]), lay["g1"], lay["b1"])
        t = jnp.asarray(36 + q[base]); n = jnp.arange(N_)
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); qd = jnp.einsum("ntd,hde->nhte", zd, lay["WQ"])
        for h in MH: qh = qh.at[n, h, t].set(qd[n, h, jnp.asarray(36 + q[donor])])
        kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1)
        oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"]).sum(1); xa = xb + oh
        hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); x3 = xa + hid @ lay["W2"] + lay["c2"]
        Lx = np.asarray(F.N.ln(x3, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[np.arange(N_), q[base]]
        P = np.exp(Lx - Lx.max(-1, keepdims=True)); return P / P.sum(-1, keepdims=True)
    P0 = run(base); res = dict(mixture_heads=MH, noop_max_abs=0.0)
    for nm, don in (("diffS", dd), ("sameS", ds)):
        Pd = run(don); obs = cen(Pd) - cen(P0)
        sL = Lg[base].sum(-1) * 0 + 1                                    # likelihood term enters via mix() through Lg directly
        Ppool, okp = mix(W[base] * np.maximum(S[don][:, None] - e[base], 0), base); Pfull, okf = mix(W[base] * np.maximum(S[don][:, None] - e[don], 0), base)
        ok = okp & okf; pred_pool = cen(Ppool) - cen(PB[base]); pred_full = cen(Pfull) - cen(PB[base])
        shuf = np.random.default_rng(1).permutation(N_)
        res[nm] = dict(mean_abs_obs=float(np.abs(obs).mean()), QUERY_POOLED=fid(obs, pred_pool, ok), QUERY_FULL=fid(obs, pred_full, ok), KEY_PERGROUP_pred0_meanabs=float(np.abs(obs).mean()),
                       shuffled=fid(obs, pred_pool[shuf], ok & ok[shuf]), n_excluded_undefined=int((~ok).sum()))
    out[sp] = res
    print(sp, "MH", MH, {nm: {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3)) for k, v in res[nm].items()} for nm in ("diffS", "sameS")}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_query_swap" + os.environ.get("QS_TAG", "") + ".json"), "w"), indent=1)
