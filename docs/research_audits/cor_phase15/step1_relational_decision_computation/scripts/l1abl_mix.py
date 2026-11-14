"""Workstream B4: what carries eligibility-dependent behaviour after layer 1 is mean-ablated, and what lies outside the E-subspace?
(1) Layer-1 mean ablation (outputs of all layer-1 heads and MLP1 replaced by their natural mean, token-wise): do the layer-2 mixture head's
    readout attention weights still track the Bayes responsibilities r_g (which depend on eligibility), i.e. does LAYER 2 reconstruct the relation
    from layer-0 information?  Measure corr(mass_g, r_g) intact vs ablated, and corr with the eligibility-free responsibilities r0_g ∝ W_g sumL_g.
    Behavioural residual: on TRUE-factorial xAP vs xP cells, the eligibility-specific decision contrast (|Δ| TRUE - |Δ| CTRL) intact vs ablated.
(2) Complement information: linear decodability (held-out logistic) of e_h, of the C=1 decision (argmax Bayes) and of the 'agreement at q' votes
    from the 442-dim complement of the E-subspace at the q7 tokens, versus from the E-subspace itself.
Usage: l1abl_mix.py RSUF SPEC... -> logs/l1abl_mix<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("LA_STREAM", 73_300_000))); N_ = 3000
d = NV.Bm.sample_fast(NV.ENV, N_, rng); X = d["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(N_, 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1, keepdims=True)
rB = W * (S - e) * Lg.sum(-1); rB /= rB.sum(1, keepdims=True); r0 = W * Lg.sum(-1); r0 /= r0.sum(1, keepdims=True)
tq = q[:, None] * 6 + np.arange(6)[None]
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    r = {}; F.forward(M, jnp.asarray(X), rec=r); A2 = np.asarray(r["pattern2"]); heads1 = np.asarray(r["heads1"]).sum(1); mlp1 = np.asarray(r["mlp_out1"]); x1 = np.asarray(r["resid1"]); x2 = np.asarray(r["resid2"])
    mass = np.array([A2[ii, h, 36 + q][ii[:, None], tq].sum(1).mean() for h in range(4)]); MH = [h for h in range(4) if mass[h] >= 0.8]
    res = dict(mixture_heads=MH)
    if MH:
        x2abl = x1 + heads1.mean(0, keepdims=True) + mlp1.mean(0, keepdims=True)
        lay = M["layers"][2]; z = F.N.ln(jnp.asarray(x2abl), lay["g1"], lay["b1"])
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); Aab = np.asarray(jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / 4.0, -1))
        def corr(A):
            m = np.mean([np.stack([A[ii, h, 36 + q][ii, q * 6 + 2 * g] + A[ii, h, 36 + q][ii, q * 6 + 2 * g + 1] for g in range(3)], 1) for h in MH], 0)
            m = m / m.sum(1, keepdims=True); return dict(B=float(np.corrcoef(m.ravel(), rB.ravel())[0, 1]), noelig=float(np.corrcoef(m.ravel(), r0.ravel())[0, 1]))
        res["intact"] = corr(A2); res["layer1_mean_ablated"] = corr(Aab)
    fE = os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy")
    if not os.path.exists(fE): fE = os.path.join(os.environ.get("DECOY_DIR", ""), f"{M['tag']}_E_R.npy")
    Q, _ = np.linalg.qr(np.load(fE)); Pc = np.eye(448) - Q @ Q.T
    z7 = x2[ii[:, None], np.concatenate([(36 + q)[:, None], tq], 1)].reshape(N_, -1); zE = z7 @ Q; zC = z7 @ Pc
    yB = GL.posterior_closed(env, X, q).argmax(-1); agree = (a_[..., 0] == a_[..., 1]).astype(int)
    def dec(Z, y):
        if len(np.unique(y[:2000])) < 2: return None
        c = LogisticRegression(max_iter=3000, C=1.0).fit(Z[:2000], y[:2000]); return float((c.predict(Z[2000:]) == y[2000:]).mean())
    res["decode"] = {nm: {"E": dec(zE, y), "complement": dec(zC, y), "majority": float(max(np.bincount(y[2000:]) / len(y[2000:])))} for nm, y in
                     (("e_g0", e[:, 0].astype(int)), ("e_g1", e[:, 1].astype(int)), ("bayes_decision", yB), ("agree_q_g0", agree[:, 0]))}
    out[sp] = res; print(sp, json.dumps({k: v for k, v in res.items()}), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l1abl_mix" + os.environ.get("LA_TAG", "") + ".json"), "w"), indent=1)
