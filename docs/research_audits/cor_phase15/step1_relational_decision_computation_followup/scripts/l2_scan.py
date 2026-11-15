"""Workstream A1 (development scan): what does layer 2 do at the readout token of the target question q?
Exact rule (C=1):  p(y) ∝ sum_g W_g (S - e_g) L_g(y)  -- a MIXTURE over 'group g is the reliable group' with posterior responsibility
   r_g = W_g (S - e_g) sum_y L_g(y) / sum_h W_h (S - e_h) sum_y L_h(y).
Candidate implementations of the mixture in a Transformer block:
  ATT-MIX : the readout query attends to group g's source tokens at q with weight ~ r_g, and the values carry g's candidate likelihood L_g(y)
  RESID   : the decision is assembled in the residual stream before layer 2 (layer 2 only transmits)
  MLP     : layer-2 MLP computes the mixture from pooled features
Measures (natural episodes, stream 72_000_000):
  (1) direct logit attribution at token 36+q: centred-logit contributions of resid2 (skip path), each layer-2 head, MLP2, with the final LayerNorm
      linearised at the actual input (exact up to the LN scale, which is shared); shares of the variance of the centred logit
  (2) per head: attention mass on group g's two tokens at q (m_hg), on g's tokens at other questions, on other tokens;
      Spearman/Pearson correlation of the normalised per-group mass with r_g under the Bayes rule and alternative weightings:
        B: W_g (S-e_g) sum L_g ;  A3: W_g (1-e_g) sum L_g ;  A0: W_g sum L_g ;  M: (S - e_g) only ;  W: W_g only
Usage: l2_scan.py SPEC... -> logs/l2_scan<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("L2_STREAM", 72_000_000))); N_ = 3000
d = NV.Bm.sample_fast(NV.ENV, N_, rng); X = d["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(N_, 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1, keepdims=True)
sL = Lg.sum(-1)
def resp(w): w = np.maximum(w, 0); return w / np.maximum(w.sum(1, keepdims=True), 1e-300)
pf = (GL.counts(X, 3)[ii, q] == 5).astype(float)
R = {"B": resp(W * (S - e) * sL), "A1": resp(W * (pf.sum(1, keepdims=True) - pf) * sL), "A3": resp(W * (1 - e) * sL), "A0": resp(W * sL), "M": resp(S - e + 1e-9), "W": resp(W)}
def rankcorr(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    rec = {}; L = np.asarray(F.forward(M, jnp.asarray(X), rec=rec)); rec = {k: np.asarray(v) for k, v in rec.items()}
    t = 36 + q; x3 = rec["resid3"][ii, t]; mu = x3.mean(-1, keepdims=True); sd = np.sqrt(((x3 - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
    gf, WA = np.asarray(M["gf"]), np.asarray(M["WA"])
    lin = lambda v: ((v - v.mean(-1, keepdims=True)) / sd * gf) @ WA                              # LN-linearised logit contribution (bias excluded)
    comps = {"resid2": rec["resid2"][ii, t], **{f"h2.{h}": rec["heads2"][ii, h, t] for h in range(4)}, "mlp2": rec["mlp_out2"][ii, t]}
    cl = {k: lin(v) for k, v in comps.items()}; cl = {k: v - v.mean(-1, keepdims=True) for k, v in cl.items()}
    tot = sum(cl.values()); Lc = L[ii, q] - L[ii, q].mean(-1, keepdims=True)
    res = dict(recon_err=float(np.abs(tot - Lc).max()), logit_var_share={k: float((v * tot).sum() / (tot * tot).sum()) for k, v in cl.items()})
    P = np.exp(Lc - Lc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True); Pb = GL.posterior_closed(env, X, q)
    res["KL_bayes_model"] = float((Pb * (np.log(Pb + 1e-12) - np.log(P + 1e-12))).sum(-1).mean())
    heads = {}
    for h in range(4):
        A = rec[f"pattern{os.environ.get('L2_LAYER', '2')}"][ii, h, t]                                                                # (N,43)
        mq = np.stack([A[ii, q * 6 + 2 * g] + A[ii, q * 6 + 2 * g + 1] for g in range(3)], 1)
        moth = np.stack([sum(A[ii, qp * 6 + 2 * g] + A[ii, qp * 6 + 2 * g + 1] for qp in range(6)) for g in range(3)], 1) - mq
        mn = mq / np.maximum(mq.sum(1, keepdims=True), 1e-12)
        heads[h] = dict(mass_q_groups=float(mq.sum(1).mean()), mass_other_q=float(moth.sum(1).mean()), mass_ro_ep=float(A[:, 36:43].sum(1).mean()),
                        corr={k: float(np.corrcoef(mn.ravel(), v.ravel())[0, 1]) for k, v in R.items()}, rankcorr={k: rankcorr(mn.ravel(), v.ravel()) for k, v in R.items()})
    res["heads"] = heads; out[sp] = res
    print(sp, "KLb", round(res["KL_bayes_model"], 3), "recon", round(res["recon_err"], 4), "shares", {k: round(v, 2) for k, v in res["logit_var_share"].items()}, flush=True)
    for h, v in heads.items(): print(f"   h2.{h} massQ {v['mass_q_groups']:.2f} othQ {v['mass_other_q']:.2f} ro {v['mass_ro_ep']:.2f} corr", {k: round(x, 2) for k, x in v["corr"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_scan" + os.environ.get("L2_TAG", "") + ".json"), "w"), indent=1)
