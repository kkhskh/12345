"""Workstream A2: how does the mixture head compute the responsibilities?  (pooled S, per-group e_g, per-group weight m_g = S - e_g, ...)
For each mixture head (l2_scan rule: readout-token mass on q-tokens >= 0.8), the pre-softmax score from the readout query (token 36+q) to group g
(mean over g's two q-tokens) is regressed on candidate features.  Bayes:  log r_g = log W_g + log(S - e_g) + log sum_y L_g(y) + const(episode).
Scores are compared WITHIN episode (subtract the episode mean over groups), which removes any pooled additive term.
  (1) full model:  score_g ~ a log W_g + c log sumL_g + cell(e_g, m_g) effects (m_g = S - e_g in {0,1,2})   -> fitted cell means
  (2) is the m_g = 0 cell (g is the ONLY eligible group: Bayes weight 0) strongly suppressed?  (Bayes: score -> -inf)
  (3) source of the eligibility dependence: query/key path swaps (natural): recompute scores with the readout QUERY from a same-q donor episode that
      differs in S, keys from the base; and with KEYS from the donor, query from the base.  Report how the m_g-cell contrast moves.
Usage: l2_scores.py SPEC... -> logs/l2_scores<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("SC_STREAM", 72_200_000))); N_ = 6000
d = NV.Bm.sample_fast(NV.ENV, N_, rng); X = d["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(N_, 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(int); S = e.sum(1, keepdims=True); m = S - e
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    def _r2(Xc):
        r = {}; F.forward(M, jnp.asarray(Xc), rec=r); return np.asarray(r["resid2"]), np.asarray(r["pattern2"])
    x2 = np.concatenate([_r2(X[i:i + 1000])[0] for i in range(0, len(X), 1000)]); A2 = _r2(X[:500])[1]
    tqk = q[:, None] * 6 + np.arange(6)[None]
    mass = np.array([A2[ii[:500], h, 36 + q[:500]][np.arange(500)[:, None], tqk[:500]].sum(1).mean() for h in range(4)]); MH = [h for h in range(4) if mass[h] >= 0.8]
    if not MH: out[sp] = dict(mixture_heads=[]); print(sp, "no mixture head"); continue
    lay = {k: np.asarray(v) for k, v in M["layers"][2].items()}
    def ln(x): mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True); return (x - mu) / np.sqrt(v + 1e-5) * lay["g1"] + lay["b1"]
    z = ln(x2); res = dict(mixture_heads=MH, heads={})
    rngd = np.random.default_rng(5); donor = np.array([rngd.choice(np.where((q == q[i]) & (S[:, 0] != S[i, 0]))[0]) if ((q == q[i]) & (S[:, 0] != S[i, 0])).any() else i for i in range(N_)])
    for h in MH:
        def scores(zq, zk):
            qv = zq[ii, 36 + q] @ lay["WQ"][h]; kv = zk[ii[:, None], tqk] @ lay["WK"][h]           # (N,e), (N,6,e)
            s = np.einsum("ne,nse->ns", qv, kv) / 4.0; return s.reshape(N_, 3, 2).mean(-1)              # per group
        s = scores(z, z); sc = s - s.mean(1, keepdims=True)
        Xf = np.stack([np.log(W), np.log(Lg.sum(-1))], -1); Xf = Xf - Xf.mean(1, keepdims=True)
        cells = [(eg, mg) for eg in (0, 1) for mg in (0, 1, 2)]
        D = np.stack([((e == eg) & (m == mg)).astype(float) for eg, mg in cells], -1)                  # (N,3,6) indicators
        Dm = D - D.mean(1, keepdims=True)
        Z = np.concatenate([Xf, Dm], -1).reshape(-1, 2 + len(cells)); y = sc.reshape(-1)
        coef, *_ = np.linalg.lstsq(Z, y, rcond=None); yhat = Z @ coef; r2 = float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        cnt = {f"e{eg}_m{mg}": int(D[..., k].sum()) for k, (eg, mg) in enumerate(cells)}
        cm = {f"e{eg}_m{mg}": float(coef[2 + k]) for k, (eg, mg) in enumerate(cells)}
        # pooled-vs-per-group: does a model with ONLY e_g (no m_g) fit as well?
        De = np.stack([(e == 1).astype(float)], -1); De = De - De.mean(1, keepdims=True); Ze = np.concatenate([Xf, De], -1).reshape(-1, 3); ce, *_ = np.linalg.lstsq(Ze, y, rcond=None)
        r2e = float(1 - ((y - Ze @ ce) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        Dmm = np.stack([np.log(np.maximum(m, 0.5))], -1).astype(float); Dmm = Dmm - Dmm.mean(1, keepdims=True); Zm = np.concatenate([Xf, Dmm], -1).reshape(-1, 3); cmm, *_ = np.linalg.lstsq(Zm, y, rcond=None)
        r2m = float(1 - ((y - Zm @ cmm) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        # query/key swaps with a donor of different S (same q): which side carries the cross-group eligibility?
        sq = scores(z[donor], z); sk = scores(z, z[donor])
        contrast = lambda ss: float(((ss - ss.mean(1, keepdims=True)) * (np.log(np.maximum(m, 0.5)) - np.log(np.maximum(m, 0.5)).mean(1, keepdims=True))).mean())
        res["heads"][h] = dict(coef_logW=float(coef[0]), coef_logsumL=float(coef[1]), cell_effects=cm, cell_counts=cnt, R2_cells=r2, R2_e_only=r2e, R2_logm=r2m,
                               logm_contrast_base=contrast(s), logm_contrast_query_from_donor=contrast(sq), logm_contrast_keys_from_donor=contrast(sk))
        v = res["heads"][h]
        print(sp, f"h2.{h} R2 cells {r2:.3f} e-only {r2e:.3f} log m {r2m:.3f} | logW {v['coef_logW']:.2f} logsumL {v['coef_logsumL']:.2f} | cells", {k: round(x, 2) for k, x in cm.items()}, "| n", cnt, "| logm contrast base/qDonor/kDonor", round(v["logm_contrast_base"], 3), round(v["logm_contrast_query_from_donor"], 3), round(v["logm_contrast_keys_from_donor"], 3), flush=True)
    out[sp] = res
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_scores" + os.environ.get("SC_TAG", "") + ".json"), "w"), indent=1)
