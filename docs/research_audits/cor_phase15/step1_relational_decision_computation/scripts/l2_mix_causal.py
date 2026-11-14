"""Workstream A1-A3: causal test of the ATTENTION-MIXTURE account of layer 2.
Account (ATT-MIX): a layer-2 'mixture head' at the readout token 36+q puts attention r_g on group g's two source tokens at q, approximating the
posterior responsibility that g is the reliable group, r_g ∝ W_g (S - e_g) sum_y L_g(y); its values carry g's candidate evidence L_g(y).
Mixture heads are identified WITHOUT using any intervention outcome: heads whose mean attention mass from the readout token on the three groups'
q-tokens is >= 0.8 (l2_scan.py).  Models with no such head are reported as 'no mixture head'.
Intervention (class II/III: the attention ROW of the readout token in each mixture head is replaced; everything else is the base run):
  donor  : row taken from a different natural episode with the same target question (random donor)
  noop   : row from the base itself (sanity)
  uniform: row replaced by uniform attention over the six q-tokens (null with the same support)
Prediction of ATT-MIX for a donor row with responsibilities r^d:  p_pred(y) ∝ sum_g r^d_g  L^b_g(y) / sum_y' L^b_g(y')   (base likelihoods).
Because the mixture head carries only part of the logit, the test is a FIDELITY REGRESSION of the observed change of centred log-probabilities
(patched - base) on the predicted change (log p_pred - log p_bayes(base)), over episodes x candidates:
  slope b, R^2, and the same regression for ALTERNATIVE predictions:
    'bayes_resp' (ATT-MIX, responsibilities r from the Bayes rule) ;  'mass_resp' (the donor's ACTUAL attention mass, renormalised over groups) ;
    'null' (predicted change 0: R^2 undefined, slope reported against bayes_resp predictions of a SHUFFLED pairing) ;
    'value_only' (predicted change if attention carried nothing and only values mattered: 0 change)
  plus the uniform-row control.  Pre-flight criterion (before any confirmation): the noop row must give |observed change| ~ 0.
Usage: l2_mix_causal.py SPEC... -> logs/l2_mix_causal<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("MC_STREAM", 72_100_000))); N_ = 2000
d = NV.Bm.sample_fast(NV.ENV, 3 * N_, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); ii = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(len(X), 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1, keepdims=True)
rB = W * (S - e) * Lg.sum(-1); rB = rB / rB.sum(1, keepdims=True); Ln = Lg / Lg.sum(-1, keepdims=True)
PB = GL.posterior_closed(env, X, q)
base = np.arange(N_); donor = np.array([rng.choice(np.where(q == q[b])[0]) for b in base])          # same-q donor
tq = lambda qq: qq[:, None] * 6 + np.arange(6)[None]
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def fid(obs, pred):
    o, p = obs.ravel(), pred.ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); r2 = float(1 - ((o - b * p) ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-12))
    return dict(slope=b, R2=r2, corr=float(np.corrcoef(o, p)[0, 1]) if p.std() > 0 and o.std() > 0 else 0.0)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    rec = {}; L0 = np.asarray(F.forward(M, jnp.asarray(X), rec=rec)); A2 = np.asarray(rec["pattern2"]); x2 = np.asarray(rec["resid2"])
    mass = np.array([np.mean([A2[i, h, 36 + q[i], tq(q[i:i + 1])[0]].sum() for i in range(500)]) for h in range(4)])
    MH = [h for h in range(4) if mass[h] >= 0.8]; res = dict(mixture_heads=MH, mass=mass.round(3).tolist())
    if not MH: out[sp] = res; print(sp, "no mixture head; masses", mass.round(2), flush=True); continue
    lay = M["layers"][2]
    def run(rows):        # rows: (N_, H, 43) replacement attention rows for the readout token (None -> keep)
        xb = jnp.asarray(x2[base]); z = F.N.ln(xb, lay["g1"], lay["b1"])
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1)
        t = jnp.asarray(36 + q[base]); n = jnp.arange(N_)
        for h in MH: A = A.at[n, h, t].set(jnp.asarray(rows[:, h]))
        oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"]).sum(1); xa = xb + oh
        hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); x3 = xa + hid @ lay["W2"] + lay["c2"]
        Lx = np.asarray(F.N.ln(x3, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[np.arange(N_), q[base]]
        P = np.exp(Lx - Lx.max(-1, keepdims=True)); return P / P.sum(-1, keepdims=True)
    own = A2[base][:, :, :, :][np.arange(N_)[:, None], np.arange(4)[None], (36 + q[base])[:, None]]          # (N,H,43) base rows
    don = A2[donor][np.arange(N_)[:, None], np.arange(4)[None], (36 + q[donor])[:, None]]
    uni = np.zeros_like(own); uni[np.arange(N_)[:, None], :, tq(q[base])] = 1 / 6
    P0 = run(own); Pd = run(don); Pu = run(uni)
    obs_d = cen(Pd) - cen(P0); obs_u = cen(Pu) - cen(P0)
    def pred_from(r):
        Pp = (r[..., None] * Ln[base]).sum(1); Pp /= Pp.sum(-1, keepdims=True); return cen(Pp) - cen(PB[base])
    mass_d = np.stack([don[:, MH, :][:, :, [2 * g, 2 * g + 1]].sum((1, 2)) if False else sum(don[np.arange(N_), h][:, None] * 0 for h in MH) for g in range(3)], 1) if False else None
    # donor's actual per-group mass (averaged over mixture heads), positions of the donor's q-tokens == base positions (same q)
    md = np.stack([np.mean([don[np.arange(N_), h][np.arange(N_)[:, None], tq(q[base])[:, [2 * g, 2 * g + 1]]].sum(1) for h in MH], 0) for g in range(3)], 1)
    md = md / md.sum(1, keepdims=True); ru = np.full((N_, 3), 1 / 3)
    shuf = np.random.default_rng(3).permutation(N_)
    res.update(noop_max_abs_change=float(np.abs(cen(P0) - cen(np.exp(L0[base, q[base]] - L0[base, q[base]].max(-1, keepdims=True)) / np.exp(L0[base, q[base]] - L0[base, q[base]].max(-1, keepdims=True)).sum(-1, keepdims=True))).max()),
               donor=dict(bayes_resp=fid(obs_d, pred_from(rB[donor])), mass_resp=fid(obs_d, pred_from(md)), shuffled_pred=fid(obs_d, pred_from(rB[donor])[shuf]),
                          mean_abs_obs=float(np.abs(obs_d).mean()), mean_abs_pred=float(np.abs(pred_from(rB[donor])).mean())),
               uniform=dict(bayes_resp=fid(obs_u, pred_from(ru)), mean_abs_obs=float(np.abs(obs_u).mean())),
               per_episode_obs_donor=obs_d.astype(np.float32).tolist(), per_episode_pred_donor=pred_from(rB[donor]).astype(np.float32).tolist())
    out[sp] = res
    print(sp, "MH", MH, "noop", round(res["noop_max_abs_change"], 4), "donor", {k: {kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3) for k, v in res["donor"].items()}, "uniform", {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3)) for k, v in res["uniform"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_mix_causal" + os.environ.get("MC_TAG", "") + ".json"), "w"))
