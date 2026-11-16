"""XIII-D Workstream B3: what happens to the UNUSED route?  For a model (any checkpoint) measures, separately:
  PRESENT  (information)   held-out ridge R2 of the candidate evidence at q, decoded from the readout-token residual after layer 0 (RO1) and after
                           layer 1 (RO2): pooled evidence (3, centred) and group-resolved evidence (9: k_g(y) for each group g, candidate y);
                           baseline: the same decoder from the readout token of a DIFFERENT question (no evidence about q expected beyond priors).
  USED     (transmission)  lik_RO1 / lik_RO2 (metrics.avail): share of the Bayes change that reaches the output when only the readout-token
                           state at that layer comes from a likelihood donor.
  MIXTURE  mix_mass of the strongest layer-2 head on the q tokens.
  REACTIVATION (optional, RS_REACT=1)  WA (+ final LN) alone retrained for 300 steps (lr 3e-3, batch 256, stream 90_300_000) with the
           mixture route BLOCKED (layer-2 readout -> own-q-source attention masked, as in l2q) during training and evaluation; accuracy vs the
           exact posterior's argmax and KL on 2000 episodes, before and after.  Control: the same with the readout token's layer-0 output
           additionally replaced by its dataset mean (RO1 information removed) -> what the early route contributes to the reactivated model.
Streams: RS_STREAM (decoding/eval, default 90_400_000), RS_SETS (likelihood pairs, 90_500_000), RS_TRAIN (reactivation training, 90_300_000).
Usage: RS_TAG=_x python3 scripts/route_status.py SPEC[@step] ...  -> appends logs/route_status<TAG>.jsonl"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP, metrics as MT
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("RS_STREAM", 90_400_000))); n = 3000
X = NV.Bm.sample_fast(NV.ENV, n, rng)["X"]; q = rng.integers(0, 6, n); qo = (q + 1 + rng.integers(0, 5, n)) % 6; i = np.arange(n)
a6 = X.reshape(n, 6, 3, 2)[i, q]; y = np.arange(3)
EVp = np.where(a6.reshape(n, 6)[..., None] == y, np.log(0.8), np.log(0.1)).sum(1); EVp -= EVp.mean(-1, keepdims=True)
EVg = (a6[..., None] == y).sum(2).reshape(n, 9).astype(float)
PB = GL.posterior_closed(RP.ENV1, X, q); S = MT.sets(int(os.environ.get("RS_SETS", 90_500_000)))
def r2(Fe, T):
    tr, te = np.arange(2000), np.arange(2000, n); Fc = np.c_[Fe, np.ones(n)]; W = np.linalg.solve(Fc[tr].T @ Fc[tr] + 1e-2 * np.eye(Fc.shape[1]), Fc[tr].T @ T[tr])
    Yh = Fc[te] @ W; return float(1 - ((T[te] - Yh) ** 2).sum() / ((T[te] - T[te].mean(0)) ** 2).sum())
MASK = np.zeros((43, 43), np.float32)
for qq in range(6): MASK[36 + qq, qq * 6:(qq + 1) * 6] = -1e9
def fwd_masked(M, X_, ro1_mean=None):
    x = F.N.embed(dict(M["emb"]), X_)
    for l in range(3):
        lay = M["layers"][l]; z = F.N.ln(x, lay["g1"], lay["b1"])
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        s = jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]) + (jnp.asarray(MASK) if l == 2 else 0.0)
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", jax.nn.softmax(s, -1), vh), lay["WO"])
        x = x + jax.nn.gelu(F.N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
        if l == 0 and ro1_mean is not None: x = x.at[:, 36:42].set(jnp.asarray(ro1_mean))
    return x
def react(M, ro1_mean=None):
    t = dict(gf=M["gf"], bf=M["bf"], WA=M["WA"]); opt = optax.adam(3e-3); st = opt.init(t); r = np.random.default_rng(int(os.environ.get("RS_TRAIN", 90_300_000)))
    feat = jax.jit(lambda X_: fwd_masked(M, X_, ro1_mean))
    def lossf(t_, H, Y_): lp = jax.nn.log_softmax(F.N.ln(H, t_["gf"], t_["bf"])[:, 36:42] @ t_["WA"], -1); return (-jnp.take_along_axis(lp, Y_[..., None], -1)[..., 0]).mean()
    g = jax.jit(jax.value_and_grad(lossf))
    def ev(t_):
        H = np.concatenate([np.asarray(feat(jnp.asarray(X[a:a + 500]))) for a in range(0, 2000, 500)])
        P = RP.sm(np.asarray(F.N.ln(jnp.asarray(H), t_["gf"], t_["bf"])[:, 36:42] @ t_["WA"])[np.arange(2000), q[:2000]])
        return float((P.argmax(-1) == PB[:2000].argmax(-1)).mean()), float((PB[:2000] * (np.log(PB[:2000] + 1e-12) - np.log(P + 1e-12))).sum(-1).mean())
    before = ev(t)
    for s_ in range(300):
        d = NV.Bm.sample_fast(NV.ENV, 256, r); H = feat(jnp.asarray(d["X"])); l_, gr = g(t, H, jnp.asarray(d["Y"])); up, st = opt.update(gr, st); t = optax.apply_updates(t, up)
    return dict(before_acc=before[0], before_KL=before[1], after_acc=ev(t)[0], after_KL=ev(t)[1])
out = os.path.join(HERE, "..", "logs", "route_status" + os.environ.get("RS_TAG", "") + ".jsonl")
for u in sys.argv[1:]:
    M = xc.load(u); rec = {}; F.forward(M, jnp.asarray(X), rec=rec); r = dict(unit=u)
    for l in (1, 2):
        Z = np.asarray(rec[f"resid{l}"]); Zq = Z[i, 36 + q]; Zo = Z[i, 36 + qo]
        r[f"present_pooled_RO{l}"] = r2(Zq, EVp); r[f"present_group_RO{l}"] = r2(Zq, EVg); r[f"present_pooled_otherRO{l}"] = r2(Zo, EVp)
    r["lik_RO1"] = MT.avail(M, S["PL"], 1); r["lik_RO2"] = MT.avail(M, S["PL"], 2)
    mass, share = RP.head_stats(M, S["Xn"][:1500], S["qn"][:1500], F, jnp); r["mix_mass"] = float(mass.max())
    if os.environ.get("RS_REACT"):
        r["react"] = react(M); mu = np.asarray(rec["resid1"])[:, 36:42].mean(0); r["react_ro1_removed"] = react(M, mu)
    with open(out, "a") as fh: fh.write(json.dumps(r) + "\n")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
