"""XIII-D Workstream A3: a SMALL, pre-declared set of mechanistically motivated step-0 measurements of the fresh readout map WA
(and, for contrast, of the fresh layer 2), computed for a (parent, L, W) initialisation BEFORE any training step.

Mechanistic rationale.  The fresh readout map is the only route by which the loss gradient enters the network: dL/dh_RO = WA (p - y) at the
readout tokens (after the final LN).  Whether the parent's EXISTING early-readout features (candidate evidence already present at the readout
token after layers 0-1, from pretraining) are immediately useful depends on how WA maps them to the three candidate logits.

Measurements (all on 1500 natural episodes, stream 90_100_000, random target q):
  S_RO1  signed slope of the step-0 centred logits obtained by applying final LN + WA to the readout-token residual after LAYER 1 (bypassing the
         fresh layer 2) on the centred pooled candidate log-evidence at q  -> does WA already read the parent's early-readout features with the
         right sign?   [principal, pre-declared]
  S_RO0  same after layer 0 only.
  S_full signed slope of the actual step-0 logits (through the fresh layer 2).
  R2_RO1 fraction of variance of the centred step-0 RO1 logits explained by the candidate evidence (|alignment| irrespective of sign).
  g_L0, g_L1, g_L2, g_WA  step-0 gradient norms of the label loss w.r.t. layer-0 / layer-1 / layer-2 / WA parameters; g_ratio = g_L0 / g_L2.
  sv_WA  singular values of the column-centred WA (the 2-dim candidate-contrast subspace it reads), and ||WA||_F.
Usage: w_features.py PARENT:L:W ... [env WF_TAG, WF_WA=transform for interventions: perm<ijk> | neg | rot<theta> | scale<s>]
-> appends logs/w_features<TAG>.jsonl"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV, ftlib as F, relpairs as RP
NL, N = NV.NL, NV.N
rng = np.random.default_rng(90_100_000); n = 1500
d = NV.Bm.sample_fast(NV.ENV, n, rng); X = d["X"]; Y = d["Y"]; q = rng.integers(0, 6, n); i = np.arange(n)
a6 = X.reshape(n, 6, 6)[i, q]; y = np.arange(3); EV = np.where(a6[..., None] == y, np.log(0.8), np.log(0.1)).sum(1); EVc = EV - EV.mean(-1, keepdims=True)
def wa_transform(WA, spec):
    if not spec: return WA
    if spec.startswith("perm"): return WA[:, [int(c) for c in spec[4:]]]
    if spec == "neg": return -WA
    if spec.startswith("scale"): return WA * float(spec[5:])
    raise ValueError(spec)
def build(parent, L, W, spec=""):
    pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
    return dict(emb={k: pf[k] for k in NL.FROZEN_KEYS}, layers=[pf["layers"][0], pf["layers"][1], fresh(L + 7777)["layers"][2]], gf=pf["gf"], bf=pf["bf"],
                WA=jnp.asarray(wa_transform(np.asarray(fresh(W + 7777)["WA"]), spec)))
def slope(Lg):
    c = Lg - Lg.mean(-1, keepdims=True); return float((c * EVc).sum() / (EVc * EVc).sum()), float(1 - ((c - (c * EVc).sum() / (EVc * EVc).sum() * EVc) ** 2).sum() / max((c * c).sum(), 1e-12))
def features(M):
    rec = {}; Lfull = np.asarray(F.forward(M, jnp.asarray(X), rec=rec))[i, q]; r = {}
    for l in (0, 1):
        h = jnp.asarray(np.asarray(rec[f"resid{l + 1}"])[i, 36 + q])[:, None]
        Lr = np.asarray(F.N.ln(h, M["gf"], M["bf"])[:, 0] @ M["WA"]); r[f"S_RO{l}"], r[f"R2_RO{l}"] = slope(Lr)
    r["S_full"], r["R2_full"] = slope(Lfull)
    def loss(layers, WA):
        Mx = dict(M); Mx["layers"] = layers; Mx["WA"] = WA; lp = jax.nn.log_softmax(F.forward(Mx, jnp.asarray(X[:500])), -1)
        return (-jnp.take_along_axis(lp, jnp.asarray(Y[:500])[..., None], -1)[..., 0]).mean()
    gl, gw = jax.grad(loss, argnums=(0, 1))(M["layers"], M["WA"])
    nrm = lambda t: float(np.sqrt(sum(float((v ** 2).sum()) for v in jax.tree_util.tree_leaves(t))))
    r.update(g_L0=nrm(gl[0]), g_L1=nrm(gl[1]), g_L2=nrm(gl[2]), g_WA=nrm(gw)); r["g_ratio"] = r["g_L0"] / r["g_L2"]
    WA = np.asarray(M["WA"]); Wc = WA - WA.mean(1, keepdims=True); r["sv_WA"] = np.linalg.svd(Wc, compute_uv=False)[:2].tolist(); r["norm_WA"] = float(np.linalg.norm(WA))
    return r
if __name__ == "__main__":
    out = os.path.join(HERE, "..", "logs", "w_features" + os.environ.get("WF_TAG", "") + ".jsonl"); spec = os.environ.get("WF_WA", "")
    for u in sys.argv[1:]:
        p, L, W = u.split(":"); r = features(build(p, int(L), int(W), spec)); r.update(unit=u, wa_transform=spec)
        with open(out, "a") as fh: fh.write(json.dumps(r) + "\n")
        print(u, spec, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("unit",)}, flush=True)
