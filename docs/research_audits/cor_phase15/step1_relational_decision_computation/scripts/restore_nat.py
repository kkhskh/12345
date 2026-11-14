"""Workstream B4 (XII): restoration after a NATURAL-activation removal (path_inA: layer 1 fed the natural xA residual; bypass keeps xAP),
instead of Investigation XI's additive completion.  Otherwise identical to restore.py.  Original docstring:
Workstream B (restoration / mediation).  After removing layer 1's conjunction (factorial_ablate.py set L1: additive completion of every
layer-1 head and MLP1 at xAP), restore ONLY the E-subspace coordinates of the intact xAP state at resid2 (q7 tokens):
   x_rest = x_abl + Q Q^T (x_intact - x_abl)          (k = 6 dims out of 448)
and recompute layer 2.  If the decision-relevant conjunction is carried by the E-subspace, the decision returns to Bayes.
Controls: restore a random 6-dim subspace; restore the orthogonal complement (all q7 coordinates EXCEPT the E-subspace).
Reports KL(Bayes || model) for intact / ablated / restored-E / restored-random / restored-complement, and the recovered fraction
   rec = (KL_abl - KL_x) / (KL_abl - KL_intact).
Usage: restore.py RSUF SPEC... -> logs/restore<TAG>.json   (FA_EVAL as in factorial.py)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
EVP = os.environ.get("FA_EVAL", "dev")
if EVP.startswith("stream:"):
    _r = np.random.default_rng(int(EVP[7:])); ev = dict(X=NV.Bm.sample_fast(NV.ENV, 6000, _r)["X"]); ev["q"] = _r.integers(0, 6, 6000)
else: ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); C = GL.counts(X, 3); a = X.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]; rng = np.random.default_rng(1104)
cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq = []
for i in range(n):
    hs = [h for h in range(3) if not eq[i, q[i], h] and C[i, q[i], h] == 4]
    if not hs: continue
    h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
    x0 = X[i].copy(); xA = x0.copy(); xA[q[i], 2 * h + 1] = xA[q[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
    for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
    qq.append(q[i])
    if len(qq) >= 500: break
cells = {k: np.array(v) for k, v in cells.items()}; qq = np.array(qq); ii = np.arange(len(qq)); Xp = cells["xAP"]
cfB = GL.posterior_closed(GL.Env(3, 1), Xp, qq)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
tokj = jnp.asarray(np.concatenate([(36 + qq)[:, None], qq[:, None] * 6 + np.arange(6)[None]], 1)); iij = jnp.asarray(ii)[:, None]
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy"))); Qr, _ = np.linalg.qr(np.random.default_rng(5).standard_normal(Q.shape))
    recs = {}
    for k, Xc in cells.items(): r = {}; F.forward(M, jnp.asarray(Xc), rec=r); recs[k] = r
    add = lambda key, idx=None: (lambda f: f(recs["xA"]) + f(recs["xP"]) - f(recs["x00"]))(lambda r: r[key] if idx is None else r[key][:, idx])
    lay1 = M["layers"][1]; xr1 = jnp.asarray(recs["xAP"]["resid1"]); z = F.N.ln(jnp.asarray(recs["xA"]["resid1"]), lay1["g1"], lay1["b1"])
    qh = jnp.einsum("ntd,hde->nhte", z, lay1["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay1["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay1["WV"])
    A_ = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / 4.0, -1); oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A_, vh), lay1["WO"]).sum(1)
    hid = jax.nn.gelu(F.N.ln(jnp.asarray(recs["xA"]["resid1"]) + oh, lay1["g2"], lay1["b2"]) @ lay1["W1"] + lay1["c1"], approximate=True)
    x_abl = xr1 + oh + hid @ lay1["W2"] + lay1["c2"]; x_int = recs["xAP"]["resid2"]
    def run_with(Pm):
        Pm = jnp.asarray(Pm)
        def fn(x):
            vb = x_abl[iij, tokj].reshape(len(qq), -1); vi = x_int[iij, tokj].reshape(len(qq), -1); vp = vb + (vi - vb) @ Pm
            return x_abl.at[iij, tokj].set(vp.reshape(len(qq), 7, 64))
        lay = M["layers"][2]; x = F.block_heads(lay, fn(x_abl), None, 2, None, None); L = np.asarray(F.N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[ii, qq]
        P = np.exp(L - L.max(-1, keepdims=True)); return P / P.sum(-1, keepdims=True)
    I448 = np.eye(448); res = {}
    for nm, Pm in (("ablated", 0 * I448), ("restore_E", Q @ Q.T), ("restore_random", Qr @ Qr.T), ("restore_complement", I448 - Q @ Q.T), ("restore_all", I448)):
        res[nm] = float(kl(cfB, run_with(Pm)).mean())
    for nm in ("restore_E", "restore_random", "restore_complement"): res[f"rec_{nm[8:]}"] = (res["ablated"] - res[nm]) / (res["ablated"] - res["restore_all"])
    out[sp] = res; print(sp, {k: round(v, 3) for k, v in res.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "restore_nat" + os.environ.get("FA_TAG", "") + ".json"), "w"), indent=1)
