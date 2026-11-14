"""Workstream B: NECESSITY of each layer's contribution to the agreement x perfectness INTERACTION (total effect, downstream recomputed).
TRUE 2x2 design of factorial.py (n_h(q) = 4 -> P makes h perfect; A makes h agree at q; e_h = 1 only in xAP).
'Interaction removal' for component set C: in the xAP run, replace each component's output (all 43 tokens) by its ADDITIVE COMPLETION
   c(xA) + c(xP) - c(x00)          (taken from the intact runs of the other three cells),
so the component keeps both main effects but loses its own conjunction; everything downstream is recomputed.  Sets:
   L0      : layer-0 heads + mlp0      L1attn : layer-1 heads (mlp1 recomputed)      mlp1 : mlp1 only      L1 : layer-1 heads + mlp1
Outcomes at xAP: remaining fraction of the E-subspace interaction  |Q^T dd|  (dd re-computed with the ablated xAP);
   decision: KL(Bayes || model) and KL(cf_e0 || model), where cf_e0 is the Bayes formula with e_h forced to 0 (what a model that misses the
   conjunction for group h should output);  pref_e0 = fraction of pairs where the model is closer to cf_e0 than to Bayes.
Usage: factorial_ablate.py RSUF SPEC... -> logs/factorial_ablate<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
EVP = os.environ.get("FA_EVAL", "dev")
if EVP.startswith("stream:"):   # fresh natural episodes (confirmation): FA_EVAL=stream:<seed>
    _r = np.random.default_rng(int(EVP[7:])); ev = dict(X=NV.Bm.sample_fast(NV.ENV, 6000, _r)["X"]); ev["q"] = _r.integers(0, 6, 6000)
else: ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); C = GL.counts(X, 3); a = X.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]; rng = np.random.default_rng(1104)
cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq, hh = [], []
for i in range(n):
    hs = [h for h in range(3) if not eq[i, q[i], h] and C[i, q[i], h] == 4]
    if not hs: continue
    h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
    x0 = X[i].copy(); xA = x0.copy(); xA[q[i], 2 * h + 1] = xA[q[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
    for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
    qq.append(q[i]); hh.append(h)
    if len(qq) >= 500: break
cells = {k: np.array(v) for k, v in cells.items()}; qq = np.array(qq); hh = np.array(hh); ii = np.arange(len(qq))
env = GL.Env(3, 1); Xp = cells["xAP"]; rel, col, noi = GL._group_lik(env, Xp); relm = rel.mean(-1); lr = np.log(relm / noi)
W = np.exp(lr.sum(1) - lr[ii, qq]); Lg = rel[ii, qq] / noi; e = ((GL.counts(Xp, 3)[ii, qq] == 5) & (Xp.reshape(len(Xp), 6, 3, 2)[ii, qq][..., 0] == Xp.reshape(len(Xp), 6, 3, 2)[ii, qq][..., 1])).astype(float)
nz = lambda P: P / P.sum(-1, keepdims=True); cfB = nz((W[..., None] * (e.sum(1, keepdims=True) - e)[..., None] * Lg).sum(1))
e0 = e.copy(); e0[ii, hh] = 0; cf0 = nz(np.maximum((W[..., None] * (e0.sum(1, keepdims=True) - e0)[..., None] * Lg).sum(1), 1e-30))
assert e[ii, hh].min() == 1
tok = lambda q_: np.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + np.arange(6)[None]], 1)
q7 = lambda Z: np.asarray(Z)[ii[:, None], tok(qq)].reshape(len(qq), -1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
SETS = {"none": [], "L0": [(0, h) for h in range(4)] + ["mlp0"], "L1attn": [(1, h) for h in range(4)], "mlp1": ["mlp1"], "L1": [(1, h) for h in range(4)] + ["mlp1"]}
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy"))); recs = {}
    for k, Xc in cells.items(): r = {}; F.forward(M, jnp.asarray(Xc), rec=r); recs[k] = r
    add = lambda key, idx=None: (lambda f: f(recs["xA"]) + f(recs["xP"]) - f(recs["x00"]))(lambda r: r[key] if idx is None else r[key][:, idx])
    base_dd = lambda zAP: (q7(zAP) - q7(recs["xA"]["resid2"]) - q7(recs["xP"]["resid2"]) + q7(recs["x00"]["resid2"])) @ Q
    I0 = np.linalg.norm(base_dd(recs["xAP"]["resid2"]), axis=1).mean(); res = {}
    allm = jnp.ones((len(qq), 43), bool)
    for nm, items in SETS.items():
        hp = {it: (allm, add(f"heads{it[0]}", it[1])) for it in items if isinstance(it, tuple)}
        mp = {int(it[3]): (allm, add(f"mlp_out{it[3]}")) for it in items if isinstance(it, str)}
        r = {}; L = np.asarray(F.forward(M, jnp.asarray(Xp), head_patch=hp or None, mlp_patch=mp or None, rec=r))[ii, qq]
        P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True); kB, k0 = kl(cfB, P), kl(cf0, P)
        res[nm] = dict(E_interaction_remaining=float(np.linalg.norm(base_dd(r["resid2"]), axis=1).mean() / I0), KL_bayes=float(kB.mean()), KL_e0=float(k0.mean()),
                       pref_e0=float((k0 < kB).mean()), argmax_bayes=float((P.argmax(-1) == cfB.argmax(-1)).mean()))
    res["ref_KL_bayes_e0"] = float(kl(cfB, cf0).mean()); res["ref_disc_frac"] = float((cfB.argmax(-1) != cf0.argmax(-1)).mean())
    out[sp] = res; print(sp, {k: (round(v["E_interaction_remaining"], 2), round(v["KL_bayes"], 3), round(v["KL_e0"], 3), round(v["pref_e0"], 2)) for k, v in res.items() if isinstance(v, dict)}, "ref", round(res["ref_KL_bayes_e0"], 3), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "factorial_ablate" + os.environ.get("FA_TAG", "") + ".json"), "w"), indent=1)
