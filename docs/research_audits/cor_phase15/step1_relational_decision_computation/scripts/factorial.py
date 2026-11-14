"""Workstream B (principal): within-episode 2x2 FACTORIAL test of WHERE agreement-at-q and perfectness are COMBINED.
For a natural episode, target question q and group h that disagrees at q:
  factor A : make h agree at q              (copy source 2h's q-report into source 2h+1)
  factor P : make h agree at the one other question q' where it disagrees   (n_h(q): 4 -> 5 = perfect)
  cells x00, xA, xP, xAP;  e_h = 1 only in xAP.
  TRUE  design: n_h(q) = 4 at base (P makes the group perfect)  -> Bayes e_h: 0,0,0,1  (pure interaction)
  CTRL  design: n_h(q) = 3 at base (P makes 3 -> 4, still imperfect) -> e_h: 0,0,0,0   (no interaction in e)
Double difference dd = z(xAP) - z(xA) - z(xP) + z(x00) at every stage (same episode, same token positions: no alignment needed).
  rel_I(stage) = mean ||dd|| / mean( ||z(xA)-z(x00)|| + ||z(xP)-z(x00)|| )          at the q7 tokens
Residual additivity: resid2 = emb + sum_l (sum_h head_lh + mlp_l), so dd(resid2) = sum_c dd(c) EXACTLY; we report each component's share
of the interaction inside the Investigation X E-subspace Q (share_c = <Q^T dd_c, Q^T dd>/|Q^T dd|^2) and in the full q7 space.
Decision: interaction of the centred logit at q of the agreed value v (= source 2h's report at q).
Usage: factorial.py RSUF SPEC...  -> logs/factorial<TAG>.json"""
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
X = ev["X"]; q = ev["q"]; n = len(X); ix = np.arange(n); C = GL.counts(X, 3); a = X.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]
rng = np.random.default_rng(1104)
def design(base_n, nmax=500):
    cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq, vv = [], []
    for i in range(n):
        hs = [h for h in range(3) if not eq[i, q[i], h] and C[i, q[i], h] == base_n]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
        x0 = X[i].copy(); xA = x0.copy(); xA[q[i], 2 * h + 1] = xA[q[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
        for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
        qq.append(q[i]); vv.append(x0[q[i], 2 * h])
        if len(qq) >= nmax: break
    return {k: np.array(v) for k, v in cells.items()}, np.array(qq), np.array(vv)
designs = {"TRUE": design(4), "CTRL": design(3)}
def e_all(Xc, qq):
    i = np.arange(len(qq)); A_ = Xc.reshape(len(Xc), 6, 3, 2)[i, qq]; return ((GL.counts(Xc, 3)[i, qq] == 5) & (A_[..., 0] == A_[..., 1])).sum(-1)
SANITY = {nm: {k: float((e_all(cl[k], qq) - e_all(cl["x00"], qq)).mean()) for k in cl} for nm, (cl, qq, vv) in designs.items()}
print("sanity: mean change in S relative to x00", SANITY, flush=True)
COMPS = ["emb"] + [f"h{l}.{h}" for l in (0, 1) for h in range(4)] + ["mlp0", "mlp1"]
tok = lambda qq: np.concatenate([(36 + qq)[:, None], qq[:, None] * 6 + np.arange(6)[None]], 1)
def q7(Z, qq): i = np.arange(len(qq)); return Z[i[:, None], tok(qq)].reshape(len(qq), -1)
out = {"sanity_dS": SANITY}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy"))); r = {}
    for nm, (cl, qq, vv) in designs.items():
        st, comp, lg, lb = {}, {}, {}, {}
        for k, Xc in cl.items():
            rec = {}; L = np.asarray(F.forward(M, jnp.asarray(Xc), rec=rec)); R = {kk: np.asarray(v) for kk, v in rec.items()}
            st[k] = {"resid0": R["resid0"], "mid0": R["resid0"] + R["heads0"].sum(1), "resid1": R["resid1"], "mid1": R["resid1"] + R["heads1"].sum(1), "resid2": R["resid2"]}
            comp[k] = {"emb": R["resid0"], **{f"h{l}.{h}": R[f"heads{l}"][:, h] for l in (0, 1) for h in range(4)}, "mlp0": R["mlp_out0"], "mlp1": R["mlp_out1"]}
            i = np.arange(len(qq)); c = F.centred(L[i, qq]); lg[k] = c[i, vv]
            lp = np.log(np.maximum(GL.posterior_closed(GL.Env(G=3, C=1), Xc, qq), 1e-12)); lb.setdefault(k, (lp - lp.mean(-1, keepdims=True))[i, vv])
        dd = lambda D, s: q7(D["xAP"][s], qq) - q7(D["xA"][s], qq) - q7(D["xP"][s], qq) + q7(D["x00"][s], qq)
        stages = {}
        for s in st["x00"]:
            S_ = {k: st[k] for k in st}; ddv = dd(S_, s); mA = q7(st["xA"][s], qq) - q7(st["x00"][s], qq); mP = q7(st["xP"][s], qq) - q7(st["x00"][s], qq)
            stages[s] = dict(rel_I=float(np.linalg.norm(ddv, axis=1).mean() / (np.linalg.norm(mA, axis=1).mean() + np.linalg.norm(mP, axis=1).mean())),
                             rel_I_E=float(np.linalg.norm(ddv @ Q, axis=1).mean() / (np.linalg.norm(mA @ Q, axis=1).mean() + np.linalg.norm(mP @ Q, axis=1).mean())) if s == "resid2" else None)
        ddE = dd(st, "resid2") @ Q; tot = (ddE ** 2).sum(); ddF = dd(st, "resid2"); totF = (ddF ** 2).sum()
        shares = {c: float((((q7(comp["xAP"][c], qq) - q7(comp["xA"][c], qq) - q7(comp["xP"][c], qq) + q7(comp["x00"][c], qq)) @ Q) * ddE).sum() / tot) for c in COMPS}
        sharesF = {c: float(((q7(comp["xAP"][c], qq) - q7(comp["xA"][c], qq) - q7(comp["xP"][c], qq) + q7(comp["x00"][c], qq)) * ddF).sum() / totF) for c in COMPS}
        ddl = lg["xAP"] - lg["xA"] - lg["xP"] + lg["x00"]
        r[nm] = dict(n=int(len(qq)), stages=stages, E_interaction_norm=float(np.linalg.norm(ddE, axis=1).mean()), per_episode_E_norm=np.linalg.norm(ddE, axis=1).astype(np.float32).tolist(), per_episode_full_norm=np.linalg.norm(ddF, axis=1).astype(np.float32).tolist(), share_E=shares, share_full=sharesF,
                     logit=dict(main_A=float((lg["xA"] - lg["x00"]).mean()), main_P=float((lg["xP"] - lg["x00"]).mean()), interaction=float(ddl.mean()),
                                interaction_se=float(ddl.std() / np.sqrt(len(ddl))), bayes_main_A=float((lb["xA"] - lb["x00"]).mean()), bayes_main_P=float((lb["xP"] - lb["x00"]).mean()),
                                bayes_interaction=float((lb["xAP"] - lb["xA"] - lb["xP"] + lb["x00"]).mean())))
    out[sp] = r
    print(sp, {nm: ({s: round(v["rel_I"], 3) for s, v in r[nm]["stages"].items()}, round(r[nm]["stages"]["resid2"]["rel_I_E"], 3), round(r[nm]["E_interaction_norm"], 3),
                    {c: round(v, 2) for c, v in r[nm]["share_E"].items() if abs(v) > 0.08}, {k: round(v, 3) for k, v in r[nm]["logit"].items()}) for nm in r}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "factorial" + os.environ.get("FA_TAG", "") + ".json"), "w"), indent=1)
