"""Workstream B5: WHEN does the agreement x perfectness conjunction emerge during fine-tuning, relative to the behavioural A1 -> B switch?
At each saved checkpoint of a full_c1 run (steps 74 ... 4000) compute, with the 2x2 factorial design of factorial.py (TRUE n_h=4, CTRL n_h=3):
  ratio(step) = mean||dd_TRUE|| / mean||dd_CTRL||   at resid2, q7 tokens, FULL space (no subspace: the final E-subspace need not apply early)
  rel_I_TRUE(mid0), rel_I_TRUE(mid1)          (where the interaction lives)
  bdist(step) = behavioural closeness: KL(Bayes || model) and KL(A1 || model) on the factorial xP cells (h perfect, disagreeing at q: B and A1 differ)
The behavioural switch step is the first checkpoint where KL(Bayes) < KL(A1).  Usage: dyn_factorial.py SPEC... -> logs/dyn_factorial<TAG>.json"""
import json, os, sys, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
EVP = os.environ.get("FA_EVAL", "dev")
if EVP.startswith("stream:"):   # fresh natural episodes (confirmation): FA_EVAL=stream:<seed>
    _r = np.random.default_rng(int(EVP[7:])); ev = dict(X=NV.Bm.sample_fast(NV.ENV, 6000, _r)["X"]); ev["q"] = _r.integers(0, 6, 6000)
else: ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); C = GL.counts(X, 3); a = X.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]
def design(base_n, seed=1104, nmax=400):
    rng = np.random.default_rng(seed); cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq, hh = [], []
    for i in range(n):
        hs = [h for h in range(3) if not eq[i, q[i], h] and C[i, q[i], h] == base_n]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
        x0 = X[i].copy(); xA = x0.copy(); xA[q[i], 2 * h + 1] = xA[q[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
        for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
        qq.append(q[i]); hh.append(h)
        if len(qq) >= nmax: break
    return {k: np.array(v) for k, v in cells.items()}, np.array(qq), np.array(hh)
D = {"TRUE": design(4), "CTRL": design(3)}
env = GL.Env(3, 1)
def rules(Xp, qq):
    ii = np.arange(len(qq)); rel, col, noi = GL._group_lik(env, Xp); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, qq]); Lg = rel[ii, qq] / noi
    A_ = Xp.reshape(len(Xp), 6, 3, 2)[ii, qq]; pf = (GL.counts(Xp, 3)[ii, qq] == 5).astype(float); e = pf * (A_[..., 0] == A_[..., 1])
    f = lambda x: (lambda P: P / P.sum(-1, keepdims=True))(np.maximum((W[..., None] * (x.sum(1, keepdims=True) - x)[..., None] * Lg).sum(1), 1e-30))
    return f(e), f(pf)
REF = rules(D["TRUE"][0]["xP"], D["TRUE"][1])     # xP: h perfect but NOT agreeing at q -> Bayes e_h = 0, A1 (perfect-only) counts h
tok = lambda q_: np.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + np.arange(6)[None]], 1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; tag = f"{parent}_{spec}_s4000_seed{int(seed or 0)}{suf}"
    steps = sorted(int(re.match(r"step(\d+)\.npz", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(F.model_dir(tag), "step*.npz")))
    traj = {}
    for stp in steps:
        M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf, step=stp); row = {}
        for nm, (cl, qq, hh) in D.items():
            ii = np.arange(len(qq)); recs = {}
            for k, Xc in cl.items(): r = {}; L = np.asarray(F.forward(M, jnp.asarray(Xc), rec=r)); recs[k] = ({s: np.asarray(v) for s, v in r.items()}, L)
            st = lambda k, s: (lambda R: {"mid0": R["resid0"] + R["heads0"].sum(1), "mid1": R["resid1"] + R["heads1"].sum(1), "resid2": R["resid2"]}[s])(recs[k][0])
            q7 = lambda Z: Z[ii[:, None], tok(qq)].reshape(len(qq), -1)
            for s in ("mid0", "mid1", "resid2"):
                dd = q7(st("xAP", s)) - q7(st("xA", s)) - q7(st("xP", s)) + q7(st("x00", s)); mA = q7(st("xA", s)) - q7(st("x00", s)); mP = q7(st("xP", s)) - q7(st("x00", s))
                row[f"{nm}_{s}_dd"] = float(np.linalg.norm(dd, axis=1).mean()); row[f"{nm}_{s}_relI"] = row[f"{nm}_{s}_dd"] / float(np.linalg.norm(mA, axis=1).mean() + np.linalg.norm(mP, axis=1).mean())
            if nm == "TRUE":
                Lq = recs["xP"][1][ii, qq]; P = np.exp(Lq - Lq.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
                row["KL_bayes"] = float(kl(REF[0], P).mean()); row["KL_A1"] = float(kl(REF[1], P).mean())
        row["ratio_resid2"] = row["TRUE_resid2_dd"] / row["CTRL_resid2_dd"]; row["ratio_mid1"] = row["TRUE_mid1_dd"] / row["CTRL_mid1_dd"]
        traj[stp] = row
    sw = next((s for s in steps if traj[s]["KL_bayes"] < traj[s]["KL_A1"]), None); em = next((s for s in steps if traj[s]["ratio_resid2"] >= 2.0), None)
    out[sp] = dict(traj=traj, behavioural_switch_step=sw, interaction_emergence_step=em)
    print(sp, "switch", sw, "emerge(ratio>=2)", em, [(s, round(traj[s]["ratio_resid2"], 2), round(traj[s]["KL_bayes"], 3), round(traj[s]["KL_A1"], 3)) for s in steps], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "dyn_factorial" + os.environ.get("FA_TAG", "") + ".json"), "w"), indent=1)
