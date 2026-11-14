"""Workstream B5: WHY does the representational conjunction precede the behavioural switch?  Per saved checkpoint of a full_c1 run:
  mag      : full-space interaction ratio  mean||dd_TRUE|| / mean||dd_CTRL||  at resid2, q7 tokens (Investigation XI measure)
  ddnorm   : mean||dd_TRUE|| (absolute magnitude of the conjunction at resid2)
  use_nat  : decision effect of the conjunction with NATURAL activations: mean KL( P(xAP) || P(xAP with layer 1 fed resid1 of xA) )
             minus the same quantity for the CTRL design (where Bayes predicts no eligibility change)       [downstream use, natural]
  use_hyb  : mean KL( P(xAP) || P(layer 2 applied to the additive completion of resid2) ) TRUE minus CTRL     [downstream use of the resid2 interaction]
  gain     : use_hyb / ddnorm^2    (readout sensitivity per unit of represented interaction; second-order since KL ~ quadratic)
  beh_B, beh_A1 : KL(Bayes || model) and KL(A1 || model) on xP cells (h perfect, disagreeing at q: the two rules disagree)
  heur     : beh_B - beh_A1 (> 0: closer to the A1 heuristic)
Predictor test: is the first checkpoint where use_hyb (TRUE-CTRL) exceeds 0.05 nats closer to the behavioural switch (first D2-B-closest
checkpoint, Investigation X traj logs) than the representational-emergence checkpoint (mag >= 2)?
Usage: dyn_mech.py TRAJFILE SPEC... -> logs/dyn_mech<TAG>.json"""
import json, os, sys, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
TRAJ = json.load(open(sys.argv[1])) if sys.argv[1] != "-" else {}
EVP = os.environ.get("FA_EVAL", "dev")
if EVP.startswith("stream:"):
    _r = np.random.default_rng(int(EVP[7:])); ev = dict(X=NV.Bm.sample_fast(NV.ENV, 6000, _r)["X"]); ev["q"] = _r.integers(0, 6, 6000)
else: ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); C = GL.counts(X, 3); a = X.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]
def design(base_n, seed=1104, nmax=300):
    rng = np.random.default_rng(seed); cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq = []
    for i in range(n):
        hs = [h for h in range(3) if not eq[i, q[i], h] and C[i, q[i], h] == base_n]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
        x0 = X[i].copy(); xA = x0.copy(); xA[q[i], 2 * h + 1] = xA[q[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
        for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
        qq.append(q[i])
        if len(qq) >= nmax: break
    return {k: np.array(v) for k, v in cells.items()}, np.array(qq)
D = {"TRUE": design(4), "CTRL": design(3)}; env = GL.Env(3, 1)
def rules(Xp, qq):
    ii = np.arange(len(qq)); rel, col, noi = GL._group_lik(env, Xp); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, qq]); Lg = rel[ii, qq] / noi
    A_ = Xp.reshape(len(Xp), 6, 3, 2)[ii, qq]; pf = (GL.counts(Xp, 3)[ii, qq] == 5).astype(float); e = pf * (A_[..., 0] == A_[..., 1])
    f = lambda x: (lambda P: P / P.sum(-1, keepdims=True))(np.maximum((W[..., None] * (x.sum(1, keepdims=True) - x)[..., None] * Lg).sum(1), 1e-30))
    return f(e), f(pf)
REF = rules(D["TRUE"][0]["xP"], D["TRUE"][1])
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
sm = lambda L: (lambda E_: E_ / E_.sum(-1, keepdims=True))(np.exp(L - L.max(-1, keepdims=True)))
tok = lambda q_: np.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + np.arange(6)[None]], 1)
def tail_from(M, x_in, L1_in=None, x2=None, qq=None):
    ii = np.arange(len(qq))
    if x2 is None:
        lay = M["layers"][1]; x = jnp.asarray(x_in); xin = jnp.asarray(L1_in) if L1_in is not None else x; z = F.N.ln(xin, lay["g1"], lay["b1"])
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        A_ = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / 4.0, -1); oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A_, vh), lay["WO"]).sum(1)
        hid = jax.nn.gelu(F.N.ln(xin + oh, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); x2 = x + oh + hid @ lay["W2"] + lay["c2"]
    x3 = F.block_heads(M["layers"][2], jnp.asarray(x2), None, 2, None, None); return np.asarray(F.N.ln(x3, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[ii, qq]
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; tag = f"{parent}_{spec}_s4000_seed{int(seed or 0)}{suf}"
    steps = sorted(int(re.match(r"step(\d+)\.npz", os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(F.model_dir(tag), "step*.npz")))
    traj = {}
    for stp in steps:
        M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf, step=stp); row = {}
        for nm, (cl, qq) in D.items():
            ii = np.arange(len(qq)); R = {}
            for c, Xc in cl.items(): r = {}; L = np.asarray(F.forward(M, jnp.asarray(Xc), rec=r)); R[c] = ({k: np.asarray(v) for k, v in r.items()}, L[ii, qq])
            q7 = lambda Z: Z[ii[:, None], tok(qq)].reshape(len(qq), -1)
            dd = q7(R["xAP"][0]["resid2"]) - q7(R["xA"][0]["resid2"]) - q7(R["xP"][0]["resid2"]) + q7(R["x00"][0]["resid2"]); row[f"{nm}_ddnorm"] = float(np.linalg.norm(dd, axis=1).mean())
            PAP = sm(R["xAP"][1])
            Lnat = tail_from(M, R["xAP"][0]["resid1"], L1_in=R["xA"][0]["resid1"], qq=qq); row[f"{nm}_use_nat"] = float(kl(PAP, sm(Lnat)).mean())
            x2add = R["xA"][0]["resid2"] + R["xP"][0]["resid2"] - R["x00"][0]["resid2"]; Lh = tail_from(M, None, x2=x2add, qq=qq); row[f"{nm}_use_hyb"] = float(kl(PAP, sm(Lh)).mean())
            if nm == "TRUE":
                P = sm(R["xP"][1]); row["beh_B"] = float(kl(REF[0], P).mean()); row["beh_A1"] = float(kl(REF[1], P).mean())
        row["mag"] = row["TRUE_ddnorm"] / row["CTRL_ddnorm"]; row["use_nat"] = row["TRUE_use_nat"] - row["CTRL_use_nat"]; row["use_hyb"] = row["TRUE_use_hyb"] - row["CTRL_use_hyb"]
        row["gain"] = row["use_hyb"] / (row["TRUE_ddnorm"] ** 2 + 1e-9); row["heur"] = row["beh_B"] - row["beh_A1"]; traj[stp] = row
    sw = None
    for key_ in (sp, sp.replace("_eval_conf9", "_eval_conf9")):
        if key_ in TRAJ: sw = next((r["step"] for r in TRAJ[key_] if r["D2"]["closest"] == "B"), None)
    em = next((s for s in steps if traj[s]["mag"] >= 2.0), None); us = next((s for s in steps if traj[s]["use_hyb"] >= 0.05), None)
    out[sp] = dict(traj=traj, behavioural_switch_step=sw, representation_emergence_step=em, use_onset_step=us)
    print(sp, "switch", sw, "emerge", em, "use_onset", us, [(s, round(traj[s]["mag"], 2), round(traj[s]["use_hyb"], 3), round(traj[s]["use_nat"], 3), round(traj[s]["gain"], 4), round(traj[s]["heur"], 3)) for s in steps], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "dyn_mech" + os.environ.get("FA_TAG", "") + ".json"), "w"), indent=1)
