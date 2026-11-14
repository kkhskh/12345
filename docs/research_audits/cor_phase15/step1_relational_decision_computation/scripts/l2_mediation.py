"""Workstreams A3/B1/B2: does the eligibility of group h reach the decision THROUGH the layer-2 mixture head's attention (a mediator identified
without DAS)?  2x2 factorial cells (XI/XII design): TRUE design n_h(q)=4 (e_h = 1 only in xAP), CTRL design n_h(q)=3 (e_h = 0 everywhere).
Natural-activation path intervention (class II): run xAP, but replace the readout-token attention ROW of each mixture head by that head's row
in the natural xP run (h perfect, NOT agreeing at q -> e_h = 0 in TRUE; also 0 in CTRL).  Values, keys elsewhere, all other heads, MLP2 from xAP.
Decoy control: the same replacement in a NON-mixture layer-2 head (lowest q-token mass).
Measures (signed, entropy-neutral): the change of the centred log-probability vector at q,
   NIE = cen(P_patched) - cen(P_xAP);   TE = cen(P_xP) - cen(P_xAP)
   ATT-MIX prediction for NIE: mixture with xP's Bayes responsibilities and xAP's likelihoods, minus Bayes(xAP)  -> fidelity slope / R^2
   eligibility-specific effect = |NIE|_TRUE - |NIE|_CTRL (difference in differences over matched designs), with episode-bootstrap CI
Usage: l2_mediation.py SPEC... -> logs/l2_mediation<TAG>.json   (FA_EVAL as in factorial.py)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, stats_xii as ST
np.seterr(all="ignore")
EVP = os.environ.get("FA_EVAL", "stream:72400000"); _r = np.random.default_rng(int(EVP[7:])); X0 = NV.Bm.sample_fast(NV.ENV, 6000, _r)["X"]; q0 = _r.integers(0, 6, 6000)
n = len(X0); C = GL.counts(X0, 3); a = X0.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]; env = GL.Env(3, 1)
def design(base_n, seed=1304, nmax=400):
    rng = np.random.default_rng(seed); cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq = []
    for i in range(n):
        hs = [h for h in range(3) if not eq[i, q0[i], h] and C[i, q0[i], h] == base_n]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q0[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
        x0 = X0[i].copy(); xA = x0.copy(); xA[q0[i], 2 * h + 1] = xA[q0[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
        for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
        qq.append(q0[i])
        if len(qq) >= nmax: break
    return {k: np.array(v) for k, v in cells.items()}, np.array(qq)
D = {"TRUE": design(4), "CTRL": design(3)}
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def bayes_parts(Xc, qq):
    ii = np.arange(len(qq)); rel, col, noi = GL._group_lik(env, Xc); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, qq]); Lg = rel[ii, qq] / noi
    A_ = Xc.reshape(len(Xc), 6, 3, 2)[ii, qq]; e = ((GL.counts(Xc, 3)[ii, qq] == 5) & (A_[..., 0] == A_[..., 1])).astype(float)
    r = W * (e.sum(1, keepdims=True) - e) * Lg.sum(-1); return r / r.sum(1, keepdims=True), Lg / Lg.sum(-1, keepdims=True), GL.posterior_closed(env, Xc, qq)
def fid(o, p):
    o, p = o.ravel(), p.ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); return dict(slope=b, R2=float(((1 - ((o - b * p) ** 2).sum() / ((o - o.mean()) ** 2).sum()) if ((o - o.mean()) ** 2).sum() > 1e-10 else 0.0)))
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); lay = M["layers"][2]; res = {}
    for nm, (cl, qq) in D.items():
        N_ = len(qq); ii = np.arange(N_); recs = {}
        for k, Xc in cl.items(): r = {}; F.forward(M, jnp.asarray(Xc), rec=r); recs[k] = {kk: np.asarray(r[kk]) for kk in ("resid2", "pattern2")}
        tq = qq[:, None] * 6 + np.arange(6)[None]
        mass = np.array([recs["xAP"]["pattern2"][ii, h, 36 + qq][ii[:, None], tq].sum(1).mean() for h in range(4)])
        if nm == "TRUE": MH = [h for h in range(4) if mass[h] >= 0.8]; DEC = [int(np.argmin(mass))]
        if not MH: break
        def run(rows_from, heads):
            xb = jnp.asarray(recs["xAP"]["resid2"]); z = F.N.ln(xb, lay["g1"], lay["b1"])
            qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
            A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1); t = jnp.asarray(36 + qq); nn = jnp.arange(N_)
            for h in heads: A = A.at[nn, h, t].set(jnp.asarray(recs[rows_from]["pattern2"][ii, h, 36 + qq]))
            oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"]).sum(1); xa = xb + oh
            hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); x3 = xa + hid @ lay["W2"] + lay["c2"]
            Lx = np.asarray(F.N.ln(x3, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[ii, qq]; P = np.exp(Lx - Lx.max(-1, keepdims=True)); return P / P.sum(-1, keepdims=True)
        P_ap = run("xAP", []); P_pat = run("xP", MH); P_dec = run("xP", DEC)
        rec_xp = {}; Lp = np.asarray(F.forward(M, jnp.asarray(cl["xP"])))[ii, qq]; P_xp = np.exp(Lp - Lp.max(-1, keepdims=True)); P_xp /= P_xp.sum(-1, keepdims=True)
        NIE = cen(P_pat) - cen(P_ap); NIEd = cen(P_dec) - cen(P_ap); TE = cen(P_xp) - cen(P_ap)
        r_xp, _, _ = bayes_parts(cl["xP"], qq); _, Ln_ap, PB_ap = bayes_parts(cl["xAP"], qq)
        pred = (r_xp[..., None] * Ln_ap).sum(1); pred /= pred.sum(-1, keepdims=True); pred = cen(pred) - cen(PB_ap)
        res[nm] = dict(n=N_, mag_NIE=np.abs(NIE).mean(1).astype(np.float32).tolist(), mag_NIE_decoy=np.abs(NIEd).mean(1).astype(np.float32).tolist(),
                       mean_mag_NIE=float(np.abs(NIE).mean()), mean_mag_NIE_decoy=float(np.abs(NIEd).mean()), mean_mag_TE=float(np.abs(TE).mean()),
                       frac_TE_mediated=float((NIE * TE).sum() / (TE * TE).sum()), fidelity_ATTMIX=fid(NIE, pred), mean_mag_pred=float(np.abs(pred).mean()))
    if not MH: out[sp] = dict(mixture_heads=[]); print(sp, "no mixture head"); continue
    T, Cc = res["TRUE"], res["CTRL"]; did = ST.ci_mean(np.array(T["mag_NIE"]) - np.mean(Cc["mag_NIE"]))
    res.update(mixture_heads=MH, decoy_heads=DEC, DiD_mag_NIE_true_minus_ctrl=did)
    out[sp] = res
    print(sp, "MH", MH, "dec", DEC, {nm: {k: (round(v, 3) if isinstance(v, float) else ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else None)) for k, v in res[nm].items() if not k.startswith("mag_") and k != "n"} for nm in ("TRUE", "CTRL")}, "DiD", {k: round(v, 3) for k, v in did.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_mediation" + os.environ.get("MD_TAG", "") + ".json"), "w"))
