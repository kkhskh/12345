"""Reviewer check: is Q5's low E-fidelity in early-readout units an artefact of the target dE = cen(cfE_Bayes) - cen(P_model_base),
which contains the model's own base-episode deviation from Bayes?  Re-score the SAVED E-trained subspaces (no retraining) against
(i) the original target, (ii) the Bayes-to-Bayes change cen(cfE) - cen(Bayes_base), (iii) the model's own fitted power-mean counterfactual change."""
import os, sys, json, time
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, jax.numpy as jnp
from scipy.special import logsumexp as lse
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore"); t0 = time.time()
def cfp(W, L, x):
    P = np.maximum((W[..., None] * (x.sum(1, keepdims=True) - x)[..., None] * L).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
def sample(n, stream, frac_disc):
    rng = np.random.default_rng(stream); need_d = int(frac_disc * n); need_o = n - need_d; acc = {k: [] for k in ("Xb", "Xs", "qb", "qs", "cfE", "cfA1", "eb", "es", "disc")}; nd = no = 0
    while nd < need_d or no < need_o:
        d = NV.Bm.sample_fast(NV.ENV, 20000, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); W, L, pf, e, S = RP.parts(X, q)
        ok_b = np.where(e.sum(1) >= 1)[0]; bi = rng.choice(ok_b, 200000); si = rng.integers(0, len(X), 200000); ri = rng.integers(0, len(X), 200000)
        okp = (e[si].sum(1) >= 1) & (pf[si].sum(1) >= 1); cE = cfp(W[bi], L[bi], e[si]); cA = cfp(W[bi], L[bi], pf[si]); disc = okp & (cE.argmax(-1) != cA.argmax(-1))
        D_ = np.where(disc)[0][:max(need_d - nd, 0)]; O_ = np.where(okp & ~disc)[0][:max(need_o - no, 0)]; sel = np.concatenate([D_, O_]); nd += len(D_); no += len(O_)
        for k, v in (("Xb", X[bi[sel]]), ("Xs", X[si[sel]]), ("qb", q[bi[sel]]), ("qs", q[si[sel]]), ("cfE", cE[sel]), ("cfA1", cA[sel]), ("eb", e[bi[sel]]), ("es", e[si[sel]]), ("disc", disc[sel])): acc[k].append(v)
        _ = cfp(W[bi[sel]], L[bi[sel]], e[ri[sel]])   # keep RNG-consumption identical to das_bal (cfRANDE computed there; no rng use)
    out = {k: np.concatenate(v) for k, v in acc.items()}; perm = rng.permutation(len(out["Xb"])); out = {k: v[perm] for k, v in out.items()}; return out
tok = lambda q_, xp=np: xp.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + xp.arange(6)[None]], 1)
ev = sample(3000, 79_400_000, 1.0)
Wb, Lb, pfb, ebb, Sb = RP.parts(ev["Xb"], ev["qb"]); assert np.array_equal(ebb, ev["eb"])
PBb = cfp(Wb, Lb, ebb); PBb2 = GL.posterior_closed(RP.ENV1, ev["Xb"], ev["qb"])
print("check Bayes base cfp vs posterior_closed max diff", float(np.abs(PBb - PBb2).max()), "n", len(ev["Xb"]), round(time.time() - t0), "s", flush=True)
PMP = json.load(open(os.path.join(PKG, "logs", "fit_powermean_conf13b_addendum.json")))
def pm_pred(th, W, L, e):
    al, ka, de, tau, rho = th; f = e.sum(1, keepdims=True) - e; z = al * np.log(W) + ka * np.log(np.maximum(f, 1)) + de * (f == 0); lw = z - lse(z, 1, keepdims=True)
    lp = tau * lse(lw[..., None] + rho * np.log(L), 1) / rho; return lp - lp.mean(-1, keepdims=True)
res = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    R = np.load(os.path.join(PKG, "logs", "das_bal_conf13b", f"{M['tag']}_R_E.npy"))
    def patched(R, Xb, Xs, qb, qs):
        rs = {}; F.forward(M, Xs, rec=rs); xs = rs["resid2"]; tb, ts = tok(qb, jnp), tok(qs, jnp); i = jnp.arange(Xb.shape[0])[:, None]
        def fn(x):
            vb = x[i, tb].reshape(Xb.shape[0], -1); vs = xs[i, ts].reshape(Xb.shape[0], -1); Q, _ = jnp.linalg.qr(R); vp = vb + ((vs - vb) @ Q) @ Q.T; return x.at[i, tb].set(vp.reshape(Xb.shape[0], 7, 64))
        return F.forward(M, Xb, resid_patch={2: fn})[jnp.arange(Xb.shape[0]), qb]
    pj = jax.jit(patched); J = jnp.asarray
    L0 = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), ev["Xb"]))[np.arange(len(ev["Xb"])), ev["qb"]]; P0 = RP.sm(L0)
    Ls = [np.asarray(pj(J(R), J(ev["Xb"][s:s + 500]), J(ev["Xs"][s:s + 500]), J(ev["qb"][s:s + 500]), J(ev["qs"][s:s + 500]))) for s in range(0, len(ev["Xb"]), 500)]
    o = RP.cen(RP.sm(np.concatenate(Ls))) - RP.cen(P0)
    dE = RP.cen(ev["cfE"]) - RP.cen(P0); dEB = RP.cen(ev["cfE"]) - RP.cen(PBb); base_err = RP.cen(PBb) - RP.cen(P0)
    th = PMP[sp]["PM"]["params"]; dPM = pm_pred(th, Wb, Lb, ev["es"]) - pm_pred(th, Wb, Lb, ebb)
    # model's own natural counterfactual is not available (no natural donor with base values + source eligibility); PM surrogate instead
    r = dict(orig=RP.fid(o, dE), bayes_to_bayes=RP.fid(o, dEB), pm_surrogate=RP.fid(o, dPM),
             share_var_target_from_base_error=float((base_err ** 2).sum() / (dE ** 2).sum()), mean_abs_dE=float(np.abs(dE).mean()), mean_abs_dEB=float(np.abs(dEB).mean()),
             mean_abs_base_err=float(np.abs(base_err).mean()), mean_abs_obs=float(np.abs(o).mean()),
             corr_obs_with_minus_base_err=float(np.corrcoef(o.ravel(), base_err.ravel())[0, 1]))
    res[sp] = r
    print(sp[:6], {k: (round(v["R2"], 3), round(v["slope"], 3)) if isinstance(v, dict) else round(v, 3) for k, v in r.items()}, round(time.time() - t0), "s", flush=True)
json.dump(res, open("/tmp/review13b_mech/das_target.json", "w"), indent=1)
