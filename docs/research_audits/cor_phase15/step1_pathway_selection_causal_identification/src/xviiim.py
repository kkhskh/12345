"""XVIII measurement library.  Loads XVIII (or XIV) fine-tuning checkpoints by directory and computes the three organisational diagnostics
(route-structural, behavioural, causal-necessity), competence, and the early-window route shares.  All definitions are XIV's/XIII-C's
(src/metrics.py, src/maskfwd.py of step1_cross_family_relational_foundations) except the classification rule below.

Classification (frozen in FREEZE_XVIII; development values in XIV_RESULT_RECONSTRUCTION.md §4):
  structural  : XIV cls1211 -- EARLY if mix_mass < 0.5 and lik_RO1 >= 0.4; MIXTURE if mix_mass >= 0.8 and lik_RO1 <= 0.15; else HYBRID
  behavioural : rho <= 0.60 -> EARLY; rho >= 0.66 -> MIXTURE; else HYBRID
  causal      : nec_E >= 0.30 and nec_M <= 0.10 -> EARLY; nec_M >= 0.30 and nec_E <= 0.15 -> MIXTURE; else HYBRID
  final       : the common label if all three agree, otherwise HYBRID (reported with the three components)
  competent   : KL(Bayes||model) <= 0.06 on the measurement stream."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); X14 = os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"); sys.path.insert(0, X14)
import numpy as np, jax, jax.numpy as jnp
import native as NV, ftlib as F, metrics as MT, maskfwd as MFW, relpairs as RP, genlib as GL
NL, N = NV.NL, NV.N


def load_dir(D, parent, step):
    pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
    tmpl = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(7777)["WA"], new=[fresh(7777)["layers"][2]], emb={k: pf[k] for k in NL.FROZEN_KEYS}, pre=[pf["layers"][l] for l in range(2)])
    z = np.load(os.path.join(D, f"step{step}.npz"))
    t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tmpl), [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
    return dict(tag=os.path.basename(D) + f"@{step}", parent=parent, spec="full_c1", emb=t["emb"], layers=list(t["pre"]) + [t["new"][0]], gf=t["gf"], bf=t["bf"], WA=t["WA"], dir=D, step=step)


def nat(stream, n=1500):
    rng = np.random.default_rng(stream + 7); X = NV.Bm.sample_fast(NV.ENV, n, rng)["X"]; q = rng.integers(0, 6, n); return X, q, GL.posterior_closed(RP.ENV1, X, q)


def kl_of(L, q, PB):
    i = np.arange(len(q)); P = RP.sm(np.asarray(L)[i, q]); return float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean())


def necessity(M, NAT):
    X, q, PB = NAT; base = kl_of(MFW.forward(M, X), q, PB)
    kE = kl_of(MFW.forward(M, X, biases=dict([MFW.edges("ro0")])), q, PB); kM = kl_of(MFW.forward(M, X, biases=dict([MFW.edges("own2")])), q, PB)
    k0 = kl_of(MFW.forward(M, X, biases=dict([MFW.edges("own0")])), q, PB)
    return dict(KL_nat=base, nec_E=kE - base, nec_M=kM - base, nec_E_own0=k0 - base)


def shares(M, X):
    """Mean readout attention share on the route edges: layer-0 own-q, layer-0 all sources, layer-2 own-q (mean and max over heads)."""
    rec = {}; F.forward(M, jnp.asarray(X), rec=rec); out = {}
    own = np.zeros((6, 43), bool)
    for q in range(6): own[q, q * 6:(q + 1) * 6] = True
    for l in (0, 2):
        A = np.asarray(rec[f"pattern{l}"])[:, :, 36:42, :]
        so = (A * own).sum(-1).mean((0, 2)); out[f"own{l}_mean"] = float(so.mean()); out[f"own{l}_maxhead"] = float(so.max())
        if l == 0: out["src0_mean"] = float(A[..., 0:36].sum(-1).mean())
    return out


def cls_struct(r): return "EARLY" if (r["mix_mass"] < 0.5 and r["lik_RO1"] >= 0.4) else ("MIXTURE" if (r["mix_mass"] >= 0.8 and r["lik_RO1"] <= 0.15) else "HYBRID")
def cls_behav(r): return "EARLY" if r["rho"] <= 0.60 else ("MIXTURE" if r["rho"] >= 0.66 else "HYBRID")
def cls_causal(r): return "EARLY" if (r["nec_E"] >= 0.30 and r["nec_M"] <= 0.10) else ("MIXTURE" if (r["nec_M"] >= 0.30 and r["nec_E"] <= 0.15) else "HYBRID")
def classify(r):
    c = (cls_struct(r), cls_behav(r), cls_causal(r)); return dict(cls_struct=c[0], cls_behav=c[1], cls_causal=c[2], cls=c[0] if len(set(c)) == 1 else "HYBRID",
                                                                  competent=bool(r["KL"] <= 0.06))


def measure_final(M, stream):
    S = MT.sets(stream); r = dict(lik_RO1=MT.avail(M, S["PL"], 1))
    mass, share = RP.head_stats(M, S["Xn"][:1500], S["qn"][:1500], F, jnp); r["mix_mass"] = float(mass.max())
    b, Ln = MT.behaviour(M, S); r.update(b); r.update(MT.rho_fit(Ln, S)); r.update(necessity(M, nat(stream))); r.update(classify(r)); return r


# ---------------------------------------------------------------- additions after the pre-experiment reviews (A §4, B §2 R7)
from scipy.optimize import minimize
from scipy.special import logsumexp as _lse


def rho_fit_lapse(Ln, S, ntr=1500):
    """XIII-B nested power-mean family (metrics.rho_fit) with a uniform lapse: P = (1 - eps) P_pm + eps / K, eps = sigmoid(th[5]).
    Pre-review B (N2): plain rho confounds lapse-type error with organisation.  Same data, split and starts as metrics.rho_fit."""
    X, q = S["Xn"], S["qn"]; n = len(q); W, Lg, pf, e, Sx = RP.parts(X, q); f = Sx - e; lW = np.log(W); lL = np.log(Lg)
    cenl = lambda lp: lp - lp.mean(-1, keepdims=True); y = cenl(np.log(RP.sm(Ln) + 1e-12)); tr, te = np.arange(ntr), np.arange(ntr, n)
    def fam(th, idx):
        al, ka, de, tau, rho, le = th; rho = float(np.clip(rho, 1e-3, 5)); eps = 1 / (1 + np.exp(-le))
        z = al * lW[idx] + ka * np.log(np.maximum(f[idx], 1)) + de * (f[idx] == 0); lw = z - _lse(z, 1, keepdims=True)
        lp = tau * _lse(lw[..., None] + rho * lL[idx], 1) / rho; lp = lp - _lse(lp, -1, keepdims=True)
        return cenl(np.log((1 - eps) * np.exp(lp) + eps / lp.shape[-1]))
    loss = lambda th: float(((fam(th, tr) - y[tr]) ** 2).mean()); best = None
    for st in ([1, 1, -5, 1, 1.0, -4.0], [0.8, 0.5, -8, 1, 0.5, -4.0], [1, 1, -5, 1.2, 1.0, -2.5]):
        r = minimize(loss, st, method="Powell", options=dict(maxiter=6000, xtol=1e-4, ftol=1e-8))
        if best is None or r.fun < best.fun: best = r
    yt = y[te]; R2 = float(1 - ((fam(best.x, te) - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    return dict(rhoL=float(np.clip(best.x[4], 1e-3, 5)), lapseL=float(1 / (1 + np.exp(-best.x[5]))), R2_pmL=R2)


def nec_rq1(M, NAT):
    """Third-route check: KL rise when layer-1 readout -> all sources is blocked at test time (XIII-D 'rq1')."""
    X, q, PB = NAT; base = kl_of(MFW.forward(M, X), q, PB); m = np.zeros((43, 43), bool); m[36:42, 0:36] = True
    return kl_of(MFW.forward(M, X, biases={1: m}), q, PB) - base


def cls_behav2(r):
    """Behavioural class after pre-review B (R7), calibrated on the 24 XIV step-1211 runs (logs/dev/rhoL_xiv.json; all 24 correct):
    EARLY if lapse >= 0.017 and rhoL <= 0.75; MIXTURE if lapse <= 0.017 and rhoL >= 0.68; else HYBRID."""
    return "EARLY" if (r["lapseL"] >= 0.017 and r["rhoL"] <= 0.75) else ("MIXTURE" if (r["lapseL"] <= 0.017 and r["rhoL"] >= 0.68) else "HYBRID")


def classify_v2(r):
    c = (cls_struct(r), cls_behav2(r), cls_causal(r)); return dict(cls2_struct=c[0], cls2_behav=c[1], cls2_causal=c[2], cls2=c[0] if len(set(c)) == 1 else "HYBRID")


def measure_v2(M, stream):
    """measure_final plus the lapse-aware behavioural fit and the third-route check (frozen in FREEZE_XVIII)."""
    S = MT.sets(stream); r = measure_final(M, stream); Ln = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), S["Xn"]))[np.arange(len(S["qn"])), S["qn"]]
    r.update(rho_fit_lapse(Ln, S)); r["nec_rq1"] = nec_rq1(M, nat(stream)); r.update(classify_v2(r)); return r
