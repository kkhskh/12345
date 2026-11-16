"""XIII-C: continuous implementation measures for one model (any checkpoint), shared by trajectory, factorial and confirmation scripts.
Definitions are XIII-B's (class_features.py, traj_resp.py, fit_powermean.py) re-implemented on fixed pair sets so that every model and every
checkpoint is measured on IDENTICAL inputs:
  lik_RO1  : fidelity slope of the Bayes change transmitted when the readout-token resid1 is taken from a likelihood donor (candidate evidence
             available at the readout token after layer 0).  ~0.85 early-readout, ~0 mixture.
  rel_RO2  : same for relational pairs at resid2.
  route_Q / route_R : share of the relational effect lost when layer-1 attention output at the q tokens / the readout token is frozen to base
             (edited group's layer-0 output at q' from the donor).
  mix_mass / mix_share : max over layer-2 heads of readout attention mass on the 6 q source tokens; direct-logit share of that head.
  l2_attn / l2_mlp : two-player Shapley shares (attention output all heads at the readout token; MLP2 output at the readout token) of the
             relational effect, donor patched into base.
  slope_W / slope_e : behavioural response slopes on W-only and e-change relational pairs (traj_resp.py).
  KL, acc  : KL(Bayes||model) and Bayes-argmax agreement on natural episodes.
  rho, R2_pm : nested power-mean exponent fitted on natural episodes (fit_powermean.py family, 2 starts)."""
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import logsumexp as lse
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")

_SETS = {}


def sets(stream=86_100_000, n=600):
    key = (stream, n)
    if key in _SETS: return _SETS[key]
    rng = np.random.default_rng(stream); PR = RP.relational_pairs(n, rng)
    d = NV.Bm.sample_fast(NV.ENV, 4 * n, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); Xd = X.copy(); hh = rng.integers(0, 3, len(X))
    for i in range(len(X)): s_ = 2 * hh[i] + rng.integers(2); Xd[i, q[i], s_] = (X[i, q[i], s_] + rng.integers(1, 3)) % 3
    PB = GL.posterior_closed(RP.ENV1, X, q); PBd = GL.posterior_closed(RP.ENV1, Xd, q); keep = np.where(np.abs(RP.cen(PBd) - RP.cen(PB)).max(-1) > 0.05)[0][:n]
    PL = dict(Xb=X[keep], Xd=Xd[keep], q=q[keep], PB=PB[keep], PBd=PBd[keep])
    Xn = NV.Bm.sample_fast(NV.ENV, 3000, rng)["X"]; qn = rng.integers(0, 6, 3000); PBn = GL.posterior_closed(RP.ENV1, Xn, qn)
    _SETS[key] = dict(PR=PR, PL=PL, Xn=Xn, qn=qn, PBn=PBn); return _SETS[key]


def _slope(L, L0, P_):
    return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), RP.cen(P_["PBd"]) - RP.cen(P_["PB"]))["slope"]


def avail(M, P_, l):
    m = len(P_["q"]); i = np.arange(m); rD = {}; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD); ro = 36 + P_["q"]; rd = np.asarray(rD[f"resid{l}"])
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"])))[i, P_["q"]]
    L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), resid_patch={l: lambda x: x.at[i, ro].set(rd[i, ro])}))[i, P_["q"]]
    return _slope(L, L0, P_)


def routes(M, P_):
    m = len(P_["q"]); i = np.arange(m); q_, qp, h = P_["q"], P_["qp"], P_["h"]; rB, rD = {}, {}
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), rec=rB))[i, q_]; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD)
    up = np.zeros((m, 43), bool); up[i, qp * 6 + 2 * h] = True; up[i, qp * 6 + 2 * h + 1] = True; up = jnp.asarray(up)
    def mk(kind):
        mm = np.zeros((m, 43), bool)
        if kind == "QS": mm[i[:, None], q_[:, None] * 6 + np.arange(6)[None]] = True
        else: mm[i, 36 + q_] = True
        return jnp.asarray(mm)
    def tr(freeze):
        hp = {(0, hd): (up, rD["heads0"][:, hd]) for hd in range(4)}; mp = {0: (up, rD["mlp_out0"])}
        if freeze:
            for hd in range(4): hp[(1, hd)] = (mk(freeze), rB["heads1"][:, hd])
        L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), head_patch=hp, mlp_patch=mp))[i, q_]; return _slope(L, L0, P_)
    s0 = tr(None); return dict(transfer=s0, route_Q=1 - tr("QS") / s0, route_R=1 - tr("RO") / s0)


def l2_split(M, P_):
    """Two-player Shapley (A = layer-2 attention output at RO, all heads; M = MLP2 output at RO), donor into base, projection on total."""
    m = len(P_["q"]); i = np.arange(m); q_ = P_["q"]; rB, rD = {}, {}
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), rec=rB))[i, q_]; L1 = np.asarray(F.forward(M, jnp.asarray(P_["Xd"]), rec=rD))[i, q_]
    ro = np.zeros((m, 43), bool); ro[i, 36 + q_] = True; ro = jnp.asarray(ro); TE = RP.cen(RP.sm(L1)) - RP.cen(RP.sm(L0))
    def run(A, Mm):
        hp = {(2, hd): (ro, rD["heads2"][:, hd]) for hd in range(4)} if A else None
        mp = {2: (ro, rD["mlp_out2"])} if Mm else None
        L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), head_patch=hp, mlp_patch=mp))[i, q_]; d = RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0))
        return float((d * TE).sum() / max((TE * TE).sum(), 1e-12))
    vA, vM, vAM = run(1, 0), run(0, 1), run(1, 1)
    return dict(l2_attn=0.5 * (vA + (vAM - vM)), l2_mlp=0.5 * (vM + (vAM - vA)), l2_both=vAM, l2_interact=vAM - vA - vM)


def behaviour(M, S):
    PR = S["PR"]; iP = np.arange(len(PR["q"])); fj = jax.jit(lambda X_: F.forward(M, X_)); ech = PR["e_change"]
    dB = RP.cen(PR["PBd"]) - RP.cen(PR["PB"]); Lb = np.asarray(F.batched(fj, PR["Xb"]))[iP, PR["q"]]; Ld = np.asarray(F.batched(fj, PR["Xd"]))[iP, PR["q"]]
    o = RP.cen(RP.sm(Ld)) - RP.cen(RP.sm(Lb)); n = len(S["qn"]); Ln = np.asarray(F.batched(fj, S["Xn"]))[np.arange(n), S["qn"]]; P = RP.sm(Ln); PB = S["PBn"]
    return dict(slope_W=RP.fid(o[~ech], dB[~ech])["slope"], slope_e=RP.fid(o[ech], dB[ech])["slope"],
                KL=float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()), acc=float((P.argmax(-1) == PB.argmax(-1)).mean())), Ln


def rho_fit(Ln, S, ntr=1500):
    X, q = S["Xn"], S["qn"]; n = len(q); W, Lg, pf, e, Sx = RP.parts(X, q); f = Sx - e; lW = np.log(W); lL = np.log(Lg)
    cenl = lambda lp: lp - lp.mean(-1, keepdims=True); y = cenl(np.log(RP.sm(Ln) + 1e-12)); tr, te = np.arange(ntr), np.arange(ntr, n)
    def fam(th, idx):
        al, ka, de, tau, rho = th; rho = float(np.clip(rho, 1e-3, 5))
        z = al * lW[idx] + ka * np.log(np.maximum(f[idx], 1)) + de * (f[idx] == 0); lw = z - lse(z, 1, keepdims=True)
        return cenl(tau * lse(lw[..., None] + rho * lL[idx], 1) / rho)
    loss = lambda th: float(((fam(th, tr) - y[tr]) ** 2).mean()); best = None
    for st in ([1, 1, -5, 1, 1.0], [0.8, 0.5, -8, 1, 0.5]):
        r = minimize(loss, st, method="Powell", options=dict(maxiter=4000, xtol=1e-4, ftol=1e-8))
        if best is None or r.fun < best.fun: best = r
    yt = y[te]; R2 = float(1 - ((fam(best.x, te) - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    return dict(rho=float(np.clip(best.x[4], 1e-3, 5)), R2_pm=R2)


def all_metrics(M, S=None, rho=True, l2=True):
    S = S or sets(); mass, share = RP.head_stats(M, S["Xn"][:1500], S["qn"][:1500], F, jnp); hm = int(np.argmax(mass))
    r = dict(lik_RO1=avail(M, S["PL"], 1), lik_RO2=avail(M, S["PL"], 2), rel_RO2=avail(M, S["PR"], 2), mix_mass=float(mass.max()), mix_share=float(share[hm]),
             n_mix=len(RP.mixture_heads(mass, share)[0]), **routes(M, S["PR"]))
    b, Ln = behaviour(M, S); r.update(b)
    if l2: r.update(l2_split(M, S["PR"]))
    if rho: r.update(rho_fit(Ln, S))
    return r
