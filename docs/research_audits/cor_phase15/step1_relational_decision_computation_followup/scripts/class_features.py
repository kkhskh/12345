"""Workstream D1 (XIII-B): CONTINUOUS solution-class features per model (natural interchange, class II).
  lik_RO1, lik_RO2 : availability at the readout token of CANDIDATE evidence -- likelihood pairs (one source value at q changed); donor resid1 / resid2
                     at the readout token transplanted; fidelity slope on the Bayes change (pre-assembled ~0.85, mixture ~0.0)
  rel_RO2          : same for RELATIONAL pairs (edit at q' != q), resid2 at the readout token
  mix_mass, mix_share (max over layer-2 heads of q-mass, and the share of the head with max mass), strict/mass-only mixture heads
  route_Q, route_R : 1 - transfer slope when the layer-1 attention output at the q tokens / at the readout token is frozen to base, with the edited
                     group's layer-0 output at q' taken from the donor (path_chain.py definitions; relational pairs)
Usage: class_features.py SPEC... (CF_STREAM default 75_700_000) -> logs/class_features<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("CF_STREAM", 75_700_000))); n = 800
PR = RP.relational_pairs(n, rng)
d = NV.Bm.sample_fast(NV.ENV, 4 * n, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); Xd = X.copy(); hh = rng.integers(0, 3, len(X))
for i in range(len(X)): s_ = 2 * hh[i] + rng.integers(2); Xd[i, q[i], s_] = (X[i, q[i], s_] + rng.integers(1, 3)) % 3
PB = GL.posterior_closed(RP.ENV1, X, q); PBd = GL.posterior_closed(RP.ENV1, Xd, q); keep = np.where(np.abs(RP.cen(PBd) - RP.cen(PB)).max(-1) > 0.05)[0][:n]
PL = dict(Xb=X[keep], Xd=Xd[keep], q=q[keep], PB=PB[keep], PBd=PBd[keep])
def avail(M, P_, l):
    m = len(P_["q"]); i = np.arange(m); rD = {}; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD); ro = 36 + P_["q"]; rd = np.asarray(rD[f"resid{l}"])
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"])))[i, P_["q"]]
    L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), resid_patch={l: lambda x: x.at[i, ro].set(rd[i, ro])}))[i, P_["q"]]
    return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), RP.cen(P_["PBd"]) - RP.cen(P_["PB"]))["slope"]
def routes(M, P_):
    m = len(P_["q"]); i = np.arange(m); q_, qp, h = P_["q"], P_["qp"], P_["h"]; rB, rD = {}, {}
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), rec=rB))[i, q_]; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD); pred = RP.cen(P_["PBd"]) - RP.cen(P_["PB"])
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
        L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), head_patch=hp, mlp_patch=mp))[i, q_]; return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), pred)["slope"]
    s0 = tr(None); return dict(transfer=s0, route_Q=1 - tr("QS") / s0, route_R=1 - tr("RO") / s0)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    nat = NV.Bm.sample_fast(NV.ENV, 1500, np.random.default_rng(75_700_001))["X"]; qn = np.random.default_rng(75_700_002).integers(0, 6, 1500)
    mass, share = RP.head_stats(M, nat, qn, F, jnp); MH, MHm = RP.mixture_heads(mass, share); hm = int(np.argmax(mass))
    r = dict(lik_RO1=avail(M, PL, 1), lik_RO2=avail(M, PL, 2), rel_RO2=avail(M, PR, 2), mix_mass=float(mass.max()), mix_share=float(share[hm]), strict_mixture=MH, mass_only=MHm,
             mass=mass.tolist(), share=list(map(float, share)), **routes(M, PR))
    out[sp] = r; print(sp[:22], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("mass", "share")}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "class_features" + os.environ.get("CF_TAG", "") + ".json"), "w"), indent=1)
