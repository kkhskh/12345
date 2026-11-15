"""XIII-C Workstream B3/B4: is candidate evidence POOLED over sources before relational gating in early-readout models?
On-support ASSIGNMENT (SWAP) edits at the target question q: two groups g != h that BOTH self-agree at q with different values (a,a),(b,b) exchange
their values.  Unchanged: the multiset of the six source values at q (so any ungated, permutation-invariant pool of per-source evidence at q),
every group's agreement at q, all other questions (so W, e, f unchanged).  Changed: WHICH group carries which candidate -> the exact posterior
changes through the group weights.  Comparison family LIK: one source value at q changed (class_features likelihood pairs).
Measures per model:
  behaviour : fidelity slope/R2 of the model's centred-log-prob change on the Bayes change, for SWAP and LIK;
  RO1 / RO2 : transplant of the readout-token resid1 / resid2 from the donor (lik_RO-type availability) -> transferred share of the model's OWN
              full change (own-normalised: 1 = the readout token carries all of it), for SWAP and LIK;
  QS1       : same for the 6 q source tokens' resid1 (control: group-resolved evidence lives there).
Prediction of the pooling hypothesis for early-readout models: RO1 transfers LIK (it carries candidate evidence) but NOT SWAP (a pooled code is
invariant to the swap), and the model's behavioural SWAP slope is attenuated relative to LIK.
Usage: BP_STREAM=86_400_000 BP_TAG=_dev python3 scripts/b4_pooling.py SPEC... -> logs/b4_pooling<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("BP_STREAM", 86_400_000))); n0 = 12000
X = NV.Bm.sample_fast(NV.ENV, n0, rng)["X"]; q = rng.integers(0, 6, n0); a = X.reshape(n0, 6, 3, 2)
Xs, Bs, qs = [], [], []
for i in range(n0):
    ag = [g for g in range(3) if a[i, q[i], g, 0] == a[i, q[i], g, 1]]
    pairs = [(g, h) for g in ag for h in ag if g < h and a[i, q[i], g, 0] != a[i, q[i], h, 0]]
    if not pairs: continue
    g, h = pairs[rng.integers(len(pairs))]; x = X[i].copy(); vg, vh = x[q[i], 2 * g], x[q[i], 2 * h]
    x[q[i], 2 * g:2 * g + 2] = vh; x[q[i], 2 * h:2 * h + 2] = vg; Xs.append(X[i]); Bs.append(x); qs.append(q[i])
Xs, Bs, qs = np.array(Xs), np.array(Bs), np.array(qs)
PB = GL.posterior_closed(RP.ENV1, Xs, qs); PBd = GL.posterior_closed(RP.ENV1, Bs, qs); keep = np.where(np.abs(RP.cen(PBd) - RP.cen(PB)).max(-1) > 0.05)[0][:800]
SW = dict(Xb=Xs[keep], Xd=Bs[keep], q=qs[keep], PB=PB[keep], PBd=PBd[keep])
d = NV.Bm.sample_fast(NV.ENV, 4000, rng); X2 = d["X"]; q2 = rng.integers(0, 6, 4000); Xd2 = X2.copy(); hh = rng.integers(0, 3, 4000)
for i in range(4000): s_ = 2 * hh[i] + rng.integers(2); Xd2[i, q2[i], s_] = (X2[i, q2[i], s_] + rng.integers(1, 3)) % 3
P2 = GL.posterior_closed(RP.ENV1, X2, q2); P2d = GL.posterior_closed(RP.ENV1, Xd2, q2); k2 = np.where(np.abs(RP.cen(P2d) - RP.cen(P2)).max(-1) > 0.05)[0][:800]
LK = dict(Xb=X2[k2], Xd=Xd2[k2], q=q2[k2], PB=P2[k2], PBd=P2d[k2])
# sanity: the ungated pooled evidence at q is identical for SWAP pairs
y = np.arange(3); pool = lambda Xx, qq: np.where(Xx.reshape(len(Xx), 6, 6)[np.arange(len(Xx)), qq][..., None] == y, np.log(0.8), np.log(0.1)).sum(1)
out = dict(n_swap=len(SW["q"]), n_lik=len(LK["q"]), swap_pool_maxdiff=float(np.abs(pool(SW["Xb"], SW["q"]) - pool(SW["Xd"], SW["q"])).max()),
           swap_bayes_mean_abs=float(np.abs(RP.cen(SW["PBd"]) - RP.cen(SW["PB"])).mean()), lik_bayes_mean_abs=float(np.abs(RP.cen(LK["PBd"]) - RP.cen(LK["PB"])).mean()), models={})
def own_transfer(M, P_, l, site):
    m = len(P_["q"]); i = np.arange(m); rD = {}; LD = np.asarray(F.forward(M, jnp.asarray(P_["Xd"]), rec=rD))[i, P_["q"]]; rd = np.asarray(rD[f"resid{l}"])
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"])))[i, P_["q"]]
    toks = (36 + P_["q"])[:, None] if site == "RO" else P_["q"][:, None] * 6 + np.arange(6)[None]
    def patch(x): return x.at[i[:, None], toks].set(rd[i[:, None], toks])
    L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), resid_patch={l: patch}))[i, P_["q"]]
    full = RP.cen(RP.sm(LD)) - RP.cen(RP.sm(L0)); part = RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0))
    return float((part * full).sum() / max((full * full).sum(), 1e-12))
for sp in sys.argv[1:]:
    M = xc.load(sp); fj = jax.jit(lambda X_: F.forward(M, X_)); r = {}
    for nm, P_ in (("SWAP", SW), ("LIK", LK)):
        i = np.arange(len(P_["q"])); o = RP.cen(RP.sm(np.asarray(F.batched(fj, P_["Xd"]))[i, P_["q"]])) - RP.cen(RP.sm(np.asarray(F.batched(fj, P_["Xb"]))[i, P_["q"]]))
        r[nm] = dict(behaviour=RP.fid(o, RP.cen(P_["PBd"]) - RP.cen(P_["PB"])), RO1=own_transfer(M, P_, 1, "RO"), RO2=own_transfer(M, P_, 2, "RO"), QS1=own_transfer(M, P_, 1, "QS"))
    out["models"][sp] = r
    print(sp, {nm: dict(slope=round(v["behaviour"]["slope"], 3), R2=round(v["behaviour"]["R2"], 3), RO1=round(v["RO1"], 3), RO2=round(v["RO2"], 3), QS1=round(v["QS1"], 3)) for nm, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "b4_pooling" + os.environ.get("BP_TAG", "") + ".json"), "w"), indent=1)
