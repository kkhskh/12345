"""Reviewer check: how much of the purely relational effect (relational pairs, values at q identical) is carried by resid2 at the
DAS site (readout token + 6 q source tokens) versus other token sets?  Slope of observed change on the Bayes change (class_features 'avail' style)."""
import os, sys, json
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, relpairs as RP
np.seterr(all="ignore")
P_ = RP.relational_pairs(800, np.random.default_rng(81_000_010)); n = len(P_["q"]); i = np.arange(n); q = P_["q"]
pred = RP.cen(P_["PBd"]) - RP.cen(P_["PB"])
def mask(kind):
    m = np.zeros((n, 43), bool)
    if kind in ("RO", "SITE7"): m[i, 36 + q] = True
    if kind in ("QS", "SITE7"): m[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = True
    if kind == "OS": m[:, :36] = True; m[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = False
    if kind == "ALL": m[:] = True
    return m
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    rD = {}; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD); r2d = np.asarray(rD["resid2"]); L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"])))[i, q]
    LDn = np.asarray(F.forward(M, jnp.asarray(P_["Xd"])))[i, q]; te = RP.cen(RP.sm(LDn)) - RP.cen(RP.sm(L0))
    r = {"model_TE_slope_on_bayes": RP.fid(te, pred)["slope"]}
    for k in ("RO", "QS", "SITE7", "OS", "ALL"):
        m = jnp.asarray(mask(k)); rep = jnp.asarray(r2d)
        L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), resid_patch={2: lambda x: jnp.where(m[..., None], rep, x)}))[i, q]
        o = RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)); r[k] = float((o * te).sum() / (te * te).sum())   # projection on the model's own total effect
    out[sp] = r; print(sp[:6], {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open("/tmp/review13b_mech/site_cover.json", "w"), indent=1)
