"""Workstream A1 (XIII-B): WHERE does the candidate decision become causally operative?  Natural interchange (class II) of single components.
For base/donor pairs with the same target question q (natural episodes), the output of ONE component, restricted to a token set, is replaced by its
value in the donor run; the observed change of centred log-probabilities at q is regressed on the Bayes change cen(PB_donor) - cen(PB_base)
(fidelity slope b: b=1 means the component at those tokens carries the WHOLE decision difference; b=0 none).  Also mean-ablation (necessity):
replace the component by its mean over natural episodes and report KL(model || intact) and the drop in agreement with Bayes.
Components: resid_l at the readout token (state BEFORE layer l, l=1,2,3 -- 'availability'), each head (l,h) and MLP_l output at token sets
RO = readout token 36+q, QS = the six source tokens of question q, OS = other source tokens, ALL.
Usage: locate.py SPEC... (env LOC_STREAM default 75_000_000, LOC_TAG) -> logs/locate<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("LOC_STREAM", 75_000_000))); N_ = int(os.environ.get("LOC_N", 1500))
d = NV.Bm.sample_fast(NV.ENV, 3 * N_, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); PB = GL.posterior_closed(env, X, q)
# XIII-B v2: RELATIONAL pairs.  donor = base with ONE group's source pair edited at ONE question q' != q (agree <-> disagree), so the values at q
# (candidate likelihoods L at q) are identical and the Bayes change is purely relational (W_h and possibly e_h).  Kept if |Bayes change| > 0.05.
FAM = os.environ.get("LOC_FAM", "rel")
Xd = X.copy(); hh = rng.integers(0, 3, len(X)); qp = (q + rng.integers(1, 6, len(X))) % 6
for i in range(len(X)):
    if FAM == "lik":                         # LIKELIHOOD pairs: one source value AT q changed (relational structure elsewhere unchanged)
        s_ = 2 * hh[i] + rng.integers(2); Xd[i, q[i], s_] = (X[i, q[i], s_] + rng.integers(1, 3)) % 3; continue
    a0, a1 = X[i, qp[i], 2 * hh[i]], X[i, qp[i], 2 * hh[i] + 1]
    Xd[i, qp[i], 2 * hh[i] + 1] = a0 if a0 != a1 else (a0 + rng.integers(1, 3)) % 3
PBd = GL.posterior_closed(env, Xd, q)
ii_ = np.arange(len(X)); a_b = X.reshape(-1, 6, 3, 2)[ii_, q]; a_d = Xd.reshape(-1, 6, 3, 2)[ii_, q]
e_b = (GL.counts(X, 3)[ii_, q] == 5) & (a_b[..., 0] == a_b[..., 1]); e_d = (GL.counts(Xd, 3)[ii_, q] == 5) & (a_d[..., 0] == a_d[..., 1])
cen_ = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
big = np.abs(cen_(PBd) - cen_(PB)).max(-1) > 0.05
sel = np.where(big)[0][:N_]; N_ = len(sel); X, q, PB, Xd, PBd = X[sel], q[sel], PB[sel], Xd[sel], PBd[sel]; ECH = (e_b[sel] != e_d[sel]).any(1)
X = np.concatenate([X, Xd]); q = np.concatenate([q, q]); PB = np.concatenate([PB, PBd]); base = np.arange(N_); donor = N_ + np.arange(N_)
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
pred = cen(PB[donor]) - cen(PB[base])
def fid(o, p):
    o, p = o.ravel(), p.ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); ss = ((o - o.mean()) ** 2).sum()
    return dict(slope=b, R2=float(1 - ((o - b * p) ** 2).sum() / ss) if ss > 1e-10 else 0.0, mean_abs_obs=float(np.abs(o).mean()))
sm = lambda L: (lambda E: E / E.sum(-1, keepdims=True))(np.exp(L - L.max(-1, keepdims=True)))
T = 43
def tokmask(kind):
    m = np.zeros((N_, T), bool); qq = q[base]
    if kind == "RO": m[base, 36 + qq] = True
    elif kind == "QS": m[base[:, None], qq[:, None] * 6 + np.arange(6)[None]] = True
    elif kind == "OS": m[:, :36] = True; m[base[:, None], qq[:, None] * 6 + np.arange(6)[None]] = False
    elif kind == "ALL": m[:] = True
    return jnp.asarray(m)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    recD = {}; F.forward(M, jnp.asarray(X[donor]), rec=recD); recB = {}; L0 = np.asarray(F.forward(M, jnp.asarray(X[base]), rec=recB))
    P0 = sm(L0[base * 0 + np.arange(N_), q[base]]); Xb = jnp.asarray(X[base]); ii = np.arange(N_)
    def run(**kw): L = np.asarray(F.forward(M, Xb, **kw))[ii, q[base]]; return sm(L)
    res = dict(intact_agree_bayes=float((P0.argmax(-1) == PB[base].argmax(-1)).mean()), intact_KL_bayes=float((PB[base] * (np.log(PB[base] + 1e-12) - np.log(P0 + 1e-12))).sum(-1).mean()))
    ro = 36 + q[base]
    for l in (1, 2, 3):           # availability: state entering layer l (3 = final residual) at the readout token, from donor
        rd = recD[f"resid{l}"]
        def fn(x, rd=rd): return x.at[ii, ro].set(rd[ii, ro])
        res[f"resid{l}_RO"] = fid(cen(run(resid_patch={l: fn})) - cen(P0), pred)
    for l in range(3):
        for kind in ("RO", "QS", "OS", "ALL"):
            m = tokmask(kind)
            for h in range(4):
                res[f"L{l}H{h}_{kind}"] = fid(cen(run(head_patch={(l, h): (m, recD[f"heads{l}"][:, h])})) - cen(P0), pred)
            res[f"L{l}MLP_{kind}"] = fid(cen(run(mlp_patch={l: (m, recD[f"mlp_out{l}"])})) - cen(P0), pred)
            res[f"L{l}ATT_{kind}"] = fid(cen(run(head_patch={(l, h): (m, recD[f"heads{l}"][:, h]) for h in range(4)})) - cen(P0), pred)
        # necessity: mean ablation of the whole attention output and of the MLP output at all tokens of layer l
        mA = jnp.ones((N_, T), bool)
        for nm, kw in (("ATT", dict(head_patch={(l, h): (mA, jnp.broadcast_to(recB[f"heads{l}"][:, h].mean(0, keepdims=True), recB[f"heads{l}"][:, h].shape)) for h in range(4)})),
                       ("MLP", dict(mlp_patch={l: (mA, jnp.broadcast_to(recB[f"mlp_out{l}"].mean(0, keepdims=True), recB[f"mlp_out{l}"].shape))}))):
            P = run(**kw); res[f"ablate_L{l}{nm}"] = dict(agree_bayes=float((P.argmax(-1) == PB[base].argmax(-1)).mean()), KL_intact=float((P0 * (np.log(P0 + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()))
    # same fidelities restricted to e-changing pairs and W-only pairs (main components only)
    out[sp] = res; res["frac_e_change"] = float(ECH.mean()); res["n_pairs"] = int(N_)
    keys = ["resid1_RO", "resid2_RO", "resid3_RO"] + [f"L{l}{c}_{k}" for l in range(3) for c in ("ATT", "MLP") for k in ("RO", "QS", "ALL")]
    print(sp[:6], {k: round(res[k]["slope"], 2) for k in keys}, flush=True)
    print("      ablate", {k: (round(v["agree_bayes"], 2), round(v["KL_intact"], 2)) for k, v in res.items() if k.startswith("ablate")}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "locate" + os.environ.get("LOC_TAG", "") + ".json"), "w"), indent=1)
