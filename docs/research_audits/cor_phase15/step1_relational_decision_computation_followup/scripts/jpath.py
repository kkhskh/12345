"""Workstream C2/C3 (XIII-B): WHERE does the cross-question identity relation enter the repaired decision?  Adapted models from adapt2.py
(beta = 0.6).  J-pairs (x, x'): one group's sources relabelled by a value permutation at every question except q (per-group statistics and the
values at q unchanged; exact posterior changes).  In the adapted model, the layer-2 input (resid2) of x' is transplanted into the run of x at a
token set T in {RO (readout token of q), QS (6 source tokens of q), OS (other source tokens), ALL}; the observed change of centred log-probs at q
is regressed on the EXACT change (fidelity slope; 1 = the J-dependence carried entirely by resid2 at T).  Also the same for resid1 (layer-1 input).
Usage: jpath.py SPEC COND [COND...]  (COND names of checkpoints/adapt2/<tag>_<COND>_b0.6.npz) -> logs/jpath<TAG>.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, genlib as GL, xii_bayes as XB, relpairs as RP
np.seterr(all="ignore")
sp = sys.argv[1]; parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M0 = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
E2 = GL.Env(3, 2, beta=0.6); rj = np.random.default_rng(79_100_000); dj = GL.sample(E2, 1200, rj); Xj = dj["X"]; qj = rj.integers(0, 6, len(Xj)); hj = rj.integers(0, 3, len(Xj))
perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]; Xj2 = Xj.copy()
for i in range(len(Xj)):
    pm = np.array(perms[rj.integers(len(perms))]); cols = [2 * hj[i], 2 * hj[i] + 1]; others = [t for t in range(6) if t != qj[i]]; Xj2[i][np.ix_(others, cols)] = pm[Xj[i][np.ix_(others, cols)]]
dB = RP.cen(XB.posterior_shared(E2, Xj2, qj)) - RP.cen(XB.posterior_shared(E2, Xj, qj)); ok = np.isfinite(dB).all(1); Xj, Xj2, qj, dB = Xj[ok], Xj2[ok], qj[ok], dB[ok]; n = len(qj); ii = np.arange(n)
def load(cond):
    z = np.load(os.path.join(HERE, "..", "checkpoints", "adapt2", f"{M0['tag']}_{cond}_b0.6.npz")); M = dict(M0); M["layers"] = [dict(l) for l in M0["layers"]]
    for k, v in z.items():
        top, *rest = k.split("/")
        if top in ("l0", "l1", "l2") and rest: M["layers"][int(top[1])][rest[0]] = jnp.asarray(v)
        elif top in ("gf", "bf", "WA"): M[top] = jnp.asarray(v)
    return M
def mask(kind):
    m = np.zeros((n, 43), bool)
    if kind == "RO": m[ii, 36 + qj] = True
    elif kind == "QS": m[ii[:, None], qj[:, None] * 6 + np.arange(6)[None]] = True
    elif kind == "OS": m[:, :36] = True; m[ii[:, None], qj[:, None] * 6 + np.arange(6)[None]] = False
    else: m[:] = True
    return m
out = {}
for cond in ["native"] + sys.argv[2:]:
    M = M0 if cond == "native" else load(cond); rA = {}; F.forward(M, jnp.asarray(Xj2), rec=rA)
    L0 = np.asarray(F.forward(M, jnp.asarray(Xj)))[ii, qj]; LA = np.asarray(F.forward(M, jnp.asarray(Xj2)))[ii, qj]; P0 = RP.sm(L0)
    r = dict(full=RP.fid(RP.cen(RP.sm(LA)) - RP.cen(P0), dB))
    for l in (1, 2):
        ra = np.asarray(rA[f"resid{l}"])
        for kind in ("RO", "QS", "OS", "ALL"):
            m = jnp.asarray(mask(kind)); src = jnp.asarray(ra)
            L = np.asarray(F.forward(M, jnp.asarray(Xj), resid_patch={l: lambda x, m=m, src=src: jnp.where(m[..., None], src, x)}))[ii, qj]
            r[f"resid{l}_{kind}"] = RP.fid(RP.cen(RP.sm(L)) - RP.cen(P0), dB)
    out[cond] = r; print(M0["tag"][:6], f"{cond:9s}", {k: round(v["slope"], 2) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "jpath" + os.environ.get("JP_TAG", "") + ".json"), "w"), indent=1)
