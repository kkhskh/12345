"""Reviewer check: own-question (route) attention share in the intervened layer under none / real / sham / location-matched noise,
on the base checkpoints at steps 25, 50, 100 (the intervention window).  Fresh random episodes (reviewer stream 91_000_000)."""
import sys, os, json
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_cross_family_relational_foundations"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, xc, native as NV, maskfwd as MF
X = NV.Bm.sample_fast(NV.ENV, 400, np.random.default_rng(91_000_000))["X"]
own = np.zeros((6, 43), bool); src = np.zeros((6, 43), bool)
for q in range(6): own[q, q*6:(q+1)*6] = True; src[q, 0:36] = True
res = {}
for name, spec, real, sham, sig in (("E", "uP160:fx:21:21:21:base", "ro0", "oq0", 2.5), ("M", "uP160:fx:23:23:23:base", "own2", "rd2", 1.25)):
    lay, em = MF.edges(real); _, sm = MF.edges(sham)
    for k in (25, 50, 100):
        M = xc.load(spec, k); out = {}
        for cond, b, nz in (("none", None, None), ("real", {lay: em}, None), ("sham", {lay: sm}, None)) + tuple((f"noise_r{r}", None, (lay, sig, em, jax.random.PRNGKey(7000 + r))) for r in range(3)):
            rec = {}; MF.forward(M, X, b, nz, rec); A = np.asarray(rec[f"pattern{lay}"])[:, :, 36:42]
            ownm = (A * own).sum(-1)   # n,H,6
            out[cond] = dict(own_mean=float(ownm.mean()), own_maxhead=float(ownm.mean((0, 2)).max()), src_mean=float((A * src).sum(-1).mean()),
                             own_frac_of_src=float(((A * own).sum(-1) / np.maximum((A * src).sum(-1), 1e-9)).mean()))
        res[f"{name}@{k}"] = out; print(name, k, {c: {a: round(v, 3) for a, v in d.items()} for c, d in out.items()}, flush=True)
json.dump(res, open("/tmp/review_b/route_share.json", "w"), indent=1)
