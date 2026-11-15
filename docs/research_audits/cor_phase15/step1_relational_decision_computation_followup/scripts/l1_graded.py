"""Workstream B4 (XIII-B): graded and resampling ablations of layer 1 with the BEHAVIOURAL RESIDUAL computed.
Targets: L1ATT (all four head outputs), L1MLP, L1ATT+MLP, and each single layer-1 head; at all tokens.
Modes: MEAN (output <- (1-a) out + a mean_over_episodes) and RESAMPLE (output <- (1-a) out + a out[pi(i)], pi a random derangement), a in {.25,.5,.75,1}.
Measures on natural episodes (stream 75_500_000): agreement with Bayes, KL(Bayes||model), KL(intact||model), ECE (calibration vs the Bayes
answer distribution sampled from the exact posterior), layer-2 mixture-head q-mass / entropy / corr(mass, Bayes r_g) (heads by relpairs rule;
mass-only list if none), and the BEHAVIOURAL RESIDUAL = relational-pair transfer slope (relpairs.relational_pairs; base and donor ablated identically)
relative to the intact slope.  -> logs/l1_graded<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("LG_STREAM", 75_500_000))); N_ = 1500
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_); PB = GL.posterior_closed(RP.ENV1, X, q)
W, Lg, pf, e, S = RP.parts(X, q); rB = RP.resp(W, Lg, S - e); Ysamp = np.array([rng.choice(3, p=p) for p in PB])
PP = RP.relational_pairs(800, rng); nP = len(PP["q"]); iP = np.arange(nP); predP = RP.cen(PP["PBd"]) - RP.cen(PP["PB"])
der = (ii + rng.integers(1, N_, N_)) % N_; derP = (iP + rng.integers(1, nP, nP)) % nP
qs = q[:, None] * 6 + np.arange(6)[None]; ALL = jnp.ones((N_, 43), bool); ALLP = jnp.ones((nP, 43), bool)
def ece(P, y):
    c = P.max(-1); h = P.argmax(-1) == y; b = np.clip((c * 10).astype(int), 0, 9); return float(sum(abs(h[b == k].mean() - c[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any()))
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    mass, share = RP.head_stats(M, X, q, F, jnp); MH, MHm = RP.mixture_heads(mass, share); HU = MH or MHm
    rN = {}; F.forward(M, jnp.asarray(X), rec=rN); rb, rd = {}, {}; F.forward(M, jnp.asarray(PP["Xb"]), rec=rb); F.forward(M, jnp.asarray(PP["Xd"]), rec=rd)
    mean_h = np.asarray(rN["heads1"]).mean(0); mean_m = np.asarray(rN["mlp_out1"]).mean(0)
    def patches(rec, target, mode, a, perm, mask):
        H = np.asarray(rec["heads1"]); Mo = np.asarray(rec["mlp_out1"]); hp, mp = {}, None
        src_h = (mean_h[None] if mode == "MEAN" else H[perm]); src_m = (mean_m[None] if mode == "MEAN" else Mo[perm])
        hs = range(4) if target in ("ATT", "ATT+MLP") else ([int(target[1])] if target.startswith("H") else [])
        for h in hs: hp[(1, h)] = (mask, jnp.asarray((1 - a) * H[:, h] + a * np.broadcast_to(src_h[:, h], H[:, h].shape)))
        if target in ("MLP", "ATT+MLP"): mp = {1: (mask, jnp.asarray((1 - a) * Mo + a * np.broadcast_to(src_m, Mo.shape)))}
        return dict(head_patch=hp, mlp_patch=mp)
    def measure(kwN, kwB, kwD):
        rr = {}; L = np.asarray(F.forward(M, jnp.asarray(X), rec=rr, **kwN))[ii, q]; P = RP.sm(L); A = np.asarray(rr["pattern2"])
        r = dict(agree_bayes=float((P.argmax(-1) == PB.argmax(-1)).mean()), KL_bayes=float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()),
                 ECE=ece(P, Ysamp))
        if HU:
            mq = np.stack([np.take_along_axis(A[ii, h, 36 + q], qs, 1) for h in HU]).mean(0); pg = np.stack([mq[:, 2 * g:2 * g + 2].sum(1) for g in range(3)], 1)
            pn = pg / np.maximum(pg.sum(1, keepdims=True), 1e-12); row = np.stack([A[ii, h, 36 + q] for h in HU]).mean(0)
            r.update(mix_mass=float(mq.sum(1).mean()), mix_entropy=float(-(row * np.log(row + 1e-12)).sum(1).mean()), mix_corr_B=float(np.corrcoef(pn.ravel(), rB.ravel())[0, 1]))
        Lb = np.asarray(F.forward(M, jnp.asarray(PP["Xb"]), **kwB))[iP, PP["q"]]; Ld = np.asarray(F.forward(M, jnp.asarray(PP["Xd"]), **kwD))[iP, PP["q"]]
        r["rel_transfer"] = RP.fid(RP.cen(RP.sm(Ld)) - RP.cen(RP.sm(Lb)), predP); return r, P
    intact, P0 = measure({}, {}, {}); res = dict(heads_used=HU, mixture_heads=MH, intact=intact); slope0 = intact["rel_transfer"]["slope"]
    for target in ("ATT", "MLP", "ATT+MLP", "H0", "H1", "H2", "H3"):
        for mode in ("MEAN", "RESAMPLE"):
            for a in ((0.25, 0.5, 0.75, 1.0) if target in ("ATT", "MLP", "ATT+MLP") else (1.0,)):
                r, P = measure(patches(rN, target, mode, a, der, ALL), patches(rb, target, mode, a, derP, ALLP), patches(rd, target, mode, a, derP, ALLP))
                r["KL_intact"] = float((P0 * (np.log(P0 + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()); r["behavioural_residual"] = r["rel_transfer"]["slope"] / slope0
                res[f"{target}|{mode}|{a}"] = r
    out[sp] = res
    print(sp[:6], "HU", HU, "intact", {k: round(v, 3) for k, v in intact.items() if isinstance(v, float)}, "slope0", round(slope0, 2), flush=True)
    for k, v in res.items():
        if "|" in k and (k.endswith("|1.0") or k.endswith("|0.5")):
            print(f"   {k:22s} agree {v['agree_bayes']:.2f} KLb {v['KL_bayes']:.2f} ECE {v['ECE']:.3f} resid {v['behavioural_residual']:.2f}" + (f" mass {v['mix_mass']:.2f} ent {v['mix_entropy']:.2f} corrB {v['mix_corr_B']:.2f}" if 'mix_mass' in v else ""), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l1_graded" + os.environ.get("LG_TAG", "") + ".json"), "w"), indent=1)
