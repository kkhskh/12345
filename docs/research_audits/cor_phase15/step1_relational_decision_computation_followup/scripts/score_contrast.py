"""Workstream B4 (representation vs use): which coordinates of the layer-2 input does the mixture head USE for the eligibility contrast?
e_g is linearly decodable both from the 6-dim E-subspace and from its 442-dim complement (l1abl_mix.py).  Here: the within-episode score contrast
of the mixture head(s) in S = 1 episodes, Δ = score(sole eligible group) - mean score(other two groups)   (Bayes: -inf; models: strongly < 0),
and the S = 2 contrast (eligible vs non-eligible; Bayes: log(1/2)),
recomputed after mean-patching (natural mean over 2000 episodes, at the q7 tokens of the target question, in the layer-2 input) of:
  E (6 dims) | random 6-dim subspace (3 draws) | the complement of E (442 dims) | nothing
Fraction of the S=1 contrast removed by each.  Usage: score_contrast.py RSUF SPEC... -> logs/score_contrast<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
rng = np.random.default_rng(int(os.environ.get("SCC_STREAM", 73_400_000))); N_ = 3000
d = NV.Bm.sample_fast(NV.ENV, N_, rng); X = d["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
a_ = X.reshape(N_, 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(int); S = e.sum(1)
tok7 = np.concatenate([(36 + q)[:, None], q[:, None] * 6 + np.arange(6)[None]], 1); tq = q[:, None] * 6 + np.arange(6)[None]
out = {}
for sp in sys.argv[1 + 1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    r = {}; F.forward(M, jnp.asarray(X), rec=r); x2 = np.asarray(r["resid2"]); A2 = np.asarray(r["pattern2"])
    mass = np.array([A2[ii, h, 36 + q][ii[:, None], tq].sum(1).mean() for h in range(4)]); MH = [h for h in range(4) if mass[h] >= 0.8]
    if not MH: out[sp] = dict(mixture_heads=[]); print(sp, "no mixture head"); continue
    fE = os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy")
    if not os.path.exists(fE): fE = os.path.join(os.environ.get("DECOY_DIR", ""), f"{M['tag']}_E_R.npy")     # XIII: E from das_decoy when no das.py fit exists
    Q, _ = np.linalg.qr(np.load(fE))
    z7 = x2[ii[:, None], tok7].reshape(N_, -1); zbar = z7.mean(0)
    lay = {k: np.asarray(v) for k, v in M["layers"][2].items()}
    def ln(x): mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True); return (x - mu) / np.sqrt(v + 1e-5) * lay["g1"] + lay["b1"]
    def contrasts(P):                     # P: (448,448) projector of the coordinates to mean-patch (None = intact)
        xx = x2.copy()
        if P is not None: zp = z7 - (z7 - zbar) @ P; xx[ii[:, None], tok7] = zp.reshape(N_, 7, 64)
        z = ln(xx); c1, c2 = [], []
        for h in MH:
            qv = z[ii, 36 + q] @ lay["WQ"][h]; kv = z[ii[:, None], tq] @ lay["WK"][h]; s = (np.einsum("ne,nse->ns", qv, kv) / 4.0).reshape(N_, 3, 2).mean(-1)
            el = e == 1; d1 = (s * el).sum(1) - (s * ~el).sum(1) / 2; d2 = (s * el).sum(1) / 2 - (s * ~el).sum(1)
            c1.append(d1[S == 1].mean()); c2.append(d2[S == 2].mean())
        return float(np.mean(c1)), float(np.mean(c2))
    base = contrasts(None); res = dict(mixture_heads=MH, intact=dict(S1=base[0], S2=base[1]))
    PE = Q @ Q.T; res["E_mean"] = dict(zip(("S1", "S2"), contrasts(PE))); res["complement_mean"] = dict(zip(("S1", "S2"), contrasts(np.eye(448) - PE)))
    rnd = [contrasts((lambda R: R @ R.T)(np.linalg.qr(np.random.default_rng(k).standard_normal((448, 6)))[0])) for k in (1, 2, 3)]
    res["random6_mean"] = dict(S1=float(np.mean([x[0] for x in rnd])), S2=float(np.mean([x[1] for x in rnd])))
    DDIR = os.environ.get("DECOY_DIR")                       # matched-budget decoy subspaces (das_decoy.py: E, A1, PERM with identical procedure)
    if DDIR:
        for rule in ("E", "A1", "PERM"):
            f = os.path.join(DDIR, f"{M['tag']}_{rule}_R.npy")
            if os.path.exists(f): Qd, _ = np.linalg.qr(np.load(f)); res[f"decoy_{rule}_mean"] = dict(zip(("S1", "S2"), contrasts(Qd @ Qd.T)))
    for k in [k for k in res if k.endswith("_mean")]: res[k]["frac_S1_removed"] = 1 - res[k]["S1"] / base[0]
    out[sp] = res; print(sp, {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "score_contrast" + os.environ.get("SCC_TAG", "") + ".json"), "w"), indent=1)
