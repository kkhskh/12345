"""Workstream B1: component-level causal profiles (natural forward patching; development models only).
For edit pairs (before -> after) of families F45_R1, D2 (create q-agreement of a perfect group) and D1 (remove it, S >= 2):
  for each component c in {head (l,h) for l in 0,1,2, h in 0..3} U {mlp l}: run the BEFORE episode with c's output (all tokens) replaced by
  its value in the AFTER episode (all upstream computation recomputed normally, so later components see the patched residual: this is the
  'total effect through c' variant of path patching); response = change of the oriented centred margin of the edited group's q-vote.
Share = component response / full response.  Also the 'direct-only' variant is NOT computed (no freezing of downstream) -- noted.
Profiles compared across models by cosine similarity of the share vectors.  Usage: head_profile.py SPEC... (parent:spec:seed:suffix)
-> logs/head_profile.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
ev = NV.load_eval(); X = ev["X"]; q = ev["q"]; n = len(X); idx = np.arange(n)
cnt = GL.counts(X, 3)[idx, q]; a_ = X.reshape(n, 6, 3, 2)[idx, q]; ag = a_[..., 0] == a_[..., 1]; e = (cnt == 5) & ag
rng = np.random.default_rng(77)
fams = {}
s = (ev["reg_F45"] == "R1") & ev["orient_F45"].astype(bool); fams["R1"] = (X[s], ev["X_F45"][s], q[s], ev["vote_F45"][s].astype(int))
for kind in ("D1", "D2"):
    bi, Xe, v = [], [], []
    for i in range(n):
        if kind == "D1" and e[i].sum() < 2: continue
        cand = [h for h in range(3) if cnt[i, h] == 5 and (ag[i, h] if kind == "D1" else not ag[i, h])]
        if not cand: continue
        h = cand[rng.integers(len(cand))]; x = X[i].copy()
        x[q[i], 2 * h + 1] = (x[q[i], 2 * h] + rng.integers(1, 3)) % 3 if kind == "D1" else x[q[i], 2 * h]
        bi.append(i); Xe.append(x); v.append(x[q[i], 2 * h])
    bi = np.array(bi[:400]); fams[kind] = (X[bi], np.array(Xe[:400]), q[bi], np.array(v[:400]))
COMPS = [("head", l, h) for l in range(3) for h in range(4)] + [("mlp", l, None) for l in range(3)]
out = {"components": [f"{c[0]}{c[1]}" + ("" if c[2] is None else f".{c[2]}") for c in COMPS], "models": {}}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    prof = {}
    for f, (Xb, Xa, qq, vv) in fams.items():
        nb = len(qq); i = np.arange(nb); Xbj, Xaj = jnp.asarray(Xb), jnp.asarray(Xa)
        ra = {}; La = np.asarray(F.forward(M, Xaj, rec=ra)); Lb = np.asarray(F.forward(M, Xbj))
        def margin(L):
            c = F.centred(L[i, qq]); oth = np.where(np.arange(3)[None] == vv[:, None], -np.inf, c).max(-1); return c[i, vv] - oth
        full = margin(La) - margin(Lb); m = jnp.ones((nb, 43), bool); res = []
        for kind, l, h in COMPS:
            if kind == "head": L = np.asarray(F.forward(M, Xbj, head_patch={(l, h): (m, ra[f"heads{l}"][:, h])}))
            else: L = np.asarray(F.forward(M, Xbj, mlp_patch={l: (m, ra[f"mlp_out{l}"])}))
            res.append(float((margin(L) - margin(Lb)).mean()))
        prof[f] = dict(full=float(full.mean()), effects=res, shares=[r / float(full.mean()) for r in res], n=nb)
    out["models"][sp] = prof; print(sp, {f: (round(v["full"], 3), [round(x, 2) for x in v["shares"]]) for f, v in prof.items()}, flush=True)
sims = {}
for f in fams:
    ks = list(out["models"])
    for a in ks:
        for b in ks:
            if a < b:
                u, v = np.array(out["models"][a][f]["effects"]), np.array(out["models"][b][f]["effects"]); sims[f"{f}|{a}|{b}"] = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
out["cosine_similarity_of_effect_profiles"] = sims
json.dump(out, open(os.path.join(HERE, "..", "logs", "head_profile.json"), "w"), indent=1)
