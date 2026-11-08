"""POST HOC: where does the last-layer readout-q attention go? Mass on question q's sources, on other questions'
sources, on readout tokens, on the episode token (per head). Usage: posthoc_attn_targets.py CKPT OUT.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import bayes as B, ckpt as C, analysis as A
ck, outp = sys.argv[1:3]; p = C.load(ck); env = B.Env("T4", K=3, Q=6); Q = env.Q; L = len(p["layers"])
X = B.sample_fast(env, 1000, np.random.default_rng(5_600_000))["X"]; c, _ = A.acts(p, X, [f"pattern{L-1}"]); pat = c[f"pattern{L-1}"]
res = {}
for h in range(pat.shape[1]):
    own, oth, rd, ep = [], [], [], []
    for q in range(Q):
        r = pat[:, h, A.rd_pos(env, q)]
        own.append(r[:, 6*q:6*q+6].sum(1)); oth.append(r[:, :6*Q].sum(1) - own[-1]); rd.append(r[:, 6*Q:7*Q].sum(1)); ep.append(r[:, 7*Q])
    res[f"head{h}"] = dict(own_question=float(np.mean(own)), other_questions=float(np.mean(oth)), readouts=float(np.mean(rd)), episode=float(np.mean(ep)))
json.dump(res, open(outp, "w"), indent=1); print(ck.split('/')[-1], {k: {a: round(b, 2) for a, b in v.items()} for k, v in res.items()})
