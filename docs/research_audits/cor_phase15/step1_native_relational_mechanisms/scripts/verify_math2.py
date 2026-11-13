"""Theorem 1-C (general number of colluders C): p(y) ∝ binom(S, C) * A_y - binom(S-1, C-1) * B_y, with S = #eligible groups,
A_y = sum_g W_g L_g(y), B_y = sum_g W_g e_g L_g(y)  (beta = 1).  Verified against brute-force enumeration.  -> logs/verify_math2.json"""
import json, os, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, genlib as GL
np.seterr(all="ignore"); out = {}
for G, C in ((3, 1), (4, 1), (4, 2), (5, 2), (5, 3), (6, 2), (6, 4)):
    env = GL.Env(G=G, C=C); d = GL.sample(env, 3000, np.random.default_rng(100 * G + C)); X = d["X"]; q = np.random.default_rng(7 + G + C).integers(0, 6, len(X))
    pb = GL.posterior_brute(env, X, q); _, parts = GL.deepsets_factor(env, X, q); S = parts["S"]
    cS = np.array([comb(int(s), C) for s in S], float); cS1 = np.array([comb(int(s) - 1, C - 1) if s >= 1 else 0 for s in S], float)
    P = cS[:, None] * parts["A"] - cS1[:, None] * parts["B"]; P = P / P.sum(-1, keepdims=True)
    out[f"G{G}_C{C}"] = dict(max_abs_diff=float(np.abs(P - pb).max()), S_dist=np.bincount(S, minlength=G + 1).tolist())
    print(G, C, out[f"G{G}_C{C}"], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "verify_math2.json"), "w"), indent=1)
