import sys, os
sys.path.insert(0, "/home/claude/12345/docs/research_audits/cor_phase15/step1_cross_family_relational_foundations/src")
import numpy as np, gatelib as GL
rng = np.random.default_rng(5); X = GL.TX.sample_gate(20000, rng)["X"]
x0, x1 = X[:, :, 0], X[:, :, 1]; g = GL.gate(X); d = x0 == x1
alt = (3 - (x0 + x1) % 3 - x0) % 3; x1t = np.where(g, x1, alt); gt = (x0 + x1t) % 3 == 0; dt = x0 == x1t
print("P(g)", g.mean(), "edited frac", (x1t != x1).mean(), "d changed frac", (dt != d).mean(), "g preserved", (gt == g).all())
print("P(g|d)", g[d].mean(), "P(g|~d)", g[~d].mean())
for v in range(3): m = x0 == v; print("x0", v, "corr(g,d)", np.corrcoef(g[m], d[m])[0, 1])
m = x0 == 2; print("held-out x0=2: P(g)", g[m].mean(), "P(d)", d[m].mean())
