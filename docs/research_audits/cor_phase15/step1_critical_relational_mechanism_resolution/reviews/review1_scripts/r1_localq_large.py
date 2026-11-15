"""Review 1: exact LOCALQ vs FULL accuracy at beta = 0.6 on 100,000 fresh episodes (exact analytic marginalisation; see r1_localq_exact.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np, genlib as GL, xii_bayes as XB
src = open(os.path.join(HERE, "r1_localq_exact.py")).read(); ns = {}
exec(src[src.index("def exact_localq"):src.index("PX = exact_localq")], dict(np=np, al=0.8, K=3), ns)
E2 = GL.Env(3, 2, beta=0.6); rng = np.random.default_rng(424242); n = 100000; d = GL.sample(E2, n, rng); X, Y = d["X"], d["Y"]; q = rng.integers(0, 6, n); y = Y[np.arange(n), q]
PF = XB.posterior_shared(E2, X, q); PX = ns["exact_localq"](X, q, 0.6); cf = PF.argmax(-1) == y; cl = PX.argmax(-1) == y; dlt = cf.astype(float) - cl
print("acc_full", cf.mean(), "acc_localq_exact", cl.mean(), "gap", dlt.mean(), "paired s.e.", dlt.std() / np.sqrt(n))
for nn in (3000,):
    print("paired s.e. of gap at n=3000", dlt.std() / np.sqrt(nn))
