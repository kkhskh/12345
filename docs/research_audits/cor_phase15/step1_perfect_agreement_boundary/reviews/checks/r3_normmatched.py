"""R3 check: norm-matched random site delta (same per-episode ||dA_site|| as F45, random direction) and sign-flipped F45 site delta."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np, boundary as B
E = B.E; STREAM = 16_000_000; NE = 3000
D = E.sample(NE, STREAM); X, cfg = D["X"], D["cfg"]; q = np.random.default_rng(STREAM + 1).integers(0, 6, NE); idx = np.arange(NE)
Xp = X.copy(); gs = np.full(NE, -1); ok = np.zeros(NE, bool)
for qq in range(6):
    s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(STREAM + 10 + qq + 400), 4, 5); Xp[s], gs[s], ok[s] = a, g_, o
vote = np.array([B.group_vote(X[i:i + 1], q[i], max(gs[i], 0))[0] for i in idx]); o = ok & (vote >= 0); v = np.maximum(vote, 0)
for m in sys.argv[1:]:
    M = B.Model(m); A0, L0 = M.acts(X)[:2]; c0 = E.clogit(L0[idx, q]); Aa, La = M.acts(Xp)[:2]; dA = Aa - A0; rng = np.random.default_rng(1)
    dfull = B.oriented_margin(E.clogit(La[idx, q]), v) - B.oriented_margin(c0, v)
    Zr = np.zeros_like(dA); Zs = np.zeros_like(dA)
    for i in idx:
        pos = E.site_positions(M.kind, q[i]); d = dA[i, pos]; r = rng.standard_normal(d.shape); Zr[i, pos] = r * np.linalg.norm(d) / np.linalg.norm(r); Zs[i, pos] = d
    res = []
    for nm, Z in (("site", Zs), ("rand_normmatched", Zr), ("site_x0.5", 0.5 * Zs)):
        dm = B.oriented_margin(E.clogit(M.run(X, Z)[idx, q]), v) - B.oriented_margin(c0, v)
        res.append(f"{nm}: mean|dm| {np.abs(dm[o]).mean():.3f} share {(dm*dfull)[o].sum()/(dfull**2)[o].sum():.2f}")
    print(m, f"F45 mean|dm_full| {np.abs(dfull[o]).mean():.3f}", " | ".join(res), flush=True)
