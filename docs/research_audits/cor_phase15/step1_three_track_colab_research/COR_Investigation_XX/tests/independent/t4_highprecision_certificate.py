"""Independent high-precision recomputation of two size-18 T4 interaction coefficients (40-digit mpmath, own posterior implementation).
Written by pre-run reviewer B (an AI agent reviewer, not the package builder); copied verbatim except this docstring.  Optional (slow; needs mpmath).
Usage: python tests/independent/t4_highprecision_certificate.py <package root>.  Reviewer result: -20.8558394526408 and 138.635712661052."""
import sys, itertools, numpy as np
sys.path.insert(0, sys.argv[1]); from cor_research import env as E; E.configure('cpu'); E.legacy_paths()
import genlib as GL, relpairs as RP
from cor_research.mathematics import t4_order as T4O
import mpmath as mp
mp.mp.dps = 40
env = RP.ENV1; K=3; A = mp.mpf(4)/5
def post_mp(x, q):
    # exact enumeration over roles (g reliable, h colluder), uniform role prior, beta=1
    G=3; out=[mp.mpf(0)]*K
    for g in range(G):
        for h in range(G):
            if h==g: continue
            w = mp.mpf(1); ok=True
            for qq in range(6):
                a0,a1 = x[qq,2*h], x[qq,2*h+1]
                if a0!=a1: ok=False; break
            if not ok: continue
            for qq in range(6):
                if qq==q: continue
                s0,s1 = x[qq,2*g], x[qq,2*g+1]
                w *= sum((A if s0==y else (1-A)/2)*(A if s1==y else (1-A)/2) for y in range(K))/K
            for y in range(K):
                s0,s1 = x[q,2*g], x[q,2*g+1]
                out[y] += w*(A if s0==y else (1-A)/2)*(A if s1==y else (1-A)/2)
    return out
rng = np.random.default_rng(0); res=[]
for b in range(2):
    d = GL.sample(env, 1, rng); x0 = d["X"][0]; x1 = (x0 + rng.integers(1, env.K, x0.shape)) % env.K
    g_rel, hs = env.roles[int(d["ridx"][0])]; free = [s for s in range(6) if s//2 not in hs]
    for size in ((18,) if b in (0,1) else ()):
        coords = [(0, free[0])] + [(1 + (j % 5), free[(j // 5) % len(free)]) for j in range(size - 1)]
        if len(set(coords)) < len(coords): continue
        tot = mp.mpf(0); n=len(coords)
        for sub in itertools.product([0,1], repeat=n):
            X = x0.copy()
            for j,(qq,s) in enumerate(coords):
                if sub[j]: X[qq,s]=x1[qq,s]
            P = post_mp(X, 0); f = mp.log(P[0])-mp.log(P[1])
            tot += (-1)**(n-sum(sub)) * f
        v64,_ = T4O.interaction(GL, env, x0, x1, coords)
        print(b, size, 'mp=', mp.nstr(tot, 15), 'float64=', v64, flush=True)
