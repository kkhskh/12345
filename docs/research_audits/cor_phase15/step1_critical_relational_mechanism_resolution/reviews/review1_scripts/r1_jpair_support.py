"""Review 1: at beta = 0.6 every report array has positive probability, so 'finite posterior' (c_math V5) is vacuous as an on-support test.
Report instead how typical J-pair edits are: log p(X') - log p(X) under the generator, and the fraction of raw arrays with p > 0."""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np, genlib as GL
K = 3; E2 = GL.Env(3, 2, beta=0.6); rng = np.random.default_rng(3)
def logmarg(Xx):
    rel, _, _ = GL._group_lik(E2, Xx); relm = rel.mean(-1); a_ = Xx.reshape(len(Xx), 6, 3, 2); terms = []
    for g in range(3):
        H = [h for h in range(3) if h != g]; srcs = a_[:, :, H, :].reshape(len(Xx), 6, 4); alleq = (srcs == srcs[..., :1]).all(-1)
        colH = E2.beta * alleq / K + (1 - E2.beta) / K ** 4; terms.append(np.log(relm[:, :, g]).sum(1) + np.log(colH).sum(1) - np.log(3))
    T = np.stack(terms); mx = T.max(0); return mx + np.log(np.exp(T - mx).sum(0))
Xu = rng.integers(0, 3, (2000, 6, 6)); print("uniform random arrays with finite log p:", np.isfinite(logmarg(Xu)).mean())
perms = [np.array(p) for p in itertools.permutations(range(3))]; nonid = perms[1:]
d = GL.sample(E2, 2000, rng); X = d["X"]; q = rng.integers(0, 6, 2000); h = rng.integers(0, 3, 2000); X2 = X.copy()
for j in range(2000):
    pm = nonid[rng.integers(5)]; cols = [2 * h[j], 2 * h[j] + 1]; others = [t for t in range(6) if t != q[j]]; X2[j][np.ix_(others, cols)] = pm[X[j][np.ix_(others, cols)]]
dl = logmarg(X2) - logmarg(X); print("J-pair log p(X') - log p(X): median %.2f, 10%% %.2f, frac < -3: %.3f" % (np.median(dl), np.percentile(dl, 10), (dl < -3).mean()))
