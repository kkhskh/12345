"""Review 1 (math): exact LOCALQ posterior vs c_math.py V4 Monte-Carlo estimate on the SAME episodes.
Replays c_math.py's RNG stream (V1, V2 rng draws, V3 sample) to regenerate X4/q4, re-runs the 48-draw SNIS estimator, and computes the exact
posterior given T = (self-agreement bits at q' != q, all values at q) by analytic marginalisation (cell sum factorises over questions)."""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np, genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); K, Q, al = 3, 6, 0.8
rng = np.random.default_rng(87_000_000)
for beta in (0.6, 1.0):
    E = GL.Env(3, 2, beta=beta); GL.sample(E, 400, rng); rng.integers(0, 6, 400)
E2 = GL.Env(3, 2, beta=0.6); d = GL.sample(E2, 20000, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
perms = [np.array(p) for p in itertools.permutations(range(3))]
for j in range(4000):
    rng.integers(5); rng.integers(3); rng.integers(6)
E1 = GL.Env(3, 2, beta=1.0); GL.sample(E1, 30000, rng); rng.integers(0, 6, 30000)
d4 = GL.sample(E2, 3000, rng); X4, Y4 = d4["X"], d4["Y"]; q4 = rng.integers(0, 6, 3000); n = 3000; i = np.arange(n); yq = Y4[i, q4]
PF = XB.posterior_shared(E2, X4, q4); print("acc_full", (PF.argmax(-1) == yq).mean())

def exact_localq(X, q, beta):
    n = len(X); a = X.reshape(n, 6, 3, 2); ag = a[..., 0] == a[..., 1]; i = np.arange(n)
    pr = al ** 2 + (1 - al) ** 2 / (K - 1); pa = 1 / K
    P = np.zeros((n, K))
    for g in range(3):
        H = [h for h in range(3) if h != g]
        lq = np.log(np.where(ag[:, :, g], pr, 1 - pr))
        both = ag[:, :, H[0]] & ag[:, :, H[1]]
        ind = np.where(ag[:, :, H[0]], pa, 1 - pa) * np.where(ag[:, :, H[1]], pa, 1 - pa)
        lq = lq + np.log(beta * both + (1 - beta) * ind)
        lq[i, q] = 0.0; lw = lq.sum(1)
        aq = a[i, q]; srcs = aq[:, H, :].reshape(n, 4); alleq = (srcs == srcs[:, :1]).all(1)
        colq = beta * alleq / K + (1 - beta) / K ** 4
        y = np.arange(K); rel = np.where(aq[:, g, 0, None] == y, al, (1 - al) / 2) * np.where(aq[:, g, 1, None] == y, al, (1 - al) / 2)
        P += np.exp(lw)[:, None] * rel * colq[:, None]
    return P / P.sum(1, keepdims=True)
PX = exact_localq(X4, q4, 0.6); acc_ex = (PX.argmax(-1) == yq).mean(); print("acc_localq_exact", acc_ex, "gap_exact", (PF.argmax(-1) == yq).mean() - acc_ex)
# cross-check exact LOCALQ against brute enumeration of the cell for a few episodes (cell is a product of per-(group,q') S3 orbits)
def logmarg(Xx):
    rel, _, _ = GL._group_lik(E2, Xx); relm = rel.mean(-1); a_ = Xx.reshape(len(Xx), 6, 3, 2); terms = []
    for g in range(3):
        H = [h for h in range(3) if h != g]; srcs = a_[:, :, H, :].reshape(len(Xx), 6, 4); alleq = (srcs == srcs[..., :1]).all(-1)
        colH = E2.beta * alleq / K + (1 - E2.beta) / K ** 4; terms.append(np.log(relm[:, :, g]).sum(1) + np.log(colH).sum(1) - np.log(3))
    T = np.stack(terms); mx = T.max(0); return mx + np.log(np.exp(T - mx).sum(0))
# replay MC with R=48 (same rng continuation as c_math) and record ESS
LW, PS_ = [], []
for r_ in range(48):
    Xr = X4.copy()
    for g in range(3):
        pm = np.stack(perms)[rng.integers(0, 6, (3000, 6))]
        for p in range(6):
            m_ = q4 != p; v = Xr[:, p, 2 * g:2 * g + 2]; Xr[m_, p, 2 * g:2 * g + 2] = np.take_along_axis(pm[m_, p], v[m_], 1)
    LW.append(logmarg(Xr)); PS_.append(XB.posterior_shared(E2, Xr, q4))
LW = np.stack(LW); w = np.exp(LW - LW.max(0)); w = w / w.sum(0); PL = (w[..., None] * np.stack(PS_)).sum(0)
ess = 1 / (w ** 2).sum(0)
print("acc_localq_mc48 (replay)", (PL.argmax(-1) == yq).mean())
print("ESS quantiles (of 48)", np.percentile(ess, [5, 25, 50, 75, 95]).round(2), "frac ESS<2", (ess < 2).mean())
print("mean |MC - exact| max-over-y", np.abs(PL - PX).max(1).mean(), "decision disagreement MC vs exact", (PL.argmax(-1) != PX.argmax(-1)).mean())
kl = (PX * (np.log(PX) - np.log(PL + 1e-300))).sum(1); print("KL(exact||MC) mean", kl.mean())
# check MC estimator on a 200-episode subset with many draws: does it converge to the exact value?
sub = np.arange(200); R2 = 1500; LW2 = []; PS2 = []
for r_ in range(R2):
    Xr = X4[sub].copy()
    for g in range(3):
        pm = np.stack(perms)[rng.integers(0, 6, (len(sub), 6))]
        for p in range(6):
            m_ = q4[sub] != p; v = Xr[:, p, 2 * g:2 * g + 2]; Xr[m_, p, 2 * g:2 * g + 2] = np.take_along_axis(pm[m_, p], v[m_], 1)
    LW2.append(logmarg(Xr)); PS2.append(XB.posterior_shared(E2, Xr, q4[sub]))
LW2 = np.stack(LW2); w2 = np.exp(LW2 - LW2.max(0)); w2 /= w2.sum(0); PL2 = (w2[..., None] * np.stack(PS2)).sum(0)
print("subset200: max|MC1500 - exact|", np.abs(PL2 - PX[sub]).max(), "mean", np.abs(PL2 - PX[sub]).max(1).mean(), "| MC48 mean err on same subset", np.abs(PL[sub] - PX[sub]).max(1).mean())
