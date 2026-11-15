"""Review 1 (math): b5_residual.py canonicalisation.  (1) Is exact C=1 Bayes constant within the canonical cells (sufficiency of the key)?
(2) Effect of b5's GEOMETRIC cell mean (mean of log-probs) vs the KL-optimal ARITHMETIC cell mean on the structured / within-cell split, on a
synthetic 'model' whose deviation from Bayes is partly a function of the cell (structured) and partly episode-specific (non-sufficient)."""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np, genlib as GL
np.seterr(all="ignore"); env = GL.Env(3, 1); rng = np.random.default_rng(11); N_ = 30000
X = GL.sample(env, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_); PB = GL.posterior_closed(env, X, q)
a6 = X.reshape(N_, 6, 3, 2); eq = a6[..., 0] == a6[..., 1]; cnt = eq.sum(1) - eq[ii, q]; vq = a6[ii, q]; PERMS = list(itertools.permutations(range(3)))
keys = []; maps = np.zeros((N_, 3), int)
for j in range(N_):
    best = None
    for pc in PERMS:
        key = tuple(sorted((int(cnt[j, g]), tuple(sorted((pc[vq[j, g, 0]], pc[vq[j, g, 1]])))) for g in range(3)))
        if best is None or key < best[0]: best = (key, pc)
    keys.append(best[0]); maps[j] = best[1]
kid = {k: n for n, k in enumerate(sorted(set(keys)))}; K = np.array([kid[k] for k in keys])
def to_canon(L): C = np.zeros_like(L); C[ii[:, None], maps] = L; return C
PBc = to_canon(PB); spread = 0.0
for c in range(len(kid)):
    m = K == c
    if m.sum() > 1: spread = max(spread, float(np.ptp(PBc[m], 0).max()))
print("cells", len(kid), "max within-cell spread of canonical exact Bayes", spread)
# stabiliser check: cells whose canonical Bayes vector is not invariant under the candidate perms that fix the key
# (2) synthetic model: structured distortion (temperature depending on cell) + episode-specific noise
tau_c = 1 + float(os.environ.get("SC", "0.3")) * np.random.default_rng(1).standard_normal(len(kid)); z = rng.standard_normal(N_)
LP = (tau_c[K] + float(os.environ.get("SC", "0.3")) * z)[:, None] * np.log(PB + 1e-12); P = np.exp(LP - LP.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
Pc = to_canon(P)
def kl(A, B): return float((A * (np.log(A + 1e-300) - np.log(B + 1e-300))).sum(-1).mean())
tot = kl(Pc, PBc)
for name in ("geometric (b5)", "arithmetic"):
    S = np.zeros((len(kid), 3)); C = np.zeros(len(kid))
    src = np.log(Pc + 1e-300) if name.startswith("geo") else Pc
    np.add.at(S, K, src); np.add.at(C, K, 1); cell = S[K] / C[K][:, None]
    cellP = np.exp(cell) / np.exp(cell).sum(-1, keepdims=True) if name.startswith("geo") else cell / cell.sum(-1, keepdims=True)
    w = kl(Pc, cellP); s = kl(cellP, PBc)
    print(f"{name:15s} total {tot:.5f}  within {w/tot:.3f}  structured {s/tot:.3f}  sum {(w+s)/tot:.3f}")
