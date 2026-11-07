"""R4 check 5: T5 unit criterion. (a) <zA,zB> lies exactly in the span of the base features;
(b) D_sum^K(unit) with a poly2 sum model vs a flexible sum model; (c) T5b gains."""
import sys; sys.path.insert(0, "../../src")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import learned_analysis as LA, learned as Lm
from semantics import sample_k, semantic_agreement
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
def poly2(Z):
    k = Z.shape[1]; return np.column_stack([Z] + [Z[:, i:i+1]*Z[:, j:j+1] for i in range(k) for j in range(i, k)])
bt = sample_k("T5a", 4000, np.random.default_rng(3000)); s = bt.zA + bt.zB
Fs = poly2(s); Fab = np.column_stack([poly2(bt.zA), poly2(bt.zB)]); dot = (bt.zA*bt.zB).sum(1)
X = np.column_stack([np.ones(4000), Fs, Fab]); res = dot - X @ np.linalg.lstsq(X, dot, rcond=None)[0]
print("(a) residual std of <zA,zB> on base features:", res.std(), " (dot std", dot.std(), ")")
for task in ["T5a", "T5b"]:
  for arch in ["TF1", "TF2"]:
    P = LA.load(task, arch, 0); L = len(P["layers"])
    bt = sample_k(task, 4000, np.random.default_rng(3000)); _, st = LA.capture(P, task, Lm.to_inp(task, bt))
    s = bt.zA + bt.zB; Fs = poly2(s); Fab = np.column_stack([poly2(bt.zA), poly2(bt.zB)])
    dot = (bt.zA*bt.zB).sum(1); sem = semantic_agreement(bt.zA, bt.zB)
    oh = lambda z: np.eye(3)[z.argmax(1)]
    Fp = np.eye(3)[bt.p]  # prior one-hot (T5b)
    for l in range(L):
      H = st[f"mlp_hidden{l}"][:, -1]; rows = []
      for u in range(H.shape[1]):
        y = H[:, u]
        if y.var() < 1e-10: continue
        ds_poly = 1 - LA.r2(Fs, y)
        h = HGB(max_iter=200).fit(s[:2000], y[:2000]); ds_flex = 1 - h.score(s[2000:], y[2000:])
        g_dot = LA.r2(np.column_stack([Fs, Fab, dot]), y) - LA.r2(np.column_stack([Fs, Fab]), y)
        base2 = np.column_stack([Fs, Fab, oh(bt.zA), oh(bt.zB), Fp])
        g_sem = LA.r2(np.column_stack([Fs, Fab, dot, sem]), y) - LA.r2(np.column_stack([Fs, Fab]), y)
        g_sem2 = LA.r2(np.column_stack([base2, sem]), y) - LA.r2(base2, y)
        rows.append((ds_poly, ds_flex, g_dot, g_sem, g_sem2))
      R = np.array(rows)
      print(f"{task} {arch} L{l}: n={len(R)}  #D_poly>=.2: {(R[:,0]>=.2).sum()}  #D_flex>=.2: {(R[:,1]>=.2).sum()}"
            f"  max gain(dot)={R[:,2].max():.2e}  max gain(code)={R[:,3].max():.3f}  max gain(+per-source argmax,+p)={R[:,4].max():.3f}")
