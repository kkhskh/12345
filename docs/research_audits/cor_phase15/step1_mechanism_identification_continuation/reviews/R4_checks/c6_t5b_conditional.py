"""R4 check 6: T5b units vs candidate-/prior-conditional agreement features the criterion omits."""
import sys; sys.path.insert(0, "../../src")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import learned_analysis as LA, learned as Lm
from semantics import sample_k, semantic_agreement
from c5_t5 import poly2
for arch in ["TF1", "TF2"]:
  for seed in [0, 3]:
    P = LA.load("T5b", arch, seed); L = len(P["layers"])
    bt = sample_k("T5b", 4000, np.random.default_rng(3000)); _, st = LA.capture(P, "T5b", Lm.to_inp("T5b", bt))
    s = bt.zA + bt.zB; sem = semantic_agreement(bt.zA, bt.zB).astype(float)
    ohA = np.eye(3)[bt.zA.argmax(1)]; ohB = np.eye(3)[bt.zB.argmax(1)]; Fp = np.eye(3)[bt.p]
    base = np.column_stack([poly2(s), poly2(bt.zA), poly2(bt.zB), (bt.zA*bt.zB).sum(1), sem])  # code's 'full'
    base_p = np.column_stack([base, ohA, ohB, Fp])
    extra = np.column_stack([sem[:, None]*ohA, (1-sem)[:, None]*Fp])   # agree-on-k, prior-if-disagree
    for l in range(L):
      H = st[f"mlp_hidden{l}"][:, -1]
      g = np.array([LA.r2(np.column_stack([base_p, extra]), H[:, u]) - LA.r2(base_p, H[:, u]) for u in range(H.shape[1]) if H[:, u].var() > 1e-10])
      print(f"T5b {arch} s{seed} L{l}: gain of [1[agree]*onehot(k*), 1[disagree]*onehot(p)] beyond code-full + per-source argmax + p:"
            f" max={g.max():.3f}  #>=0.2: {(g>=.2).sum()}  #>=0.1: {(g>=.1).sum()}")
