"""R4 check 3: (a) the agreement-probe direction at the last residual vs the centred readout
direction; (b) whether sign(ab)-probe accuracy at the last site equals model accuracy on sign(ab);
(c) T2: is ab decodable via a^2, b^2, (a+b)^2 features?"""
import sys; sys.path.insert(0, "../../src")
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
import learned_analysis as LA
from semantics import sample_binary
for task in ["T1", "T2", "T3", "T4"]:
  for arch in ["TF1", "TF2", "TF2AO"]:
    for seed in [0, 3]:
      P = LA.load(task, arch, seed); L = len(P["layers"])
      bt = sample_binary(task, 4000, np.random.default_rng(3000))
      lg, st = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p))
      X = st[f"resid_post{L-1}"][:, -1]; s = (bt.a*bt.b > 0).astype(int)
      tr, te = slice(0, 2000), slice(2000, 4000)
      w = LogisticRegression(C=10.0, max_iter=3000).fit(X[tr], s[tr]).coef_[0]; w /= np.linalg.norm(w)
      v = np.asarray(P["gf"]*(P["WU"][:,0]-P["WU"][:,1])); vc = v - v.mean(); vc /= np.linalg.norm(vc)
      m = lg[:,0]-lg[:,1]
      line = f"{task} {arch} s{seed}: |cos(probe, centred readout)|={abs(w@vc):.3f}"
      if task == "T2":
        def R(y): return Ridge(alpha=1e-3).fit(X[tr], y[tr]).score(X[te], y[te])
        a, b = bt.a, bt.b
        line += f"  R2: ab={R(a*b):.3f} a^2={R(a**2):.3f} b^2={R(b**2):.3f} (a+b)^2={R((a+b)**2):.3f}"
        # decode ab using only the three decoded nonrelational features
        F = np.column_stack([Ridge(alpha=1e-3).fit(X[tr], y[tr]).predict(X) for y in (a**2, b**2, (a+b)**2)])
        line += f"  ab-from-3-decoded={Ridge(alpha=1e-6).fit(F[tr], (a*b)[tr]).score(F[te], (a*b)[te]):.3f}"
      print(line, flush=True)
