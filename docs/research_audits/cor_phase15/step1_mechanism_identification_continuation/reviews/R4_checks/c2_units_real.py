"""R4 check 2: on real checkpoints, compare the code's unit criterion with (i) a backfit
non-relational residual share and (ii) an even/odd split; and test the final-LN centring in step 6."""
import sys; sys.path.insert(0, "../../src")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import learned_analysis as LA, learned as Lm
from c1_iagree_synthetic import code_crit, backfit_resid_share
A, B, g = LA.grid(); n = len(g)
tot = dict(code=0, backfit=0, code_only=0, backfit_only=0, oddmiss=0)
for task in ["T1", "T3", "T4"]:
  for arch in ["TF1", "TF2"]:
    for seed in range(5):
      P = LA.load(task, arch, seed); L = len(P["layers"])
      pv = 1.0 if task == "T3" else 0.0
      _, st = LA.capture(P, task, LA.inp_bin(A, B, np.full_like(A, pv)))
      agp, cfp = LA.matched_pairs()
      _, sta = LA.capture(P, task, LA.inp_bin(agp[:,0], agp[:,1], np.full(len(agp), pv)))
      _, stc = LA.capture(P, task, LA.inp_bin(cfp[:,0], cfp[:,1], np.full(len(cfp), pv)))
      v = np.asarray(P["gf"] * (P["WU"][:, 0] - P["WU"][:, 1])); vc = v - v.mean()
      c = dict(code=0, backfit=0, code_only=0, backfit_only=0, oddmiss=0, signflip_centre=0, nrel=0)
      for l in range(L):
        H = st[f"mlp_hidden{l}"][:, -1]; W2 = np.asarray(P["layers"][l]["W2"])
        for u in range(H.shape[1]):
          y = H[:, u]
          if y.var() < 1e-10: continue
          ds, ia = code_crit(y); br = backfit_resid_share(y, it=60)
          Z = y.reshape(n, n); odd = (Z - Z[::-1, ::-1]) / 2   # (a,b)->(-a,-b) odd part
          oddshare = odd.var() / y.var()
          fc = ds >= .2 and ia >= .2; fb = ds >= .2 and br >= .2
          c["code"] += fc; c["backfit"] += fb; c["code_only"] += fc and not fb; c["backfit_only"] += fb and not fc
          if fb and not fc and oddshare > .5: c["oddmiss"] += 1
          if fc:
            c["nrel"] += 1
            cond = np.mean(sta[f"mlp_hidden{l}"][:, -1, u] - stc[f"mlp_hidden{l}"][:, -1, u])
            if np.sign(W2[u] @ v) != np.sign(W2[u] @ vc): c["signflip_centre"] += 1
      print(task, arch, seed, c, flush=True)
