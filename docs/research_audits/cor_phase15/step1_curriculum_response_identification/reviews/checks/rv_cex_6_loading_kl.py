"""Counterexample reviewer, check 6: is the per-model one-factor loading (rv_cex_4) a function of KL alone, across arms?
If coherence were 'explained by approximation quality', a model's loading should be predicted by its KL irrespective of arm.
Uses logs/review_checks/rv_cex_4_factor.json and logs/conf/primary.json.  Usage: python rv_cex_6_loading_kl.py"""
import json, os
import numpy as np
from scipy.stats import spearmanr
PK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
L = json.load(open(os.path.join(PK, "logs", "review_checks", "rv_cex_4_factor.json")))["observed"]["loadings"]
Q = json.load(open(os.path.join(PK, "logs", "conf", "primary.json")))["P2_adjusted"]["quality"]
m = list(L); l = np.array([L[k] for k in m]); kl = np.log(np.array([Q[k]["KL_bayes"] for k in m])); T = np.array([k[1] == "T" for k in m])
for k in m: print(f"  {k}: loading {L[k]:.2f}  KL {Q[k]['KL_bayes']:.3f}")
print("Spearman(loading, log KL) all", round(spearmanr(l, kl)[0], 2), "| within T1", round(spearmanr(l[T], kl[T])[0], 2), "| within P", round(spearmanr(l[~T], kl[~T])[0], 2))
X = np.c_[np.ones(len(m)), kl]; b = np.linalg.lstsq(X, l, rcond=None)[0]; res = l - X @ b
X2 = np.c_[X, T]; b2 = np.linalg.lstsq(X2, l, rcond=None)[0]
print(f"loading ~ log KL: slope {b[1]:+.3f}, R2 {1 - (res**2).sum() / ((l - l.mean())**2).sum():.2f}; adding arm: arm coef {b2[2]:+.3f}, "
      f"mean residual T {res[T].mean():+.3f} P {res[~T].mean():+.3f}")
