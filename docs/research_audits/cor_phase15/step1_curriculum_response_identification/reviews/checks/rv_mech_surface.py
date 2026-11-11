"""V-surface realisation vs natural episodes (generator only): cross-pair agreement statistics in the realised questions
(not a sufficient statistic for Bayes, but available to a model), and the fraction of R pairs that create a second count-5 group."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "..", "logs", "conf")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); import cri as R; E = R.E
G = np.load(os.path.join(D, "generator.npz")); X, q, cfg = G["X"], G["qsel"], G["cfg"]; Y = G["Y"]
rng = np.random.default_rng(5); N = 2000
Xr = R.realize(X[:N], 0, E.loo(X[:N])[:, 0], rng)          # same counts, fresh contents (question 0 kept)
def stats(Z, Yt):
    a = [Z[:, 1:, s] for s in (0, 2, 4)]                    # first source of each pair, questions 1..5
    cross = np.mean([(a[i] == a[j]).mean() for i in range(3) for j in range(i + 1, 3)])
    truth = np.mean([(a[i] == Yt[:, 1:]).mean() for i in range(3)])
    return cross, truth
print("natural   : cross-pair first-source agreement %.3f, first source = truth %.3f" % stats(X[:N], Y[:N]))
print("realised  : cross-pair first-source agreement %.3f, first source = truth %.3f" % stats(Xr, Y[:N]))
h = R.true_h(cfg); mk = G["ok_R"].astype(bool); n0, n1 = E.loo(X)[np.arange(len(X)), q], E.loo(G["X_R"])[np.arange(len(X)), q]
two = ((n1 == 5).sum(1) >= 2) & ((n0 == 5).sum(1) == 1); fr = ((n0 == 5).sum(1) >= 2)
print("R valid pairs: base already |H|>=2 %.3f ; edit creates second count-5 group %.3f" % (fr[mk].mean(), two[mk].mean()))
# Cm magnitude relative to N (f_Cm in the package is a projection only)
nrm = lambda a: np.sqrt((a ** 2).sum(-1)).mean(); mk = G["ok_N"].astype(bool)
for m in ["cT61", "cT62", "cT63", "cT64", "cT65", "cT67", "cP61", "cP62", "cP63", "cP64", "cP66", "cP67"]:
    z = np.load(os.path.join(D, f"model_{m}.npz")); dN = (z["c_N"] - z["c_base"])[mk]; dC = (z["c_Cm"] - z["c_base"])[mk]
    print(m, "||dCm||/||dN|| = %.2f   f_Cm(proj) = %.3f" % (nrm(dC) / nrm(dN), (dC * dN).sum() / (dN * dN).sum()))
