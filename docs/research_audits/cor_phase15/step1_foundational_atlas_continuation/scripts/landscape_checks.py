"""Two derived predictions checked exactly (mandate §10):
 (a) rel<->coll role-swap symmetry at rho*(alpha): E[P(true reliable)] = E[P(true colluder is reliable)] at rho* only.
 (b) 'a copying pair is one vote': in a reliable-vs-informative-colluder conflict (reliable pair agrees on v, colluder pair
     copies c != v) the Bayes answer follows the colluder iff 2 a_c/(1-a_c) > (2 alpha/(1-alpha))^2 (K = 3, other evidence
     ignored); threshold a_c* = 0.970 for alpha = 0.8.  Writes logs/landscape/checks.json."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, roles_engine as RE
K = 3; out = {}
def gen(roles):
    ps = sorted(set(itertools.permutations(roles))); return RE.Gen(ps, [1.0] * len(ps), K, 6)
for alpha in (0.7, 0.8, 0.9):
    a0 = alpha ** 2 + (1 - alpha) ** 2 / 2; rs = (3 * a0 - 1) / 2; rows = []
    for rho in (rs - 0.15, rs, rs + 0.15):
        g = gen([("rel", alpha, 0.0, 0.0), ("coll", float(rho), None), ("noise",)]); d = g.sample(4000, np.random.default_rng(8_500_000)); pc = g.posterior(d["X"])["post_cfg"]
        ri = lambda c: [r[0] for r in c].index("rel"); ci = lambda c: [r[0] for r in c].index("coll")
        tr = np.array([ri(g.configs[c]) for c in d["cfg"]]); tc = np.array([ci(g.configs[c]) for c in d["cfg"]]); rel_of = np.array([ri(c) for c in g.configs])
        P_true = np.array([pc[i, rel_of == tr[i]].sum() for i in range(4000)]); P_coll = np.array([pc[i, rel_of == tc[i]].sum() for i in range(4000)])
        rows.append(dict(rho=float(rho), E_P_true_rel=float(P_true.mean()), E_P_colluder_called_rel=float(P_coll.mean())))
    out[f"swap_alpha{alpha}"] = rows; print(alpha, [(round(r["rho"], 3), round(r["E_P_true_rel"], 3), round(r["E_P_colluder_called_rel"], 3)) for r in rows], flush=True)
alpha = 0.8; thr = None
odds_rel = (2 * alpha / (1 - alpha)) ** 2; ac_star = odds_rel / (odds_rel + 2); out["a_c_star_predicted"] = ac_star; rows = []
for ac in (0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995):
    g = gen([("rel", alpha, 0.0, 0.0), ("coll", 1.0, ac), ("noise",)]); d = g.sample(8000, np.random.default_rng(8_600_000)); X = d["X"]; py = g.posterior(X)["post_y"].argmax(-1)
    cf = [g.configs[c] for c in d["cfg"]]; A_ = np.array(RE.GROUPS); n = np.arange(len(X))
    ci = np.array([[r[0] for r in c].index("coll") for c in cf]); ri = np.array([[r[0] for r in c].index("rel") for c in cf])
    xc = X[n, :, A_[ci, 0]]; r0 = X[n, :, A_[ri, 0]]; r1 = X[n, :, A_[ri, 1]]; conf = (r0 == r1) & (r0 != xc)
    rows.append(dict(a_c=ac, follow_colluder=float((py == xc)[conf].mean()), n=int(conf.sum()))); print(rows[-1], flush=True)
out["one_vote_rows"] = rows; print("predicted a_c* =", round(ac_star, 4))
json.dump(out, open(os.path.join(HERE, "..", "logs", "landscape", "checks.json"), "w"), indent=1)
