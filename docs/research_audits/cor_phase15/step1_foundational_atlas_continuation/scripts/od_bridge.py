"""O/D restricted bridge (mandate §14).  D_c(x_q) = number of sources whose report at q differs from candidate c
(the per-candidate discrepancy of the original O/D formalism in its simplest form).  Theorem OD1: if all sources are
conditionally independent given Y with one common accuracy alpha > 1/K and symmetric errors, then
log P(c | x_q) = const - D_c log(alpha / e) exactly, so the discrepancy ranking equals the Bayes ranking.
Checked: agreement of argmin D_c (ties broken by Bayes, i.e. counted as agreement) with the Bayes argmax, single-question
Bayes and full-episode Bayes, for generators that satisfy or violate the assumptions.  Writes logs/od_bridge.json."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, roles_engine as RE
K = 3; out = {}
def gen(roles, Q=6):
    ps = sorted(set(itertools.permutations(roles))); return RE.Gen(ps, [1.0] * len(ps), K, Q)
cases = {"OD1_all_reliable_equal_alpha": [("rel", 0.8, 0.0, 0.0)] * 3,
         "unequal_alpha_0.9_0.7_0.55": [("rel", 0.9, 0.0, 0.0), ("rel", 0.7, 0.0, 0.0), ("rel", 0.55, 0.0, 0.0)],
         "correlated_pair_kappa0.7": [("rel", 0.8, 0.0, 0.7), ("rel", 0.8, 0.0, 0.0), ("rel", 0.8, 0.0, 0.0)],
         "biased_errors_0.8": [("rel", 0.8, 0.8, 0.0)] * 3,
         "T1_rel_noise_noise": [("rel", 0.8, 0.0, 0.0), ("noise",), ("noise",)],
         "T4_rel_coll_noise": [("rel", 0.8, 0.0, 0.0), ("coll", 1.0, None), ("noise",)]}
for nm, roles in cases.items():
    g = gen(roles); d = g.sample(5000, np.random.default_rng(9_000_000)); X = d["X"]; post = g.posterior(X)["post_y"]   # leave-one-out-weighted Bayes per question
    D = np.stack([(X != c).sum(-1) for c in range(K)], -1)                  # N,Q,K
    best = np.nan_to_num(post, nan=-1).argmax(-1)
    tie_ok = D[np.arange(len(X))[:, None], np.arange(X.shape[1])[None], best] == D.min(-1)   # Bayes choice is among the min-D set
    strict = (D == D.min(-1, keepdims=True)).sum(-1) == 1
    out[nm] = dict(argminD_consistent_with_bayes=float(tie_ok.mean()), on_unique_minD=float(tie_ok[strict].mean()), frac_unique=float(strict.mean()),
                   acc_bayes=float((best == d["Y"]).mean()), acc_minD_random_tiebreak=float(np.mean([(np.random.default_rng(1).permutation(np.flatnonzero(D[i, q] == D[i, q].min()))[0] == d["Y"][i, q]) for i in range(1500) for q in range(6)])))
    print(nm, {k: round(v, 3) for k, v in out[nm].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "od_bridge.json"), "w"), indent=1)
