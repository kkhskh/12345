"""POST HOC (exploratory; motivated by PP2's outcome): agreement of each model with three references on the PP2 shifts
(same streams): true Bayes, global plug-in Bayes (training generator), local factorised plug-in (src/local_bayes.py).
Also on-distribution (T4) agreement. Usage: pp2b_local_posthoc.py OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src")); sys.path.insert(0, HERE)
import numpy as np
import gen_family as G, local_bayes as LB
from pp2_shift import model_lp, m4, m1   # noqa  (pp2_shift runs on import; guard below)
plug4 = G.Fam(pc=1.0, rho=0.999); plug1 = G.Fam(pc=0.0)
shifts = {"T4_on_distribution": G.Fam(pc=1.0, rho=1.0), "rho0.9": G.Fam(pc=1.0, rho=0.9), "rho0.7": G.Fam(pc=1.0, rho=0.7), "pc0_T1data": G.Fam(pc=0.0)}
seeds = {"T4_on_distribution": 6_250_000, "rho0.9": 6_200_000, "rho0.7": 6_200_001, "pc0_T1data": 6_200_002}
res = {}
for sn, f in shifts.items():
    d = G.sample(f, 6000, np.random.default_rng(seeds[sn])); X, Y = d["X"], d["Y"]
    refs = {"true": G.posterior(f, X)["post_y"], "global_plugin": G.posterior(plug4, X)["post_y"], "local_plugin": LB.local_post(plug4, X)}
    am = {k: np.nan_to_num(v, nan=0).argmax(-1) for k, v in refs.items()}
    R = {f"acc_{k}": float((v == Y).mean()) for k, v in am.items()}
    for name, (kind, path) in m4.items():
        a = model_lp(kind, path, X).argmax(-1)
        R[name] = {f"agree_{k}": float((a == v).mean()) for k, v in am.items()}
        dis = am["global_plugin"] != am["local_plugin"]
        R[name]["n_global_vs_local_disagree"] = int(dis.sum())
        R[name]["side_local_when_global_local_disagree"] = float((a == am["local_plugin"])[dis].mean()) if dis.any() else None
        R[name]["side_global_when_global_local_disagree"] = float((a == am["global_plugin"])[dis].mean()) if dis.any() else None
    res[sn] = R
    print(sn, {k: round(v, 3) for k, v in R.items() if not isinstance(v, dict)})
    for name in m4: print("   ", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[name].items()})
json.dump(res, open(sys.argv[1], "w"), indent=1)
