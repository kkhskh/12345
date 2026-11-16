"""Post-review reconstruction of the R1 power simulation (the original pre-freeze run that wrote
logs/A_power.json was executed inline and its script was not preserved; disclosed in reviews/ADJUDICATION.md).
Same method as recorded in A_power.json: Gaussian latent with correlation c shared by the two contexts of each
seed; 8 seeds x 2 parents; frozen R1 decision rule. Adds the Review-A heterogeneous scenarios (transfer in one
parent only). Writes logs/A_power_reproduced.json."""
import json, os, numpy as np
from scipy.stats import rankdata
def sp(a, b):
    ra = rankdata(a, axis=-1); rb = rankdata(b, axis=-1)
    ra -= ra.mean(-1, keepdims=True); rb -= rb.mean(-1, keepdims=True)
    return (ra*rb).sum(-1)/np.sqrt((ra*ra).sum(-1)*(rb*rb).sum(-1)+1e-300)
rng = np.random.default_rng(0)
def run(c1, c2, reps=300, NP=1000, NB=1000):
    out = []
    for _ in range(reps):
        pr = []
        for c in (c1, c2):
            z = rng.standard_normal(8)
            a = np.sqrt(c)*z+np.sqrt(1-c)*rng.standard_normal(8); b = np.sqrt(c)*z+np.sqrt(1-c)*rng.standard_normal(8)
            pr.append((a, b))
        obs = np.mean([sp(a, b) for a, b in pr])
        perm = np.mean([sp(np.broadcast_to(a, (NP, 8)), np.array([b[rng.permutation(8)] for _ in range(NP)])) for a, b in pr], 0)
        pv = (np.sum(perm >= obs-1e-12)+1)/(NP+1)
        bs = [sp(a[I], b[I]) for (a, b), I in ((ab, rng.integers(0, 8, (NB, 8))) for ab in pr)]
        ub = np.percentile(np.nanmean(bs, 0), 95)
        out.append('T' if (pv < 0.05 and obs >= 0.3) else ('N' if (pv >= 0.05 and ub < 0.6) else 'I'))
    out = np.array(out); return {k: float(np.mean(out == k)) for k in 'TNI'}
res = {f"c160={a},c161={b}": run(a, b) for a, b in ((0, 0), (0.3, 0.3), (0.6, 0.6), (0.4, 0), (0.6, 0), (0.8, 0))}
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "logs", "A_power_reproduced.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
