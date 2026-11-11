"""EXPLORATORY (post hoc, after the confirmatory outcomes): route selectivity of Bayesian-null vs Bayesian-relevant responses.
For each model, from the frozen path-patching arrays: remaining fraction of the N (and R) response when the readout's own
layer-2 input (Rq) is held at its base value while every other position takes its counterfactual value
(rem = <Δ_block, Δ_full>/||Δ_full||²), and the coherence (mean pairwise Pearson) of N responses carried by each route
(Rq-only patch vs Rq-blocked) within each arm.  Usage: route_selectivity.py PHASE OUTNAME MODEL [...]"""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, cri as R
PH, OUT, names = sys.argv[1], sys.argv[2], sys.argv[3:]; D = os.path.join(HERE, "..", "logs", PH)
G = np.load(os.path.join(D, "generator.npz")); res = {}; parts = {}
for m in names:
    z = np.load(os.path.join(D, f"model_{m}.npz")); r = {}
    for F in ("N", "R"):
        mk = G[f"ok_{F}"].astype(bool); full = (z[f"c_{F}"] - z["c_base"])[mk]
        for nm in ("block_Rq", "patch_Rq", "block_Qs"):
            d = (z[f"c_{F}_{nm}"] - z["c_base"])[mk]; r[f"{F}|{nm}"] = float((d * full).sum() / (full * full).sum())
        r[f"{F}|mean_abs"] = float(np.abs(full).mean())
    mk = G["ok_N"].astype(bool); parts[m] = {k: (z[f"c_N_{k}"] - z["c_base"])[mk].ravel() for k in ("patch_Rq", "block_Rq")}
    res[m] = r
coh = {}
for part in ("patch_Rq", "block_Rq"):
    for a in ("T", "P"):
        ms = [m for m in names if R.kind_of(m) == a]
        coh[f"{part}|{a}"] = float(np.mean([np.corrcoef(parts[x][part], parts[y][part])[0, 1] for x, y in itertools.combinations(ms, 2)]))
json.dump(dict(per_model=res, coherence_by_route=coh), open(os.path.join(D, f"{OUT}.json"), "w"), indent=1)
for m, r in res.items(): print(m, {k: round(v, 2) for k, v in r.items()})
print(coh)
