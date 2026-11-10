"""Group-level summaries with bootstrap CIs (episodes resampled; 2000 replicates).  Usage: analyze.py PHASE
Decision-response equivalence: for each response family, mean Pearson r of model-pair responses within the P-count
curriculum, within the T1 curriculum, and across curricula; also each model's r with the exact Bayes response
(random_support family).  Writes logs/PHASE/analysis.json."""
import itertools, json, os, sys
import numpy as np
PH = sys.argv[1]; D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", PH)
Z = np.load(os.path.join(D, "responses.npz")); R = json.load(open(os.path.join(D, "results.json")))
names = R["models"]; kind = {m: ("P" if m[0] in "PS" else "T") for m in names}
fams = sorted({k.split("|")[0] for k in Z.files if "|" in k and not k.startswith("mask")})
rng = np.random.default_rng(0); out = {}
def r(a, b): a, b = a.ravel(), b.ravel(); return np.corrcoef(a, b)[0, 1]
for fn in fams:
    mk = Z[f"mask|{fn}"]; idx = np.where(mk)[0]; resp = {m: Z[f"{fn}|{m}"] for m in names}
    if fn == "random_support":
        dB = Z["dB"]
        for m in names:
            beta = (resp[m][mk] * dB[mk]).sum() / (dB[mk] ** 2).sum(); resp[m] = resp[m] - beta * dB   # residual from Bayes
    groups = {"within_P": [], "within_T": [], "cross": []}
    for a, b in itertools.combinations(names, 2):
        g = "cross" if kind[a] != kind[b] else f"within_{kind[a]}"; groups[g].append((a, b))
    def stat(ii):
        return {g: float(np.mean([r(resp[a][ii], resp[b][ii]) for a, b in prs])) if prs else float("nan") for g, prs in groups.items()}
    pt = stat(idx); boots = [stat(rng.choice(idx, len(idx))) for _ in range(300)]
    o = {g: dict(mean_r=pt[g], ci95=[float(np.nanpercentile([b[g] for b in boots], 2.5)), float(np.nanpercentile([b[g] for b in boots], 97.5))], n_pairs=len(groups[g])) for g in groups}
    dif = [b["cross"] - b["within_T"] for b in boots]; o["cross_minus_within_T"] = dict(point=pt["cross"] - pt["within_T"], ci95=[float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))])
    dif = [b["cross"] - b["within_P"] for b in boots]; o["cross_minus_within_P"] = dict(point=pt["cross"] - pt["within_P"], ci95=[float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))])
    o["n_episodes"] = int(mk.sum()); o["pairs"] = {f"{a}~{b}": float(r(resp[a][idx], resp[b][idx])) for a, b in itertools.combinations(names, 2)}
    out[fn] = o
out["bayes_tracking_r"] = {m: R["decision_response"]["random_support"].get(f"{m}|r_with_bayes") for m in names}
json.dump(out, open(os.path.join(D, "analysis.json"), "w"), indent=1)
for fn in fams: print(fn, {g: (round(v["mean_r"], 3), [round(x, 3) for x in v["ci95"]]) for g, v in out[fn].items() if isinstance(v, dict) and "mean_r" in v}, "x-T", round(out[fn]["cross_minus_within_T"]["point"], 3))
print(out["bayes_tracking_r"])
