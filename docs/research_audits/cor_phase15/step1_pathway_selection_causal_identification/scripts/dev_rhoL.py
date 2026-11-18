"""XVIII development: lapse-aware rho on the 24 XIV step-1211 checkpoints (threshold calibration) and on exact Bayes + lapse references."""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); sys.path.insert(0, os.path.join(H, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, xviiim as XM, xc, metrics as MT, relpairs as RP, genlib as GL
S = MT.sets(86_100_000); out = {}
PB = S["PBn"]
for eps in (0.0, 0.05, 0.10, 0.15):
    P = (1 - eps) * PB + eps / 3; out[f"bayes_lapse{eps}"] = XM.rho_fit_lapse(np.log(P), S); print("bayes+lapse", eps, out[f"bayes_lapse{eps}"], flush=True)
for l in open(os.path.join(H, "..", "logs", "dev", "behav_xiv.jsonl")):
    u = json.loads(l)["unit"]; M = xc.load(u, 1211); b, Ln = MT.behaviour(M, S); r = XM.rho_fit_lapse(Ln, S); r["rho"] = json.loads(l)["rho"]; out[u] = r
    print(u, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(H, "..", "logs", "dev", "rhoL_xiv.json"), "w"), indent=1)
