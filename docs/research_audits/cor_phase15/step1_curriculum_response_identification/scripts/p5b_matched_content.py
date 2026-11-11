"""Prediction P5(b) exactly as frozen in FROZEN_PREDICTIONS.md (the estimand was frozen; this code was written after the freeze
because cri_stats.py did not implement it): f_Cm = <Δ_Cm, Δ_N> / ||Δ_N||² per model on the N-valid set, and the two-sided
exact permutation test of the arm difference in mean f_Cm.  Usage: p5b_matched_content.py PHASE OUTNAME MODEL [...]"""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, cri as R
PH, OUT, names = sys.argv[1], sys.argv[2], sys.argv[3:]; D = os.path.join(HERE, "..", "logs", PH)
G = np.load(os.path.join(D, "generator.npz")); mk = G["ok_N"].astype(bool); f = {}
for m in names:
    z = np.load(os.path.join(D, f"model_{m}.npz")); dN = (z["c_N"] - z["c_base"])[mk]; dC = (z["c_Cm"] - z["c_base"])[mk]
    f[m] = float((dC * dN).sum() / (dN * dN).sum())
v = np.array([f[m] for m in names]); arm = np.array([R.kind_of(m) == "T" for m in names]); obs = v[arm].mean() - v[~arm].mean()
null = []
for c in itertools.combinations(range(len(v)), arm.sum()):
    l = np.zeros(len(v), bool); l[list(c)] = True; null.append(v[l].mean() - v[~l].mean())
p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
json.dump(dict(f_Cm=f, diff_T_minus_P=float(obs), perm_p_two_sided=p), open(os.path.join(D, f"{OUT}.json"), "w"), indent=1)
print({k: round(x, 3) for k, x in f.items()}, "diff", round(obs, 3), "p", round(p, 4))
