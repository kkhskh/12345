"""Positive-theory review check 3: strength of the N-vs-R route separation.
sel = (R share remaining with Rq blocked) - (N share remaining with Rq blocked); remaining = <d_block, d_full>/<d_full, d_full>.
(i) per-model episode-bootstrap CI (within-model inference is legitimate here: the claim is about each model's pathway);
(ii) exact arm permutation over the 12 successful models (two-sided); (iii) the same statistic at 250/500 transfer steps
(training stage) and for failed runs; (iv) does route separation track N-response alignment / KL?"""
import itertools, json, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "conf")
G = np.load(os.path.join(D, "generator.npz")); mN = G["ok_N"].astype(bool); mR = G["ok_R"].astype(bool)
rng = np.random.default_rng(3)
def frac(z, F, mk, tag, idx=None):
    full = (z[f"c_{F}"] - z["c_base"])[mk]; d = (z[f"c_{F}_{tag}"] - z["c_base"])[mk]
    if idx is not None: full, d = full[idx], d[idx]
    return float((d * full).sum() / (full * full).sum())
def sel_stats(m, B=500):
    z = np.load(os.path.join(D, f"model_{m}.npz"))
    nN, nR = mN.sum(), mR.sum()
    s = frac(z, "R", mR, "block_Rq") - frac(z, "N", mN, "block_Rq")
    bs = [frac(z, "R", mR, "block_Rq", rng.integers(0, nR, nR)) - frac(z, "N", mN, "block_Rq", rng.integers(0, nN, nN)) for _ in range(B)]
    KL = json.load(open(os.path.join(D, f"model_{m}.json")))["quality"]["KL_bayes"]
    return s, np.percentile(bs, [2.5, 97.5]), frac(z, "N", mN, "block_Rq"), frac(z, "R", mR, "block_Rq"), frac(z, "R", mR, "patch_Rq"), KL
T = [61, 62, 63, 64, 65, 67]; P = [61, 62, 63, 64, 66, 67]
sets = {"T2000": [f"cT{s}" for s in T], "P": [f"cP{s}" for s in P], "T500": [f"bcT{s}_b500" for s in T],
        "T250": [f"bcT{s}_b250" for s in T], "failed": ["cT60", "cT66", "cP60", "cP65"]}
out = {}
for g, ms in sets.items():
    print(f"== {g}")
    for m in ms:
        s, ci, nb, rb, rp, kl = sel_stats(m); out[m] = s
        print(f"  {m:12s} KL={kl:.3f} N_rem|blockRq={nb:.2f} R_rem|blockRq={rb:.2f} R_via_Rq={rp:.2f} sel={s:+.2f} CI[{ci[0]:+.2f},{ci[1]:+.2f}]")
v = np.array([out[m] for m in sets["T2000"] + sets["P"]]); lab = np.r_[np.ones(6), np.zeros(6)]
obs = v[lab == 1].mean() - v[lab == 0].mean(); null = []
for c in itertools.combinations(range(12), 6):
    l = np.zeros(12); l[list(c)] = 1; null.append(v[l == 1].mean() - v[l == 0].mean())
null = np.array(null)
print(f"\narm difference in sel (T2000 - P): {obs:+.3f}, exact two-sided perm p = {(np.abs(null) >= abs(obs) - 1e-12).mean():.4f}")
print("P5(a) excess, T vs P (frozen values from FROZEN_PREDICTIONS OUTCOMES), exact one-sided perm:")
ex = np.array([0.17, 1.01, 0.05, 0.72, 0.53, 0.52, 0.12, 0.20, 0.04, -0.03, 0.04, 0.08]); o2 = ex[:6].mean() - ex[6:].mean()
n2 = np.array([ex[np.array(c)].mean() - np.delete(ex, c).mean() for c in itertools.combinations(range(12), 6)])
print(f"  diff {o2:+.3f}, p = {(n2 >= o2 - 1e-12).mean():.4f}")
print("\nmean sel by set:", {g: round(float(np.mean([out[m] for m in ms])), 3) for g, ms in sets.items()})
print("same-parent sel across stages (T250, T500, T2000):")
for s in T: print(f"  seed {s}: {out[f'bcT{s}_b250']:+.2f} {out[f'bcT{s}_b500']:+.2f} {out[f'cT{s}']:+.2f}")
