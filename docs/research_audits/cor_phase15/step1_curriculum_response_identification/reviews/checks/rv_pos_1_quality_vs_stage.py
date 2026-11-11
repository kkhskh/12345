"""Positive-theory review check 1: is the T1 null-response coherence a function of approximation quality (KL) alone,
or of training stage / curriculum at matched KL?  Uses archived N responses (centred logits, ok_N mask) only."""
import itertools, json, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "conf")
G = np.load(os.path.join(D, "generator.npz")); mk = G["ok_N"].astype(bool)
seeds_T = [61, 62, 63, 64, 65, 67]; seeds_P = [61, 62, 63, 64, 66, 67]
groups = {"T2000": [f"cT{s}" for s in seeds_T], "T500": [f"bcT{s}_b500" for s in seeds_T],
          "T250": [f"bcT{s}_b250" for s in seeds_T], "P": [f"cP{s}" for s in seeds_P],
          "failed": ["cT60", "cT66", "cP60", "cP65"]}
names = sum(groups.values(), []); grp = {m: g for g, ms in groups.items() for m in ms}
R = {}; KL = {}
for m in names:
    z = np.load(os.path.join(D, f"model_{m}.npz")); R[m] = (z["c_N"] - z["c_base"])[mk].ravel()
    KL[m] = json.load(open(os.path.join(D, f"model_{m}.json")))["quality"]["KL_bayes"]
M = np.array([R[m] for m in names]); C = np.corrcoef(M); ix = {m: i for i, m in enumerate(names)}
def coh(ms): return float(np.mean([C[ix[a], ix[b]] for a, b in itertools.combinations(ms, 2)]))
print("== within-group coherence and KL")
for g, ms in groups.items():
    print(f"{g:7s} coh={coh(ms):.3f}  KL median={np.median([KL[m] for m in ms]):.3f}  range=[{min(KL[m] for m in ms):.3f},{max(KL[m] for m in ms):.3f}]")
# (a) KL-matched subsets
lo = [m for m in groups["T500"] if KL[m] < 0.06]; mid = [m for m in groups["T500"] if 0.06 <= KL[m] <= 0.19]
print("\n== (a) T1 at 500 steps split by KL")
print("T500 KL<0.06:", lo, f"coh={coh(lo):.3f}")
print("T500 KL in P-count range:", mid, f"coh={coh(mid):.3f}" if len(mid) > 1 else "")
print("P-count (KL 0.068-0.188) coh=%.3f" % coh(groups["P"]))
best5 = [f"cT{s}" for s in (61, 62, 63, 64, 65)]
print("T2000 without cT67 (KL<=0.021):", f"coh={coh(best5):.3f}")
# (b) same-parent persistence across stage
print("\n== (b) same-parent vs other-parent correlation across stages")
for a, b in (("T500", "T2000"), ("T250", "T2000"), ("T250", "T500")):
    same = [C[ix[x], ix[y]] for x, y in zip(groups[a], groups[b])]
    oth = [C[ix[x], ix[y]] for i, x in enumerate(groups[a]) for j, y in enumerate(groups[b]) if i != j]
    print(f"{a}~{b}: same parent mean={np.mean(same):.3f} {np.round(same, 2).tolist()}  other parents mean={np.mean(oth):.3f}")
# (c) alignment with the near-Bayes T1 consensus, leave-one-out, vs KL, for every model
print("\n== (c) corr with leave-one-out consensus of the five near-Bayes T2000 models, vs log KL")
rows = []
for m in names:
    ref = [x for x in best5 if x != m]; cons = M[[ix[x] for x in ref]].mean(0)
    r = float(np.corrcoef(R[m], cons)[0, 1]); rows.append((m, grp[m], KL[m], r))
for m, g, k, r in sorted(rows, key=lambda t: t[2]): print(f"{m:12s} {g:7s} KL={k:.3f} r_cons={r:.3f}")
# regression r_cons ~ logKL + T1-lineage indicator (exploratory; models not all independent: budgets share parents)
X = np.array([[1, np.log(k), 1.0 if g != "P" and not m.startswith("cP") else 0.0] for m, g, k, r in rows if m not in best5])
y = np.array([r for m, g, k, r in rows if m not in best5])
b = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ b
print("\nr_cons ~ 1 + logKL + T1lineage (non-reference models):", np.round(b, 3).tolist(), "resid sd", round(float(res.std()), 3))
X0 = X[:, :2]; b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
print("r_cons ~ 1 + logKL:", np.round(b0, 3).tolist(), "resid sd", round(float((y - X0 @ b0).std()), 3))
# KL-matched comparison of r_cons: T1-lineage vs P-count models with KL in [0.05, 0.2]
tm = [r for m, g, k, r in rows if g in ("T500", "T250", "T2000") and m not in best5 and 0.05 <= k <= 0.2]
pm = [r for m, g, k, r in rows if g == "P" and 0.05 <= k <= 0.2]
print("KL in [0.05,0.2]: T1-lineage r_cons mean=%.3f (n=%d)  P-count mean=%.3f (n=%d)" % (np.mean(tm), len(tm), np.mean(pm), len(pm)))
# (d) parent-aware version: consensus excludes the model's own parent (budget variants share the parent's errors)
print("\n== (d) parent-aware leave-out consensus (exclude own parent seed)")
def seed(m): return int(''.join(ch for ch in m.split('_')[0] if ch.isdigit()))
rows2 = []
for m in names:
    ref = [x for x in best5 if seed(x) != seed(m) or x == m and False]; ref = [x for x in ref if x != m]
    if m.startswith("cP"): ref = best5
    cons = M[[ix[x] for x in ref]].mean(0); rows2.append((m, grp[m], KL[m], float(np.corrcoef(R[m], cons)[0, 1])))
for m, g, k, r in sorted(rows2, key=lambda t: t[2]): print(f"{m:12s} {g:7s} KL={k:.3f} r_cons={r:.3f}")
sel = [(m, g, k, r) for m, g, k, r in rows2 if m not in best5]
X = np.array([[1, np.log(k), 0.0 if m.startswith("cP") else 1.0] for m, g, k, r in sel]); y = np.array([r for *_, r in sel])
b = np.linalg.lstsq(X, y, rcond=None)[0]; print("r_cons ~ 1 + logKL + T1lineage:", np.round(b, 3).tolist())
# independent units only: P-count (6) vs T500 (6): lineage coefficient with permutation over these 12
Xi = [(m, k, r) for m, g, k, r in rows2 if g in ("P", "T500")]
lab = np.array([0 if m.startswith("cP") else 1 for m, _, _ in Xi]); lk = np.log([k for _, k, _ in Xi]); yy = np.array([r for *_, r in Xi])
def coefT(l): return np.linalg.lstsq(np.c_[np.ones(len(l)), lk, l], yy, rcond=None)[0][2]
obs = coefT(lab); null = []
for c in itertools.combinations(range(12), 6):
    l = np.zeros(12); l[list(c)] = 1; null.append(coefT(l))
null = np.array(null)
print("P vs T500 (12 independent parents): logKL slope=%.3f, T1 coef=%.3f, two-sided perm p=%.3f" %
      (np.linalg.lstsq(np.c_[np.ones(12), lk, lab], yy, rcond=None)[0][1], obs, float((np.abs(null) >= abs(obs) - 1e-12).mean())))
print("corr(logKL, r_cons) over P+T500 =", round(float(np.corrcoef(lk, yy)[0, 1]), 3),
      "; over all non-reference models =", round(float(np.corrcoef(X[:, 1], y)[0, 1]), 3))
print("\n== (e) T500 coherence without bcT67_b500 (KL 0.226, failed-level quality):",
      round(coh([m for m in groups["T500"] if m != "bcT67_b500"]), 3),
      "; median KL of the five:", round(float(np.median([KL[m] for m in groups["T500"] if m != "bcT67_b500"])), 3))
