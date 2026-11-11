"""Positive-theory review check 2: how informative is the P1 failure?  (i) leave-one-model-out sensitivity of DELTA,
(ii) power of the frozen joint rule (perm p<0.05 AND bootstrap lower>0) at 6+6 for true effects 0.15 and 0.45, under a
Gaussian factor model whose pair-correlation structure is matched to the observed 6+6 matrix (common factor + arm factor,
model-specific loadings resampled from the observed one-factor loadings)."""
import itertools, json, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "conf")
P1 = json.load(open(os.path.join(D, "primary.json")))
names = P1["models"]; arm = np.array([1 if n.startswith("cT") else 0 for n in names]); nm = 12
S = np.eye(nm)
for k, v in P1["P1_primary_N_input_clogit_pearson"]["pairs"].items():
    a, b = k.split("~"); i, j = names.index(a), names.index(b); S[i, j] = S[j, i] = v
def coh(S, lab):
    t = [S[i, j] for i, j in itertools.combinations(np.where(lab == 1)[0], 2)]
    p = [S[i, j] for i, j in itertools.combinations(np.where(lab == 0)[0], 2)]
    return np.mean(t) - np.mean(p)
print("observed DELTA", round(coh(S, arm), 3))
print("== leave-one-model-out DELTA")
for d in range(nm):
    keep = [k for k in range(nm) if k != d]; print(f"  drop {names[d]}: {coh(S[np.ix_(keep, keep)], arm[keep]):.3f}")
LABS = []
for c in itertools.combinations(range(nm), 6):
    l = np.zeros(nm, int); l[list(c)] = 1; LABS.append(l)
LABS = np.array(LABS)
pairs = np.array(list(itertools.combinations(range(nm), 2)))
def all_stats(S):
    v = S[pairs[:, 0], pairs[:, 1]]
    TT = LABS[:, pairs[:, 0]] * LABS[:, pairs[:, 1]]; PP = (1 - LABS[:, pairs[:, 0]]) * (1 - LABS[:, pairs[:, 1]])
    return (TT @ v) / TT.sum(1) - (PP @ v) / PP.sum(1)
obs_lab_idx = int(np.where((LABS == arm).all(1))[0][0])
rng = np.random.default_rng(7)
def boot_lo(S, reps=1000):
    T_, P_ = np.where(arm == 1)[0], np.where(arm == 0)[0]; vals = []
    for _ in range(reps):
        dt, dp = rng.choice(T_, 6), rng.choice(P_, 6)
        ct = [S[i, j] for i, j in itertools.combinations(dt, 2) if i != j]; cp = [S[i, j] for i, j in itertools.combinations(dp, 2) if i != j]
        if ct and cp: vals.append(np.mean(ct) - np.mean(cp))
    return np.percentile(vals, 2.5)
st = all_stats(S); print("check: exact perm p =", round(float((st >= st[obs_lab_idx] - 1e-12).mean()), 4))
lam = np.array(list(P1["P1b_one_factor_N"]["loadings"].values()))
def sim(delta_target, nsim=300, dim=1500):
    # within-T extra factor variance b2 chosen so mean T-T excess ≈ delta_target (+ observed P-P excess over cross)
    passes = 0; obs_deltas = []
    for _ in range(nsim):
        l = rng.choice(lam, nm) * 0.95
        b = np.sqrt(max(delta_target + 0.012, 0)) * rng.uniform(0.5, 1.5, nm) * (arm == 1) + np.sqrt(0.012) * (arm == 0)
        f0, fT, fP = rng.standard_normal((3, dim))
        X = l[:, None] * f0 + b[:, None] * np.where(arm[:, None] == 1, fT, fP)
        X += np.sqrt(np.clip(1 - l ** 2 - b ** 2, 0.05, None))[:, None] * rng.standard_normal((nm, dim))
        Ss = np.corrcoef(X); st = all_stats(Ss); p = (st >= st[obs_lab_idx] - 1e-12).mean(); obs_deltas.append(st[obs_lab_idx])
        if p < 0.05 and boot_lo(Ss) > 0: passes += 1
    return passes / nsim, float(np.mean(obs_deltas))
for dt in (0.0, 0.15, 0.30, 0.45):
    pw, md = sim(dt); print(f"true T-specific variance≈{dt:.2f}: mean simulated DELTA={md:.3f}, P(pass frozen rule)={pw:.2f}")
