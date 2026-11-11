"""Counterexample reviewer, check 4: what the within-arm coherence contrast, the KL adjustment and P1b can distinguish.
(i)  Real cohort pair matrix (logs/conf/primary.json): fit a ONE common factor with model-specific (quality-dependent) loadings,
     S_ij ~ l_i l_j, by least squares over all 66 pairs; compare within/cross structure; statistic
     D_cross = cross - sqrt(coh_T coh_P) (0 under one factor with homogeneous within-arm loadings) with a model bootstrap.
(ii) Parametric simulation under that one-factor null (no curriculum-specific component; loadings = fitted, i.e. coherence is
     purely 'quality'): run the package's P1b procedure (leave-pair-out factor, EXCESS_TT, exact permutation over 924 labelings)
     -> false-positive rate, and the null distribution of the package's DELTA and of D_cross.
(iii) World B: two arm-SPECIFIC orthogonal shared components of equal size at matched noise (curriculum-specific structure, equal
     quality): DELTA, the KL-adjusted contrast (KL a function of total error size, identical across arms) and cross coherence.
Usage: python rv_cex_4_factor.py"""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); PK = os.path.join(HERE, "..", "..")
P = json.load(open(os.path.join(PK, "logs", "conf", "primary.json"))); names = P["models"]; nm = len(names)
arm = np.array([1 if m[1] == "T" else 0 for m in names]); S = np.eye(nm)
for k, v in P["P1_primary_N_input_clogit_pearson"]["pairs"].items():
    a, b = k.split("~"); i, j = names.index(a), names.index(b); S[i, j] = S[j, i] = v
prs = list(itertools.combinations(range(nm), 2)); rng = np.random.default_rng(7)


def coh(S, lab):
    T = [S[i, j] for i, j in prs if lab[i] == lab[j] == 1]; Pp = [S[i, j] for i, j in prs if lab[i] == lab[j] == 0]
    C = [S[i, j] for i, j in prs if lab[i] != lab[j]]; return np.mean(T), np.mean(Pp), np.mean(C)


def dcross(S, lab):
    t, p, c = coh(S, lab); return c - np.sqrt(max(t, 1e-9) * max(p, 1e-9))


t, p, c = coh(S, arm); print(f"(i) observed coh_T {t:.3f} coh_P {p:.3f} cross {c:.3f}; sqrt(T*P) {np.sqrt(t*p):.3f}; D_cross {dcross(S, arm):+.3f}")
l0 = minimize(lambda l: sum((S[i, j] - l[i] * l[j]) ** 2 for i, j in prs), np.full(nm, 0.4), method="L-BFGS-B", bounds=[(0, 1)] * nm).x
res1 = np.array([S[i, j] - l0[i] * l0[j] for i, j in prs])
grp = np.array(["TT" if arm[i] + arm[j] == 2 else ("PP" if arm[i] + arm[j] == 0 else "TP") for i, j in prs])
print("    one-factor loadings:", dict(zip(names, np.round(l0, 2))), "| mean residual by pair type:",
      {g: round(float(res1[grp == g].mean()), 3) for g in ("TT", "PP", "TP")})
bt = []
for _ in range(4000):
    T_ = rng.choice(np.where(arm == 1)[0], 6); P_ = rng.choice(np.where(arm == 0)[0], 6)
    tl = [S[i, j] for i, j in itertools.combinations(T_, 2) if i != j]; pl = [S[i, j] for i, j in itertools.combinations(P_, 2) if i != j]
    if not tl or not pl: continue
    tt, pp = np.mean(tl), np.mean(pl); cc = np.mean([S[i, j] for i in T_ for j in P_]); bt.append(cc - np.sqrt(max(tt, 1e-9) * max(pp, 1e-9)))
print(f"    D_cross model-bootstrap 95% CI [{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}]")

# ------------------------------------------------------------------ (ii)
NV = 1642 * 2                                                       # N-valid entries x 2 free centred-logit dims
labs = []
for cc in itertools.combinations(range(nm), 6):
    l = np.zeros(nm, int); l[list(cc)] = 1; labs.append(l)
labs = np.array(labs)


def p1b(Z):
    Zs = [(x - x.mean()) / x.std() for x in Z]; res = {}
    tot = np.sum(Zs, 0)
    for i, j in prs:
        f = (tot - Zs[i] - Zs[j]) / (nm - 2); li, lj = np.corrcoef(Zs[i], f)[0, 1], np.corrcoef(Zs[j], f)[0, 1]
        res[(i, j)] = np.corrcoef(Zs[i], Zs[j])[0, 1] - li * lj
    r = np.array([res[pq] for pq in prs]); Sm = np.eye(nm)
    for (i, j) in prs: Sm[i, j] = Sm[j, i] = np.corrcoef(Zs[i], Zs[j])[0, 1]
    TTm = np.array([[l[i] == l[j] == 1 for i, j in prs] for l in labs], float)
    ex = (TTm @ r) / TTm.sum(1); obs = ex[np.where((labs == arm).all(1))[0][0]]
    return obs, float((ex >= obs - 1e-12).mean()), Sm


fp, deltas, dcs, exs = 0, [], [], []
NSIM = 200
for s in range(NSIM):
    f = rng.standard_normal(NV); Z = [l0[m] * f + np.sqrt(1 - l0[m] ** 2) * rng.standard_normal(NV) for m in range(nm)]
    ex, pv, Sm = p1b(Z); fp += pv < 0.05; exs.append(ex); tt, pp, cc = coh(Sm, arm); deltas.append(tt - pp); dcs.append(dcross(Sm, arm))
print(f"(ii) one-factor null (quality-dependent loadings only), {NSIM} sims: P1b EXCESS_TT mean {np.mean(exs):+.3f} "
      f"(observed +0.135; parametric P(EXCESS_TT >= 0.135) = {np.mean(np.array(exs) >= 0.135):.3f}), false-positive rate at p<0.05: {fp / NSIM:.2f}; DELTA mean {np.mean(deltas):+.3f}; "
      f"D_cross mean {np.mean(dcs):+.3f}, 2.5-97.5% [{np.percentile(dcs, 2.5):+.3f}, {np.percentile(dcs, 97.5):+.3f}] (observed {dcross(S, arm):+.3f})")

# ------------------------------------------------------------------ (iii)
a = 0.45                                                            # shared-component loading in each arm
fT, fP = rng.standard_normal(NV), rng.standard_normal(NV)
Z = [a * (fT if arm[m] else fP) + np.sqrt(1 - a * a) * rng.standard_normal(NV) for m in range(nm)]
Sm = np.corrcoef(np.array(Z)); tt, pp, cc = coh(Sm, arm)
KL = np.ones(nm)                                                    # equal total error size in both arms -> equal KL
pr_ = prs; y = np.array([Sm[i, j] for i, j in pr_]); TT = np.array([arm[i] == arm[j] == 1 for i, j in pr_], float); PP = np.array([arm[i] == arm[j] == 0 for i, j in pr_], float)
Xd = np.c_[np.ones(len(pr_)), TT, PP, [KL[i] + KL[j] for i, j in pr_], [abs(KL[i] - KL[j]) for i, j in pr_]]
b = np.linalg.lstsq(Xd, y, rcond=None)[0]
print(f"(iii) arm-specific orthogonal shared components, equal size and quality: coh_T {tt:.3f} coh_P {pp:.3f} DELTA {tt-pp:+.3f}; "
      f"KL-adjusted DELTA {b[1]-b[2]:+.3f}; cross {cc:+.3f}")
json.dump(dict(observed=dict(coh_T=t, coh_P=p, cross=c, D_cross=dcross(S, arm), D_cross_ci=[np.percentile(bt, 2.5), np.percentile(bt, 97.5)],
                             loadings=dict(zip(names, l0.tolist()))),
               one_factor_null=dict(EXCESS_TT_mean=np.mean(exs), fp_rate=fp / NSIM, DELTA_mean=np.mean(deltas), D_cross_q=[np.percentile(dcs, 2.5), np.percentile(dcs, 97.5)]),
               worldB=dict(coh_T=tt, coh_P=pp, cross=cc, KL_adj=b[1] - b[2])),
          open(os.path.join(PK, "logs", "review_checks", "rv_cex_4_factor.json"), "w"), indent=1, default=float)
