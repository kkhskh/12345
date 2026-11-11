"""Counterexample reviewer, check 2: analytic models evaluated on the package's exact conf N (Bayes-null) and R pairs.
(A) Shared algorithm, different irrelevant-count responses: two exact F_mix members, phi_rel(n) = n log rho +- kappa (n - 2.5)^2,
    phi_col and tau exact.  Same factorization, matched quality; report null-response r, R-response r, KL.
(B) Coherence without any shared error function: each model = C tanh(tau (c* + sigma(x) eps_M(x)) / C), eps_M iid N(0,1)
    drawn independently per model and per input, sigma(x) decreasing in the total leave-q agreement count (heteroscedastic
    'less evidence, noisier code'); C = a shared output ceiling.  Within-group coherence (mean pairwise Pearson, the package's
    statistic, centred logits and probabilities), KL, and the sign of the 'more agreement -> more confidence' regression
    coefficient (dir) of null_structure.py.  Controls: no ceiling; homoscedastic noise.
Usage: python rv_cex_2_null.py"""
import itertools, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__)); PK = os.path.join(HERE, "..", ".."); sys.path.insert(0, os.path.join(PK, "src"))
import numpy as np
import cri as R
E = R.E
Gd = np.load(os.path.join(PK, "logs", "conf", "generator.npz")); q = Gd["qsel"]; NE = len(q); idx = np.arange(NE)
okN, okR = Gd["ok_N"].astype(bool), Gd["ok_R"].astype(bool)


def JV(X):
    J = np.zeros((len(X), 6, 3)); v = E.loo(X)[idx, q]
    for k in range(6):
        s = q == k; J[s] = R.J_at(X[s], k)
    return J, v


D = {f: JV(Gd[x]) for f, x in (("base", "X"), ("N", "X_N"), ("R", "X_R"))}
cB = {"base": Gd["cB_base"], "N": Gd["cB_N"], "R": Gd["cB_R"]}
for f in D: assert np.abs(R.mix_exact(*D[f]) - cB[f]).max() < 1e-6
soft = lambda c: np.exp(c) / np.exp(c).sum(-1, keepdims=True)
Pb = soft(cB["base"])
KL = lambda c: float((Pb * (np.log(Pb + 1e-12) - np.log(soft(c) + 1e-12))).sum(-1).mean())


def coh(resps, mask, space="clogit"):
    rs = [r[mask].ravel() for r in resps]
    return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in itertools.combinations(rs, 2)]))


out = {}
# ---------------------------------------------------------------- (A)
lr = np.log(E.RHO); n = np.arange(6)
def mix_member(kappa):
    phi = n * lr + kappa * (n - 2.5) ** 2; th = np.r_[phi[1:] - phi[0], np.full(5, -50.0), 0.0]
    return {f: R.mix_logits(th, *D[f]) for f in D}
A = {}
for kap in (0.1, 0.25):
    a, b = mix_member(+kap), mix_member(-kap)
    rN = [a["N"] - a["base"], b["N"] - b["base"]]; rR = [a["R"] - a["base"], b["R"] - b["base"]]
    A[kap] = dict(null_r=coh(rN, okN), R_r=coh(rR, okR), KL_convex=KL(a["base"]), KL_concave=KL(b["base"]),
                  mean_abs_null=[float(np.abs(r[okN]).mean()) for r in rN],
                  R_r_with_bayes=[float(np.corrcoef(r[okR].ravel(), (cB["R"] - cB["base"])[okR].ravel())[0, 1]) for r in rR])
    print(f"(A) kappa {kap}: F_mix members (convex/concave phi_rel): null-response r {A[kap]['null_r']:+.3f}; R-response r {A[kap]['R_r']:.3f}; "
          f"KL {A[kap]['KL_convex']:.3f}/{A[kap]['KL_concave']:.3f}; |null| {np.round(A[kap]['mean_abs_null'], 3)}; R r with Bayes {np.round(A[kap]['R_r_with_bayes'], 3)}")
out["A_shared_algorithm"] = A

# ---------------------------------------------------------------- (B)
tot = {f: D[f][1].sum(-1) for f in D}
n0 = D["base"][1]; n1 = D["N"][1]
H = np.array([np.where(v == 5)[0][0] if (v == 5).sum() == 1 else -1 for v in n0])
oth = np.array([[g for g in range(3) if g != h] if h >= 0 else [0, 1] for h in H])
dirn = np.sign((n1 - n0)[idx, oth[:, 0]]); u = cB["base"] / (np.linalg.norm(cB["base"], axis=-1, keepdims=True) + 1e-12)
def dir_coef(r):
    s = (r * u).sum(-1)[okN]; Xd = np.c_[np.ones(okN.sum()), dirn[okN]]; return float(np.linalg.lstsq(Xd, s, rcond=None)[0][1])


def arm(seed, nmod, sigma0, gamma, C, tau, hetero=True):
    rng = np.random.default_rng(seed); rN, rNp, rR, kl, dc = [], [], [], [], []
    for m in range(nmod):
        o = {}
        for f in D:
            sg = sigma0 * np.exp(gamma * (10 - tot[f]) / 5.0) if hetero else np.full(NE, sigma0 * np.exp(gamma))
            pre = tau * (cB[f] + sg[:, None] * rng.standard_normal((NE, 3)))
            c = C * np.tanh(pre / C) if np.isfinite(C) else pre; o[f] = c - c.mean(-1, keepdims=True)
        rN.append(o["N"] - o["base"]); rNp.append(soft(o["N"]) - soft(o["base"])); rR.append(o["R"] - o["base"])
        kl.append(KL(o["base"])); dc.append(dir_coef(o["N"] - o["base"]))
    return dict(coh_N=coh(rN, okN), coh_N_prob=coh(rNp, okN), coh_Rerr=None, KL=float(np.median(kl)), dir_coef=float(np.mean(dc)),
                mean_abs_N=float(np.mean([np.abs(r[okN]).mean() for r in rN])), _rN=rN)


B = {}; print("\n(B) independent noise x shared output ceiling (6 models per setting; coherence = mean pairwise Pearson on N pairs)")
settings = [("hetero, C=1.5, tau=1.0", 0.4, 0.6, 1.5, 1.0, True), ("hetero, C=1.5, tau=0.75", 0.4, 0.6, 1.5, 0.75, True),
            ("hetero, C=2.5, tau=1.0", 0.4, 0.6, 2.5, 1.0, True), ("hetero, C=inf, tau=1.0", 0.4, 0.6, np.inf, 1.0, True),
            ("homosc, C=1.5, tau=1.0", 0.4, 0.6, 1.5, 1.0, False), ("hetero, C=1.5, tau=1.0, small noise", 0.15, 0.6, 1.5, 1.0, True),
            ("hetero, C=1.5, tau=1.0, large noise", 0.8, 0.6, 1.5, 1.0, True)]
for i, (nm, s0, g, C, tau, het) in enumerate(settings):
    r = arm(100 + i, 6, s0, g, C, tau, het); B[nm] = {k: v for k, v in r.items() if not k.startswith("_")}
    print(f"  {nm:38s} coh_N {r['coh_N']:+.3f} (prob {r['coh_N_prob']:+.3f}) KL {r['KL']:.3f} |N| {r['mean_abs_N']:.3f} dir-coef {r['dir_coef']:+.3f}")
# cross-group coherence between a tau=1 and a tau=0.75 group with the same ceiling and noise law
g1 = arm(200, 6, 0.4, 0.6, 1.5, 1.0)["_rN"]; g2 = arm(201, 6, 0.4, 0.6, 1.5, 0.75)["_rN"]
cross = float(np.mean([np.corrcoef(a[okN].ravel(), b[okN].ravel())[0, 1] for a in g1 for b in g2]))
B["cross_tau1_vs_tau075"] = cross; print(f"  cross coherence tau=1 group vs tau=0.75 group: {cross:+.3f}")
out["B_noise_ceiling"] = B
json.dump(out, open(os.path.join(PK, "logs", "review_checks", "rv_cex_2_null.json"), "w"), indent=1, default=float)
