"""Workstream C: mathematical checks for Investigation XII (deterministic; writes logs/math_xii.json).
M1  General-C eligibility theorem (independent copying, beta=1):  p(y) ∝ sum_g W_g L_g(y) e_C(e_{-g}),  e_C = elementary symmetric
    polynomial of degree C in the OTHER groups' eligibility indicators.  Checked against brute-force enumeration for (G,C) in
    {(3,1),(3,2),(4,1),(4,2),(5,2),(5,3)}.
M2  Shared copying (the actual genlib sampler, one copied value per question for all colluders): exact posterior replaces prod_{h in H} e_h by
    the JOINT indicator 1[all sources of H agree with each other at every question].  Gap between e_C rule and the exact posterior.
M3  e is a deterministic function of (W, L) on the support:  W_g determines n_g, L_g determines agreement at q.  Consequence: every
    informative E-interchange (e_src != e_base with W, L from the base) produces a high-level state that NO input realizes; the same holds
    for VS.  Both are valid do-interventions in an SCM in which e (or S) is a node; neither is an input-realizable counterfactual.
M4  False-positive bound: for any subspace R and rules V, V',  |IIA_V(R) - IIA_V'(R)| <= P_pairs[argmax cf_V != argmax cf_V'].  Computed for
    V' in {A1, PERM, RANDE} vs V = E on the decoy-DAS pair distribution, and compared with the observed development decoy IIA values.
M5  Preservation power: among pairs on which decoy V' is UNCHANGED (its preservation set), the fraction on which the true variable E
    changes (only these pairs can expose the decoy) and the fraction on which the Bayes decision changes."""
import json, os, sys, itertools, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); out = {}
kl = lambda P, Q: (P * (np.log(P + 1e-300) - np.log(Q + 1e-300))).sum(-1)
def elig(env, X, q):
    n = len(X); ii = np.arange(n); A_ = X.reshape(n, X.shape[1], env.G, 2)[ii, q]
    return ((GL.counts(X, env.G)[ii, q] == X.shape[1] - 1) & (A_[..., 0] == A_[..., 1])).astype(float)
def esym_rule(env, X, q, joint=False):
    n = len(X); ii = np.arange(n); rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi; e = elig(env, X, q); a_ = X.reshape(n, X.shape[1], env.G, 2); m = np.zeros((n, env.G))
    for g in range(env.G):
        for H in itertools.combinations([h for h in range(env.G) if h != g], env.C):
            if joint:
                s = a_[:, :, list(H), :].reshape(n, X.shape[1], -1); m[:, g] += (s == s[..., :1]).all(-1).all(-1)
            else: m[:, g] += np.prod(e[:, list(H)], 1)
    P = np.maximum((W[..., None] * m[..., None] * Lg).sum(1), 1e-300); return P / P.sum(-1, keepdims=True)
# M1 / M2
m1 = {}; m2 = {}
for G, C in ((3, 1), (3, 2), (4, 1), (4, 2), (5, 2), (5, 3)):
    env = GL.Env(G, C); r = np.random.default_rng(80_000_000 + 10 * G + C); d = GL.sample(env, 3000, r); q = r.integers(0, 6, 3000)
    Pb = GL.posterior_brute(env, d["X"], q); Pr = esym_rule(env, d["X"], q); m1[f"G{G}C{C}"] = float(np.abs(Pb - Pr).max())
    Ps = XB.posterior_shared(env, d["X"], q); Pj = esym_rule(env, d["X"], q, joint=True); Y = d["Y"][np.arange(3000), q]
    m2[f"G{G}C{C}"] = dict(max_abs_shared_vs_jointrule=float(np.abs(Ps - Pj).max()), KL_shared_vs_esym=float(kl(Ps, Pr).mean()),
                           acc_shared=float((Ps.argmax(-1) == Y).mean()), acc_esym=float((Pr.argmax(-1) == Y).mean()))
out["M1_independent_copy_esym_theorem_max_abs_err"] = m1; out["M2_shared_copy"] = m2
# M3: e = f(W, L) and off-image interchange
env = GL.Env(3, 1); r = np.random.default_rng(80_100_000); d = GL.sample(env, 20000, r); X = d["X"]; q = r.integers(0, 6, len(X)); ii = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi; e = elig(env, X, q)
keyWL = np.round(np.concatenate([np.log(W)[..., None], np.log(Lg)], -1), 8)                      # (n,G,1+K)
tab = {}
coll = 0
for i in range(len(X)):
    for g in range(3):
        k = tuple(keyWL[i, g]); v = e[i, g]
        if k in tab and tab[k] != v: coll += 1
        tab[k] = v
out["M3_distinct_(W_g,L_g)_values"] = len(tab); out["M3_collisions_e_not_determined_by_WL"] = coll
# realizability of (e_src, W_base, L_base): e_src must equal f(W_base, L_base) = e_base  ->  informative pairs are never realizable
rr = np.random.default_rng(3); bi = rr.integers(0, len(X), 20000); si = rr.integers(0, len(X), 20000); inf_ = np.abs(e[si] - e[bi]).sum(1) > 0
out["M3_frac_informative_E_pairs_input_realizable"] = 0.0 if inf_.any() else None
out["M3_frac_pairs_informative"] = float(inf_.mean())
# M4 / M5 on the decoy-DAS pair distribution (same construction as das_decoy.py, stream 64_500_000)
import native as NV
rng = np.random.default_rng(64_500_000); dd = NV.Bm.sample_fast(NV.ENV, 24000, rng); X = dd["X"]; q = rng.integers(0, 6, len(X)); ii = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
cnt = GL.counts(X, 3)[ii, q]; a_ = X.reshape(len(X), 6, 3, 2)[ii, q]; pf = (cnt == 5).astype(float); e = pf * (a_[..., 0] == a_[..., 1])
S = e.sum(1); two = np.where(S >= 2)[0]; one = np.where(S == 1)[0]; n = 6000
bi = np.concatenate([rng.choice(two, n // 2), rng.choice(one, n - n // 2)]); rng.shuffle(bi)
si = np.where(rng.random(n) < 0.6, rng.choice(two, n), rng.integers(0, len(X), n)); ri = rng.integers(0, len(X), n)
def cf(x, b): P = np.maximum((W[b][..., None] * (x.sum(1, keepdims=True) - x)[..., None] * Lg[b]).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
cfE = cf(e[si], bi); dec = {"A1": pf[si], "PERM": np.roll(e[si], 1, 1), "RANDE": e[ri]}
m4 = {}; m5 = {}
for nm, xs in dec.items():
    diffE = np.abs(e[si] - e[bi]).sum(1) > 0; cfD = cf(xs, bi); m4[nm] = dict(P_argmax_differ_all=float((cfD.argmax(-1) != cfE.argmax(-1)).mean()),
                                                                          P_argmax_differ_on_informative=float((cfD.argmax(-1) != cfE.argmax(-1))[diffE].mean()))
    xb = pf[bi] if nm == "A1" else e[bi]; pres = (np.abs(xs - xb).sum(1) == 0) & (si != bi)
    m5[nm] = dict(n_pres=int(pres.sum()), frac_E_changes=float(diffE[pres].mean()), frac_bayes_decision_changes=float((cfE.argmax(-1) != cf(e[bi], bi).argmax(-1))[pres].mean()))
out["M4_false_positive_bound"] = m4; out["M5_preservation_power"] = m5
obs = {}
for f in glob.glob(os.path.join(HERE, "..", "..", "step1_native_causal_construction", "logs", "decoy*", "*_full_c1_*.json")):
    j = json.load(open(f)); obs.setdefault(j["rule"], []).append(j["trained"]["IIA"])
out["M4_observed_full_model_decoy_IIA_range_XI"] = {k: [min(v), max(v)] for k, v in obs.items()}
json.dump(out, open(os.path.join(HERE, "..", "logs", "math_xii.json"), "w"), indent=1); print(json.dumps(out, indent=1))
