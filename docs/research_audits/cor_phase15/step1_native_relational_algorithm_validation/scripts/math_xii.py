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
# realizable iff some input has e = e_src together with (W_base, L_base); since e = f(W, L) (0 collisions above), that requires
# e_src[g] == tab[(W_base,g, L_base,g)] for every g  -> measured on the pairs:
real = np.array([all(tab[tuple(keyWL[b, g])] == e[s_, g] for g in range(3)) for b, s_ in zip(bi[:5000], si[:5000])]); inf5 = inf_[:5000]
out["M3_frac_informative_E_pairs_input_realizable"] = float(real[inf5].mean()); out["M3_frac_noninformative_pairs_realizable"] = float(real[~inf5].mean())
out["M3_note"] = "requires alpha != 1/K (otherwise W, L carry no information about agreement)"
out["M3_frac_pairs_informative"] = float(inf_.mean())
# M4 / M5 on the decoy-DAS pair distributions, reproducing das_decoy.sample EXACTLY per rule (XII correction after independent review:
# the first version reused the E-stratified selection for A1; das_decoy stratifies the base by the A1 count P_ when RULE == A1).
import native as NV
def decoy_pairs(RULE, n=6000, stream=64_500_000):
    rng = np.random.default_rng(stream); dd = NV.Bm.sample_fast(NV.ENV, 4 * n, rng); X = dd["X"]; q = rng.integers(0, 6, len(X)); ii = np.arange(len(X))
    rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
    cnt = GL.counts(X, 3)[ii, q]; a_ = X.reshape(len(X), 6, 3, 2)[ii, q]; pf = (cnt == 5).astype(float); e = pf * (a_[..., 0] == a_[..., 1])
    key = pf.sum(1) if RULE == "A1" else e.sum(1); two = np.where(key >= 2)[0]; one = np.where(key == 1)[0]
    bi = np.concatenate([rng.choice(two, n // 2), rng.choice(one, n - n // 2)]); rng.shuffle(bi)
    si = np.where(rng.random(n) < 0.6, rng.choice(two, n), rng.integers(0, len(X), n)); ri = rng.integers(0, len(X), n)
    xs = {"E": e[si], "PERM": np.roll(e[si], 1, axis=1), "A1": pf[si], "RANDE": e[ri]}[RULE]; xb = pf[bi] if RULE == "A1" else e[bi]
    def cf(x): P = np.maximum((W[bi][..., None] * (x.sum(1, keepdims=True) - x)[..., None] * Lg[bi]).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
    return dict(cfD=cf(xs), cfE=cf(e[si]), cfB0=cf(e[bi]), diff=np.abs(xs - xb).sum(1) > 0, pres=(np.abs(xs - xb).sum(1) == 0) & (si != bi), Echg=np.abs(e[si] - e[bi]).sum(1) > 0)
m4 = {}; m5 = {}
for nm in ("A1", "PERM", "RANDE"):
    D = decoy_pairs(nm); dif = D["cfD"].argmax(-1) != D["cfE"].argmax(-1)
    m4[nm] = dict(delta_on_rule_IIA_set=float(dif[D["diff"]].mean()), delta_all_pairs=float(dif.mean()), n_IIA_set=int(D["diff"].sum()))
    pres = D["pres"]; klE = kl(D["cfE"], D["cfB0"])                                                                           # an ideal E mediator's output change on that pair
    m5[nm] = dict(n_pres=int(pres.sum()), frac_E_changes=float(D["Echg"][pres].mean()), frac_bayes_decision_changes=float((D["cfE"].argmax(-1) != D["cfB0"].argmax(-1))[pres].mean()),
                  ideal_E_mediator_preserve_KL=float(klE[pres].mean()))
out["M4_false_positive_bound"] = m4; out["M5_preservation_power"] = m5
out["M4_M5_note"] = "XII-3 bound applies to IIA computed on the SAME pair set for both rules; rule-specific diff sets are reported separately"
obs = {}
for f in glob.glob(os.path.join(HERE, "..", "..", "step1_native_causal_construction", "logs", "decoy*", "*_full_c1_*.json")):
    j = json.load(open(f)); obs.setdefault(j["rule"], []).append(j["trained"]["IIA"])
out["M4_observed_full_model_decoy_IIA_range_XI"] = {k: [min(v), max(v)] for k, v in obs.items()}
json.dump(out, open(os.path.join(HERE, "..", "logs", "math_xii.json"), "w"), indent=1); print(json.dumps(out, indent=1))
