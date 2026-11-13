"""Exact mathematical verification for Investigation IX (numpy only).  -> logs/verify_math.json
 V1  genlib brute-force posterior == latent_reliability bayes.episode_posterior on T4 episodes drawn by bayes.sample_fast
 V2  genlib sampler (G=3,C=1) reproduces bayes.sample_fast statistics (agreement rates by role, report accuracy)
 V3  closed form with m_g (no vote matching) == brute force, for G = 3, 4, 5, 6 ; vote-matching m' != brute force
 V4  Theorem 1 deep-sets factorisation  p(y) ∝ S*A_y - B_y  == brute force for G = 3..6
 V5  ANOVA (Moebius) interaction order of the evidence function f_y(x_1..x_G) = p(X, Y_q=y)/prod_g p_noise(x_g):
     exact order 1 + C (C colluders), computed on random product grids of group states, G = 4, C = 1, 2
 V6  partial copying (beta < 1): the hard gate becomes a soft per-group factor; deep-sets order-2 form still exact with
     e_g replaced by the per-group likelihood ratio (Theorem 1'); verified against brute force
"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_latent_reliability", "src"))
import numpy as np, genlib as GL, bayes as Bm
np.seterr(divide="ignore", invalid="ignore")
out = {}; rng = np.random.default_rng(9_000_001)
# ---- V1
envT = Bm.Env("T4", K=3, Q=6); d = Bm.sample_fast(envT, 3000, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
ref = Bm.episode_posterior(envT, X)["post_y"][np.arange(len(X)), q]
e3 = GL.Env(G=3, C=1); pb = GL.posterior_brute(e3, X, q)
out["V1_max_abs_diff_vs_bayes_episode_posterior"] = float(np.abs(pb - ref).max())
# ---- V2
dg = GL.sample(e3, 20000, np.random.default_rng(2)); dt = Bm.sample_fast(envT, 20000, np.random.default_rng(3))
def stats(X):
    a = X.reshape(*X.shape[:2], 3, 2); return np.sort((a[..., 0] == a[..., 1]).mean((0, 1)))
out["V2_group_agreement_rates_sorted"] = dict(genlib=stats(dg["X"]).tolist(), bayes_sample_fast=stats(dt["X"]).tolist())
# pooled over groups the rates are mixtures; compare also the per-episode count distribution of all-agreeing groups
def nperf(X):
    a = X.reshape(*X.shape[:2], 3, 2); return np.bincount((a[..., 0] == a[..., 1]).all(1).sum(1), minlength=4) / len(X)
out["V2_number_of_all_agreeing_groups_dist"] = dict(genlib=nperf(dg["X"]).tolist(), bayes_sample_fast=nperf(dt["X"]).tolist())
# ---- V3/V4
for G in (3, 4, 5, 6):
    env = GL.Env(G=G, C=1); dd = GL.sample(env, 3000, np.random.default_rng(10 + G)); Xg = dd["X"]; qg = np.random.default_rng(20 + G).integers(0, 6, len(Xg))
    pb = GL.posterior_brute(env, Xg, qg); pc = GL.posterior_closed(env, Xg, qg); pd, _ = GL.deepsets_factor(env, Xg, qg)
    # mandate's incorrect vote-matching m': h perfect and h's (agreeing) vote equals one of g's reports
    idx = np.arange(len(Xg)); a_ = Xg.reshape(len(Xg), 6, G, 2)[idx, qg]; n = GL.counts(Xg, G)[idx, qg]
    eh = (n == 5) & (a_[..., 0] == a_[..., 1])
    match = (a_[:, None, :, 0] == a_[:, :, None, 0]) | (a_[:, None, :, 0] == a_[:, :, None, 1])   # [n, g, h]: h's vote in g's reports
    mprime = (eh[:, None, :] & match & ~np.eye(G, dtype=bool)[None]).sum(2)
    rel, col, noi = GL._group_lik(env, Xg); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[idx, qg]); Lg = rel[idx, qg] / noi
    Pm = (W[..., None] * mprime[..., None] * Lg).sum(1); Pm = Pm / np.maximum(Pm.sum(-1, keepdims=True), 1e-300)
    m, _ = GL.eligibility(Xg, G, qg)
    out[f"V3_G{G}"] = dict(closed_vs_brute=float(np.abs(pc - pb).max()), deepsets_vs_brute=float(np.abs(pd - pb).max()),
                           vote_matching_vs_brute=float(np.nanmax(np.abs(Pm - pb))), agree_m_vs_mprime=float((m == mprime).mean()),
                           frac_two_or_more_eligible=float((eh.sum(1) >= 2).mean()))
# ---- V5 interaction order (Moebius on a product grid of group states at the target question, others marginal)
def evidence_fn(env, states, y, q=0):
    """states: tuple of per-group full report matrices (Q,2) ; returns f_y = p(X, Y_q=y) / prod_g p_noise(x_g)."""
    X = np.concatenate(states, 1)[None]                         # 1,Q,2G
    rel, col, noi = GL._group_lik(env, X)
    tot = 0.0
    for g, hs in env.roles:
        val = 1.0
        for qq in range(env.Q):
            r = rel[0, qq, g, y] / env.K if qq == q else rel[0, qq, g].mean()
            val *= r / noi
            for h in hs: val *= col[0, qq, h] / noi
        tot += val
    return tot
def moebius_orders(env, ngrid=3, seed=0, y=0):
    r = np.random.default_rng(seed); G = env.G
    # per group a small set of candidate states: include an all-agreeing pair-matrix and random ones
    S = []
    for g in range(G):
        st = [np.repeat(r.integers(0, env.K, (env.Q, 1)), 2, 1)]          # all agree
        while len(st) < ngrid: st.append(r.integers(0, env.K, (env.Q, 2)))
        S.append(st)
    F = np.zeros([ngrid] * G)
    for ix in itertools.product(range(ngrid), repeat=G): F[ix] = evidence_fn(env, tuple(S[g][ix[g]] for g in range(G)), y)
    # Moebius / ANOVA with base point 0: interaction term for subset T = sum_{U subset T} (-1)^{|T|-|U|} F(x_U, base_rest)
    res = {}
    for k in range(1, G + 1):
        mx = 0.0
        for T in itertools.combinations(range(G), k):
            for vals in itertools.product(range(1, ngrid), repeat=k):
                acc = 0.0
                for u in range(k + 1):
                    for U in itertools.combinations(range(k), u):
                        ix = [0] * G
                        for j in U: ix[T[j]] = vals[j]
                        acc += (-1) ** (k - u) * F[tuple(ix)]
                mx = max(mx, abs(acc))
        res[k] = mx / np.abs(F).max()
    return res
for C in (1, 2):
    env = GL.Env(G=4, C=C, Q=3)
    out[f"V5_interaction_order_G4_C{C}"] = {str(k): v for k, v in moebius_orders(env, 3, seed=C).items()}
# ---- V6 partial copying
env = GL.Env(G=4, C=1, beta=0.7); dd = GL.sample(env, 3000, np.random.default_rng(77)); Xg = dd["X"]; qg = np.random.default_rng(78).integers(0, 6, len(Xg))
pb = GL.posterior_brute(env, Xg, qg); idx = np.arange(len(Xg))
rel, col, noi = GL._group_lik(env, Xg); relm = rel.mean(-1)
lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[idx, qg]); Lg = rel[idx, qg] / noi
lc = np.log(col / noi); Ce = np.exp(lc.sum(1))                                     # soft eligibility: all-question colluder LR
P = Ce.sum(1)[:, None] * (W[..., None] * Lg).sum(1) - (W[..., None] * Ce[..., None] * Lg).sum(1)
P = P / P.sum(-1, keepdims=True)
out["V6_partial_copy_beta0.7_softgate_deepsets_vs_brute"] = float(np.abs(P - pb).max())
out["V6_hard_gate_closed_form_vs_brute_beta0.7"] = None
try:
    m, e = GL.eligibility(Xg, 4, qg); Ph = (W[..., None] * m[..., None] * Lg).sum(1); Ph = Ph / np.maximum(Ph.sum(-1, keepdims=True), 1e-300)
    out["V6_hard_gate_closed_form_vs_brute_beta0.7"] = float(np.nanmax(np.abs(Ph - pb)))
    out["V6_frac_no_eligible_group_beta0.7"] = float((e.sum(1) == 0).mean())
except Exception as ex: out["V6_err"] = str(ex)
json.dump(out, open(os.path.join(HERE, "..", "logs", "verify_math.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
