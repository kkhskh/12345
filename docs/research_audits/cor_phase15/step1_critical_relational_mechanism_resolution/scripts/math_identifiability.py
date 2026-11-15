"""XIII-C Workstream B1 (mathematics, numerically verified).  All claims are about the C = 1, beta = 1 T4 generator (G = 3, K = 3, Q = 6,
alpha = 0.8) and the XIII-B nested power-mean family
    PM(theta):  log p(y) = (tau/rho) log sum_g w_g L_g(y)^rho + c,   w = softmax_g(a log W_g + k log max(f_g,1) + d [f_g = 0]).
Statements checked here (proofs in DECISION_LAW_AND_MATHEMATICAL_IDENTIFIABILITY.md):
 T1 (tempering identity)    PM(rho, tau) == MIX of rho-TEMPERED evidence L_g^rho followed by output exponent tau/rho.          [exact]
 T2 (reliability identity)  Because L_g(y) = prod_{s in g} c_s(y)/noi with c in {alpha, (1-alpha)/2}, L_g^rho equals, up to a factor that
                            is identical for every group, the likelihood of the SAME reports under a misstated per-source reliability alpha'
                            with  alpha'/((1-alpha')/2) = (alpha/((1-alpha)/2))^rho.  Hence PM(rho, tau) == exact-form mixture with
                            alpha'(rho) at the target question, followed by output exponent tau/rho.                              [exact]
 T3 (non-identifiability boundary)  On episodes where only one group has non-zero weight, PM does not depend on rho at all (log p = tau log L_g):
                            rho is identified only by episodes with >= 2 effectively weighted groups whose likelihood profiles differ.
 T4 (lapse confound)        A Bayes posterior with a lapse floor  p = (1-lam) p_B + lam/3  is NOT in the PM family, but the best PM fit to it
                            has rho < 1: a lapse floor alone moves the fitted exponent (numerical).
 T5 (pure output temperature) p ∝ p_B^tau is in the PM family with rho = 1; the fit recovers rho = 1 (numerical; tau is not confounded with rho).
 T6 (E / A1 / A2 separability)  on-support relational and q-edits that separate the three eligibility rules exist under the generator.
-> logs/math_identifiability.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp as lse
import native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
out = {}; rng = np.random.default_rng(86_200_000); n = 6000
X = NV.Bm.sample_fast(NV.ENV, n, rng)["X"]; q = rng.integers(0, 6, n); i = np.arange(n)
W, Lg, pf, e, S = RP.parts(X, q); f = S - e; lW, lL = np.log(W), np.log(Lg)
cen = lambda a: a - a.mean(-1, keepdims=True)
def wts(a, k, d, idx=slice(None)):
    z = a * lW[idx] + k * np.log(np.maximum(f[idx], 1)) + d * (f[idx] == 0); return z - lse(z, 1, keepdims=True)
def PM(th, idx=slice(None), lL_=None):
    a, k, d, tau, rho = th; lL_ = lL if lL_ is None else lL_; return cen(tau * lse(wts(a, k, d, idx)[..., None] + rho * lL_[idx], 1) / rho)
# ---- T1
th = (1.0, 1.0, -6.0, 1.3, 0.6); lhs = PM(th)
rhs = cen((th[3] / th[4]) * np.log((np.exp(wts(*th[:3]))[..., None] * Lg ** th[4]).sum(1)))
out["T1_max_abs_diff"] = float(np.abs(lhs - rhs).max())
# ---- T2: rebuild L_g under alpha'(rho) from the raw reports
al = 0.8; r0 = al / ((1 - al) / 2)
def L_alpha(X, q, alpha):
    a_ = X.reshape(len(X), 6, 3, 2)[np.arange(len(X)), q]; y = np.arange(3); b = (1 - alpha) / 2
    c1 = np.where(a_[..., 0, None] == y, alpha, b); c2 = np.where(a_[..., 1, None] == y, alpha, b); return c1 * c2 * 9.0
T2 = {}
for rho in (0.3, 0.6, 0.9):
    rp = r0 ** rho; alp = rp / (rp + 2); Lp = L_alpha(X, q, alp)
    pm = PM((1.0, 1.0, -6.0, 1.0, rho)); mix_alp = cen((1.0 / rho) * lse(wts(1.0, 1.0, -6.0)[..., None] + np.log(Lp), 1))
    T2[str(rho)] = dict(alpha_prime=float(alp), max_abs_diff=float(np.abs(pm - mix_alp).max()))
out["T2_reliability_identity"] = T2
# ---- T3: single-weighted-group episodes
# effective weights of the exact rule (a = 1, k = 1, d -> -inf): w_g ∝ W_g f_g.  "single" = one group carries > 99.9% of the weight.
th1 = lambda rho: (1.0, 1.0, -60.0, 1.0, rho); wB = np.exp(wts(1.0, 1.0, -60.0)); single = wB.max(1) > 0.999; multi = ~single
d_single = np.abs(PM(th1(0.3))[single] - PM(th1(1.0))[single]); d_multi = np.abs(PM(th1(0.3))[multi] - PM(th1(1.0))[multi])
# exact single-group case: put all weight on one group -> log p = tau * log L_g (rho cancels)
w1 = np.full((n, 3), -1e6); w1[:, 0] = 0.0
exact_single = max(float(np.abs(cen(lse(w1[..., None] + r * lL, 1) / r) - cen(lL[:, 0])).max()) for r in (0.2, 0.6, 1.0, 2.0))
out["T3"] = dict(frac_effectively_single=float(single.mean()), rho_effect_single_mean=float(d_single.mean()) if single.any() else None,
                 rho_effect_multi_mean=float(d_multi.mean()), exact_single_group_rho_invariance_maxdiff=exact_single,
                 n_weighted_groups_hist={str(k): float(((wB > 0.01).sum(1) == k).mean()) for k in (1, 2, 3)})
# ---- fitting helper (KL objective: model distribution as the 'data', proper scoring rule)
tr, te = i[:3000], i[3000:]
def fitPM(target_lp, starts=((1, 1, -5, 1, 1.0), (0.8, 0.5, -8, 1, 0.5), (1.2, 1.5, -3, 1.2, 0.3), (1, 1, -10, 0.8, 1.5))):
    P = np.exp(target_lp - lse(target_lp, -1, keepdims=True))
    def kl(th, idx):
        lq = PM(th, idx); lq = lq - lse(lq, -1, keepdims=True); return float((P[idx] * (np.log(P[idx] + 1e-300) - lq)).sum(-1).mean())
    res = [minimize(lambda t: kl(t, tr), s, method="Powell", options=dict(maxiter=8000, xtol=1e-5, ftol=1e-10)) for s in starts]
    b = min(res, key=lambda r: r.fun)
    return dict(rho=float(b.x[4]), tau=float(b.x[3]), KL_train=float(b.fun), KL_test=kl(b.x, te), start_spread=float(np.ptp([r.fun for r in res])),
                rho_by_start=[float(r.x[4]) for r in res])
lpB = np.log(GL.posterior_closed(RP.ENV1, X, q) + 1e-300)
out["fit_bayes"] = fitPM(lpB)
out["T4_lapse"] = {str(lam): fitPM(np.log((1 - lam) * np.exp(lpB) + lam / 3)) for lam in (0.01, 0.03, 0.1)}
out["T5_temperature"] = {str(t): fitPM(t * lpB) for t in (0.7, 1.5)}
# ---- T6: existence and counts of on-support separating edits (relational q' edits for n_h 5<->4 and 4<->3, q-edits for n=5 / n=4 groups)
cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(n, 6, 3, 2); agq = a_[i, q, :, 0] == a_[i, q, :, 1]
allag = (a_[..., 0] == a_[..., 1]).all(1)                                      # a group agreeing at all 6 questions (admissible colluder)
def rule(cn, ag, which):
    return ((cn == 5) & ag) if which == "E" else (cn == 5) if which == "A1" else ((cn >= 4) & ag)
sep = {}
for kind, cond in (("rel_5to4", (cnt == 5) & agq), ("rel_4to3", (cnt == 4) & agq), ("q_flip_n5", cnt == 5), ("q_flip_n4", cnt == 4)):
    rows = []
    for ii_, h in zip(*np.where(cond)):
        cn2, ag2 = cnt[ii_].copy(), agq[ii_].copy()
        if kind.startswith("rel"): cn2[h] -= 1
        else: ag2[h] = ~ag2[h]
        al6 = allag[ii_].copy(); al6[h] = False if (kind == "rel_5to4" or (kind == "q_flip_n5" and agq[ii_, h])) else al6[h]
        if not al6.any(): continue                                            # off-support after the edit
        ch = {w: bool((rule(cnt[ii_], agq[ii_], w) != rule(cn2, ag2, w)).any()) for w in ("E", "A1", "A2")}; rows.append(ch)
    sep[kind] = dict(n=len(rows), **{f"changes_{w}": float(np.mean([r[w] for r in rows])) for w in ("E", "A1", "A2")})
out["T6_separating_edits"] = sep
json.dump(out, open(os.path.join(HERE, "..", "logs", "math_identifiability.json"), "w"), indent=1); print(json.dumps(out, indent=1))
