"""Investigation V library: exact generator families, input-level V-surfaces, operator families, covariate-corrected encoders.
Builds on Investigation IV's eqv.py (imported read-only from the sibling package)."""
import itertools, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_cross_implementation_causal_equivalence", "src"))
import numpy as np
import eqv as E
from scipy.optimize import minimize

G, F4, Q, K = E.G, E.F4, E.Q, E.K
CFG = E.CFG; H_OF = np.array([h for g, h in CFG]); G_OF = np.array([g for g, h in CFG])
SUPP = np.array([v for v in itertools.product(range(6), repeat=3) if 5 in v])


def kind_of(name):
    """'P' (P-count) or 'T' (T1) from a model name: P2, T10, cP60, cT61, bdevT2_b250, bcT60_b500 ..."""
    base = name[1:] if name.startswith("b") else name
    base = base[3:] if base.startswith("dev") else (base[1:] if base.startswith("c") else base)
    return base[0]


def ckpt_path(name):
    """Development: names of Investigation IV (P0..P4, T0, T1, T2, T10, T11).  Cohort: 'cT60'..'cT67', 'cP60'..'cP67'.
    Budget variants: 'b' + the file name in checkpoints/budget (e.g. bdevT2_b250, bcT60_b250)."""
    if name.startswith("b"):
        return os.path.join(HERE, "..", "checkpoints", "budget", name[1:] + ".npz")
    if name.startswith("c"):
        arm, s = name[1], name[2:]
        return os.path.join(HERE, "..", "checkpoints", f"ft_{'T1' if arm == 'T' else 'P'}_s{s}.npz")
    return E.ckpt_path(name)


class Model(E.Model):
    def __init__(self, name):
        E.ckpt_path_orig = E.ckpt_path
        p = ckpt_path(name)
        E.ckpt_path = lambda _n: p
        try: super().__init__(name)
        finally: E.ckpt_path = E.ckpt_path_orig
        self.kind = kind_of(name)


# ------------------------------------------------------------------ generator-side quantities
def true_h(cfg): return H_OF[cfg]


def bayes_logodds(p):
    """centred log-probabilities of a probability vector (the gauge used for all model/Bayes comparisons)."""
    lp = np.log(np.clip(p, 1e-12, 1)); return lp - lp.mean(-1, keepdims=True)


def realize(X, q, V, rng):
    """Replace X_{-q} by a fresh realisation with leave-q counts V (N,3): for each group g, a uniformly random set of
    V[i,g] of the five other questions agree; reports: x_a uniform, x_b = x_a if agree else one of the other two values.
    X_q is untouched.  Every realisation has positive probability under every config whose colluder h has V_h = 5
    (the colluder pair must agree in all questions; reliable and noise pairs can produce any pattern)."""
    N = len(X); Xp = X.copy(); oth = np.array([k for k in range(Q) if k != q])
    for g, (a, b) in enumerate(G.GROUPS):
        perm = np.argsort(rng.random((N, 5)), 1); agree = perm < V[:, g:g + 1]           # random subset of size V_g
        xa = rng.integers(0, K, (N, 5)); xb = np.where(agree, xa, (xa + 1 + rng.integers(0, K - 1, (N, 5))) % K)
        Xp[:, oth, a] = xa; Xp[:, oth, b] = xb
    return Xp


def support_labels(X, Xp, q, cfg):
    """per pair: marginal support of x', same world (true colluder agrees in all questions of x'), X_q fixed,
    exact Bayes at q preserved, number of changed reports, agreement pattern changed."""
    pb, pa = E.bayes_from(X, E.loo(X))[:, q], E.bayes_from(Xp, E.loo(Xp))[:, q]
    return dict(marginal=E.episode_support(Xp), same_world=E.same_world(Xp, cfg), xq_fixed=(Xp[:, q] == X[:, q]).all(-1),
                bayes_preserved=np.abs(pb - pa).max(-1) < 1e-12, n_reports_changed=(Xp != X).sum((1, 2)),
                agreement_pattern_changed=(E.agree(Xp) != E.agree(X)).any((1, 2)))


def family_N(X, q, rng):
    """Bayesian-null minimal edits: two non-colluder components shifted together (Inv IV pi_preserving_edit)."""
    return E.pi_preserving_edit(X, q, rng)


def family_R(X, q, cfg, rng):
    """Bayesian-relevant minimal edits: one A_{k,g} toggled (k != q), same world required (the true colluder may not be
    toggled), exact Bayes at q must change."""
    N = len(X); Xp = X.copy(); ok = np.zeros(N, bool); A = E.agree(X); h = true_h(cfg)
    for i in range(N):
        cands = [(g, k) for g in range(3) if g != h[i] for k in range(Q) if k != q]
        Z = X[i:i + 1].copy(); g, k = cands[rng.integers(len(cands))]; E._toggle(Z, 0, k, g, A[i, k, g] == 0, rng)
        Xp[i] = Z[0]; ok[i] = True
    pb, pa = E.bayes_from(X, E.loo(X))[:, q], E.bayes_from(Xp, E.loo(Xp))[:, q]
    return Xp, ok & (np.abs(pb - pa).max(-1) > 1e-9)


def family_C(X, q, rng, kind):
    """V-preserving controls.  C1 relabel: other questions' reports cyclically relabelled (Inv IV);  C2 swap: the two
    sources of every pair swapped in other questions;  C3 resample: X_{-q} re-drawn with the SAME per-question agreement
    pattern (fresh report values)."""
    Xp = X.copy(); oth = [k for k in range(Q) if k != q]
    if kind == "C1": return E.relabel_pairs(X, q)
    if kind == "C2":
        for a, b in G.GROUPS: Xp[:, oth, a], Xp[:, oth, b] = X[:, oth, b], X[:, oth, a]
        return Xp
    A = E.agree(X)
    for g, (a, b) in enumerate(G.GROUPS):
        xa = rng.integers(0, K, (len(X), Q)); xb = np.where(A[:, :, g] == 1, xa, (xa + 1 + rng.integers(0, K - 1, (len(X), Q))) % K)
        Xp[:, oth, a] = xa[:, oth]; Xp[:, oth, b] = xb[:, oth]
    return Xp


def matched_content_control(X, Xn, rng):
    """Family Cm, matched to a minimal edit X -> Xn: for every (question k, pair g) whose reports the edit touched, change the
    same pair's reports WITHOUT changing its agreement status (agreeing pair (a,a) -> (a',a') with a' != a; disagreeing pair:
    the source the edit changed is set to the third value, still disagreeing).  V, exact Bayes and the latent world are
    unchanged; the report tokens touched are the same as in the edit."""
    Xc = X.copy(); diff = (Xn != X)
    for i, k, s in zip(*np.where(diff)):
        g = s // 2; a, b = G.GROUPS[g]
        if Xc[i, k, a] == X[i, k, a] and Xc[i, k, b] == X[i, k, b]:
            if X[i, k, a] == X[i, k, b]:
                nv = (X[i, k, a] + 1 + rng.integers(0, K - 1)) % K; Xc[i, k, a] = Xc[i, k, b] = nv
            else:
                other = a if s == b else b; third = 3 - X[i, k, a] - X[i, k, b]; Xc[i, k, s] = third
                if Xc[i, k, a] == Xc[i, k, b]: Xc[i, k, s] = X[i, k, s]
    return Xc


def surface_vs(h):
    """same-world target count vectors for an episode whose true colluder is h: v_h = 5, others free (36 vectors)."""
    return np.array([v for v in itertools.product(range(6), repeat=3) if v[h] == 5])


# ------------------------------------------------------------------ operator families fitted to V-surfaces
def J_at(X, q): return G.joint(F4, X[:, q:q + 1])[:, 0]                  # (N, C, K) = P(X_q, y | R)


def mix_logits(theta, Jq, Vs):
    """Bayes-mixture family ('distorted Bayes'): w(g,h) = exp(phi_rel[v_g] + phi_col[v_h]); P(y) ∝ sum w J; output tau*log P.
    theta = (phi_rel[1..5], phi_col[0..4] relative to phi_col[5] = 0, log tau); phi_rel[0] = 0 (gauge)."""
    pr = np.r_[0, theta[0:5]]; pc = np.r_[theta[5:10], 0]; tau = np.exp(theta[10])
    w = np.exp(pr[Vs[:, G_OF]] + pc[Vs[:, H_OF]])                          # (M, C)
    P = (w[:, :, None] * Jq).sum(1); L = tau * np.log(P + 1e-300); return L - L.mean(-1, keepdims=True)


def mix_exact(Jq, Vs, tau=1.0):
    th = np.r_[np.arange(1, 6) * np.log(E.RHO), np.full(5, -50.0), np.log(tau)]; return mix_logits(th, Jq, Vs)


def vote_logits(theta, Xq, Vs):
    """Weighted-voting family (alternative factorization): logit(y) = sum_s a(v_{g(s)}, [v_{g(s)} = 5]) [x_{q,s} = y],
    a over 6 count values for non-perfect groups and one value for perfect (count-5) groups; plus a common scale."""
    a = np.r_[theta[0:6]]; votes = np.zeros((len(Xq), K))
    for g, (sa, sb) in enumerate(G.GROUPS):
        wg = a[Vs[:, g]]
        for s in (sa, sb): votes[np.arange(len(Xq)), Xq[:, s]] += wg
    return votes - votes.mean(-1, keepdims=True)


def fit_family(kind, target, Jq, Xq, Vs, theta0=None):
    """least-squares fit of centred logits.  Returns theta, R² over all entries, R² of the within-episode (V-dependent) part."""
    if kind == "mix":
        f = lambda th: mix_logits(th, Jq, Vs); th0 = np.r_[np.arange(1, 6) * np.log(E.RHO), np.full(5, -3.0), 0.0] if theta0 is None else theta0
    else:
        f = lambda th: vote_logits(th, Xq, Vs); th0 = np.ones(6) if theta0 is None else theta0
    loss = lambda th: ((f(th) - target) ** 2).mean()
    r = minimize(loss, th0, method="L-BFGS-B", options=dict(maxiter=3000)); pred = f(r.x)
    return r.x, pred


def r2(pred, target, groups=None):
    """R² of all entries; if groups (episode ids) given, also R² of the within-episode deviations (V-dependent part)."""
    out = dict(r2_all=float(1 - ((target - pred) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()))
    if groups is not None:
        def dm(Z):
            Z = Z.copy()
            for gid in np.unique(groups):
                s = groups == gid; Z[s] -= Z[s].mean(0)
            return Z
        t, p = dm(target), dm(pred); out["r2_within"] = float(1 - ((t - p) ** 2).sum() / (t ** 2).sum())
    return out


# ------------------------------------------------------------------ covariate-corrected encoders (Inv IV finding ME1)
def fit_encoder_cov(kind, Xf, Af):
    """count encoder fitted jointly with one-hot own-question reports of the token's question as covariates; only the count
    coefficients are returned (so abstract interventions do not carry own-evidence correlates)."""
    n = E.loo(Xf); out = {}
    def oh_q(k): return E.onehot(Xf[:, k], 3)
    if kind == "P_full":
        for q in range(Q):
            Fc = E.onehot(n[:, q]); Fz = oh_q(q); W = E.lstsq(np.c_[Fc, Fz], Af[:, 36 + q]); out[36 + q] = W[:Fc.shape[1]]
    else:
        for t in range(36):
            k, g = t // 6, (t % 6) // 2; Fc = E.onehot(n[:, k, g][:, None]); Fz = oh_q(k)
            W = E.lstsq(np.c_[Fc, Fz], Af[:, t]); out[t] = W[:Fc.shape[1]]
    return out
