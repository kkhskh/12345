"""T4 generator, exact Bayes reference and the candidate decision rules (Track B, B1/B2), v2 after review B (MAJ-4, MAJ-5).
Thin wrappers over the vendored, unchanged historical modules (genlib = generator + closed-form posterior; relpairs.ENV1 = T4).

T4: G = 3 groups x 2 sources; roles: one reliable group (alpha = 0.8), one colluding group (both sources always report the same value),
one noise group; K = 3; Q = 6.  Exact Bayes (genlib.posterior_closed, checked against brute-force enumeration over role assignments
in cor_research.mathematics.run_track_c): p(y | x, q) ∝ Σ_g W_g m_g L_g(y).
Candidate rules (all on the SAME on-support inputs):
  bayes     exact posterior
  local     exact posterior from question q's six reports only (cross-question evidence discarded)
  noelig    drops the colluder-eligibility factor m_g:      p ∝ Σ_g W_g L_g
  nocount   drops the cross-question reliability weight W_g:  p ∝ Σ_g m_g L_g
  maprole   FULL role MAP: p(y | x, r*) with r* = argmax_r p(r | x) over the role assignments; ties (relative 1e-9) are AVERAGED
  softrole  geometric pooling over roles: p ∝ exp(Σ_r p(r|x) log p(y | x, r)) (mean-field limit; listed as calibration-only, not tested)
  bayes_a   exact Bayes computed with a wrong reliability alpha' (parametric-precision null; Env(alpha=alpha'))
The role-conditional terms replicate genlib.posterior_brute's per-role numerators exactly (same expressions; verified: their
role-sum equals posterior_brute to 1e-12 in tests)."""
import numpy as np
from cor_research import env as E
E.legacy_paths()
import genlib as GL, relpairs as RP
ENV = RP.ENV1; ENVQ1 = GL.Env(3, 1, Q=1); RULES = ("local", "noelig", "nocount", "maprole")
DIR_RULES = ("local", "noelig")    # review B r2 N1: nocount's and maprole's argmax differences from Bayes are (almost) all exact ties broken toward class 0


def decide(P, tol=1e-9):
    """tie-aware decision: argmax, or -1 when the top two are within tol (relative)"""
    s = np.sort(P, -1); d = P.argmax(-1); return np.where(s[:, -1] - s[:, -2] > tol * np.maximum(s[:, -1], 1e-300), d, -1)


def sample(n, seed):
    rng = np.random.default_rng(seed); d = GL.sample(ENV, n, rng); q = rng.integers(0, ENV.Q, n); return d["X"], q, d["ridx"], d["Y"]


def bayes(X, q, env=ENV): return GL.posterior_closed(env, X, q)


def role_terms(X, q, env=ENV):
    """log numerators L[r, n, y] of genlib.posterior_brute (one per role assignment r)"""
    n = len(X); idx = np.arange(n); rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); out = []
    for g, hs in env.roles:
        others = [h for h in range(env.G) if h != g and h not in hs]
        lq = np.log(relm[:, :, g]) + sum(np.log(col[:, :, h]) for h in hs) + len(others) * np.log(noi)
        mask = np.ones_like(lq, bool); mask[idx, q] = False; tot = np.where(mask, lq, 0.0).sum(1)
        lyq = np.log(rel[idx, q, g, :]) + sum(np.log(col[idx, q, h]) for h in hs)[:, None] + len(others) * np.log(noi)
        out.append(tot[:, None] + lyq)
    return np.stack(out)


def role_rules(X, q):
    L = role_terms(X, q); m = L.max((0, 2), keepdims=True); W = np.exp(L - m)            # r, n, y
    pr = W.sum(-1); pr = pr / pr.sum(0, keepdims=True)                                    # p(r | x)
    Z = W.sum(-1, keepdims=True); py_r = np.where(Z > 0, W / np.where(Z > 0, Z, 1.0), 1.0 / W.shape[-1])   # p(y | x, r); impossible roles get weight 0
    top = pr.max(0, keepdims=True); tie = pr >= top * (1 - 1e-9)
    MAP = (py_r * tie[..., None]).sum(0) / tie.sum(0)[:, None]
    lg = (pr[..., None] * np.log(np.maximum(py_r, 1e-300))).sum(0); S = np.exp(lg - lg.max(-1, keepdims=True)); S /= S.sum(-1, keepdims=True)
    return dict(maprole=MAP, softrole=S, role_posterior=pr, tie_frac=float((tie.sum(0) > 1).mean()))


def rules(X, q, alpha_prime=None):
    n = len(X); i = np.arange(n); PB = GL.posterior_closed(ENV, X, q)
    PL = GL.posterior_closed(ENVQ1, X[i, q][:, None], np.zeros(n, int))
    rel, col, noi = GL._group_lik(ENV, X); relm = rel.mean(-1); m, e = GL.eligibility(X, ENV.G, q)
    lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi
    nrm = lambda P: P / P.sum(-1, keepdims=True)
    PN = nrm((W[..., None] * Lg).sum(1)); PC = nrm((m[..., None] * Lg).sum(1) + 1e-12)
    RR = role_rules(X, q); out = dict(bayes=PB, local=PL, noelig=PN, nocount=PC, maprole=RR["maprole"], softrole=RR["softrole"])
    if alpha_prime is not None: out["bayes_a"] = GL.posterior_closed(GL.Env(3, 1, alpha=alpha_prime), X, q)
    return out


def top2_gap(P): s = np.sort(P, -1); return s[:, -1] - s[:, -2]


def discriminating(n_pool, seed, gap=0.05):
    """On-support discriminating inputs (tie-aware, review B r2 N1): natural episodes with a non-tied Bayes decision (top-2 gap >= gap) on which
    at least one directed rule (local, noelig) has a UNIQUE decision different from Bayes's."""
    X, q, r, Y = sample(n_pool, seed); R = rules(X, q); dB = R["bayes"].argmax(-1)
    keep = (top2_gap(R["bayes"]) >= gap) & np.any([(decide(R[k]) >= 0) & (decide(R[k]) != dB) for k in DIR_RULES], 0)
    return X[keep], q[keep], dict(n_pool=n_pool, n_kept=int(keep.sum()), per_rule_unique_disagreement={k: float(((decide(R[k]) >= 0) & (decide(R[k]) != dB))[keep].mean()) for k in RULES})


def equivalence_table(n=12000, seed=20_000_001):
    """mandate B1 (tie-aware, review B r2 N4): on-support decision disagreement where BOTH decisions are unique, the tie rate of each rule,
    KL from Bayes, and the frozen category of each candidate"""
    X, q, _, _ = sample(n, seed); R = rules(X, q, alpha_prime=0.7); names = ["bayes", "local", "noelig", "nocount", "maprole", "softrole", "bayes_a"]
    dec = {a: decide(R[a]) for a in names}
    D = {a: {b: float(((dec[a] >= 0) & (dec[b] >= 0) & (dec[a] != dec[b])).mean()) for b in names} for a in names}
    kl = {a: float((R["bayes"] * (np.log(R["bayes"] + 1e-12) - np.log(R[a] + 1e-12))).sum(-1).mean()) for a in names}
    cat = dict(bayes="reference", local="genuinely different on support (cross-question evidence discarded)", noelig="genuinely different on support (posterior combination without colluder eligibility)",
               nocount="differs mainly at exact ties (unique-decision disagreement only); not used for directed tests", maprole="posterior-combination variant; its argmax differences are exact ties: not used for directed tests",
               softrole="calibration-only (geometric pooling)", bayes_a="calibration/precision-only (wrong reliability parameter)")
    return dict(n=n, unique_decision_disagreement=D, tie_rate={a: float((dec[a] < 0).mean()) for a in names}, KL_from_bayes=kl, category=cat, maprole_role_tie_fraction=role_rules(X, q)["tie_frac"])


def relational_pairs(n, seed):
    """Role-preserving relational counterfactuals: base x and donor x' drawn with the SAME role assignment; the hybrid x_h has question
    q's six reports from x and every other question from x'.  Because roles agree, x_h is on-support; its exact Bayes posterior is the
    ideal reference for a transfer of cross-question (relational) evidence."""
    X, q, r, Y = sample(4 * n, seed); idx = []; rng = np.random.default_rng(seed + 1)
    for g in np.unique(r):
        w = np.where(r == g)[0]; p = rng.permutation(w); idx += [(a, b) for a, b in zip(p[: len(p) // 2], p[len(p) // 2:])]
    idx = np.array(idx[:n]); a, b = idx[:, 0], idx[:, 1]; Xa, Xb, qa = X[a], X[b], q[a]
    Xh = Xb.copy(); Xh[np.arange(len(a)), qa] = Xa[np.arange(len(a)), qa]
    assert np.all(r[a] == r[b])
    return dict(X=Xa, Xd=Xb, Xh=Xh, q=qa, PB=bayes(Xa, qa), PBh=bayes(Xh, qa))
