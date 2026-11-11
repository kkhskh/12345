"""Investigation VI library: exact-Bayes boundary quantities and the generator-defined contrast families.
Generator: gen_family.Fam() (T4: K = 3 candidates, Q = 6 questions, 3 source groups of 2, alpha = 0.8, colluder agreement 1,
one colluding group).  Re-uses Investigation V's cri.py / Investigation IV's eqv.py (read-only)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_curriculum_response_identification", "src"))
import numpy as np
import cri as R
E, G, F4, Q, K = R.E, R.G, R.F4, R.Q, R.K
CFG = R.CFG


def oriented_margin(c, cand):
    """m = c[cand] − mean of the other two centred log-probabilities (gauge-free; a candidate log-probability ratio:
    m = log p(cand) − ½[log p(c′) + log p(c″)])."""
    idx = np.arange(len(c)); oth = c.sum(-1) - c[idx, cand]
    return c[idx, cand] - oth / (K - 1)


def group_vote(X, q, g):
    """candidate c_g reported by both sources of group g at question q if they agree, else −1."""
    a, b = G.GROUPS[g]; xa, xb = X[:, q, a], X[:, q, b]
    return np.where(xa == xb, xa, -1)


def count_edit(X, q, cfg, rng, frm, to):
    """Single-report edit raising one group's leave-q count from `frm` to `to` = frm + 1, X_q fixed: pick (uniformly, from the
    generator only) a group g with n^{(-q)}_g == frm and one of its disagreeing questions k != q; set the second source equal to the
    first.  Same world always holds for frm <= 4 (a count below 5 identifies a non-colluder; reliable and noise pairs can agree).
    Returns X', g, k, ok."""
    N = len(X); Xp = X.copy(); n = E.loo(X)[:, q]; A = E.agree(X); gs = np.full(N, -1); ks = np.full(N, -1); ok = np.zeros(N, bool)
    for i in range(N):
        cand = np.where(n[i] == frm)[0]
        if not len(cand): continue
        g = cand[rng.integers(len(cand))]; dis = [k for k in range(Q) if k != q and A[i, k, g] == 0]
        k = dis[rng.integers(len(dis))]; a, b = G.GROUPS[g]; Xp[i, k, b] = Xp[i, k, a]; gs[i], ks[i], ok[i] = g, k, True
    return Xp, gs, ks, ok


def token_control(X, gs, ks):
    """F0: the same single token as the count edit (second source of group g in question k) is changed to the third value, so
    the pair still disagrees: V, exact Bayes and the latent world are unchanged; exactly one report changes."""
    Xc = X.copy()
    for i in np.where(gs >= 0)[0]:
        a, b = G.GROUPS[gs[i]]; xa, xb = X[i, ks[i], a], X[i, ks[i], b]; Xc[i, ks[i], b] = 3 - xa - xb
    return Xc


def bayes_c(X, q_idx):
    """centred exact log-posterior at the scored decision of each episode."""
    P = E.bayes_from(X, E.loo(X)); return R.bayes_logodds(P[np.arange(len(X)), q_idx])


def role_posterior(X, q_idx):
    """posterior over the 6 configurations (g reliable, h colluder) given X_{-q} (pi) and given all of X (full role posterior)."""
    n = E.loo(X)[np.arange(len(X)), q_idx]; pi = E.pi_of_n(n)
    J = G.joint(F4, X)[np.arange(len(X)), q_idx]                      # (N, C, K)
    full = pi * J.sum(-1); full = full / full.sum(-1, keepdims=True); return pi, full


# ------------------------------------------------------------------ models
def model_path(name):
    """Investigation VI cohort: 'xT70@4000' (T1 seed 70, snapshot 4000), 'xP80@500' ...; anything else is resolved by
    Investigation V's cri.ckpt_path (development models P0-P4, T0-T2, T10, T11, cT61..., bcT61_b500 ...)."""
    if name.startswith("x"):
        arm, rest = name[1], name[2:]; seed, step = rest.split("@")
        return os.path.join(HERE, "..", "checkpoints", f"x{'T1' if arm == 'T' else 'P'}{seed}", f"step{step}.npz")
    return R.ckpt_path(name)


def kind_of(name): return name[1] if name.startswith("x") else R.kind_of(name)


class Model(E.Model):
    def __init__(self, name):
        path = model_path(name); orig = E.ckpt_path; E.ckpt_path = lambda _n: path
        try: super().__init__(name)
        finally: E.ckpt_path = orig
        self.kind = kind_of(name)
