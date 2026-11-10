"""Shared library for Investigation IV (cross-implementation causal equivalence).

Generator-side objects are defined from the T4 generator only (gen_family.Fam(): K=3, Q=6, alpha=0.8, rho=1, pc=1):
  A[n,k,g]   within-pair agreement of group g in question k
  loo(X)     leave-k count vector n^{(-k)}_g = sum_{j != k} A[j,g]   (N,Q,3), values 0..5
  pi_of_n    P(R | X_{-q}) as a function of n^{(-q)}   (exact; verified in scripts/info_theory.py)
  bayes_from P(Y_q | X_q, n) for an arbitrary (possibly intervened) count vector n
Model-side objects: the residual stream entering layer 2 ("site"; output of the frozen transferred layers) of a T4
transfer model.  P-count site for decision q: readout token 36+q.  T1 site for decision q: question q's six source
tokens 6q..6q+5.  Encoders map generator features to activations (least squares on natural episodes); decoders map
activations to counts (multinomial logistic).  Nothing here is fitted on intervention outcomes."""
import itertools, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_relational_mechanism_deepening/src", "step1_foundational_atlas_continuation/src",
           "step1_relational_computation_atlas/src", "step1_latent_reliability/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd3 as F3, ckpt as C
from scipy.stats import binom

F4 = G.Fam(); Q, T, K = 6, 43, 3; CFG = F4.configs                        # 6 configs (g,h)
CROSS = [(1, 2), (3, 4), (5, 0)]                                          # source pairs straddling groups (null feature)


def sample(n, seed): return G.sample(F4, n, np.random.default_rng(seed))
def agree(X, pairs=G.GROUPS): return np.stack([(X[:, :, a] == X[:, :, b]) for a, b in pairs], -1).astype(int)
def loo(X, pairs=G.GROUPS): A = agree(X, pairs); return A.sum(1, keepdims=True) - A
def onehot(n, m=6): return np.eye(m)[n].reshape(n.shape[:-1] + (n.shape[-1] * m,))


def pair_marg_rel():
    e = (1 - F4.alpha) / (K - 1)
    return ((F4.alpha ** 2 + (K - 1) * e ** 2) / K, (2 * F4.alpha * e + (K - 2) * e ** 2) / K)


RHO = pair_marg_rel()[0] / pair_marg_rel()[1]


def pi_of_n(n):
    n = np.asarray(n); w = np.stack([RHO ** n[..., g] * (n[..., h] == 5) for g, h in CFG], -1).astype(float)
    s = w.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return w / s


def in_support(n): return (np.asarray(n) == 5).any(-1)


def bayes_from(X, n, J=None):
    """P(Y_q | X_q, n_q) for every q; n (N,Q,3) arbitrary.  nan where pi undefined (no group with count 5)."""
    J = G.joint(F4, X) if J is None else J
    py = (pi_of_n(n)[..., None] * J).sum(2)
    with np.errstate(invalid="ignore"):
        return py / py.sum(-1, keepdims=True)


def p_counts_given_R(allN):
    ap = G.agree_probs(F4); out = np.zeros((len(allN), len(CFG)))
    for c, (g, h) in enumerate(CFG):
        pr = np.ones(len(allN))
        for k in range(3):
            if k == h: pr = pr * (allN[:, k] == 5)
            else: pr = pr * binom.pmf(allN[:, k], 5, ap["rel"] if k == g else ap["noise"])
        out[:, c] = pr
    return out


def episode_support(X):
    """marginal support: some group agrees in all six questions (the colluder candidate)."""
    return (agree(X).sum(1) == 6).any(-1)


def same_world(X, cfg):
    h = np.array([CFG[c][1] for c in cfg]); return agree(X).sum(1)[np.arange(len(X)), h] == 6


def _toggle(X, i, k, g, make_agree, rng):
    a, b = G.GROUPS[g]
    X[i, k, b] = X[i, k, a] if make_agree else (X[i, k, a] + 1 + rng.integers(0, K - 1)) % K


def single_group_edit(X, q, rng, cfg=None):
    """For each episode pick a group g and a question k != q and toggle group g's agreement in question k, so that
    n^{(-q)}_g changes by +-1 and the other two components and X_q are unchanged.  Only edits that keep the episode in
    the generator's marginal support are accepted (exact check).  Returns X', g, ok, same_world flag."""
    N = len(X); Xp = X.copy(); A = agree(X); gs = np.full(N, -1); ok = np.zeros(N, bool)
    for i in range(N):
        cands = [(g, k) for g in range(3) for k in range(Q) if k != q]
        for j in rng.permutation(len(cands)):
            g, k = cands[j]; Z = X[i:i + 1].copy(); _toggle(Z, 0, k, g, A[i, k, g] == 0, rng)
            if episode_support(Z)[0]:
                Xp[i] = Z[0]; gs[i] = g; ok[i] = True; break
    sw = same_world(Xp, cfg) if cfg is not None else None
    return Xp, gs, ok, sw


def pi_preserving_edit(X, q, rng):
    """Change two count components by the same amount (both +1 or both -1) while the set {h: n_h = 5} is unchanged and
    has one element: pi(n) is unchanged exactly (pi depends on n only through differences n_g - n_g' and that set).
    Requires both non-colluder components in 0..3 (increment) or 1..4 (decrement)."""
    N = len(X); Xp = X.copy(); n = loo(X)[:, q]; A = agree(X); ok = np.zeros(N, bool)
    for i in range(N):
        H = np.where(n[i] == 5)[0]
        if len(H) != 1: continue
        o = [g for g in range(3) if g != H[0]]
        opts = []
        if all(n[i, g] <= 3 for g in o): opts.append(+1)
        if all(n[i, g] >= 1 for g in o): opts.append(-1)
        if not opts: continue
        dirn = opts[rng.integers(len(opts))]
        for g in o:
            ks = [k for k in range(Q) if k != q and A[i, k, g] == (0 if dirn > 0 else 1)]
            k = ks[rng.integers(len(ks))]; _toggle(Xp, i, k, g, dirn > 0, rng)
        ok[i] = True
    return Xp, ok


def cond_pairs(X, q, rng):
    import bayes as B, analysis as A_
    return A_.pairs(B.Env("T4", K=3, Q=6), X, q, "cond", rng)


def relabel_pairs(X, q):
    Xp = X.copy(); oth = [k for k in range(Q) if k != q]
    Xp[:, oth] = (X[:, oth] + 1 + (np.arange(len(X)) % 2)[:, None, None]) % K; return Xp


# ---------------------------------------------------------------------------------------------- model wrapper
CK = {"P": "step1_relational_mechanism_deepening/checkpoints/ft_pcount_s{}.npz",
      "P0": "step1_foundational_atlas_continuation/checkpoints/ft_pcount.npz",
      "T": "step1_relational_mechanism_deepening/checkpoints/ref_T1p{}.npz",
      "T0": "step1_foundational_atlas_continuation/checkpoints/ref_T1.npz",
      "S": "step1_cross_implementation_causal_equivalence/checkpoints/ft_pcsrc_s{}.npz"}


def ckpt_path(name):
    """'P0','P1'..'P4' P-count transfer seeds; 'T0','T1','T2','T10','T11' T1 transfer parents; 'S5'.. site-test arm."""
    kind, idx = name[0], name[1:]
    if name in ("P0", "T0"): return os.path.join(ROOT, CK[name])
    return os.path.join(ROOT, CK[kind].format(idx))


class Model:
    def __init__(self, name):
        self.name = name; p = C.load(ckpt_path(name)); p.pop("WX", None)
        self.p = jax.tree_util.tree_map(jnp.asarray, p); self.d = int(self.p["Eread"].shape[0]); pp = self.p

        @jax.jit
        def _acts(X):
            rec = {}
            def hk(n, a):
                if n in ("resid_pre2", "attn_out2", "mlp_out2"): rec[n] = a
                return a
            o = F3.forward(pp, X, hk); return rec["resid_pre2"], o["answer"], rec["attn_out2"], rec["mlp_out2"]

        @jax.jit
        def _run(X, delta):
            return F3.forward(pp, X, lambda n, a: a + delta if n == "resid_pre2" else a)["answer"]

        @jax.jit
        def _run_frozen(X, delta, attn_fix, mlp_fix, ma, mm):
            """add delta at the site; at positions with mask ma (mm) replace layer-2 attention (MLP) output by the given
            values (path patching: the effect may then pass only through the remaining paths)."""
            def hk(n, a):
                if n == "resid_pre2": return a + delta
                if n == "attn_out2": return a * (1 - ma[None, :, None]) + attn_fix * ma[None, :, None]
                if n == "mlp_out2": return a * (1 - mm[None, :, None]) + mlp_fix * mm[None, :, None]
                return a
            return F3.forward(pp, X, hk)["answer"]
        self._acts, self._run, self._run_frozen = _acts, _run, _run_frozen

    def acts(self, X, bs=1000):
        outs = [self._acts(jnp.asarray(X[i:i + bs])) for i in range(0, len(X), bs)]
        return tuple(np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(4))

    def run(self, X, delta, bs=500):
        return np.concatenate([np.asarray(self._run(jnp.asarray(X[i:i + bs]), jnp.asarray(delta[i:i + bs], jnp.float32)))
                               for i in range(0, len(X), bs)])

    def run_frozen(self, X, delta, attn_fix, mlp_fix, ma, mm, bs=500):
        f = lambda Z: jnp.asarray(Z, jnp.float32)
        return np.concatenate([np.asarray(self._run_frozen(jnp.asarray(X[i:i + bs]), f(delta[i:i + bs]), f(attn_fix[i:i + bs]),
                               f(mlp_fix[i:i + bs]), f(ma), f(mm))) for i in range(0, len(X), bs)])


def clogit(logits): return logits - logits.mean(-1, keepdims=True)
def softmax(l): e = np.exp(l - l.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------------------------- encoders
def lstsq(Fe, A):
    """per-position least squares with intercept.  Fe (N,f), A (N,d) -> W (f,d) (intercept dropped: only differences used)."""
    return np.linalg.lstsq(np.c_[np.ones(len(Fe)), Fe], A, rcond=None)[0][1:]


def site_features(kind, X, pairs=G.GROUPS):
    """generator features for the encoders.  Returns dict position -> (N,f) features at that position.
    kind: 'P_full'   readout token 36+q gets onehot(n^{(-q)}) (18)
          'T_own'    source token (k,s) gets onehot(n^{(-k)}_{g(s)}) (6)
          'T_full'   source token (k,s) gets onehot(n^{(-k)}) (18)
          'P_own'/'T..' variants with pairs=CROSS give the matched-capacity wrong-variable null."""
    n = loo(X, pairs); out = {}
    if kind == "P_full":
        for q in range(Q): out[36 + q] = onehot(n[:, q])
    elif kind == "T_own":
        for t in range(36): out[t] = onehot(n[:, t // 6, (t % 6) // 2][:, None])
    elif kind == "T_full":
        for t in range(36): out[t] = onehot(n[:, t // 6])
    elif kind == "T_wronggroup":
        for t in range(36): out[t] = onehot(n[:, t // 6, ((t % 6) // 2 + 1) % 3][:, None])
    else:
        raise ValueError(kind)
    return out


def fit_encoder(kind, Xfit, Afit, pairs=G.GROUPS):
    Fd = site_features(kind, Xfit, pairs); return {t: lstsq(Fd[t], Afit[:, t]) for t in Fd}


def encoder_delta(enc, kind, X0, n0, n1, d, positions=None, pairs=G.GROUPS):
    """activation change implementing the abstract intervention n0 -> n1 (both (N,Q,3) leave-k count arrays, i.e. what
    each position's feature would be).  positions: subset of encoder positions to write (default all)."""
    N = len(X0); dl = np.zeros((N, T, d), np.float32)
    for t, W in enc.items():
        if positions is not None and t not in positions: continue
        if kind == "P_full": f0, f1 = onehot(n0[:, t - 36]), onehot(n1[:, t - 36])
        elif kind in ("T_own",): g = (t % 6) // 2; f0, f1 = onehot(n0[:, t // 6, g][:, None]), onehot(n1[:, t // 6, g][:, None])
        elif kind == "T_wronggroup": g = ((t % 6) // 2 + 1) % 3; f0, f1 = onehot(n0[:, t // 6, g][:, None]), onehot(n1[:, t // 6, g][:, None])
        elif kind == "T_full": f0, f1 = onehot(n0[:, t // 6]), onehot(n1[:, t // 6])
        dl[:, t] = (f1 - f0) @ W
    return dl


def site_positions(model_kind, q):
    return [36 + q] if model_kind == "P" else list(range(6 * q, 6 * q + 6))


# ---------------------------------------------------------------------------------------------- decoders
def fit_decoder(model_kind, Xfit, Afit, q_all=True):
    """alpha_M: activations at M's decision-q site -> n^{(-q)} (3 multinomial logistic classifiers, shared over q).
    P: features a[36+q] (d).  T: features concat of question q's six source tokens (6d)."""
    from sklearn.linear_model import LogisticRegression
    n = loo(Xfit); feats, labs = [], []
    for q in range(Q):
        pos = site_positions(model_kind, q); feats.append(Afit[:, pos].reshape(len(Xfit), -1)); labs.append(n[:, q])
    Fe = np.concatenate(feats); L = np.concatenate(labs); mu, sd = Fe.mean(0), Fe.std(0) + 1e-6
    clf = [LogisticRegression(max_iter=2000, C=1.0).fit((Fe - mu) / sd, L[:, g]) for g in range(3)]
    return dict(kind=model_kind, mu=mu, sd=sd, clf=clf)


def decode(dec, A, q):
    """returns expected one-hot (N,18) and argmax counts (N,3) for decision q."""
    pos = site_positions(dec["kind"], q); Fe = (A[:, pos].reshape(len(A), -1) - dec["mu"]) / dec["sd"]
    probs = []
    for g in range(3):
        c = dec["clf"][g]; pr = np.zeros((len(A), 6)); pr[:, c.classes_] = c.predict_proba(Fe); probs.append(pr)
    P = np.stack(probs, 1); return P.reshape(len(A), 18), P.argmax(-1)


# ---------------------------------------------------------------------------------------------- metrics
def score(base_logit, nat_logit, patch_logit, mask=None):
    """IIA on informative cases (argmax under natural counterfactual differs from base), unconditional agreement with
    the natural counterfactual, and Pearson r between patch and natural changes of centred logits (all candidates)."""
    m = np.ones(len(base_logit), bool) if mask is None else mask
    b, nt, pt = base_logit[m], nat_logit[m], patch_logit[m]
    inf = b.argmax(-1) != nt.argmax(-1)
    dn, dp = clogit(nt) - clogit(b), clogit(pt) - clogit(b)
    r = float(np.corrcoef(dn.ravel(), dp.ravel())[0, 1]) if dn.std() > 0 and dp.std() > 0 else float("nan")
    return dict(iia=float((pt.argmax(-1) == nt.argmax(-1))[inf].mean()) if inf.any() else float("nan"), n_inf=int(inf.sum()),
                agree_all=float((pt.argmax(-1) == nt.argmax(-1)).mean()), n=int(m.sum()), r_dlogit=r,
                mean_abs_dlogit_nat=float(np.abs(dn).mean()), mean_abs_dlogit_patch=float(np.abs(dp).mean()))


def adjusted(s, floor, ceil):
    """(s - floor) / (ceil - floor): fraction of the above-floor headroom recovered."""
    return float((s - floor) / (ceil - floor)) if ceil - floor > 1e-9 else float("nan")
