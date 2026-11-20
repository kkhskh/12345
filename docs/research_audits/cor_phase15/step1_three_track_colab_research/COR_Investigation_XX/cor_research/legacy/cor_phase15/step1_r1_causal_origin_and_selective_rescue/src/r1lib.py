"""Investigation VII library.  Exact T4 quantities in closed form, restricted evidence-update rules, supported edit families
(F45/F34/F23/F0, nested 3->4->5 sequences, matched controls), and model wrappers.  Re-uses Investigation VI's boundary.py
(and through it Investigation V's cri.py / IV's eqv.py), read-only."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_perfect_agreement_boundary", "src"))
import numpy as np
import boundary as B
R, E, G, F4 = B.R, B.E, B.G, B.F4
K, Q, RHO, ALPHA = 3, 6, float(E.RHO), 0.8
EPS = (1 - ALPHA) / (K - 1)
GROUPS = G.GROUPS


def votes(X, q):
    """(N,3) common vote of each group at the scored question q (int or per-episode array), -1 if the pair disagrees."""
    Xq = X[np.arange(len(X)), q] if np.ndim(q) else X[:, q]
    return np.stack([np.where(Xq[:, a] == Xq[:, b], Xq[:, a], -1) for a, b in GROUPS], 1)


def Lrel(X, q):
    """(N,3,K) reliable-pair likelihood of each group's q-reports: prod over its two sources of (alpha if x==y else eps)."""
    out = np.ones((len(X), 3, K)); ys = np.arange(K); Xq = X[np.arange(len(X)), q] if np.ndim(q) else X[:, q]
    for g, (a, b) in enumerate(GROUPS):
        for s in (a, b):
            out[:, g] *= np.where(Xq[:, s][:, None] == ys[None], ALPHA, EPS)
    return out


def closed_form_weights(n, agree_q):
    """Exact Bayes mixture weights over 'which group is reliable' (pc = 1, rho = 1):
    W_g = RHO^{n_g} * m_g,  m_g = #{h != g : n_h == 5 and h's pair agrees at q}  (h must be a possible colluder).
    n: (N,3) leave-q counts, agree_q: (N,3) bool."""
    perf = (n == 5) & agree_q
    m = perf.sum(1, keepdims=True) - perf
    return RHO ** n * m, m


def mixture_logits(W, L):
    p = (W[:, :, None] * L).sum(1); p = p / p.sum(-1, keepdims=True)
    with np.errstate(divide="ignore"): lp = np.log(p)
    return lp - lp.mean(-1, keepdims=True)


def rule_logits(rule, X, q, **kw):
    """Restricted evidence-update rules, all of the form p(y) ∝ sum_g W_g * L_g(y) (L = exact reliable-pair likelihood):
    bayes        : W_g = RHO^n_g * m_g                        (exact)
    countonly    : W_g = RHO^n_g                              (independent reliability weighting, no collusion: T1's Bayes)
    exclude5     : W_g = RHO^n_g * [n_g < 5]                  (categorical perfect-agreement discount: a perfect group is a colluder)
    groupwise    : W_g = w[n_g]                               (any own-count-only weighting; kw w = length-6 array)
    softdisc     : W_g = RHO^n_g * delta^{[n_g = 5]}            (groupwise, perfect agreement discounted by delta)
    hcount       : W_g = w[n_g] * u[|H|]                       (own count x global number of perfect groups; still separable)
    interaction  : W_g = w[n_g] * v[m_g]                       (own count x number of OTHER perfect groups: Bayes has w=RHO^n, v=m)"""
    idx = np.arange(len(X)); n = E.loo(X)[idx, q]; agq = votes(X, q) >= 0; L = Lrel(X, q)
    if rule == "bayes": W = closed_form_weights(n, agq)[0]
    elif rule == "countonly": W = RHO ** n
    elif rule == "exclude5": W = RHO ** n * (n < 5)
    elif rule == "groupwise": W = np.asarray(kw["w"])[n]
    elif rule == "softdisc": W = RHO ** n * np.where(n == 5, kw["delta"], 1.0)
    elif rule == "hcount":
        H = ((n == 5) & agq).sum(1, keepdims=True); W = np.asarray(kw["w"])[n] * np.asarray(kw["u"])[H]
    elif rule == "interaction":
        m = closed_form_weights(n, agq)[1]; W = np.asarray(kw["w"])[n] * np.asarray(kw["v"])[m]
    else: raise ValueError(rule)
    W = W.astype(float); W = np.where(W.sum(1, keepdims=True) > 0, W, 1.0)   # degenerate all-zero -> uniform over groups
    return mixture_logits(W, L)


def regime_labels(X, q, gs, vote):
    """Generator-derived regime of a 4->5 (or k->k+1) pair, from the BEFORE episode: R1 c_g agrees with a group already at 5
    (agreeing at q); R2 c_g agrees only with a group below 5; R3 c_g agrees with no group; '' if not oriented."""
    N = len(X); idx = np.arange(N); n = E.loo(X)[idx, q]; V = votes(X, q); out = np.full(N, "", object)
    for i in range(N):
        if gs[i] < 0 or vote[i] < 0: continue
        oth = [h for h in range(3) if h != gs[i]]
        if any(n[i, h] == 5 and V[i, h] == vote[i] for h in oth): out[i] = "R1"
        elif any(n[i, h] < 5 and V[i, h] == vote[i] for h in oth): out[i] = "R2"
        else: out[i] = "R3"
    return out


def nested_sequence(X, q, cfg, rng):
    """Nested 3->4->5 sequences: choose (from the generator only) a non-colluder group g with leave-q count 3 whose q-pair agrees;
    its two disagreeing questions k1, k2 (!= q) are repaired one at a time (second source := first).  X3 -> X4 -> X5 share all other
    reports; g and c_g are the same at every stage.  Returns X3, X4, X5, g, ok."""
    N = len(X); qv = np.broadcast_to(q, (N,)); n = E.loo(X)[np.arange(N), qv]; A = E.agree(X); V = votes(X, qv)
    X4, X5 = X.copy(), X.copy(); gs = np.full(N, -1); ok = np.zeros(N, bool)
    for i in range(N):
        cand = [g for g in range(3) if n[i, g] == 3 and V[i, g] >= 0]
        if not cand: continue
        g = cand[rng.integers(len(cand))]; dis = [k for k in range(Q) if k != qv[i] and A[i, k, g] == 0]
        order = rng.permutation(dis); a, b = GROUPS[g]
        X4[i, order[0], b] = X4[i, order[0], a]; X5[i] = X4[i]; X5[i, order[1], b] = X5[i, order[1], a]; gs[i] = g; ok[i] = True
    return X, X4, X5, gs, ok


# ------------------------------------------------------------------ evaluation sets (generator side)
def loc_control(X, q, rng):
    """FLOC (agreement-location control, Bayes-null, edit distance 2): a group g with leave-q count 4 moves its single disagreement
    from question k1 to another question k2 (both != q): repair k1 (second source := first) and break k2 (second source := the value
    differing from the first, chosen uniformly).  Counts, V and exact Bayes are unchanged."""
    N = len(X); qv = np.broadcast_to(q, (N,)); n = E.loo(X)[np.arange(N), qv]; A = E.agree(X); Xc = X.copy(); gs = np.full(N, -1); ok = np.zeros(N, bool)
    for i in range(N):
        cand = [g for g in range(3) if n[i, g] == 4]
        if not cand: continue
        g = cand[rng.integers(len(cand))]; a, b = GROUPS[g]
        k1 = [k for k in range(Q) if k != qv[i] and A[i, k, g] == 0][0]; k2s = [k for k in range(Q) if k != qv[i] and k != k1 and A[i, k, g] == 1]
        k2 = k2s[rng.integers(len(k2s))]; Xc[i, k1, b] = Xc[i, k1, a]; Xc[i, k2, b] = (Xc[i, k2, a] + rng.integers(1, K)) % K; gs[i] = g; ok[i] = True
    return Xc, gs, ok


def id_control(X, q, rng):
    """FID (report-identity control, Bayes-null, edit distance 2): in a question k != q where a non-perfect group g agrees on v, both
    sources are changed to the same v' != v.  Agreement pattern, counts and exact Bayes are unchanged."""
    N = len(X); qv = np.broadcast_to(q, (N,)); n = E.loo(X)[np.arange(N), qv]; A = E.agree(X); Xc = X.copy(); gs = np.full(N, -1); ok = np.zeros(N, bool)
    for i in range(N):
        cand = [g for g in range(3) if 1 <= n[i, g] <= 4]
        if not cand: continue
        g = cand[rng.integers(len(cand))]; a, b = GROUPS[g]; ks = [k for k in range(Q) if k != qv[i] and A[i, k, g] == 1]
        k = ks[rng.integers(len(ks))]; v = (X[i, k, a] + rng.integers(1, K)) % K; Xc[i, k, a] = v; Xc[i, k, b] = v; gs[i] = g; ok[i] = True
    return Xc, gs, ok


def other_perfect_drop(X, q, rng):
    """F54o (diagnostic of cross-group dependence): episodes whose leave-q state has exactly two perfect groups that agree with each
    other at q (the R1 'after' state).  One of the two (chosen uniformly) loses perfect agreement: in one question k != q its second
    source is set to a different value (count 5 -> 4).  Orientation: the common vote c of the two groups.  Returns X', dropped group,
    the remaining perfect group, ok."""
    N = len(X); qv = np.broadcast_to(q, (N,)); n = E.loo(X)[np.arange(N), qv]; V = votes(X, qv); Xc = X.copy()
    hd = np.full(N, -1); hk = np.full(N, -1); ok = np.zeros(N, bool)
    for i in range(N):
        perf = [g for g in range(3) if n[i, g] == 5 and V[i, g] >= 0]
        if len(perf) != 2 or V[i, perf[0]] != V[i, perf[1]]: continue
        j = rng.integers(2); h, keep = perf[j], perf[1 - j]; a, b = GROUPS[h]; ks = [k for k in range(Q) if k != qv[i]]; k = ks[rng.integers(len(ks))]
        Xc[i, k, b] = (Xc[i, k, a] + rng.integers(1, K)) % K; hd[i], hk[i], ok[i] = h, keep, True
    return Xc, hd, hk, ok


def build_eval(stream, ne=6000):
    """All generator-side evaluation material for one stream: natural episodes, one scored decision each, all edit families,
    exact-Bayes centred logits, orientation candidates and regimes."""
    D = E.sample(ne, stream); X, cfg = D["X"], D["cfg"]; q = np.random.default_rng(stream + 1).integers(0, 6, ne); idx = np.arange(ne)
    out = dict(X=X, Y=D["Y"], cfg=cfg, q=q); V0 = votes(X, q)
    for name, frm in (("F45", 4), ("F34", 3), ("F23", 2)):
        Xp = X.copy(); gs = np.full(ne, -1); ks = gs.copy(); ok = np.zeros(ne, bool)
        for qq in range(6):
            s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(stream + 10 + qq + 100 * frm), frm, frm + 1); Xp[s], gs[s], ks[s], ok[s] = a, g_, k_, o
        vote = V0[idx, np.maximum(gs, 0)]; orient = ok & (vote >= 0)
        out[f"X_{name}"] = Xp; out[f"vote_{name}"] = np.maximum(vote, 0); out[f"orient_{name}"] = orient; out[f"reg_{name}"] = regime_labels(X, q, np.where(orient, gs, -1), vote).astype(str)
        if name == "F45":
            out["X_F0"] = B.token_control(X, gs, ks); out["vote_F0"] = out["vote_F45"]; out["orient_F0"] = orient; out["reg_F0"] = out["reg_F45"]
    for name, fn in (("FLOC", loc_control), ("FID", id_control)):
        Xc, gs, ok = fn(X, q, np.random.default_rng(stream + 500 + len(name))); vote = V0[idx, np.maximum(gs, 0)]
        out[f"X_{name}"] = Xc; out[f"vote_{name}"] = np.maximum(vote, 0); out[f"orient_{name}"] = ok & (vote >= 0); out[f"reg_{name}"] = np.where(ok & (vote >= 0), "all", "").astype(str)
    X3, X4, X5, gs, ok = nested_sequence(X, q, cfg, np.random.default_rng(stream + 77)); vote = V0[idx, np.maximum(gs, 0)]
    out["X_S4"], out["X_S5"] = X4, X5; out["vote_S"] = np.maximum(vote, 0); out["orient_S"] = ok; out["reg_S"] = regime_labels(X4, q, np.where(ok, gs, -1), np.where(ok, vote, -1)).astype(str)
    # F54o is built on its own natural pool (the R1-after state is ~3 % of decisions): 30000 extra episodes, keep the valid ones
    D2 = E.sample(30000, stream + 900); X2 = D2["X"]; q2 = np.random.default_rng(stream + 901).integers(0, 6, 30000)
    Xd, hd, hk, ok2 = other_perfect_drop(X2, q2, np.random.default_rng(stream + 902))
    out["X_O"] = X2[ok2]; out["q_O"] = q2[ok2]; out["X_F54o"] = Xd[ok2]; out["vote_F54o"] = votes(X2, q2)[np.arange(30000), np.maximum(hk, 0)][ok2]
    out["drop_F54o"] = hd[ok2]; out["keep_F54o"] = hk[ok2]
    for k in [k for k in list(out) if k.startswith("X")]:
        qq = out["q_O"] if k in ("X_O", "X_F54o") else q; out["cB_" + k[2:] if k != "X" else "cB_base"] = B.bayes_c(out[k], qq)
    return out


FAMS = ("F45", "F34", "F23", "F0", "FLOC", "FID")


def evaluate_logits(ev, lg):
    """lg: dict name -> centred logits (N,3) at the scored decision for 'base' and every edited array ('F45', ..., 'S4', 'S5', 'O', 'F54o').
    Returns per-family statistics (Bayes-corrected excess E and raw responses by regime, F0-type magnitude statistics)."""
    res = {}; om = B.oriented_margin
    def stats(dm, db, sel, regs):
        r = {}
        for rg in regs:
            s = sel & (ev_reg == rg) if rg != "all" else sel
            if s.sum() < 5: continue
            e = (dm - db)[s]
            r[rg] = dict(n=int(s.sum()), dmM=float(dm[s].mean()), dmB=float(db[s].mean()), E=float(e.mean()), absE=float(np.abs(e).mean()),
                         abs_dmM=float(np.abs(dm[s]).mean()), rms_dmM=float(np.sqrt((dm[s] ** 2).mean())), med_abs_dmM=float(np.median(np.abs(dm[s]))),
                         q95_abs_dmM=float(np.quantile(np.abs(dm[s]), .95)), frac_sign_agree=float((np.sign(dm[s]) == np.sign(db[s]))[np.abs(db[s]) > 1e-9].mean()) if (np.abs(db[s]) > 1e-9).any() else None)
        return r
    for f in FAMS:
        v = ev[f"vote_{f}"]; o = ev[f"orient_{f}"]; ev_reg = ev[f"reg_{f}"]
        dm = om(lg[f], v) - om(lg["base"], v); db = om(ev[f"cB_{f}"], v) - om(ev["cB_base"], v)
        res[f] = stats(dm, db, o, ("R1", "R2", "R3", "all"))
    v = ev["vote_S"]; o = ev["orient_S"]; ev_reg = ev["reg_S"]
    for a, b, nm in (("base", "S4", "S34"), ("S4", "S5", "S45")):
        dm = om(lg[b], v) - om(lg[a], v); db = om(ev["cB_" + b], v) - om(ev["cB_" + a] if a != "base" else ev["cB_base"], v)
        res[nm] = stats(dm, db, o, ("R1", "R2", "R3", "all"))
    v = ev["vote_F54o"]; o = np.ones(len(v), bool); ev_reg = np.full(len(v), "all")
    dm = om(lg["F54o"], v) - om(lg["O"], v); db = om(ev["cB_F54o"], v) - om(ev["cB_O"], v); res["F54o"] = stats(dm, db, o, ("all",))
    return res


def quality(ev, L_all):
    """L_all: model centred logits for all 6 questions of the natural base episodes (N,6,3)."""
    post = E.bayes_from(ev["X"], E.loo(ev["X"])).reshape(-1, 3); Pm = E.softmax(L_all.reshape(-1, 3)); Y = ev["Y"].reshape(-1)
    return dict(acc=float((Pm.argmax(-1) == Y).mean()), KL=float((post * (np.log(post + 1e-12) - np.log(Pm + 1e-12))).sum(-1).mean()),
                NLL=float(-np.log(Pm[np.arange(len(Y)), Y] + 1e-12).mean()))


def eval_names(): return ["base"] + list(FAMS) + ["S4", "S5", "O", "F54o"]


def eval_inputs(ev):
    m = {"base": (ev["X"], ev["q"])}
    for f in FAMS: m[f] = (ev[f"X_{f}"], ev["q"])
    m["S4"], m["S5"] = (ev["X_S4"], ev["q"]), (ev["X_S5"], ev["q"]); m["O"], m["F54o"] = (ev["X_O"], ev["q_O"]), (ev["X_F54o"], ev["q_O"])
    return m
