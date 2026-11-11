"""Exact-Bayes reference for the 4→5 boundary and the control transitions (mandate §4), plus the Proposition 3 repair checks (§3).
No learned model.  Writes logs/exact_boundary.json.
Part A — boundary regimes.  On 20,000 natural episodes (stream 14_000_000), one scored decision q each, the generator-defined edits
F45 / F34 / F23 (one group's leave-q count raised by one through a single report) are applied; for each pair: support checks,
exact Bayes oriented response Δm_B toward the edited group's own candidate c_g (when its q-sources agree), role-posterior changes,
rank changes, and a regime classification:
  context H_before = groups already at count 5 (the edited group becomes a colluder candidate at 5);
  c_g relation to the other groups' votes at q (agrees with the colluder candidate's vote / with the third group / with none).
Part B — Proposition 3 repair: (i) the reported counterexample (phi_rel = 0 on 0..4, phi_rel(5) = 1, phi_col ≡ 0) gives zero response
on every minimal null edit and non-zero responses on F45 edits; (ii) the corrected characterisation: under nondegeneracy (the six
configuration likelihood vectors determine the normalised mixture weights from P(y | X_q) over X_q — checked numerically), a member of
the Bayes-mixture class is null-invariant iff either (a) the gate is hard on 0..4 and phi_rel is affine on 0..4, or (b) phi_rel and
phi_col are both constant on 0..4; (iii) random members of each regime and random violators are tested."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import boundary as B
R, E, G = B.R, B.E, B.G
OUT = {}
D = E.sample(20000, 14_000_000); X, cfg = D["X"], D["cfg"]; rng = np.random.default_rng(14_000_001); q = rng.integers(0, 6, len(X)); idx = np.arange(len(X))
c0 = B.bayes_c(X, q)
for fam, (frm, to) in {"F45": (4, 5), "F34": (3, 4), "F23": (2, 3)}.items():
    Xp = X.copy(); gs = np.full(len(X), -1); ks = gs.copy(); ok = np.zeros(len(X), bool)
    for qq in range(6):
        s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(14_000_010 + qq + 100 * frm), frm, to)
        Xp[s], gs[s], ks[s], ok[s] = a, g_, k_, o
    c1 = B.bayes_c(Xp, q); vote = np.array([B.group_vote(X[i:i + 1], q[i], max(gs[i], 0))[0] for i in idx])
    orient = ok & (vote >= 0)
    dm = B.oriented_margin(c1, np.maximum(vote, 0)) - B.oriented_margin(c0, np.maximum(vote, 0))
    pi0, f0 = B.role_posterior(X, q); pi1, f1 = B.role_posterior(Xp, q)
    n0 = E.loo(X)[idx, q]; Hb = (n0 == 5)
    # regime: relation of c_g to the votes of the other two groups at q
    def votes(Z): return np.stack([B.group_vote(Z, None, g) if False else np.array([B.group_vote(Z[i:i + 1], q[i], g)[0] for i in idx]) for g in range(3)], 1)
    V = votes(X); rel = np.full(len(X), "none", object)
    for i in np.where(orient)[0]:
        oth = [h for h in range(3) if h != gs[i]]; col = [h for h in oth if Hb[i, h]]; thr = [h for h in oth if not Hb[i, h]]
        if any(V[i, h] == vote[i] for h in col): rel[i] = "agrees_with_colluder_candidate"
        elif any(V[i, h] == vote[i] for h in thr): rel[i] = "agrees_with_third_group"
        else: rel[i] = "agrees_with_no_group"
    nH = Hb.sum(1)
    reg = {}
    for h_ in (1, 2):
        for r_ in ("agrees_with_colluder_candidate", "agrees_with_third_group", "agrees_with_no_group"):
            m = orient & (nH == h_) & (rel == r_)
            if m.sum() < 20: continue
            d = dm[m]; reg[f"H_before={h_}|{r_}"] = dict(n=int(m.sum()), mean_dm=float(d.mean()), frac_pos=float((d > 1e-9).mean()),
                                                     frac_neg=float((d < -1e-9).mean()), frac_zero=float((np.abs(d) <= 1e-9).mean()),
                                                     q10=float(np.percentile(d, 10)), q90=float(np.percentile(d, 90)))
    dpi = np.abs(pi1 - pi0).sum(-1); dfull = np.abs(f1 - f0).sum(-1); dprob = np.abs(np.exp(c1) / np.exp(c1).sum(-1, keepdims=True) - np.exp(c0) / np.exp(c0).sum(-1, keepdims=True)).sum(-1)
    rank = (c1.argmax(-1) != c0.argmax(-1))
    OUT[fam] = dict(valid=float(ok.mean()), oriented=float(orient.mean()),
                    marginal_support=float(E.episode_support(Xp)[ok].mean()), same_world=float(E.same_world(Xp, cfg)[ok].mean()),
                    xq_fixed=bool((Xp[idx, q] == X[idx, q]).all()), reports_changed=int(np.median((Xp != X).sum((1, 2))[ok])),
                    counts_changed_exactly_one_group_by_one=bool((np.abs(E.loo(Xp)[idx, q] - n0)[ok].sum(1) == 1).all()),
                    mean_dm=float(dm[orient].mean()), frac_dm_pos=float((dm[orient] > 1e-9).mean()), frac_dm_neg=float((dm[orient] < -1e-9).mean()),
                    frac_pi_unchanged=float((dpi[ok] < 1e-12).mean()), frac_role_post_changed_but_prob_unchanged=float(((dfull[ok] > 1e-9) & (dprob[ok] < 1e-9)).mean()),
                    frac_prob_changed_without_rank_change=float(((dprob[ok] > 1e-9) & ~rank[ok]).mean()), frac_rank_reversal=float(rank[ok].mean()),
                    regimes=reg)
    if fam == "F45":
        Xc = B.token_control(X, gs, ks); cc = B.bayes_c(Xc, q)
        OUT["F0"] = dict(valid=float(ok.mean()), V_preserved=bool((E.loo(Xc)[idx, q] == n0)[ok].all()), bayes_preserved_max=float(np.abs(cc - c0)[ok].max()),
                         same_world=float(E.same_world(Xc, cfg)[ok].mean()), reports_changed=int(np.median((Xc != X).sum((1, 2))[ok])),
                         same_token_as_F45=bool(((Xc != X) == (Xp != X))[ok].all()))
# ---------------------------------------------------------------- Part B: Proposition 3 repair
SUPP = np.array([v for v in itertools.product(range(6), repeat=3) if 5 in v])
Xq = D["X"][:3000]; qq = np.zeros(3000, int); Jq = np.concatenate([R.J_at(Xq[i:i + 1], 0) for i in range(3000)])
def mix(th_rel, th_col, Vs):                                       # th_rel, th_col: length-6 arrays over counts 0..5
    w = np.exp(np.asarray(th_rel)[Vs[:, R.G_OF]] + np.asarray(th_col)[Vs[:, R.H_OF]])
    P = (w[:, :, None] * Jq).sum(1); L = np.log(P); return L - L.mean(-1, keepdims=True)
def null_pairs():
    pr = []
    for v in SUPP:
        H = np.where(v == 5)[0]
        if len(H) != 1: continue
        o = [g for g in range(3) if g != H[0]]
        for t in (1, -1):
            w = v.copy(); w[o] += t
            if all(0 <= w[g] <= 4 for g in o): pr.append((v, w))
    return pr
NP = null_pairs()
Aq = E.agree(Xq)[:, 0]                                             # own-question agreements of the 3000 own-question patterns
def natural(v): return ((np.asarray(v)[None] == 5) & (Aq == 1)).any(1)   # P(X_q, v) > 0 under the generator
def max_null_resp(fr, fc):
    out = 0.0
    for v, w in NP:
        m = natural(v) & natural(w)
        if m.any(): out = max(out, float(np.abs(mix(fr, fc, np.repeat(v[None], 3000, 0)) - mix(fr, fc, np.repeat(w[None], 3000, 0)))[m].max()))
    return out
def f45_resp(fr, fc):
    out = 0.0
    for v in SUPP:
        for g in range(3):
            if v[g] == 4:
                w = v.copy(); w[g] = 5; m = natural(v) & natural(w)
                if m.any(): out = max(out, float(np.abs(mix(fr, fc, np.repeat(v[None], 3000, 0)) - mix(fr, fc, np.repeat(w[None], 3000, 0)))[m].max()))
    return out
cx_rel, cx_col = [0, 0, 0, 0, 0, 1.0], [0] * 6
OUT["prop3_counterexample"] = dict(max_abs_null_response=max_null_resp(cx_rel, cx_col), max_abs_F45_response=f45_resp(cx_rel, cx_col), n_null_pairs=len(NP))
r = np.random.default_rng(5); tests = {}
HARD = -60.0
def rnd(kind):
    if kind == "hard_affine": a, b = r.normal(), r.normal(); return [a + b * n for n in range(5)] + [r.normal() * 3], [HARD] * 5 + [0]
    if kind == "finite_constant": return [r.normal()] * 5 + [r.normal() * 3], [r.normal()] * 5 + [r.normal()]
    if kind == "hard_nonaffine": return list(r.normal(size=5)) + [r.normal()], [HARD] * 5 + [0]
    if kind == "finite_affine_rel_constant_col": a, b = r.normal(), 0.3 + abs(r.normal()); return [a + b * n for n in range(5)] + [r.normal()], [r.normal()] * 5 + [0]
    if kind == "finite_constant_rel_affine_col": a, b = r.normal(), 0.3 + abs(r.normal()); return [r.normal()] * 5 + [r.normal()], [a + b * n for n in range(5)] + [0]
tests["exact_bayes"] = [max_null_resp([n * np.log(E.RHO) for n in range(6)], [HARD] * 5 + [0])]
for kind in ("hard_affine", "finite_constant", "hard_nonaffine", "finite_affine_rel_constant_col", "finite_constant_rel_affine_col"):
    tests[kind] = [max_null_resp(*rnd(kind)) for _ in range(5)]
OUT["prop3_characterisation_tests"] = tests
# nondegeneracy: can two different normalised weight vectors give identical P(y | X_q) for every X_q?  Check rank of the linear system
# sum_R (w_R - lam(x) w'_R) J(x,R,y) = 0 for random w: identifiability <=> the only solution is w' ∝ w.
Xall = np.array(list(itertools.product(range(3), repeat=6)))[:, None, :].repeat(6, 1)             # all 729 own-question patterns
Jall = G.joint(B.F4, Xall)[:, 0]                                                                  # (729, 6, 3)
keep = Jall.sum((1, 2)) > 0; Jall = Jall[keep]                                                  # own-question patterns with positive probability
w = r.random(6); w /= w.sum(); Pw = (w[None, :, None] * Jall).sum(1); Pw /= Pw.sum(-1, keepdims=True)
# unknowns: w' (6) and lam(x) (729): equations J(x)^T w' - lam(x) Pw(x) = 0  (729*3 equations)
nx = len(Jall); A = np.zeros((nx * 3, 6 + nx))
for x in range(nx):
    A[3 * x:3 * x + 3, :6] = Jall[x].T; A[3 * x:3 * x + 3, 6 + x] = -Pw[x]
sv = np.linalg.svd(A, compute_uv=False)
OUT["nondegeneracy"] = dict(n_unknowns=A.shape[1], rank=int((sv > 1e-10 * sv[0]).sum()), null_dim=int(A.shape[1] - (sv > 1e-10 * sv[0]).sum()),
                            note="null_dim == 1 means the mixture weights are identified up to scale from P(y|X_q) over all X_q")
json.dump(OUT, open(os.path.join(HERE, "..", "logs", "exact_boundary.json"), "w"), indent=1, default=float)
print(json.dumps({k: (v if k not in ("F45", "F34", "F23") else {kk: vv for kk, vv in v.items() if kk != "regimes"}) for k, v in OUT.items()}, indent=1, default=float))
for k in ("F45", "F34", "F23"): print(k, json.dumps(OUT[k]["regimes"], indent=0))
