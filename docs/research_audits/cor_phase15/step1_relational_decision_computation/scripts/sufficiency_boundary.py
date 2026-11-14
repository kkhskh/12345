"""Workstream C3: the mathematical boundary of the per-group (deep-sets) sufficient statistic.
Regimes evaluated against the exact shared-copy posterior (xii_bayes.posterior_shared, verified exhaustively in verify_shared_posterior.py):
  R1  C=1, beta=1   : closed form  p ∝ sum_g W_g L_g (S - e_g)               (should be exact)
  R2  C=1, beta<1   : the hard (S - e_g) rule vs the graded per-group rule  p ∝ sum_g W_g L_g prod_{h!=g} col_h/noi   (per-group, exact)
  R3  C=2, beta=1   : best PER-GROUP-PRODUCT rule (independent copying; genlib.posterior_brute) and the C=1 rule vs exact; split by S=3 vs S<=2
  R4  C=2, beta<1   : same
plus the rank argument: the shared colluder factor chi(x_h1, x_h2) restricted to self-agreeing pairs is beta/K * I_K + const, rank K for beta>0,
hence not a product psi(x_h1) psi(x_h2) -- any per-group-product rule must be wrong somewhere (counterexample pair printed).
-> logs/sufficiency_boundary.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); N = 100000; out = {}
def c1_rule(X, q, beta_rule):
    """C=1 rule with eligibility e_g (hard, beta_rule=1) or graded colluder factor (beta_rule<1), applied regardless of the true C."""
    env1 = GL.Env(3, 1, beta=beta_rule); return GL.posterior_brute(env1, X, q)
def summ(P, Pt, y, S):
    kl = (Pt * (np.log(Pt + 1e-12) - np.log(P + 1e-12))).sum(-1); hit = P.argmax(-1) == y; hitB = Pt.argmax(-1) == y
    r = dict(KL=float(kl.mean()), acc=float(hit.mean()), bayes_acc=float(hitB.mean()), agree_bayes=float((P.argmax(-1) == Pt.argmax(-1)).mean()))
    for nm, m in (("S3", S == 3), ("S12", (S >= 1) & (S <= 2)), ("S0", S == 0)):
        if m.any(): r[nm] = dict(frac=float(m.mean()), KL=float(kl[m].mean()), acc=float(hit[m].mean()), bayes_acc=float(hitB[m].mean()))
    return r
for C, beta in ((1, 1.0), (1, 0.6), (2, 1.0), (2, 0.6)):
    env = GL.Env(3, C, beta=beta); rng = np.random.default_rng(76_000_000 + 10 * C + int(10 * beta)); d = GL.sample(env, N, rng); X = d["X"]
    q = rng.integers(0, 6, N); y = d["Y"][np.arange(N), q]; Pt = XB.posterior_shared(env, X, q)
    a_ = X.reshape(N, 6, 3, 2); S = ((GL.counts(X, 3)[np.arange(N), q] == 5) & (a_[np.arange(N), q, :, 0] == a_[np.arange(N), q, :, 1])).sum(-1)
    r = dict(hard_C1_rule=summ(c1_rule(X, q, 1.0), Pt, y, S))
    if beta < 1: r["graded_C1_rule_true_beta"] = summ(c1_rule(X, q, beta), Pt, y, S)
    if C == 2: r["independent_copy_product_rule"] = summ(GL.posterior_brute(env, X, q), Pt, y, S)
    out[f"C{C}_beta{beta}"] = r
    print(f"C{C} beta{beta}", json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else {a: round(b, 3) for a, b in vv.items()}) for kk, vv in v.items()} for k, v in r.items()}), flush=True)
# rank argument + explicit counterexample (C=2, beta=1): chi restricted to self-agreeing colluder pairs (a,a),(b,b)
K = 3; beta = 1.0; chi = np.array([[beta * (a == b) / K + (1 - beta) / K ** 4 for b in range(K)] for a in range(K)])
out["rank_argument"] = dict(chi_selfagreeing=chi.tolist(), rank=int(np.linalg.matrix_rank(chi)), singular_values=np.linalg.svd(chi)[1].tolist())
# counterexample (beta < 1): two episodes with IDENTICAL per-group statistics for every group -- e_g = 1 (all groups self-agreeing at all six
# questions, hence identical W_g) and identical values at q (hence identical L_g(y)) -- that differ ONLY in cross-group identity at q' != q.
# Any rule of the form F({(e_g, W_g, L_g)}_g) must give them the same output; the exact posteriors differ.  (At beta = 1 the second episode
# is off-support: there the cross-group equality AT q already fixes the colluder pair on the support -- see the document.)
A = [0, 1, 2, 0, 1, 2]; B2 = [0, 2, 0, 1, 2, 0]; Cc = [1, 0, 0, 2, 0, 1]
mk = lambda a, b, c: np.array([[[a[t]] * 2 + [b[t]] * 2 + [c[t]] * 2 for t in range(6)]]).reshape(1, 6, 6)
x1, x2 = mk(A, A, Cc), mk(A, B2, Cc)
ce = {}
for beta in (0.6, 1.0):
    env = GL.Env(3, 2, beta=beta); p1 = XB.posterior_shared(env, x1, np.array([0])); p2 = XB.posterior_shared(env, x2, np.array([0]))
    pb1 = GL.posterior_brute(env, x1, np.array([0])); pb2 = GL.posterior_brute(env, x2, np.array([0]))
    ce[f"beta{beta}"] = dict(exact_x1=p1[0].tolist(), exact_x2=p2[0].tolist(), indep_x1=pb1[0].tolist(), indep_x2=pb2[0].tolist())
    print("counterexample beta", beta, "exact", np.round(p1[0], 3), np.round(p2[0], 3), "| per-group-product", np.round(pb1[0], 3), np.round(pb2[0], 3))
out["counterexample"] = dict(x1=x1[0].tolist(), x2=x2[0].tolist(), q=0, **ce)
print("rank", out["rank_argument"]["rank"])
json.dump(out, open(os.path.join(HERE, "..", "logs", "sufficiency_boundary.json"), "w"), indent=1)
