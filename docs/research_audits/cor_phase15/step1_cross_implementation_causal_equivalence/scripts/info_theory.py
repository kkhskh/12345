"""Exact information structure of the T4 generator (mandate §5.1, theorem target A).  No learned model is used.
Checks, by exhaustive enumeration and on sampled episodes (stream 12_000_000):
 S1  P(X_k | R) depends on X_k only through the agreement indicators A_k (and a constant): the relational component
     pi_{-q} = P(R | X_{-q}) is a function of the leave-q count vector n = n^{(-q)} in {0..5}^3; formula
     pi(n)(g,h) ∝ rho^{n_g} [n_h = 5], rho = f(1)/f(0).   (verified against the generator's exact posterior)
 S2  minimal sufficient statistic: the classes of n with equal pi; number of classes; examples of distinct n with equal pi.
 S3  insufficiency of one component: Bayes accuracy with (X_q, n) vs (X_q, n_g) for each g vs X_q alone, exact
     conditionals from P(n | R) = prod_g Binom(5, a_role(g)) (colluder: point mass at 5).
 S4  the tuple of own-group components carried by question q's six source tokens, (n_{g(s)})_{s=0..5}, is a bijective
     image of n (each component twice); any proper sub-tuple missing a group is insufficient (S3).
 S5  count-preserving and pi-preserving edits: frequency on natural data of episodes admitting an input-level
     pi-preserving count change, and exact invariance of Bayes under them.
Output: logs/info_theory.json"""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eqv as E

f = E.F4; out = {}
# S1 ------------------------------------------------------------------------------------------------------------
fr = E.pair_marg_rel(); out["f_rel(agree)"], out["f_rel(disagree)"] = fr; out["rho"] = fr[0] / fr[1]
D = E.sample(4000, 12_000_000); X = D["X"]; post = E.G.posterior(f, X); n = E.loo(X)
pi_formula = E.pi_of_n(n); err = float(np.nanmax(np.abs(pi_formula - post["pi_minus"])))
out["S1_max_abs_err_pi_formula_vs_generator"] = err
py = E.bayes_from(X, n); out["S1_max_abs_err_bayes_from_counts"] = float(np.abs(py - post["post_y"]).max())
# S2 ------------------------------------------------------------------------------------------------------------
allN = np.array(list(itertools.product(range(6), repeat=3)))
supp = (allN == 5).any(1); P = E.pi_of_n(allN[supp][:, None, :])[:, 0]
keys = {}
for v, pv in zip(allN[supp], P):
    keys.setdefault(tuple(np.round(pv, 12)), []).append(tuple(int(x) for x in v))
out["S2_support_count_vectors"] = int(supp.sum()); out["S2_distinct_pi"] = len(keys)
out["S2_example_classes"] = [c for c in keys.values() if len(c) > 1][:6]
# S3 ------------------------------------------------------------------------------------------------------------
Y = D["Y"]; J = E.G.joint(f, X)                                             # N,Q,C,K
Pn_R = E.p_counts_given_R(allN)                                             # 216 x C
prior = f.prior
def acc_of(py): return float((py.argmax(-1) == Y).mean())
res = dict(full=acc_of(post["post_y"]), own_only=acc_of(post["own_y"]))
idx = {tuple(v): i for i, v in enumerate(allN)}
for g in range(3):                       # P(R | n_g) ∝ prior(R) * P(n_g | R)
    pr = np.zeros(n.shape[:2] + (len(prior),))
    for c in range(len(prior)):
        marg = np.zeros(6)
        for i, v in enumerate(allN): marg[v[g]] += Pn_R[i, c]
        pr[..., c] = prior[c] * marg[n[..., g]]
    pr /= pr.sum(-1, keepdims=True)
    pyg = (pr[..., None] * J).sum(2); pyg /= pyg.sum(-1, keepdims=True); res[f"single_component_g{g}"] = acc_of(pyg)
for pair in itertools.combinations(range(3), 2):
    pr = np.zeros(n.shape[:2] + (len(prior),))
    for c in range(len(prior)):
        marg = np.zeros((6, 6))
        for i, v in enumerate(allN): marg[v[pair[0]], v[pair[1]]] += Pn_R[i, c]
        pr[..., c] = prior[c] * marg[n[..., pair[0]], n[..., pair[1]]]
    pr /= pr.sum(-1, keepdims=True)
    pyg = (pr[..., None] * J).sum(2); pyg /= pyg.sum(-1, keepdims=True); res[f"two_components_{pair}"] = acc_of(pyg)
out["S3_bayes_accuracy"] = res
# S5 ------------------------------------------------------------------------------------------------------------
Xe, ok = E.pi_preserving_edit(X, 0, np.random.default_rng(12_000_010))
out["S5_frac_episodes_admitting_pi_preserving_edit_q0"] = float(ok.mean())
pe = E.bayes_from(Xe[ok], E.loo(Xe[ok]))[:, 0]; pb = post["post_y"][ok][:, 0]
out["S5_max_abs_bayes_change_under_pi_preserving_edit"] = float(np.abs(pe - pb).max())
out["S5_mean_count_change_L1"] = float(np.abs(E.loo(Xe[ok])[:, 0] - n[ok][:, 0]).sum(-1).mean())
json.dump(out, open(os.path.join(HERE, "..", "logs", "info_theory.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
