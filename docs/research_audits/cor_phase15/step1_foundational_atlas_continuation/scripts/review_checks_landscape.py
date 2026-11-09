"""Review-response checks (R2 stat F1/F3/F9/F11/F15, formal F1/F7/F13). Writes logs/landscape/review_checks.json.
 (1) rho -> 1 flip rate measured only on cells where the posterior is defined (NaN-guarded), steps 0.9 -> 0.95 -> 0.99 -> 0.999 -> 1.
 (2) one-vote threshold under biased errors, split by conflict direction (v = c+1 vs v = c+2), with the predicted thresholds.
 (3) Theorem U scope, EXACT at fixed other-group counts (count sufficiency + binomial likelihoods; no Monte Carlo).
 (4) misleading-relation reference: T4-Bayes plug-in disagreement with true Bayes on p_c = 0 episodes with a perfect group."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_computation_atlas", "src"))
import numpy as np, roles_engine as RE, gen_family as G
from scipy.stats import binom
K = 3; out = {}
def gen(roles):
    ps = sorted(set(itertools.permutations(roles))); return RE.Gen(ps, [1.0] * len(ps), K, 6)
# (1)
X = gen([("rel", 0.8, 0.0, 0.0), ("coll", 0.6, None), ("noise",)]).sample(3000, np.random.default_rng(8_400_000))["X"]
prev = None; rows = []
for rho in (0.9, 0.95, 0.99, 0.999, 1.0):
    with np.errstate(all="ignore"):
        py = gen([("rel", 0.8, 0.0, 0.0), ("coll", float(rho), None), ("noise",)]).posterior(X)["post_y"]
    ok = np.isfinite(py).all(-1); a = np.where(ok, np.nan_to_num(py, nan=0).argmax(-1), -1)
    if prev is not None:
        both = ok & prev[1]; rows.append(dict(rho=rho, frac_defined=float(ok.mean()), flip_rate_on_defined=float((a != prev[0])[both].mean())))
    prev = (a, ok)
out["rho_to_1_guarded"] = rows; print(rows, flush=True)
# (2)
rows = []
for ac in (0.85, 0.9, 0.905, 0.91, 0.92, 0.97, 0.995, 0.999):
    g = gen([("rel", 0.8, 0.8, 0.0), ("coll", 1.0, ac), ("noise",)]); d = g.sample(8000, np.random.default_rng(8_600_001)); Xs = d["X"]
    with np.errstate(all="ignore"):
        py = g.posterior(Xs)["post_y"].argmax(-1)
    cf = [g.configs[c] for c in d["cfg"]]; A_ = np.array(RE.GROUPS); n = np.arange(len(Xs))
    ci = np.array([[r[0] for r in c].index("coll") for c in cf]); ri = np.array([[r[0] for r in c].index("rel") for c in cf])
    xc = Xs[n, :, A_[ci, 0]]; r0 = Xs[n, :, A_[ri, 0]]; r1 = Xs[n, :, A_[ri, 1]]; conf = (r0 == r1) & (r0 != xc)
    up = conf & (r0 == (xc + 1) % K); dn = conf & (r0 == (xc + 2) % K)
    rows.append(dict(a_c=ac, follow_when_v_eq_c_plus_1=float((py == xc)[up].mean()), follow_when_v_eq_c_plus_2=float((py == xc)[dn].mean()), n_up=int(up.sum()), n_dn=int(dn.sum())))
# error mass from y = c to v: bias b = 0.8 sends (1-alpha)*b to (c+1); the rest (1-alpha)(1-b) is split uniformly over the K-1 wrong values
alpha, b = 0.8, 0.8
e_to_c1 = (1 - alpha) * (b + (1 - b) / 2); e_to_c2 = (1 - alpha) * (1 - b) / 2
def thr(e):
    r = (alpha / e) ** 2                 # pair likelihood ratio for y = v vs y = c
    return r / (r + (K - 1))             # colluder's single-vote ratio a_c (K-1)/(1-a_c) must exceed r
out["biased_one_vote"] = dict(rows=rows, predicted_threshold_v_c_plus_1=float(thr(e_to_c1)), predicted_threshold_v_c_plus_2=float(thr(e_to_c2)))
print(out["biased_one_vote"], flush=True)
# (3)
def a_of(spec):
    if spec[0] == "rel": al = spec[1]; return al ** 2 + (1 - al) ** 2 / 2
    if spec[0] == "noise": return 1 / 3
    return spec[1] + (1 - spec[1]) / 3
cfgs = []
for al in (0.45, 0.99):
    cfgs += sorted(set(itertools.permutations([("rel", al, 0.0, 0.0), ("coll", 0.6, None), ("noise",)])))
viol = 0; tot = 0; examples = []
for n1 in range(7):
    for n2 in range(7):
        v = []
        for n0 in range(7):
            w = np.array([np.prod([binom.pmf(n, 6, a_of(s)) for n, s in zip((n0, n1, n2), c)]) for c in cfgs])
            v.append(w[[c[0][0] == "rel" for c in cfgs]].sum() / w.sum())
        v = np.array(v)
        qc = all(not (v[i] > v[i + 1] + 1e-12 and (v[i + 2:] > v[i + 1] + 1e-12).any()) for i in range(5))
        tot += 1; viol += (not qc)
        if not qc and len(examples) < 3: examples.append(dict(n1=n1, n2=n2, P_rel_any_by_n0=[round(float(x), 4) for x in v]))
single = 0
for al in (0.45, 0.99):
    cf1 = sorted(set(itertools.permutations([("rel", al, 0.0, 0.0), ("coll", 0.6, None), ("noise",)])))
    for n1 in range(7):
        for n2 in range(7):
            v = []
            for n0 in range(7):
                w = np.array([np.prod([binom.pmf(n, 6, a_of(s)) for n, s in zip((n0, n1, n2), c)]) for c in cf1])
                v.append(w[[c[0][0] == "rel" for c in cf1]].sum() / w.sum())
            v = np.array(v); single += not all(not (v[i] > v[i + 1] + 1e-12 and (v[i + 2:] > v[i + 1] + 1e-12).any()) for i in range(5))
out["U_scope_exact_fixed_context"] = dict(contexts=tot, non_quasi_concave_union=int(viol), non_quasi_concave_single_role_generators=int(single), examples=examples)
print(out["U_scope_exact_fixed_context"], flush=True)
# (4)
f1, plug4 = G.Fam(pc=0.0), G.Fam(pc=1.0, rho=0.999); D = G.sample(f1, 3000, np.random.default_rng(9_400_000)); X1 = D["X"]
tb = G.posterior(f1, X1)["post_y"].argmax(-1); pb4 = np.nan_to_num(G.posterior(plug4, X1)["post_y"], nan=0).argmax(-1)
perf = np.stack([(X1[:, :, a] == X1[:, :, b]).all(1) for a, b in G.GROUPS], -1).any(-1)
out["misleading_reference_T4_plugin"] = dict(perfect_group=float((pb4 != tb)[perf].mean()), other=float((pb4 != tb)[~perf].mean())); print(out["misleading_reference_T4_plugin"])
json.dump(out, open(os.path.join(HERE, "..", "logs", "landscape", "review_checks.json"), "w"), indent=1)
