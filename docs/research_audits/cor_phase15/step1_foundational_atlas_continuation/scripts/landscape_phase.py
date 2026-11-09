"""Relational generative landscape (mandate §10): exact-Bayes phase diagrams.  Writes logs/landscape/*.json
 (1) reliable-vs-colluder nonidentifiability line: rho*(alpha) where the colluder's agreement rate equals the reliable
     pair's; Bayes accuracy and posterior mass on the true configuration along rho, for alpha in {0.7, 0.8, 0.9}.
 (2) Theorem S without (O): two answer-dependent groups -> search for episodes with equal per-group counts and different
     Bayes posteriors.
 (3) Theorem U scope: mixture over the reliable accuracy -> is P(group = reliable | its count) quasi-concave / monotone?
 (4) biased honest errors and an informative colluder (a_coll vs a_rel): when does the Bayes answer follow the colluder?
 (5) argmax discontinuity along rho: fraction of fixed episodes whose Bayes answer flips as rho moves across a grid."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, roles_engine as RE
OUT = os.path.join(HERE, "..", "logs", "landscape"); K, Q = 3, 6
def perms3(roles):
    return [tuple(p) for p in set(itertools.permutations(roles))]
def gen(roles_list, priors=None, Q=6):
    cf = []; pr = []
    for i, roles in enumerate(roles_list):
        ps = perms3(roles); cf += ps; pr += [(1.0 if priors is None else priors[i]) / len(ps)] * len(ps)
    return RE.Gen(cf, pr, K, Q)
res = {}
# (1)
R1 = {}
for alpha in (0.7, 0.8, 0.9):
    a0 = alpha ** 2 + (1 - alpha) ** 2 / (K - 1); rstar = (K * a0 - 1) / (K - 1); rows = []
    for rho in np.round(np.linspace(0.0, 1.0, 21), 3):
        g = gen([[("rel", alpha, 0.0, 0.0), ("coll", float(rho), None), ("noise",)]])
        d = g.sample(3000, np.random.default_rng(8_000_000 + int(rho * 100))); po = g.posterior(d["X"])
        rows.append(dict(rho=float(rho), bayes_acc=float((po["post_y"].argmax(-1) == d["Y"]).mean()), mass_true_cfg=float(po["post_cfg"][np.arange(3000), d["cfg"]].mean()),
                         mass_true_reliable_group=float(np.mean([po["post_cfg"][i, [j for j, c in enumerate(g.configs) if [r[0] for r in c].index("rel") == [r[0] for r in g.configs[d["cfg"][i]]].index("rel")]].sum() for i in range(3000)]))))
    R1[f"alpha{alpha}"] = dict(rho_star=float(rstar), rows=rows)
    mn = min(rows, key=lambda r: r["mass_true_reliable_group"]); ma = min(rows, key=lambda r: r["bayes_acc"]); print("alpha", alpha, "rho*", round(rstar, 3), "min rel-mass at", mn["rho"], round(mn["mass_true_reliable_group"], 3), "min acc at", ma["rho"], round(ma["bayes_acc"], 3), flush=True)
res["nonidentifiability_line"] = R1
# (2) two answer-dependent groups: rel(0.9), rel(0.6), noise
g2 = gen([[("rel", 0.9, 0.0, 0.0), ("rel", 0.6, 0.0, 0.0), ("noise",)]])
d = g2.sample(20000, np.random.default_rng(8_100_000)); X = d["X"]; po = g2.posterior(X)["post_y"][:, 0]
cnt = np.stack([(X[:, 1:, a] == X[:, 1:, b]).sum(1) for a, b in RE.GROUPS], -1); own = X[:, 0]
key = [tuple(c) + tuple(o) for c, o in zip(cnt, own)]
from collections import defaultdict
grp = defaultdict(list)
for i, k in enumerate(key): grp[k].append(i)
spread = [np.abs(po[v] - po[v].mean(0)).max() for v in grp.values() if len(v) > 1]
res["S_without_O"] = dict(max_posterior_spread_within_equal_counts_and_own_reports=float(max(spread)), median_spread=float(np.median(spread)),
                          n_classes=len(spread), note="nonzero spread = leave-q counts plus own reports are NOT sufficient when two groups are answer-dependent")
gt = gen([[("rel", 0.8, 0.0, 0.0), ("coll", 1.0, None), ("noise",)]]); dt = gt.sample(20000, np.random.default_rng(8_100_001)); Xt = dt["X"]; pt = gt.posterior(Xt)["post_y"][:, 0]
cnt = np.stack([(Xt[:, 1:, a] == Xt[:, 1:, b]).sum(1) for a, b in RE.GROUPS], -1); grp = defaultdict(list)
for i, k in enumerate([tuple(c) + tuple(o) for c, o in zip(cnt, Xt[:, 0])]): grp[k].append(i)
res["S_without_O"]["control_T4_spread"] = float(max(np.abs(pt[v] - pt[v].mean(0)).max() for v in grp.values() if len(v) > 1))
print("S without O", res["S_without_O"], flush=True)
# (3) mixture over reliable accuracy: P(rel | count) along counts 0..6 for one group, others noise/coll
R3 = {}
for nm, roles_list, pri in [("single_alpha", [[("rel", 0.8, 0.0, 0.0), ("noise",), ("noise",)]], None),
                            ("mixture_alpha_0.4_0.97", [[("rel", 0.4, 0.0, 0.0), ("noise",), ("noise",)], [("rel", 0.97, 0.0, 0.0), ("noise",), ("noise",)]], [0.5, 0.5]),
                            ("mixture_with_colluder", [[("rel", 0.45, 0.0, 0.0), ("coll", 0.6, None), ("noise",)], [("rel", 0.99, 0.0, 0.0), ("coll", 0.6, None), ("noise",)]], [0.5, 0.5])]:
    g = gen(roles_list, pri); rows = []
    # exact: P(group0 role is rel-type | count c of group0), marginalising other groups by Monte Carlo over episodes
    d = g.sample(60000, np.random.default_rng(8_200_000)); pc = g.posterior(d["X"])["post_cfg"]
    isrel0 = np.array([cfg[0][0] == "rel" for cfg in g.configs]); c0 = (d["X"][:, :, 0] == d["X"][:, :, 1]).sum(1)
    prel = pc[:, isrel0].sum(1)
    for c in range(Q + 1):
        s = c0 == c; rows.append(dict(count=c, n=int(s.sum()), mean_P_rel=float(prel[s].mean()) if s.any() else None))
    v = [r["mean_P_rel"] for r in rows if r["mean_P_rel"] is not None]
    unimodal = all(not (v[i] > v[i + 1] and v[i + 1] < v[i + 2]) for i in range(len(v) - 2))
    R3[nm] = dict(rows=rows, quasi_concave_or_monotone=bool(unimodal)); print(nm, [round(x, 3) for x in v], unimodal, flush=True)
res["U_scope_mixture"] = R3
# (4) informative colluder and biased errors: P(Bayes answer = colluder's common value | colluder agrees and disagrees with reliable pair's agreement)
R4 = []
for acoll in (0.2, 0.5, 0.8, 0.9):
    for bias in (0.0, 0.8):
        g = gen([[("rel", 0.8, bias, 0.0), ("coll", 1.0, acoll), ("noise",)]]); d = g.sample(6000, np.random.default_rng(8_300_000)); X = d["X"]; py = g.posterior(X)["post_y"].argmax(-1)
        cfgs = [g.configs[c] for c in d["cfg"]]; ci = np.array([[r[0] for r in c].index("coll") for c in cfgs]); ri = np.array([[r[0] for r in c].index("rel") for c in cfgs])
        A_ = np.array(RE.GROUPS); xc = X[np.arange(len(X)), :, A_[ci, 0]]; xr0 = X[np.arange(len(X)), :, A_[ri, 0]]; xr1 = X[np.arange(len(X)), :, A_[ri, 1]]
        conflict = (xr0 == xr1) & (xr0 != xc)
        R4.append(dict(a_coll=acoll, rel_error_bias=bias, a_rel=0.8, follow_colluder_in_conflict=float((py == xc)[conflict].mean()), bayes_acc=float((py == d["Y"]).mean())))
        print(R4[-1], flush=True)
res["informative_colluder_and_bias"] = R4
# (5) argmax discontinuity along rho
X = gen([[("rel", 0.8, 0.0, 0.0), ("coll", 0.6, None), ("noise",)]]).sample(3000, np.random.default_rng(8_400_000))["X"]; prev = None; flips = []
for rho in np.linspace(0.3, 1.0, 15):
    a = gen([[("rel", 0.8, 0.0, 0.0), ("coll", float(rho), None), ("noise",)]]).posterior(X)["post_y"].argmax(-1)
    if prev is not None: flips.append(dict(rho=float(rho), frac_answers_flipped_vs_previous=float((a != prev).mean())))
    prev = a
res["argmax_discontinuity_rho"] = flips; print(flips)
json.dump(res, open(os.path.join(OUT, "phase.json"), "w"), indent=1)
