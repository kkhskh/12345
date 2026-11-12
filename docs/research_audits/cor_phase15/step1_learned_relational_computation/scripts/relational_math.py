"""Priority 1: exact checks for MATHEMATICAL_RELATIONAL_OPERATION.md.  Stream 18_000_000 (40,000 episodes, all 6 decisions).
(1) Inv VII m vs the mandate's written formula m': agreement rate; Bayes with m' substituted (R1/R2/R3 signs).
(2) Sufficiency of sub-statistics for the exact predictive (functional dependence test: group decisions by the statistic, max spread of the
    exact posterior within groups): per-group (n, q-reports); + perfect set; (m, q-reports) without n; (RHO^n m, q-reports).
(3) Log-additive aggregation theorem check: W_g = f(n_g) exp(sum_{h != g} psi(n_h, a_h)) on random f, psi -> R1/R3 same sign.
(4) Ties: fraction of F45 R1/R2/R3 pairs whose exact-Bayes argmax is tied (|margin top1-top2| < 1e-9) or near-tied (< 0.05 nats) before/after."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, v8lib as V
L, E, B = V.L, V.E, V.B
S = 18_000_000; D = E.sample(40000, S); X = D["X"]; out = {}
m7 = V.eligibility_inv7(X); mm = V.eligibility_mandate(X); out["m_agreement_rate"] = float((m7 == mm).mean())
out["m_disagree_cases"] = dict(m7_gt_mm=float((m7 > mm).mean()), m7_lt_mm=float((m7 < mm).mean()))
# Bayes with mandate m' substituted, evaluated on the dev F45 pairs (Inv VII eval set)
ev = dict(np.load(os.path.join(HERE, "..", "..", "step1_r1_causal_origin_and_selective_rescue", "logs", "eval_dev.npz"), allow_pickle=True))
def rule_with(mfun, Xa, q):
    idx = np.arange(len(Xa)); n = E.loo(Xa)[idx, q]; m = mfun(Xa)[idx, q]; W = L.RHO ** n * m; W = np.where(W.sum(1, keepdims=True) > 0, W, 1.0)
    return L.mixture_logits(W.astype(float), L.Lrel(Xa, q))
resp = {}
for nm, mf in (("inv7_m", V.eligibility_inv7), ("mandate_m", V.eligibility_mandate)):
    r = {}
    for f in ("F45", "F34"):
        c0, c1 = rule_with(mf, ev["X"], ev["q"]), rule_with(mf, ev[f"X_{f}"], ev["q"]); v = ev[f"vote_{f}"]
        with np.errstate(invalid="ignore"): dm = B.oriented_margin(c1, v) - B.oriented_margin(c0, v)
        for rg in ("R1", "R2", "R3"):
            s = ev[f"orient_{f}"] & (ev[f"reg_{f}"] == rg) & np.isfinite(dm)
            if s.sum(): r[f"{f}_{rg}"] = float(dm[s].mean())
    c = rule_with(mf, ev["X"], ev["q"]); cb = ev["cB_base"]; fin = np.isfinite(c).all(1) & np.isfinite(cb).all(1)
    r["max_abs_diff_from_exact_base"] = float(np.abs(c - cb)[fin].max()); r["frac_nonfinite"] = float((~fin).mean())
    resp[nm] = r
out["bayes_with_m"] = resp
# (2) sufficiency
n = E.loo(X); Vt = np.stack([L.votes(X, qq) for qq in range(6)], 1); post = E.bayes_from(X, n)   # N,6,3
Rq = np.stack([X[:, :, [a, b]] for a, b in L.GROUPS], 2).reshape(len(X), 6, 6)                   # q-reports of all six sources
def spread(keyarr):
    keys = [tuple(k) for k in keyarr.reshape(-1, keyarr.shape[-1])]; P = post.reshape(-1, 3); d = {}
    for k, p in zip(keys, P): d.setdefault(k, []).append(p)
    return float(max(np.ptp(np.array(v), 0).max() for v in d.values())), len(d)
stats = {
    "counts+qreports": np.concatenate([n, Rq], -1),
    "perfectset+qreports": np.concatenate([(n == 5) & (Vt >= 0), Rq], -1).astype(int),
    "m+qreports (no counts)": np.concatenate([m7, Rq], -1),
    "RHO^n*m (as n,m) + qreports": np.concatenate([np.where(m7 > 0, n, -1), m7, Rq], -1),
    "counts only": n, "qreports only": Rq,
}
out["sufficiency_max_posterior_spread"] = {k: dict(zip(("max_spread", "n_classes"), spread(v))) for k, v in stats.items()}
# (3) log-additive aggregation theorem
idx = np.arange(len(ev["X"])); qv = ev["q"]; Xb, Xa, v, o, reg = ev["X"], ev["X_F45"], ev["vote_F45"], ev["orient_F45"], ev["reg_F45"]
H1 = ((E.loo(Xb)[idx, qv] == 5) & (L.votes(Xb, qv) >= 0)).sum(1) == 1
def logadd(Xx, f, psi):
    nn = E.loo(Xx)[idx, qv]; aa = (L.votes(Xx, qv) >= 0).astype(int); P = psi[nn, aa]; W = f[nn] * np.exp(P.sum(1, keepdims=True) - P)
    return L.mixture_logits(W, L.Lrel(Xx, qv))
rng = np.random.default_rng(11); viol = 0; same = 0; tot = 0
for t in range(300):
    f = np.exp(rng.normal(0, 2, 6)); psi = rng.normal(0, 2, (6, 2))
    with np.errstate(invalid="ignore"): dm = B.oriented_margin(logadd(Xa, f, psi), v) - B.oriented_margin(logadd(Xb, f, psi), v)
    sgn = np.sign(np.log(f[5]) - np.log(f[4]) - (psi[5, 1] - psi[4, 0]) * 0)   # own factor f(n) and own psi cancel? see doc
    for rg in ("R1", "R3"):
        s = o & H1 & (reg == rg); z = np.abs(dm[s]) < 1e-10; tot += 1
    r1 = dm[o & H1 & (reg == "R1")]; r3 = dm[o & H1 & (reg == "R3")]
    s1 = np.sign(r1[np.abs(r1) > 1e-10]); s3 = np.sign(r3[np.abs(r3) > 1e-10])
    if len(s1) and len(s3) and (len(np.unique(np.r_[s1, s3])) > 1): viol += 1
out["log_additive_theorem"] = dict(n_rules=300, rules_with_R1_R3_opposite_signs=viol)
# (4) ties
for f in ("F45",):
    cb0, cb1 = ev["cB_base"], ev[f"cB_{f}"]; T = {}
    for rg in ("R1", "R2", "R3"):
        s = ev[f"orient_{f}"] & (ev[f"reg_{f}"] == rg)
        def gap(c): cs = np.sort(c[s], -1); return cs[:, -1] - cs[:, -2]
        g0, g1 = gap(cb0), gap(cb1)
        T[rg] = dict(n=int(s.sum()), exact_tie_before=float((g0 < 1e-9).mean()), exact_tie_after=float((g1 < 1e-9).mean()),
                     near_tie_before=float((g0 < 0.05).mean()), near_tie_after=float((g1 < 0.05).mean()))
    out["ties"] = T
json.dump(out, open(os.path.join(HERE, "..", "logs", "relational_math.json"), "w"), indent=1); print(json.dumps(out, indent=1))
