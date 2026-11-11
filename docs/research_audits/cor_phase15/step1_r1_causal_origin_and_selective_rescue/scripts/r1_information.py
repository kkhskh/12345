"""Stage I (exact; no models).  (1) closed-form Bayes check; (2) explaining-away weights before/after a 4->5 edit by regime;
(3) restricted rules' signed responses by regime (F45, F34, F23) and the groupwise sign theorem check; (4) nested 3->4->5 sequences;
(5) prevalence of the R1-type 'after' state among natural decisions.  Stream 17_000_000.  Writes logs/r1_information.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import r1lib as L
E, B = L.E, L.B
NE = 40000; S = 17_000_000; D = E.sample(NE, S); X, cfg = D["X"], D["cfg"]; q = np.random.default_rng(S + 1).integers(0, 6, NE); idx = np.arange(NE)
out = {}
# (1) closed form = exact posterior
cb = B.bayes_c(X, q); cf = L.rule_logits("bayes", X, q); fin = np.isfinite(cb) & np.isfinite(cf)
out["closed_form_max_abs_diff"] = float(np.abs(cb - cf)[fin].max()); out["closed_form_nonfinite_mismatch"] = int((np.isfinite(cb) != np.isfinite(cf)).sum())
# families
fam = {}
for name, frm in (("F45", 4), ("F34", 3), ("F23", 2)):
    Xp = X.copy(); gs = np.full(NE, -1); ks = gs.copy(); ok = np.zeros(NE, bool)
    for qq in range(6):
        s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(S + 10 + qq + 100 * frm), frm, frm + 1); Xp[s], gs[s], ks[s], ok[s] = a, g_, k_, o
    vote = L.votes(X, q)[idx, np.maximum(gs, 0)]; orient = ok & (vote >= 0)
    fam[name] = dict(X=Xp, g=gs, k=ks, vote=np.maximum(vote, 0), orient=orient, reg=L.regime_labels(X, q, np.where(orient, gs, -1), vote),
                     H=((E.loo(X)[idx, q] == 5) & (L.votes(X, q) >= 0)).sum(1))
# (2) explaining-away weights (posterior over which group is reliable) before / after, by regime
f = fam["F45"]; ew = {}
for rg in ("R1", "R2", "R3"):
    s = f["orient"] & (f["reg"] == rg) & (f["H"] == 1)
    n0 = E.loo(X)[idx, q][s]; n1 = E.loo(f["X"])[idx, q][s]; a0 = (L.votes(X, q) >= 0)[s]; a1 = (L.votes(f["X"], q) >= 0)[s]
    W0 = L.closed_form_weights(n0, a0)[0]; W1 = L.closed_form_weights(n1, a1)[0]
    gg = f["g"][s]; ii = np.arange(s.sum())
    hstar = np.array([[h for h in range(3) if h != gg[j] and n0[j, h] == 5][0] for j in ii]); fth = 3 - gg - hstar
    nrm = lambda W: W / W.sum(1, keepdims=True)
    P0, P1 = nrm(W0), nrm(W1)
    ew[rg] = dict(n=int(s.sum()), before=dict(g_reliable=float(P0[ii, gg].mean()), hstar_reliable=float(P0[ii, hstar].mean()), third_reliable=float(P0[ii, fth].mean())),
                  after=dict(g_reliable=float(P1[ii, gg].mean()), hstar_reliable=float(P1[ii, hstar].mean()), third_reliable=float(P1[ii, fth].mean())),
                  # posterior mass on hypotheses whose reliable group votes c_g at q
                  mass_supporting_cg_before=float((P0 * (L.votes(X, q)[s] == f["vote"][s][:, None])).sum(1).mean()),
                  mass_supporting_cg_after=float((P1 * (L.votes(f["X"], q)[s] == f["vote"][s][:, None])).sum(1).mean()))
out["explaining_away_weights_H1"] = ew
# (3) restricted rules
def resp(rule, name, **kw):
    f = fam[name]; c0 = L.rule_logits(rule, X, q, **kw); c1 = L.rule_logits(rule, f["X"], q, **kw)
    with np.errstate(invalid="ignore"): dm = B.oriented_margin(c1, f["vote"]) - B.oriented_margin(c0, f["vote"])
    r = {}
    for rg in ("R1", "R2", "R3"):
        s = f["orient"] & (f["reg"] == rg) & np.isfinite(dm)
        if s.sum(): r[rg] = dict(n=int(s.sum()), mean=float(dm[s].mean()), frac_pos=float((dm[s] > 1e-12).mean()), frac_neg=float((dm[s] < -1e-12).mean()))
    return r
rules = {"bayes": {}, "countonly": {}, "exclude5": {}, "softdisc_0.5": dict(delta=0.5), "softdisc_0.1": dict(delta=0.1), "softdisc_2": dict(delta=2.0)}
out["rules"] = {}
for nm, kw in rules.items():
    rule = nm.split("_")[0]; out["rules"][nm] = {fn: resp(rule, fn, **kw) for fn in ("F45", "F34", "F23")}
# groupwise sign theorem: for random positive w, sign(R1) == sign(R3) == sign(w5 - w4) pairwise?
rng = np.random.default_rng(7); th = []
for t in range(200):
    w = np.exp(rng.normal(0, 2, 6)); r = {}
    f = fam["F45"]; c0 = L.rule_logits("groupwise", X, q, w=w); c1 = L.rule_logits("groupwise", f["X"], q, w=w)
    with np.errstate(invalid="ignore"): dm = B.oriented_margin(c1, f["vote"]) - B.oriented_margin(c0, f["vote"])
    sgn = np.sign(w[5] - w[4])
    row = dict(sign_w5_minus_w4=float(sgn))
    for rg in ("R1", "R2", "R3"):
        s = f["orient"] & (f["reg"] == rg) & (f["H"] == 1) & np.isfinite(dm)
        row[rg] = dict(frac_same_sign=float((np.sign(np.round(dm[s], 12)) == sgn).mean()), frac_zero=float((np.abs(dm[s]) < 1e-12).mean()), mean=float(dm[s].mean()))
    th.append(row)
out["groupwise_theorem_check"] = dict(n_rules=len(th), min_frac_same_sign={rg: float(min(r[rg]["frac_same_sign"] for r in th)) for rg in ("R1", "R2", "R3")},
                                      max_frac_zero={rg: float(max(r[rg]["frac_zero"] for r in th)) for rg in ("R1", "R2", "R3")},
                                      R1_R3_mean_same_sign=float(np.mean([np.sign(r["R1"]["mean"]) == np.sign(r["R3"]["mean"]) for r in th])))
# interaction / hcount families can produce the Bayes sign pattern
# (4) nested sequences
X3, X4, X5, gs, ok = L.nested_sequence(X, q, cfg, np.random.default_rng(S + 77))
vote = np.maximum(L.votes(X, q)[idx, np.maximum(gs, 0)], 0)
c3, c4, c5 = (B.bayes_c(Z, q) for Z in (X3, X4, X5)); m = lambda c: B.oriented_margin(c, vote)
d34, d45 = m(c4) - m(c3), m(c5) - m(c4); reg5 = L.regime_labels(X4, q, np.where(ok, gs, -1), np.where(ok, vote, -1))
n3, n4, n5 = (E.loo(Z)[idx, q][idx, np.maximum(gs, 0)] for Z in (X3, X4, X5))
seq = dict(n_sequences=int(ok.sum()), counts_ok=bool(((n3 == 3) & (n4 == 4) & (n5 == 5))[ok].all()), Xq_fixed=bool((X3[idx, q] == X5[idx, q]).all()),
           reports_changed_per_step=[int((X4 != X3).reshape(NE, -1).sum(1)[ok].max()), int((X5 != X4).reshape(NE, -1).sum(1)[ok].max())],
           same_world_stage5=float(np.mean([E.same_world(X5[i:i + 1], cfg[i:i + 1])[0] for i in np.where(ok)[0][:2000]])) if hasattr(E, "same_world") else None)
for rg in ("R1", "R2", "R3"):
    s = ok & (reg5 == rg)
    if s.sum(): seq[rg] = dict(n=int(s.sum()), bayes_d34=float(d34[s].mean()), bayes_d45=float(d45[s].mean()), frac_d45_pos=float((d45[s] > 1e-12).mean()), frac_d34_nonneg=float((d34[s] >= -1e-12).mean()))
out["nested_345"] = seq
np.savez_compressed(os.path.join(HERE, "..", "logs", "nested_345_eval17.npz"), X3=X3, X4=X4, X5=X5, g=gs, ok=ok, q=q, vote=vote, reg5=reg5.astype(str))
# (5) prevalence of decision types among natural decisions (all 6 q per episode)
n = E.loo(X); Vs = np.stack([L.votes(X, qq) for qq in range(6)], 1)               # N,6,3
perf = (n == 5) & (Vs >= 0); H = perf.sum(-1)
two_agree = np.zeros_like(H, bool); two_dis = np.zeros_like(H, bool)
for a, b in ((0, 1), (0, 2), (1, 2)):
    both = perf[..., a] & perf[..., b]; two_agree |= both & (Vs[..., a] == Vs[..., b]); two_dis |= both & (Vs[..., a] != Vs[..., b])
out["prevalence"] = dict(H1=float((H == 1).mean()), H2=float((H == 2).mean()), H3=float((H == 3).mean()), two_perfect_agree=float(two_agree.mean()), two_perfect_disagree_only=float((two_dis & ~two_agree).mean()))
json.dump(out, open(os.path.join(HERE, "..", "logs", "r1_information.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
