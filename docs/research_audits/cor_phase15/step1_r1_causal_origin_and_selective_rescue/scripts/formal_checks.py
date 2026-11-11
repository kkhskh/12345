"""Exact numerical checks of the restricted theorems (FORMAL_THEOREMS_AND_LIMITS.md).  Stream 17_000_000 pairs (as r1_information.py).
Theorem 1 (separable weights): p(y) ∝ sum_g w(n_g) L_g(y): on every |H|=1 F45 pair, sign(dm) in {sign(w5 - w4), 0}, in R1, R2 and R3.
Theorem 2 (interaction family with exclusion): W_g = w(n_g) v(m_g), v(0) = 0: on |H|=1 F45 pairs, R2/R3 responses < 0 and the R1 response
has sign( 2 w5 v1^2 - v2 w4 v1 ) = sign( w5/w4 - v2/(2 v1) ) (0 when the third group is unanimous on c_g)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, r1lib as L
E, B = L.E, L.B
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_dev.npz"), allow_pickle=True))
X, Xa, q, v, o, reg = ev["X"], ev["X_F45"], ev["q"], ev["vote_F45"], ev["orient_F45"], ev["reg_F45"]
idx = np.arange(len(X)); H1 = ((E.loo(X)[idx, q] == 5) & (L.votes(X, q) >= 0)).sum(1) == 1; sel = o & H1
def resp(rule, **kw):
    c0 = L.rule_logits(rule, X, q, **kw); c1 = L.rule_logits(rule, Xa, q, **kw)
    with np.errstate(invalid="ignore"): return B.oriented_margin(c1, v) - B.oriented_margin(c0, v)
rng = np.random.default_rng(3); out = dict(T1=[], T2=[])
for t in range(300):
    w = np.exp(rng.normal(0, 2, 6)); dm = resp("groupwise", w=w); s = np.sign(w[5] - w[4]); row = {}
    for rg in ("R1", "R2", "R3"):
        m = sel & (reg == rg) & np.isfinite(dm); z = np.abs(dm[m]) < 1e-10; row[rg] = bool(np.all(z | (np.sign(dm[m]) == s)))
    out["T1"].append(row)
    w = np.exp(rng.normal(0, 2, 6)); vv = np.r_[0.0, 1.0, np.exp(rng.normal(np.log(2), 1.5))]; dm = resp("interaction", w=w, v=vv)
    pred = np.sign(w[5] / w[4] - vv[2] / (2 * vv[1])); row = {}
    m = sel & (reg == "R1") & np.isfinite(dm); z = np.abs(dm[m]) < 1e-10; row["R1"] = bool(np.all(z | (np.sign(dm[m]) == pred))); row["R1_frac_zero"] = float(z.mean())
    for rg in ("R2", "R3"):
        m = sel & (reg == rg) & np.isfinite(dm); row[rg] = bool(np.all(dm[m] < 1e-10))
    out["T2"].append(row)
summ = dict(T1_all_hold={rg: all(r[rg] for r in out["T1"]) for rg in ("R1", "R2", "R3")}, T2_all_hold={rg: all(r[rg] for r in out["T2"]) for rg in ("R1", "R2", "R3")},
            T2_R1_max_frac_zero=max(r["R1_frac_zero"] for r in out["T2"]), n_rules=300, n_pairs={rg: int((sel & (reg == rg)).sum()) for rg in ("R1", "R2", "R3")})
json.dump(summ, open(os.path.join(HERE, "..", "logs", "formal_checks.json"), "w"), indent=1); print(summ)
