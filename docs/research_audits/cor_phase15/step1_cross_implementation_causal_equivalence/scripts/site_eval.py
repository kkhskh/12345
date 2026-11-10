"""Prediction V evaluation (training-objective site test).  Usage: site_eval.py MODEL [MODEL ...]
For each model: cond-family abstract-variable interchange at the READOUT site (P_full encoder at 36+q) and at the
SOURCE site (question q's six source tokens; T_own and T_full encoders), floor-adjusted exactly as in equiv_run.py
(floor = mismatched counterfactual counts, ceiling = full-site patch of the same site), plus the pathway fractions.
Streams as conf (fit 12_300_000, eval 12_400_000).  Writes logs/site/site_eval.json (merged across calls)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eqv as E
FIT, EVAL, NF, NE = 12_300_000, 12_400_000, 4000, 1000
Xf = E.sample(NF, FIT)["X"]; X = E.sample(NE, EVAL)["X"]; n0 = E.loo(X)
cps = {q: E.cond_pairs(X, q, np.random.default_rng(EVAL + 100 + q)) for q in range(E.Q)}
OUTF = os.path.join(HERE, "..", "logs", "site", "site_eval.json"); os.makedirs(os.path.dirname(OUTF), exist_ok=True)
R = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for m in sys.argv[1:]:
    M = E.Model(m); d = M.d; Af = M.acts(Xf)[0]; A0, L0, _, _ = M.acts(X); out = {}
    for site, ek in (("readout", "P_full"), ("source", "T_own"), ("source", "T_full")):
        enc = E.fit_encoder(ek, Xf, Af); agg = {}
        for q in range(E.Q):
            pos = E.site_positions("P" if site == "readout" else "T", q); Xp = cps[q]; n1 = E.loo(Xp)
            Ap, Lp, _, _ = M.acts(Xp); sh = np.roll(np.arange(NE), 1)
            full = np.zeros_like(A0); full[:, pos] = Ap[:, pos] - A0[:, pos]
            for nm, dl in (("enc", E.encoder_delta(enc, ek, X, n0, n1, d, pos)), ("floor", E.encoder_delta(enc, ek, X, n0, n1[sh], d, pos)), ("ceil", full)):
                agg.setdefault(nm, []).append((L0[:, q], Lp[:, q], M.run(X, dl)[:, q]))
        sc = {nm: E.score(*(np.concatenate([x[i] for x in v]) for i in range(3))) for nm, v in agg.items()}
        out[f"{site}|{ek}"] = dict(iia=sc["enc"]["iia"], floor=sc["floor"]["iia"], ceiling=sc["ceil"]["iia"],
                                   adjusted=E.adjusted(sc["enc"]["iia"], sc["floor"]["iia"], sc["ceil"]["iia"]), n_inf=sc["enc"]["n_inf"])
    R[m] = out; print(m, json.dumps(out), flush=True)
    json.dump(R, open(OUTF, "w"), indent=1)
