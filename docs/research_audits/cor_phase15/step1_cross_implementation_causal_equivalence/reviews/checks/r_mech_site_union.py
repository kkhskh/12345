"""Reviewer check: full-patch ceilings of the cond family at unions of candidate sites (site_eval.py streams:
fit 12_300_000, eval 12_400_000 x1000, cond rng(EVAL+100+q)).  Sites: readout 36+q; q's six source tokens; their union;
all 36 source tokens; all source tokens + readout.  Usage: r_mech_site_union.py M ..."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np
import eqv as E
t0 = time.time(); X = E.sample(1000, 12_400_000)["X"]
cps = {q: E.cond_pairs(X, q, np.random.default_rng(12_400_000 + 100 + q)) for q in range(E.Q)}
for m in sys.argv[1:]:
    M = E.Model(m); A0, L0, _, _ = M.acts(X); agg = {}
    for q in range(E.Q):
        Ap, Lp, _, _ = M.acts(cps[q]); qs = list(range(6 * q, 6 * q + 6))
        sites = {"readout": [36 + q], "q_src": qs, "readout+q_src": qs + [36 + q], "all_src": list(range(36)),
                 "all_src+readout": list(range(36)) + [36 + q]}
        for nm, pp in sites.items():
            dl = np.zeros_like(A0); dl[:, pp] = Ap[:, pp] - A0[:, pp]
            agg.setdefault(nm, []).append((L0[:, q], Lp[:, q], M.run(X, dl)[:, q]))
    sc = {nm: round(E.score(*(np.concatenate([x[i] for x in v]) for i in range(3)))["iia"], 4) for nm, v in agg.items()}
    print(m, json.dumps(sc), f"[{time.time() - t0:.0f}s]", flush=True)
