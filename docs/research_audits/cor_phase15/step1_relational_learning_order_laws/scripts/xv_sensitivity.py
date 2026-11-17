"""XV threshold-sensitivity analysis (exploratory, post-scoring): recompute the P3 stage calls and the per-pair P1/P2 log-ratios under
alternative thresholds for G (0.4, 0.5, 0.6) and A_p (0.4, 0.5, 0.6), from the frozen trajectory files.  Writes logs/xv_sensitivity.json."""
import json, os, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); D = json.load(open(os.path.join(PKG, "logs", "XV_design.json")))
T = {(r["cond"], r["seed"]): json.load(open(os.path.join(PKG, r["traj"])))["traj"] for r in D["runs"]}; out = {}
for gth, ath in itertools.product((0.4, 0.5, 0.6), (0.4, 0.5, 0.6)):
    tG, tI = {}, {}
    for k, tr in T.items():
        n = len(tr); tG[k] = next((tr[i]["step"] for i in range(n - 1) if tr[i]["G"] >= gth and tr[i + 1]["G"] >= gth), None)
        tI[k] = next((tr[i]["step"] for i in range(n - 1) if tr[i]["A_p"] >= ath and tr[i]["G"] <= 0.1 and tr[i + 1]["A_p"] >= ath and tr[i + 1]["G"] <= 0.1), None)
    g = lambda k: tG[k] or 4000
    calls = {c: sum(tI[(c, s)] is not None and (tG[(c, s)] is None or tI[(c, s)] < tG[(c, s)]) for s in D["seeds"]) for c in "ABCD"}
    p1 = [np.log(g((hi, s)) / g((lo, s))) for lo, hi in (("B", "C"), ("A", "D")) for s in D["seeds"]]
    p2 = [np.log(g((lo, s)) / g((hi, s))) for lo, hi in (("A", "B"), ("D", "C")) for s in D["seeds"]]
    out[f"G{gth}_A{ath}"] = dict(stage_counts=calls, P1_pairs_ge_ln1p5=int(sum(x >= np.log(1.5) for x in p1)), P2_pairs_ge_ln1p25=int(sum(x >= np.log(1.25) for x in p2)),
                                  P2_arity2_pairs_ge_ln1p25=int(sum(x >= np.log(1.25) for x in p2[:4])))
json.dump(out, open(os.path.join(PKG, "logs", "xv_sensitivity.json"), "w"), indent=1)
for k, v in out.items(): print(k, v)
