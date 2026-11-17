"""XV independent verification of the task-family mathematics (gate for confirmation).  Writes logs/xv_math_verified.json.
Checks for each confirmation condition: (1) sampler frequencies match the exact joint (300k questions, max |diff| < 0.002);
(2) Bayes from src/xvtask.bayes equals the brute-force posterior from the exact joint; (3) the gate-free formula equals brute-force
marginalisation over g; (4) every single source, and (a = 3) every pair of gate sources, has zero mutual information with g;
(5) the minimal decision order is a + 1 and the order-a optimum has gate-flip slope 0 (logs/xv_math_dev.json); (6) usefulness matching:
gate gain (order-a optimum minus Bayes) of the arity-3 condition within 2% of its arity-2 partner, intermediate gain within 10%;
(7) the reference cache logs/refs_*.npz equals a fresh refit (max |diff| < 1e-6)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, xvtask as T, xv_math_core as MC
L = os.path.join(HERE, "..", "logs"); M = json.load(open(os.path.join(L, "xv_math_dev.json"))); out = {}; ok = True
key = {"2:0.9:0.3333333333333333": "2:0.9", "3:0.9:0.3333333333333333": "3:0.9", "2:0.9:0.05": "2:0.9:0.05", "3:0.9:0.05": "3:0.9:0.05"}
for cond, mk in key.items():
    a, b, nu = int(cond.split(":")[0]), float(cond.split(":")[1]), float(cond.split(":")[2]); px, PB = T.joint(a, b, nu); r = {}
    X = T.sample(50000, np.random.default_rng(5), a, b, nu)["X"].reshape(-1, 6); f = np.bincount((X * np.array([243, 81, 27, 9, 3, 1])).sum(1), minlength=729) / len(X)
    r["sampler_max_abs"] = float(np.abs(f - px).max()); r["bayes_max_abs"] = float(np.abs(T.bayes(T.XALL, a, b) - PB).max())
    e = (1 - b) / 2; GF = np.zeros((729, 3))
    for i, x in enumerate(T.XALL):
        for y in range(3):
            g1 = (b if x[3] == y else e) * (b if x[4] == y else e) / 3; g0 = (b if x[5] == y else e) * ((nu if x[3] == x[4] else (1 - nu) / 2) / 3); GF[i, y] = 0.5 * (g1 + g0)
    GF /= GF.sum(1, keepdims=True); r["gatefree_max_abs"] = float(np.abs(GF - T.gatefree(T.XALL, a, b, nu)).max())
    m = M[mk]; r["MI_single_max"] = max(m["MI_g_single"].values()); r["MI_gatepairs_max"] = max(m["MI_g_gatepairs"].values()) if m["MI_g_gatepairs"] else 0.0
    r["min_decision_order"] = m["min_decision_order"]; r["order_a_zero_G"] = m["order_a_has_zero_G"]; r["gain_gate"] = m["gain_gate"]; r["gain_intermediate"] = m["gain_intermediate"]
    R = np.load(os.path.join(L, "refs_" + cond.replace(":", "_") + ".npz")); P1, _ = MC.fit(1, px, PB); Pa, _ = MC.fit(a, px, PB)
    r["refs_cache_max_abs"] = float(max(np.abs(R["order1"] - P1).max(), np.abs(R["ordera"] - Pa).max(), np.abs(R["bayes"] - PB).max()))
    r["pass"] = bool(r["sampler_max_abs"] < 0.002 and r["bayes_max_abs"] < 1e-12 and r["gatefree_max_abs"] < 1e-12 and r["MI_single_max"] < 1e-12 and r["MI_gatepairs_max"] < 1e-12
                     and r["min_decision_order"] == a + 1 and r["order_a_zero_G"] and r["refs_cache_max_abs"] < 1e-6)
    out[cond] = r; ok &= r["pass"]
for lo, hi in (("2:0.9:0.3333333333333333", "3:0.9:0.3333333333333333"), ("2:0.9:0.05", "3:0.9:0.05")):
    g = abs(out[hi]["gain_gate"] / out[lo]["gain_gate"] - 1); i = abs(out[hi]["gain_intermediate"] / out[lo]["gain_intermediate"] - 1)
    out[f"match {lo} vs {hi}"] = dict(gate_gain_rel_diff=g, intermediate_gain_rel_diff=i, pass_=bool(g < 0.02 and i < 0.10)); ok &= g < 0.02 and i < 0.10
out["all_pass"] = bool(ok); json.dump(out, open(os.path.join(L, "xv_math_verified.json"), "w"), indent=1); print(json.dumps(out, indent=1))
