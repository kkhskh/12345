"""XVII Stage 1 [proof + num]: the source-level discrepancy of the original O/D formalism (as used in the Phase 1.5 bridge,
step1_foundational_atlas_continuation/scripts/od_bridge.py): D_c(x) = number of answer-reporting sources whose value differs from candidate c.
Its decision rule argmin_c D_c is a sum of single-source terms, hence an order-1 rule (also with any per-source weights, the 'repaired'
weighted vote).  Every order <= a rule ignores the gate (XVI V5/V6), so F_G(argmin D) = 0 exactly and D cannot express any GATE condition.
Exact over all 729 inputs per condition: accuracy of argmin D (ties split uniformly) with D over the evidence sources (3,4,5), with D over all
six sources, the best weighted vote (= order-1 optimum, XVI references), Bayes; gate-flip following of argmin-D (decision changes under gate flips).
-> logs/xvii_od_check.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, xvitask as T, xvidiag as XD
K = 3; XA = T.xall(6); IDX = np.array([243, 81, 27, 9, 3, 1]); out = {}
for c in ["2:0.9:0.9:0.3333333333333333", "2:0.9:0.9:0.05", "3:0.9:0.9:0.3333333333333333", "3:0.9:0.9:0.05", "2:0.9:0.55:0.05"]:
    a, b1, b0, nu = XD.parse(c); px, PB = T.joint(a, b1, b0, nu); R = XD.refs(c, os.path.join(HERE, "..", "logs"))
    def acc_D(srcs):
        D = np.stack([(XA[:, srcs] != k).sum(1) for k in range(K)], 1); m = D == D.min(1, keepdims=True); P = m / m.sum(1, keepdims=True)
        return float((px[:, None] * PB * P).sum()), P
    acc_ev, Pev = acc_D([3, 4, 5]); acc_all, _ = acc_D([0, 1, 2, 3, 4, 5])
    # gate-flip invariance of D over evidence sources: flipping the last gate source never changes D (D does not read it) -> exact
    rng = np.random.default_rng(0); Xf = T.gate_flip(XA[:, None, :], a, rng)[:, 0]
    changed = float((np.abs(Pev - Pev[(Xf * IDX).sum(1)]).max(1) > 0).mean())
    acc = lambda P: float((px * PB[np.arange(len(P)), P.argmax(1)]).sum())
    out[c] = dict(acc_minD_evidence=acc_ev, acc_minD_all_sources=acc_all, acc_order1_optimum=acc(R["order1"]), acc_gatefree_optimum=acc(R["ordera"]), acc_bayes=acc(PB),
                  fraction_inputs_where_gate_flip_changes_minD=changed)
    print(c, {k: round(v, 4) for k, v in out[c].items()})
json.dump(out, open(os.path.join(HERE, "..", "logs", "xvii_od_check.json"), "w"), indent=1)
