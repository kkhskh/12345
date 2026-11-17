"""XVI independent mathematical verification [num] -> logs/xvi_math_verified.json (all_pass must be true for the frozen scorer).
Checks:
  V1  generator identity: xvitask.sample(a, b, b, nu) reproduces XV's xvtask.sample(a, b, nu) bit for bit, and the random-number
      consumption does not depend on (b1, b0, nu) (so seed-paired XVI value conditions see identical gates, labels and noise draws).
  V2  sampler law: empirical frequencies of 600k sampled questions match the exact joint (chi-square p > 1e-3) for every frozen condition.
  V3  exact joint by a second, loop-based enumeration (independent code path) equals the vectorised joint (max diff < 1e-14).
  V4  frozen value contrast (arity 2): identical agreement law (P(agree | g) equal to 1e-12), gain_intermediate relative difference <= 2%,
      gain_gate ratio >= 1.5, minimal decision order 3 in both, every proper gate subset and every single source has zero MI with g.
  V5  arity 4 (S = 7): minimal decision order exactly 5 (order-4 KL > 1e-3, order-5 KL < 1e-6), proper gate subsets MI 0, order-k optima
      for k <= 4 have gate-flip slope 0, gate value matched to arity 3 (S = 7, same b, nu) within 0.5%.
  V6  diagnostic references (src/xvidiag): F_G = 0 for the gate-free optimum and 1 for Bayes at temperatures 0.5, 1, 2; A_conf of the
      tempered order-a optimum constant across temperatures (spread < 1e-6), A_conf(order-1) = A_conf(Bayes) = 0 (|.| < 1e-6).
  V7  cached reference tables logs/refs_*.npz equal a fresh computation (max diff < 1e-9)."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_learning_order_laws", "src"))
import numpy as np
from scipy.stats import chi2
import xvitask as T, xvtask as XV, xvi_math as XM, xvidiag as XD
LOGS = os.path.join(HERE, "..", "logs"); K = 3
FROZEN = json.load(open(os.path.join(LOGS, "XVI_conditions.json")))
out = dict(checks={}); ok = lambda k, v, det=None: out["checks"].__setitem__(k, dict(pass_=bool(v), detail=det))
# V1
same = True
for a, nu in [(2, 1 / 3), (3, 1 / 3), (2, 0.05), (3, 0.05)]:
    r1, r2 = np.random.default_rng(11), np.random.default_rng(11); d1 = T.sample(400, r1, a, 0.9, 0.9, nu); d2 = XV.sample(400, r2, a, 0.9, nu)
    same &= all((d1[k] == d2[k]).all() for k in "XYg") and r1.random() == r2.random()
r1, r2 = np.random.default_rng(12), np.random.default_rng(12); d1 = T.sample(400, r1, 2, 0.9, 0.55, 0.05); d2 = T.sample(400, r2, 2, 0.9, 0.9, 0.05)
cons = r1.random() == r2.random() and (d1["g"] == d2["g"]).all() and (d1["Y"] == d2["Y"]).all() and (d1["X"][..., :3] == d2["X"][..., :3]).all()
ok("V1_generator_identity_and_rng_consumption", same and cons, dict(xv_identical=bool(same), consumption_independent_of_reliabilities=bool(cons)))
# V2
det = {}
for c in FROZEN["all_neural_conditions"]:
    a, b1, b0, nu = XD.parse(c); px, PB = T.joint(a, b1, b0, nu); d = T.sample(100000, np.random.default_rng(13), a, b1, b0, nu)
    idx = (d["X"].reshape(-1, 6) * np.array([243, 81, 27, 9, 3, 1])).sum(1) * K + d["Y"].ravel(); cnt = np.bincount(idx, minlength=729 * K)
    e = (px[:, None] * PB).ravel() * len(idx); m = e > 0; stat = float(((cnt[m] - e[m]) ** 2 / e[m]).sum()); p = float(chi2.sf(stat, m.sum() - 1))
    det[c] = dict(p=p, zero_prob_hits=int(cnt[~m].sum()))
ok("V2_sampler_law", all(v["p"] > 1e-3 and v["zero_prob_hits"] == 0 for v in det.values()), det)
# V3
def joint_loop(a, b1, b0, nu, S):
    XA = T.xall(S); e1, e0 = (1 - b1) / (K - 1), (1 - b0) / (K - 1); pxy = np.zeros((len(XA), K))
    for i, x in enumerate(XA):
        g = x[:a].sum() % K == 0; pg = (1 / K) ** (a - 1) * 0.5 * (1.0 if g else 0.5) * (1 / K) ** (S - 3 - a)
        for y in range(K):
            if g: pe = (b1 if x[S - 3] == y else e1) * (b1 if x[S - 2] == y else e1) / K
            else: pe = (b0 if x[S - 1] == y else e0) * (nu if x[S - 3] == x[S - 2] else (1 - nu) / (K - 1)) / K
            pxy[i, y] = pg * pe / K
    px = pxy.sum(1); return px, pxy / px[:, None]
mx = 0.0
for spec in FROZEN["all_neural_conditions"] + [FROZEN["arity4"]["spec"]]:
    sp = spec.split(":"); a, b1, b0, nu = int(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]); S = int(sp[4]) if len(sp) > 4 else 6
    p1, q1 = T.joint(a, b1, b0, nu, S); p2, q2 = joint_loop(a, b1, b0, nu, S); mx = max(mx, float(abs(p1 - p2).max()), float(abs(q1 - q2).max()))
ok("V3_joint_two_code_paths", mx < 1e-14, dict(max_diff=mx))
# V4
lo, hi = XM.analyse(FROZEN["value_low"]), XM.analyse(FROZEN["value_high"])
d4 = dict(gate_low=lo["gain_gate"], gate_high=hi["gain_gate"], ratio=hi["gain_gate"] / lo["gain_gate"], int_low=lo["gain_intermediate"], int_high=hi["gain_intermediate"],
          int_rel_diff=abs(hi["gain_intermediate"] - lo["gain_intermediate"]) / hi["gain_intermediate"], agree_law=[lo["P_agree_g1"], hi["P_agree_g1"], lo["P_agree_g0"], hi["P_agree_g0"]],
          min_order=[lo["min_decision_order"], hi["min_decision_order"]], mi_max=max(max(lo["MI_g_single"].values()), max(hi["MI_g_single"].values()), lo["MI_g_proper_gate_subsets_max"], hi["MI_g_proper_gate_subsets_max"]),
          residual=dict(bayes_nll=[lo["bayes"]["NLL"], hi["bayes"]["NLL"]], bayes_acc=[lo["bayes"]["acc"], hi["bayes"]["acc"]], order1_acc=[lo["order1"]["acc"], hi["order1"]["acc"]],
                        order1_nll=[lo["order1"]["NLL"], hi["order1"]["NLL"]], acc_gain_gate=[lo["acc_gain_gate"], hi["acc_gain_gate"]], MI_agree=[lo["MI_g_agree"], hi["MI_g_agree"]]))
ok("V4_value_contrast", d4["ratio"] >= 1.5 and d4["int_rel_diff"] <= 0.02 and abs(lo["P_agree_g1"] - hi["P_agree_g1"]) < 1e-12 and abs(lo["P_agree_g0"] - hi["P_agree_g0"]) < 1e-12
   and d4["min_order"] == [3, 3] and d4["mi_max"] < 1e-12, d4)
# V5
a4 = XM.analyse(FROZEN["arity4"]["spec"]); a3 = XM.analyse(FROZEN["arity4"]["matched_arity3_spec"], kmax=4)
d5 = dict(KL_by_order={k: a4[f"order{k}"]["KL"] for k in range(1, 6)}, G_by_order={k: a4[f"order{k}"]["G"] for k in range(1, 6)}, mi_proper=a4["MI_g_proper_gate_subsets_max"],
          mi_single_max=max(a4["MI_g_single"].values()), gate4=a4["gain_gate"], gate3=a3["gain_gate"], rel=abs(a4["gain_gate"] - a3["gain_gate"]) / a3["gain_gate"],
          int4=a4["gain_intermediate"], int3=a3["gain_intermediate"], acc_gain4=a4["acc_gain_gate"], acc_gain3=a3["acc_gain_gate"], converged={k: a4[f"order{k}"]["converged"] for k in range(1, 6)})
ok("V5_arity4", a4["min_decision_order"] == 5 and a4["order4"]["KL"] > 1e-3 and a4["order5"]["KL"] < 1e-6 and d5["mi_proper"] < 1e-12 and d5["mi_single_max"] < 1e-12
   and all(abs(a4[f"order{k}"]["G"]) < 1e-6 for k in range(1, 5)) and d5["rel"] < 0.005, d5)
# V6
d6 = {}
for c in FROZEN["all_neural_conditions"]:
    D = XD.Diag(c, 99_700_000, 1500, LOGS); R = D.ref_scores()
    d6[c] = dict(FG_ordera=[R[k]["F_G"] for k in ("ordera", "ordera_T0.5", "ordera_T2.0")], FG_bayes=[R[k]["F_G"] for k in ("bayes", "bayes_T0.5", "bayes_T2.0")],
                 Aconf_ordera=[R[k]["A_conf"] for k in ("ordera", "ordera_T0.5", "ordera_T2.0")], Aconf_o1_bayes=[R["order1"]["A_conf"], R["bayes"]["A_conf"]],
                 KL_ref=dict(order1=R["order1"]["KL"], ordera=R["ordera"]["KL"]))
ok("V6_diagnostic_references", all(v["FG_ordera"] == [0.0] * 3 and v["FG_bayes"] == [1.0] * 3 and np.ptp(v["Aconf_ordera"]) < 1e-6 and max(abs(x) for x in v["Aconf_o1_bayes"]) < 1e-6 for v in d6.values()), d6)
# V7
import xvi_math_core as M
mx = 0.0
for c in FROZEN["all_neural_conditions"]:
    z = np.load(os.path.join(LOGS, "refs_" + c.replace(":", "_") + ".npz")); a, b1, b0, nu = XD.parse(c); px, PB = T.joint(a, b1, b0, nu); XA = T.xall(6)
    o1, _ = M.fit(1, px, PB, XA); oa, _ = M.fit(a, px, PB, XA)
    mx = max(mx, float(abs(z["bayes"] - PB).max()), float(abs(z["order1"] - o1).max()), float(abs(z["ordera"] - oa).max()), float(abs(z["gatefree"] - T.gatefree(XA, a, b1, b0, nu)).max()))
ok("V7_reference_cache", mx < 1e-9, dict(max_diff=mx))
out["all_pass"] = all(v["pass_"] for v in out["checks"].values())
json.dump(out, open(os.path.join(LOGS, "xvi_math_verified.json"), "w"), indent=1, default=float); print({k: v["pass_"] for k, v in out["checks"].items()}, "ALL_PASS", out["all_pass"])
