"""XIII-D FREEZE-1 scorer (P2a, P2b, P2c).  Inputs: logs/unit_metrics_f1.jsonl, logs/route_status_f1.jsonl, logs/route_status_f1_react.jsonl.
Rules: missing required input -> arm FAILS; non-competent run -> arm NON-DIAGNOSTIC (reported); prediction CONFIRMED iff >= 1 diagnostic arm and all
diagnostic arms pass; FALSIFIED if any diagnostic arm fails; NON-DIAGNOSTIC otherwise.  Usage: score_f1.py [LOGDIR] -> LOGDIR/score_f1.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs")
P = json.load(open(os.path.join(HERE, "predictions_13d_f1.json"))); T = {p["id"]: p["thresholds"] for p in P["predictions"]}
def jl(f): p = os.path.join(LD, f); return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
UM = {}; RS = {}; RR = {}
for r in jl("unit_metrics_f1.jsonl"): UM.setdefault(r["unit"], {})[r["step"]] = r
for r in jl("route_status_f1.jsonl"): u, s = (r["unit"].split("@") + ["4000"])[:2]; RS.setdefault(u, {})[int(s)] = r
for r in jl("route_status_f1_react.jsonl"): RR[r["unit"].split("@")[0]] = r
U = dict(F1a="uP160:fx:21:21:21:ro0T100", F1b="uP160:fx:21:21:21:ro0T250", F1c="uP160:fx:23:23:23:l2qT100", F1d="uP160:fx:23:23:23:l2qT250",
         F1e="uP160:fx:21:21:21:ro0T0_br546_w546-1146", F1f="uP160:fx:23:23:23:l2qT0_br546_w546-1146")
def fin(u, k):
    try: return UM[u][4000][k]
    except KeyError: return None
def competent(u): k = fin(u, "KL"); return None if k is None else k <= 0.02
def cls(u):
    l, m = fin(u, "lik_RO1"), fin(u, "mix_mass")
    if l is None or m is None: return None
    return "mixture" if (m >= 0.8 and l <= 0.15) else "early" if (l >= 0.6 and m < 0.5) else "hybrid"
def arm(name, diag, passed, det): return dict(arm=name, diagnostic=bool(diag), passed=None if passed is None else bool(passed), missing=passed is None, detail=det)
def verdict(arms):
    d = [a for a in arms if a["diagnostic"]]
    return "NON-DIAGNOSTIC" if not d else ("CONFIRMED" if all(a["passed"] for a in d) else "FALSIFIED")
res = {}; t = T["P2a"]; arms = []
for k, u in U.items():
    c, cp = cls(u), competent(u)
    if c is None or cp is None: arms.append(arm(f"{k}", True, None, "missing unit metrics")); continue
    if not (cp and c == "mixture"): arms.append(arm(k, False, True, dict(cls=c, competent=cp))); continue
    r = RR.get(u)
    if r is None: arms.append(arm(k, True, None, "missing route_status react")); continue
    g = r["react"]["after_acc"] - r["react"]["before_acc"]; g0 = r["react_ro1_removed"]["after_acc"] - r["react_ro1_removed"]["before_acc"]
    ok = r["present_pooled_RO1"] >= t["min_present"] and r["lik_RO1"] <= t["max_lik"] and g >= t["min_gain"] and g0 <= t["max_gain_removed"] and g0 < 0.5 * g
    arms.append(arm(k, True, ok, dict(present=r["present_pooled_RO1"], lik=r["lik_RO1"], gain=g, gain_removed=g0)))
res["P2a"] = dict(arms=arms, verdict=verdict(arms))
arms = []
for k, want in (("F1a", "early"), ("F1c", "mixture")):
    u = U[k]; c, cp = cls(u), competent(u)
    if c is None or cp is None: arms.append(arm(k, True, None, "missing")); continue
    arms.append(arm(k, cp, c == want, dict(cls=c, want=want, competent=cp, lik=fin(u, "lik_RO1"), mix=fin(u, "mix_mass"), KL=fin(u, "KL"))))
for k in ("F1b", "F1d", "F1e", "F1f"):
    u = U[k]; arms.append(arm(k, False, True, dict(cls=cls(u), competent=competent(u), lik=fin(u, "lik_RO1"), mix=fin(u, "mix_mass"), KL=fin(u, "KL"), note="reported, no prediction")))
res["P2b"] = dict(arms=arms, verdict=verdict(arms))
t = T["P2c"]; arms = []
for k in ("F1a", "F1b", "F1c", "F1d"):
    u = U[k]; c, cp = cls(u), competent(u)
    if c is None or cp is None: arms.append(arm(k, True, None, "missing")); continue
    if not (cp and c == "mixture"): arms.append(arm(k, False, True, dict(cls=c, competent=cp))); continue
    s = RS.get(u, {})
    if not all(x in s for x in (246, 546, 4000)): arms.append(arm(k, True, None, "missing route_status steps")); continue
    pk_p = max(s[246]["present_pooled_RO1"], s[546]["present_pooled_RO1"]); pk_l = max(s[246]["lik_RO1"], s[546]["lik_RO1"])
    if pk_l < t["min_peak_lik"]: arms.append(arm(k, False, True, dict(peak_lik=pk_l, note="early route never used: not diagnostic"))); continue
    ok = s[4000]["present_pooled_RO1"] >= t["present_ratio"] * pk_p and s[4000]["lik_RO1"] <= t["use_ratio"] * pk_l
    arms.append(arm(k, True, ok, dict(peak_present=pk_p, final_present=s[4000]["present_pooled_RO1"], peak_lik=pk_l, final_lik=s[4000]["lik_RO1"])))
res["P2c"] = dict(arms=arms, verdict=verdict(arms))
json.dump(dict(scored=res, summary={k: v["verdict"] for k, v in res.items()}), open(os.path.join(LD, "score_f1.json"), "w"), indent=1, default=float)
print({k: v["verdict"] for k, v in res.items()})
