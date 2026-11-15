"""XIII-C frozen scorer: R1-R8 on the confirmation bundle (collect_13c.py).  Thresholds are read from predictions_13c.json.
Reporting rules (carried from XIII-B RB1-RB4): a missing required input FAILS that arm (never 'inconclusive'); an arm whose diagnostic condition is
not met is NON-DIAGNOSTIC and does not count towards confirmation; a prediction is CONFIRMED only if every diagnostic arm passes and at least one
arm is diagnostic; FALSIFIED if any diagnostic arm fails; NON-DIAGNOSTIC if no arm is diagnostic.
Usage: score_13c.py BUNDLE.json [OUT.json]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "predictions_13c.json"))); T = {p["id"]: p["thresholds"] for p in P["predictions"]}
B = json.load(open(sys.argv[1])); UM = B["unit_metrics"]
def fin(u, k):
    try: return UM[u]["4000"][k]
    except KeyError: return None
def at(u, s, k):
    try: return UM[u][str(s)][k]
    except KeyError: return None
def vec(u):
    a, b = fin(u, "lik_RO1"), fin(u, "mix_mass"); return None if a is None or b is None else np.array([a, b])
def arm(name, diagnostic, passed, detail):
    return dict(arm=name, diagnostic=bool(diagnostic), passed=(bool(passed) if passed is not None else False), missing=passed is None, detail=detail)
def verdict(arms):
    d = [a for a in arms if a["diagnostic"]]
    if not d: return "NON-DIAGNOSTIC"
    return "CONFIRMED" if all(a["passed"] for a in d) else "FALSIFIED"
res = {}
# ---------------- R1 initialisation vs data order (a) ; L2-init vs WA-init (b)
t = T["R1"]; arms = []
for cell, I, D in t["cells"]:
    v, vI, vD = vec(cell), vec(I), vec(D)
    if v is None or vI is None or vD is None: arms.append(arm(f"a:{cell}", True, None, "missing")); continue
    sep = float(np.linalg.norm(vI - vD)); dI = float(np.linalg.norm(v - vI)); dD = float(np.linalg.norm(v - vD))
    arms.append(arm(f"a:{cell}", sep >= t["min_sep"], dI < dD and dI <= t["max_dI"], dict(sep=sep, d_init=dI, d_data=dD, v=v.tolist(), v_init=vI.tolist(), v_data=vD.tolist())))
for cell, L2ref, otherref in t["split_cells"]:
    v, vL, vO = vec(cell), vec(L2ref), vec(otherref)
    if v is None or vL is None or vO is None: arms.append(arm(f"b:{cell}", True, None, "missing")); continue
    sep = float(np.linalg.norm(vL - vO)); dL = float(np.linalg.norm(v - vL)); dO = float(np.linalg.norm(v - vO))
    arms.append(arm(f"b:{cell}", sep >= t["min_sep"], dL < dO, dict(sep=sep, d_L2ref=dL, d_otherref=dO, v=v.tolist())))
res["R1"] = dict(arms=arms, verdict=verdict(arms))
# ---------------- R2 learning-pathway competition: (a) ro0 -> mixture, (b) l2q -> early readout retained, (c) early mixture mass predicts organisation
t = T["R2"]; arms = []
for u, ref in t["ro0_units"]:
    m, l, mr, lr_ = fin(u, "mix_mass"), fin(u, "lik_RO1"), fin(ref, "mix_mass"), fin(ref, "lik_RO1")
    if None in (m, l, mr, lr_): arms.append(arm(f"a:{u}", True, None, "missing")); continue
    diag = mr < 0.5 and lr_ >= 0.6                                                                 # the reference must really be early-readout
    arms.append(arm(f"a:{u}", diag, m >= t["ro0_min_mix"] and l <= t["ro0_max_lik"], dict(mix=m, lik=l, ref_mix=mr, ref_lik=lr_)))
for u, ref in t["l2q_units"]:
    m, l, mr, lr_ = fin(u, "mix_mass"), fin(u, "lik_RO1"), fin(ref, "mix_mass"), fin(ref, "lik_RO1")
    if None in (m, l, mr, lr_): arms.append(arm(f"b:{u}", True, None, "missing")); continue
    diag = mr >= 0.8 and lr_ <= 0.15                                                               # the reference must really be a mixture model
    arms.append(arm(f"b:{u}", diag, l >= t["l2q_min_lik"], dict(mix=m, lik=l, ref_mix=mr, ref_lik=lr_)))
for u in t["predictor_units"]:
    m5, l4 = at(u, 546, "mix_mass"), fin(u, "lik_RO1")
    if m5 is None or l4 is None: arms.append(arm(f"c:{u}", True, None, "missing")); continue
    if m5 >= t["hi_mix546"]: arms.append(arm(f"c:{u}", True, l4 <= t["hi_max_lik"], dict(mix546=m5, lik_final=l4, predicted="mixture")))
    elif m5 <= t["lo_mix546"]: arms.append(arm(f"c:{u}", True, l4 >= t["lo_min_lik"], dict(mix546=m5, lik_final=l4, predicted="early")))
    else: arms.append(arm(f"c:{u}", False, True, dict(mix546=m5, lik_final=l4, predicted="none (intermediate)")))
res["R2"] = dict(arms=arms, verdict=verdict(arms))
# ---------------- R3 routing -> decision law (causal): intervention shifts rho_KL in the direction of the organisation change
t = T["R3"]; FF = B["fair_fits"]; arms = []
for u, ref, sign in t["pairs"]:
    r, rr = FF.get(u, {}).get("rho_PM"), FF.get(ref, {}).get("rho_PM")
    lu, lr_ = fin(u, "lik_RO1"), fin(ref, "lik_RO1")
    if None in (r, rr, lu, lr_): arms.append(arm(f"a:{u}", True, None, "missing")); continue
    changed = abs(lu - lr_) >= t["min_org_change"]                                                 # diagnostic only if the organisation changed
    arms.append(arm(f"a:{u}", changed, (r - rr) * sign >= t["min_shift"], dict(rho=r, rho_ref=rr, shift=r - rr, lik=lu, lik_ref=lr_)))
xs = [(fin(u, "lik_RO1"), FF[u]["rho_PM"]) for u in t["corr_units"] if u in FF and fin(u, "lik_RO1") is not None]
c = float(np.corrcoef(*zip(*xs))[0, 1]) if len(xs) >= 6 else None
arms.append(arm("b:corr(lik_RO1, rho_KL)", True, None if c is None else c <= t["max_corr"], dict(n=len(xs), corr=c)))      # < 6 units = missing = FAIL
v3 = verdict(arms)
if not any(a["diagnostic"] for a in arms if a["arm"].startswith("a:")): v3 = "NON-DIAGNOSTIC"     # the causal arm must be diagnostic; corr alone cannot confirm
res["R3"] = dict(arms=arms, verdict=v3)
# ---------------- R4 accessibility / capacity (two-colluder repair, beta 0.6)
t = T["R4"]; AD = B["adapt3"]; arms = []
if not t: t = dict(late_unit="?", early_unit="?", late_arms=[], early_arms=[])
def js_(u, a):
    try: return AD[u.replace(":", "_")][a]["J_DM"]["slope"]
    except (KeyError, TypeError): return None
lu = t["late_unit"]; late_ok = True if None in (fin(lu, "mix_mass"), fin(lu, "lik_RO1")) else (fin(lu, "mix_mass") >= 0.8 and fin(lu, "lik_RO1") <= 0.15)
for a, cmp_, thr in t["late_arms"]:
    v = js_(lu, a); arms.append(arm(f"late:{a}", late_ok, None if v is None else (v < thr if cmp_ == "<" else v >= thr), dict(J_slope=v, rule=f"{cmp_}{thr}")))
eu = t["early_unit"]; early_ok = True if None in (fin(eu, "mix_mass"), fin(eu, "lik_RO1")) else (fin(eu, "lik_RO1") >= 0.6 and fin(eu, "mix_mass") < 0.5)
for a, cmp_, thr in t["early_arms"]:
    v = js_(eu, a); arms.append(arm(f"early:{a}", early_ok, None if v is None else (v < thr if cmp_ == "<" else v >= thr), dict(J_slope=v, rule=f"{cmp_}{thr}")))
res["R4"] = dict(arms=arms, verdict=verdict(arms))
# ---------------- R5 constructed joint information: learned low-dim causal subspace, not the decodable probe directions; frozen rows carry it
t = T["R5"] or dict(unit="?"); CC = B["c_causal"].get(f"{t['unit']}|early01"); arms = []
okU = True if fin(t["unit"], "mix_mass") is None else fin(t["unit"], "mix_mass") >= 0.8
if CC is None: arms.append(arm("all", True, None, "missing"))
else:
    das3 = CC["C3"]["das_QSRO"]["k3"]["test_fid"]["slope"]; das8 = CC["C3"]["das_QSRO"]["k8"]["test_fid"]["slope"]
    mprobe = max(CC["C3"][f"probe_{s}"]["interchange_m"]["slope"] for s in ("RO", "QS", "OS")); resp = CC["C3"]["resp_QS"]["interchange_resp3"]["slope"]
    arms.append(arm("a:low-dim causal subspace", okU, das3 >= t["min_das3"] and das8 >= t["min_das8"], dict(das3=das3, das8=das8)))
    arms.append(arm("b:probe directions not causal", okU, mprobe <= t["max_probe"] and resp <= t["max_probe"], dict(m_probe=mprobe, resp_probe=resp)))
    mh = CC["C4"]["mixture_heads"]; corr = [CC["C4"]["corr_dmass_drole"][str(h)] for h in mh]
    arms.append(arm("c:frozen mixture rows carry it", okU and len(mh) > 0, CC["C4"]["shapley"]["A"] >= t["min_rows"] and (min(corr) if corr else -1) >= t["min_corr"], dict(rows=CC["C4"]["shapley"]["A"], corr=corr)))
res["R5"] = dict(arms=arms, verdict=verdict(arms))
# ---------------- R6 attention-row specificity under rebuilt controls
t = T["R6"]; DR = B["d_rows"]; arms = []
for u in t["units"]:
    r = DR.get(u)
    if r is None: arms.append(arm(u, True, None, "missing")); continue
    if fin(u, "lik_RO1") is None: arms.append(arm(u, True, None, "missing unit metrics")); continue
    pure = fin(u, "lik_RO1") <= 0.15 and len(r["mixture_heads"]) > 0
    if not pure: arms.append(arm(u, False, True, dict(note="not a pure mixture unit", mh=r["mixture_heads"]))); continue
    A = r["arms"]; ok = (A["TRUE"]["R2"] >= t["min_true_R2"] and A["SHUFD"]["R2"] <= t["max_decoy_R2"] and A["NOOP"]["max_abs_dlogit_vs_unmodified"] == 0.0
                         and max(A["SHUF"]["R2"], A["PMEAN"]["R2"], A["SHUFB"]["R2"]) <= t["max_offdist_R2"] and A["TRUE"]["R2"] - A.get("OTHER", {"R2": 0})["R2"] >= t["min_gap_other"])
    arms.append(arm(u, True, ok, {k: A[k]["R2"] for k in A} | dict(noop=A["NOOP"]["max_abs_dlogit_vs_unmodified"])))
res["R6"] = dict(arms=arms, verdict=verdict(arms))
# ---------------- R7 early-readout eligibility identification at the covering site (one set of arms per early unit)
t = T["R7"] or dict(units=[], site="?", k=8); arms = []
for u in t["units"]:
    DD = B["d_das"].get(u); okE = True if fin(u, "lik_RO1") is None or fin(u, "mix_mass") is None else (fin(u, "lik_RO1") >= 0.6 and fin(u, "mix_mass") < 0.5)
    if DD is None: arms.append(arm(f"{u}:all", True, None, "missing")); continue
    e = DD["sites"][t["site"]][f"k{t['k']}"]; D_, S_ = e["DAS"], e["SRCFREE"]
    excess = float(np.mean([D_[f]["own_share"] - S_[f]["own_share"] for f in D_]))
    arms.append(arm(f"{u}:a source-specific transfer", okE, excess >= t["min_excess"], dict(excess=excess)))
    arms.append(arm(f"{u}:b E over A2 (E5, E3, QF4)", okE, all(D_[f]["R2_E"] - D_[f]["R2_A2"] >= t["min_margin"] for f in ("E5", "E3", "QF4")), {f: (D_[f]["R2_E"], D_[f]["R2_A2"]) for f in D_}))
    arms.append(arm(f"{u}:c E over A1 (QF5)", okE, D_["QF5"]["R2_E"] - D_["QF5"]["R2_A1"] >= t["min_margin"], dict(QF5=(D_["QF5"]["R2_E"], D_["QF5"]["R2_A1"]))))
    arms.append(arm(f"{u}:d random subspaces small", okE, max(max(r[f]["own_share"] for f in r) for r in e["RAND"]) <= t["max_rand"], {}))
res["R7"] = dict(arms=arms, verdict=verdict(arms))
# ---------------- R8 route structure: parallel additive routes Q and R in mixture units; direct route O in early units
t = T["R8"]; DT = B["d_routes"]; arms = []
for u in t["units"]:
    r = DT.get(u)
    if r is None: arms.append(arm(u, True, None, "missing")); continue
    l, m = fin(u, "lik_RO1"), fin(u, "mix_mass"); s = r["sufficiency"]
    if l is None or m is None: arms.append(arm(u, True, None, "missing unit metrics")); continue
    if l <= 0.15 and m >= 0.8:
        arms.append(arm(f"mix:{u}", True, s["Q"] >= t["min_suff"] and s["R"] >= t["min_suff"] and abs(r["redundancy"]) <= t["max_abs_red"], dict(suff=s, red=r["redundancy"])))
    elif l >= 0.6 and m < 0.5:
        arms.append(arm(f"early:{u}", True, s["O"] >= t["min_suff_O_early"] and abs(r["redundancy"]) <= t["max_abs_red"], dict(suff=s, red=r["redundancy"])))
    else: arms.append(arm(f"other:{u}", False, True, dict(suff=s, red=r["redundancy"], note="hybrid/unclassified: not diagnostic")))
res["R8"] = dict(arms=arms, verdict=verdict(arms))
out = dict(scored=res, summary={k: v["verdict"] for k, v in res.items()})
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "logs", "score_13c.json"), "w"), indent=1, default=float)
print(json.dumps(out["summary"]))
