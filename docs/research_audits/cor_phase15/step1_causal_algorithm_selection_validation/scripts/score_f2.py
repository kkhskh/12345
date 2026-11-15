"""XIII-D FREEZE-2 scorer (P1, P3, P4, P6, P7, P8; P1b and P5 are frozen in freeze 3).  Thresholds and units from predictions_13d_f2.json.  Inputs (LOGDIR):
unit_metrics_f2.jsonl, c_carrier_f2.json, d_das2_f2.json, d_rows_f2.json, e4_noninv_f2.json, d_routes2_f2.json.
Rules: missing required input -> arm FAILS; arm diagnostic conditions as written; CONFIRMED iff >= 1 diagnostic arm and all diagnostic arms pass;
FALSIFIED if any diagnostic arm fails; NON-DIAGNOSTIC otherwise.  Malformed inputs are caught and FAIL the arm (never crash).
Usage: score_f2.py [LOGDIR] -> LOGDIR/score_f2.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs")
P = json.load(open(os.path.join(HERE, "predictions_13d_f2.json"))); T = {p["id"]: p["thresholds"] for p in P["predictions"]}
def js(f):
    p = os.path.join(LD, f); return json.load(open(p)) if os.path.exists(p) else {}
UM = {}
p_ = os.path.join(LD, "unit_metrics_f2.jsonl")
if os.path.exists(p_):
    for l in open(p_): r = json.loads(l); UM.setdefault(r["unit"], {})[r["step"]] = r
def fin(u, k):
    try: return UM[u][4000][k]
    except KeyError: return None
def arm(name, diag, passed, det): return dict(arm=name, diagnostic=bool(diag), passed=None if passed is None else bool(passed), missing=passed is None, detail=det)
def verdict(arms):
    d = [a for a in arms if a["diagnostic"]]
    return "NON-DIAGNOSTIC" if not d else ("CONFIRMED" if all(a["passed"] for a in d) else "FALSIFIED")
def safe(fn, name):
    try: return fn()
    except Exception as ex: return [arm(name, True, None, f"error: {ex.__class__.__name__}: {ex}")]
def vec(u): a, b = fin(u, "lik_RO1"), fin(u, "mix_mass"); return None if a is None or b is None else np.array([a, b])
res = {}
# ---- P1 readout-map causality
def p1():
    t = T["P1"]; arms = []; R = {k: np.array(v) for k, v in t["refs"].items()}
    for cell, wref, oref in t["w_only_cells"]:
        v, vw, vo = vec(cell), R[wref], R[oref]
        if v is None: arms.append(arm(f"a:{cell}", True, None, "missing")); continue
        kl = fin(cell, "KL"); comp = kl is not None and kl <= t["max_KL"]
        sep = float(np.linalg.norm(vw - vo)); dw, do = float(np.linalg.norm(v - vw)), float(np.linalg.norm(v - vo))
        arms.append(arm(f"a:{cell}", comp and sep >= t["min_sep"], dw < do and dw <= t["max_dw"], dict(v=v.tolist(), d_W=dw, d_other=do, sep=sep, KL=kl, competent=comp)))
    return arms
res["P1"] = dict(arms=safe(p1, "P1"))
# ---- P3 / P4 carrier
CA = js("c_carrier_f2.json")
def p3():
    t = T["P3"]; arms = []
    for u in t["units"]:
        r = CA.get(u)
        if r is None: arms.append(arm(u, True, None, "missing")); continue
        S = r["sets"]; d = S["DAS"]
        arms.append(arm(f"{u}:a sufficiency beyond source-free", True, d["SUFF"]["slope"] >= t["min_suff"] and d["SUFF"]["slope"] >= d["SRCF"]["slope"] + t["min_excess_srcf"], dict(suff=d["SUFF"]["slope"], srcf=d["SRCF"]["slope"])))
        arms.append(arm(f"{u}:b null preservation", True, d["NULL_JD0"]["rel_to_JDM_patch"] <= t["max_null_D0"] and d["NULL_DEC"]["rel_to_JDM_patch"] <= t["max_null_DEC"],
                        dict(D0=d["NULL_JD0"]["rel_to_JDM_patch"], DEC=d["NULL_DEC"]["rel_to_JDM_patch"])))
        arms.append(arm(f"{u}:c encoding vs decoding", True, S["ENC_m"]["SUFF"]["slope"] >= t["min_enc"] and S["DEC_m"]["SUFF"]["slope"] <= t["max_dec"], dict(enc=S["ENC_m"]["SUFF"]["slope"], dec=S["DEC_m"]["SUFF"]["slope"])))
        c = r["CONTENT"]; arms.append(arm(f"{u}:d content", True, c["m"]["linear"] >= t["min_content"] and c["logr"]["linear"] >= t["min_content"], dict(m=c["m"]["linear"], logr=c["logr"]["linear"])))
        tr = r["TRACE"]; cr = [tr["corr_dscore_dlogr"][h] for h in tr["corr_dscore_dlogr"]]
        arms.append(arm(f"{u}:e trace", len(cr) > 0, tr["keys_and_queries"] >= t["min_qk"] and tr["values_only"] <= t["max_values"] and len(cr) > 0 and min(cr) >= t["min_corr"], dict(qk=tr["keys_and_queries"], values=tr["values_only"], corr=cr)))
    return arms
res["P3"] = dict(arms=safe(p3, "P3"))
def p4():
    t = T["P4"]; arms = []
    for u in t["units"]:
        r = CA.get(u)
        if r is None: arms.append(arm(u, True, None, "missing")); continue
        S = r["sets"]
        arms.append(arm(f"{u}:a natural-donor necessity", True, S["DAS"]["NEC"]["reverse"]["remaining_slope"] <= t["max_reverse_das"] and S["RAND0"]["NEC"]["reverse"]["remaining_slope"] >= t["min_reverse_rand"],
                        dict(das=S["DAS"]["NEC"]["reverse"]["remaining_slope"], rand=S["RAND0"]["NEC"]["reverse"]["remaining_slope"])))
        arms.append(arm(f"{u}:b synthetic ablation is not specific", True, S["DAS"]["NEC"]["ablate"]["damage_KL"] >= t["min_damage"], dict(damage=S["DAS"]["NEC"]["ablate"]["damage_KL"])))
    return arms
res["P4"] = dict(arms=safe(p4, "P4"))
# ---- P6 corrected mediation
DD, DR = js("d_das2_f2.json"), js("d_rows_f2.json")
def p6():
    t = T["P6"]; arms = []
    for u in t["elig_units"]:
        r = DD.get(u)
        if r is None: arms.append(arm(u, True, None, "missing")); continue
        e = r["sites"]["ALLX2"]["k8"]; D_, S_ = e["DAS"], e["SRCFREE"]; ex = float(np.mean([D_[f]["own_share"] - S_[f]["own_share"] for f in D_]))
        ok = ex >= t["min_excess"] and all(D_[f]["R2_E"] - D_[f]["R2_A2"] >= t["min_margin"] for f in ("E5", "E3", "QF4")) and D_["QF5"]["R2_E"] - D_["QF5"]["R2_A1"] >= t["min_margin"] \
             and all(D_[f]["R2_E"] - D_[f]["R2_G"] >= t["min_margin"] for f in ("E5", "QF4"))
        arms.append(arm(f"elig:{u}", True, ok, dict(excess=ex, R2={f: {w: D_[f][f"R2_{w}"] for w in ("E", "A1", "A2", "G")} for f in D_})))
    for u in t["row_units"]:
        r = DR.get(u)
        if r is None: arms.append(arm(u, True, None, "missing")); continue
        if not r["mixture_heads"]: arms.append(arm(f"rows:{u}", False, True, "no mixture head")); continue
        A = r["arms"]; ok = A["TRUE"]["R2"] >= t["min_true"] and A["SHUFD"]["R2"] <= t["max_decoy"] and A["SHUFD_vs_borrowed"]["R2"] <= t["max_decoy"] and A["NOOP"]["max_abs_dlogit_vs_unmodified"] <= t["noop_tol"]
        arms.append(arm(f"rows:{u}", True, ok, dict(TRUE=A["TRUE"]["R2"], SHUFD=A["SHUFD"]["R2"], borrowed=A["SHUFD_vs_borrowed"]["R2"], noop=A["NOOP"]["max_abs_dlogit_vs_unmodified"])))
    return arms
res["P6"] = dict(arms=safe(p6, "P6"))
# ---- P7 decision-error mechanism (positional-asymmetry hypothesis predicted FALSE)
E4 = js("e4_noninv_f2.json")
def p7():
    t = T["P7"]; arms = []
    for u in t["units"]:
        r = E4.get("models", {}).get(u)
        if r is None: arms.append(arm(u, True, None, "missing")); continue
        arms.append(arm(u, True, r["SYM1"]["MOVE_reduction"] <= t["max_reduction"] and r["SYM1"]["KL_nat"] > r["BASE"]["KL_nat"], dict(red=r["SYM1"]["MOVE_reduction"], KL=(r["BASE"]["KL_nat"], r["SYM1"]["KL_nat"]))))
    return arms
res["P7"] = dict(arms=safe(p7, "P7"))
# ---- P8 routes: components of one distributed mechanism
DT = js("d_routes2_f2.json")
def p8():
    t = T["P8"]; arms = []
    for u, kind in t["units"]:
        r = DT.get(u)
        if r is None: arms.append(arm(u, True, None, "missing")); continue
        b = r["block"]; base = b["none"]["rel_slope"]; ints = r["small"]["interactions"]
        singles = dict(Q=b["RO"]["rel_slope"] / base, R=b["QO"]["rel_slope"] / base, O=b["QR"]["rel_slope"] / base)
        keys = ("Q", "R") if kind == "mixture" else ("Q", "R", "O")
        ok = max(abs(v) for v in ints.values()) <= t["max_interaction"] and all(singles[k] < t["max_single_route"] for k in keys)
        arms.append(arm(f"{kind}:{u}", True, ok, dict(interactions=ints, single_open=singles)))
    return arms
res["P8"] = dict(arms=safe(p8, "P8"))
for k in res: res[k]["verdict"] = verdict(res[k]["arms"])
json.dump(dict(scored=res, summary={k: v["verdict"] for k, v in res.items()}), open(os.path.join(LD, "score_f2.json"), "w"), indent=1, default=float)
print({k: v["verdict"] for k, v in res.items()})
