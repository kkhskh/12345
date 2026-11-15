"""XIII-C mandatory preflight of the frozen scorer (run BEFORE the freeze).  Builds synthetic bundles and checks that
  P0  a prediction-consistent bundle (values taken from development measurements, mapped onto confirmation names) is scored CONFIRMED for every
      prediction (the scorer can pass);
  N1  data-order null (every crossed cell equals its DATA diagonal; split cells equal their non-L2 reference)        -> R1 FALSIFIED
  N2  no-effect null (every intervention run equals its unintervened reference)                                      -> R2 FALSIFIED, R3 NON-DIAGNOSTIC or FALSIFIED
  N3  no-op success null (decoy rows as good as true rows; no-op changes logits)                                    -> R6 FALSIFIED
  N4  capacity null (every adaptation arm equally good)                                                              -> R4 FALSIFIED
  N5  'decodable = causal' null (probe interchange as large as the learned subspace)                                -> R5 FALSIFIED
  N6  source-free null (SRCFREE as large as DAS; A2 fits as well as E)                                               -> R7 FALSIFIED
  N7  redundancy null (each route alone sufficient)                                                                  -> R8 FALSIFIED
  N8  missing inputs (empty bundle)                                                                                  -> every prediction FALSIFIED (never 'inconclusive')
-> logs/preflight_13c.json; exits non-zero if any expectation fails."""
import json, os, sys, copy, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
P = json.load(open(os.path.join(HERE, "predictions_13c.json"))); T = {p["id"]: p["thresholds"] for p in P["predictions"]}
def um(l, m, mix546=None):
    d = {"4000": dict(lik_RO1=l, mix_mass=m)}
    if mix546 is not None: d["546"] = dict(lik_RO1=l, mix_mass=mix546)
    return d
# development-anchored values: mixture (0.006, 0.966) ~ uP161 s21; early (0.80, 0.17) ~ uP160 s21; hybrid (0.42, 0.92) ~ uP161 s23 / (0.32, 0.91) uP163 s21
MIX, EAR, HYB, HYB3 = (0.006, 0.966), (0.80, 0.17), (0.42, 0.92), (0.32, 0.91)
base = {"uP161:s:20": um(*MIX), "uP161:s:23": um(*HYB), "uP163:s:21": um(*HYB3), "uP163:s:22": um(0.80, 0.25), "uP160:s:21": um(*EAR), "uP161:s:21": um(*MIX), "uP160:s:23": um(0.03, 0.97)}
good = dict(base); good.update({"uP161:fx:20:20:23": um(0.01, 0.96, 0.92), "uP161:fx:23:23:20": um(0.40, 0.91, 0.47), "uP163:fx:21:21:22": um(0.30, 0.90, 0.5),
             "uP163:fx:22:22:21": um(0.79, 0.24, 0.2), "uP163:fx:22:21:21": um(0.78, 0.22, 0.21), "uP163:fx:21:22:22": um(0.33, 0.9, 0.5),
             "uP160:fx:21:21:21:ro0T600": um(0.00, 0.95), "uP161:fx:21:21:21:l2qT600": um(0.55, 0.6), "uP160:fx:23:23:23:l2qT600": um(0.6, 0.5)})
ff = {u: dict(rho_PM=0.93 - 0.25 * v["4000"]["lik_RO1"]) for u, v in good.items()}
rowsT = dict(mixture_heads=[1, 3], arms=dict(TRUE=dict(R2=0.95), SHUFD=dict(R2=0.003), NOOP=dict(R2=0.0, max_abs_dlogit_vs_unmodified=0.0), SHUF=dict(R2=0.04), PMEAN=dict(R2=0.05), SHUFB=dict(R2=0.03), OTHER=dict(R2=0.02), WHOLE=dict(R2=0.97)))
routesM = dict(sufficiency=dict(Q=0.45, R=0.5, O=0.02), redundancy=0.0); routesE = dict(sufficiency=dict(Q=0.12, R=0.4, O=0.46), redundancy=0.0)
ARMS_ = ("L2", "early01", "L1only", "L0only", "L2plus", "supL2", "supG", "supFrozen")
def adapt(*vals): return {a: {"J_DM": {"slope": (vals[i] if i < len(vals) else vals[-1])}} for i, a in enumerate(ARMS_)}
cc = {"C3": {"das_QSRO": {"k3": {"test_fid": {"slope": 0.85}}, "k8": {"test_fid": {"slope": 0.92}}}, "probe_RO": {"interchange_m": {"slope": 0.0}}, "probe_QS": {"interchange_m": {"slope": 0.01}},
             "probe_OS": {"interchange_m": {"slope": 0.0}}, "resp_QS": {"interchange_resp3": {"slope": 0.015}}}, "C4": {"shapley": {"A": 0.8}, "mixture_heads": [1, 3], "corr_dmass_drole": {"1": 0.89, "3": 0.87}}}
dd = {"sites": {}}
def das_entry(dshare, sshare, rE, rA1, rA2, rand=0.1):
    fam = ("E5", "E3", "QF5", "QF4")
    return dict(DAS={f: dict(own_share=dshare, R2_E=rE, R2_A1=(rA1 if f == "QF5" else rE), R2_A2=(rE if f == "QF5" else rA2)) for f in fam},
                SRCFREE={f: dict(own_share=sshare) for f in fam}, RAND=[{f: dict(own_share=rand) for f in fam}])
def bundle_good():
    b = dict(unit_metrics=copy.deepcopy(good), fair_fits=copy.deepcopy(ff), c_causal={}, d_rows={}, d_routes={}, d_das={}, adapt3={})
    for u, v in good.items():
        l, m = v["4000"]["lik_RO1"], v["4000"]["mix_mass"]
        b["d_rows"][u] = copy.deepcopy(rowsT) if l <= 0.15 else dict(mixture_heads=[], arms=dict(WHOLE=dict(R2=0.97)))
        b["d_routes"][u] = copy.deepcopy(routesM if l <= 0.15 else routesE)
    t4 = T.get("R4", {}); t5 = T.get("R5", {}); t7 = T.get("R7", {})
    if t4.get("late_unit"):
        b["adapt3"][t4["late_unit"].replace(":", "_")] = adapt(0.0, 0.92, 0.9, 0.0, 0.9, 0.7); b["adapt3"][t4["early_unit"].replace(":", "_")] = adapt(0.6, 0.9, 0.9, 0.7, 0.9, 0.7)
        for a, c_, thr in t4["late_arms"]: b["adapt3"][t4["late_unit"].replace(":", "_")][a]["J_DM"]["slope"] = (thr - 0.2 if c_ == "<" else thr + 0.1)
        for a, c_, thr in t4["early_arms"]: b["adapt3"][t4["early_unit"].replace(":", "_")][a]["J_DM"]["slope"] = (thr - 0.2 if c_ == "<" else thr + 0.1)
    if t5.get("unit"): b["c_causal"][f"{t5['unit']}|early01"] = copy.deepcopy(cc)
    for u in t7.get("units", []):
        e = das_entry(0.7, 0.3, 0.9, 0.3, 0.3); b["d_das"][u] = {"sites": {t7["site"]: {f"k{t7['k']}": e}}}
    return b
def score(b, name):
    p = os.path.join(L, f"_pf_{name}.json"); json.dump(b, open(p, "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "score_13c.py"), p, os.path.join(L, f"_pf_{name}_score.json")], capture_output=True, text=True)
    os.remove(p); s = json.load(open(os.path.join(L, f"_pf_{name}_score.json"))); os.remove(os.path.join(L, f"_pf_{name}_score.json")); return s["summary"], s
checks = {}; ok_all = True
def expect(name, summary, want):
    global ok_all
    good_ = {k: summary.get(k) in v for k, v in want.items()}; checks[name] = dict(summary=summary, expected=want, ok=all(good_.values()), detail=good_); ok_all &= all(good_.values())
ids = [p["id"] for p in P["predictions"]]
s, _ = score(bundle_good(), "P0"); expect("P0", s, {k: ["CONFIRMED"] for k in ids})
b = bundle_good()
for cell, I, D in T["R1"]["cells"]: b["unit_metrics"][cell] = copy.deepcopy(b["unit_metrics"][D])
for cell, L2r, O in T["R1"]["split_cells"]: b["unit_metrics"][cell] = copy.deepcopy(b["unit_metrics"][O])
s, _ = score(b, "N1"); expect("N1", s, {"R1": ["FALSIFIED"]})
b = bundle_good()
for u, ref in T["R2"]["ro0_units"] + T["R2"]["l2q_units"]: b["unit_metrics"][u] = copy.deepcopy(b["unit_metrics"][ref]); b["fair_fits"][u] = dict(b["fair_fits"][ref])
s, _ = score(b, "N2"); expect("N2", s, {"R2": ["FALSIFIED"], "R3": ["NON-DIAGNOSTIC", "FALSIFIED"]})
b = bundle_good()
for u, r in b["d_rows"].items():
    if r["mixture_heads"]: r["arms"]["SHUFD"]["R2"] = r["arms"]["TRUE"]["R2"]; r["arms"]["NOOP"]["max_abs_dlogit_vs_unmodified"] = 0.3
s, _ = score(b, "N3"); expect("N3", s, {"R6": ["FALSIFIED"]})
if T.get("R4", {}).get("late_unit"):
    b = bundle_good()
    for u in b["adapt3"]: b["adapt3"][u] = adapt(*([0.5] * 6))
    s, _ = score(b, "N4"); expect("N4", s, {"R4": ["FALSIFIED"]})
if T.get("R5", {}).get("unit"):
    b = bundle_good(); c = b["c_causal"][f"{T['R5']['unit']}|early01"]; c["C3"]["probe_QS"]["interchange_m"]["slope"] = 0.85; c["C3"]["resp_QS"]["interchange_resp3"]["slope"] = 0.85
    s, _ = score(b, "N5"); expect("N5", s, {"R5": ["FALSIFIED"]})
if T.get("R7", {}).get("units"):
    b = bundle_good()
    for u in T["R7"]["units"]: b["d_das"][u] = {"sites": {T["R7"]["site"]: {f"k{T['R7']['k']}": das_entry(0.6, 0.6, 0.8, 0.8, 0.8, rand=0.5)}}}
    s, _ = score(b, "N6"); expect("N6", s, {"R7": ["FALSIFIED"]})
b = bundle_good()
for u, r in b["d_routes"].items(): r["sufficiency"] = dict(Q=0.9, R=0.9, O=0.9); r["redundancy"] = 1.7
s, _ = score(b, "N7"); expect("N7", s, {"R8": ["FALSIFIED"]})
s, _ = score(dict(unit_metrics={}, fair_fits={}, c_causal={}, d_rows={}, d_routes={}, d_das={}, adapt3={}), "N8"); expect("N8", s, {k: ["FALSIFIED"] for k in ids})
json.dump(dict(ok=ok_all, checks=checks), open(os.path.join(L, "preflight_13c.json"), "w"), indent=1)
for k, v in checks.items(): print(k, "OK" if v["ok"] else "FAIL", v["summary"])
sys.exit(0 if ok_all else 1)
