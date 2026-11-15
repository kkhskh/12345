"""Preflight of score_f2.py on synthetic inputs built from the DEVELOPMENT outputs relabelled to the confirmation unit names (no confirmation
output is read).  P0 dev-consistent -> all CONFIRMED.  Negative controls, each must flip its target:
N1 W-only cells land on the other diagonal -> P1 FALSIFIED; N2 decoder direction carries the effect (decodable = used) -> P3 FALSIFIED;
N3 natural-donor reverse leaves the effect -> P4 FALSIFIED; N4 graded rule G explains as well as E -> P6 FALSIFIED; N5 symmetrising removes the
error -> P7 FALSIFIED; N6 one route alone suffices -> P8 FALSIFIED; N7 empty logs -> all FALSIFIED; N8 all A-C runs non-competent -> P1
NON-DIAGNOSTIC; N9 row units without a mixture head -> P6 still scored on eligibility arms (CONFIRMED); N10 malformed carrier file -> P3/P4
FALSIFIED without crashing.  Writes logs/preflight_f2.json; exits 1 if any check fails."""
import json, os, sys, subprocess, tempfile, copy
HERE = os.path.dirname(os.path.abspath(__file__)); LG = os.path.join(HERE, "..", "logs")
P = {p["id"]: p for p in json.load(open(os.path.join(HERE, "predictions_13d_f2.json")))["predictions"]}
dev = {f: json.load(open(os.path.join(LG, f))) for f in ("c_carrier_dev.json", "d_das_dev.json", "d_rows_dev.json", "e4_noninv_dev.json", "d_routes2_dev.json")}
def build():
    t1 = P["P1"]["thresholds"]; um = []
    for cell, wref, _ in t1["w_only_cells"]:
        l, m = t1["refs"][wref]; um.append(dict(unit=cell, step=4000, lik_RO1=l + 0.03, mix_mass=m - 0.03, KL=0.004))
    ca = {u: copy.deepcopy(dev["c_carrier_dev.json"]["uP161:s:21"]) for u in P["P3"]["thresholds"]["units"]}
    dd = {u: copy.deepcopy(dev["d_das_dev.json"]["uP161:s:22"]) for u in P["P6"]["thresholds"]["elig_units"]}
    dr = {u: copy.deepcopy(dev["d_rows_dev.json"]["uP161:s:21"]) for u in P["P6"]["thresholds"]["row_units"]}
    e4 = dict(models={u: copy.deepcopy(dev["e4_noninv_dev.json"]["models"]["uP161:s:22"]) for u in P["P7"]["thresholds"]["units"]})
    rt = {u: copy.deepcopy(dev["d_routes2_dev.json"]["uP161:s:21" if k == "mixture" else "uP161:s:22"]) for u, k in P["P8"]["thresholds"]["units"]}
    return dict(um=um, ca=ca, dd=dd, dr=dr, e4=e4, rt=rt)
def score(D):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "unit_metrics_f2.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in D.get("um", [])))
    for k, f in (("ca", "c_carrier_f2.json"), ("dd", "d_das2_f2.json"), ("dr", "d_rows_f2.json"), ("e4", "e4_noninv_f2.json"), ("rt", "d_routes2_f2.json")):
        if k in D: open(os.path.join(d, f), "w").write(D[k] if isinstance(D[k], str) else json.dumps(D[k]))
    r = subprocess.run([sys.executable, os.path.join(HERE, "score_f2.py"), d], capture_output=True, text=True)
    if r.returncode: return dict(error=r.stderr[-400:])
    return json.load(open(os.path.join(d, "score_f2.json")))["summary"]
checks = {}
def chk(nm, s, want): checks[nm] = dict(summary=s, ok="error" not in s and all(s.get(k) in v for k, v in want.items()))
ALL = ("P1", "P3", "P4", "P6", "P7", "P8")
chk("P0", score(build()), {k: ["CONFIRMED"] for k in ALL})
D = build(); t1 = P["P1"]["thresholds"]
for r, (cell, wref, oref) in zip(D["um"], t1["w_only_cells"]): r["lik_RO1"], r["mix_mass"] = t1["refs"][oref]
chk("N1", score(D), {"P1": ["FALSIFIED"]})
D = build()
for u in D["ca"]: D["ca"][u]["sets"]["DEC_m"]["SUFF"]["slope"] = 0.6
chk("N2", score(D), {"P3": ["FALSIFIED"]})
D = build()
for u in D["ca"]: D["ca"][u]["sets"]["DAS"]["NEC"]["reverse"]["remaining_slope"] = 0.8
chk("N3", score(D), {"P4": ["FALSIFIED"]})
D = build()
for u in D["dd"]:
    for f, v in D["dd"][u]["sites"]["ALLX2"]["k8"]["DAS"].items(): v["R2_G"] = v["R2_E"]
chk("N4", score(D), {"P6": ["FALSIFIED"]})
D = build()
for u in D["e4"]["models"]: D["e4"]["models"][u]["SYM1"]["MOVE_reduction"] = 0.5
chk("N5", score(D), {"P7": ["FALSIFIED"]})
D = build()
for u in D["rt"]: D["rt"][u]["block"]["QO"]["rel_slope"] = D["rt"][u]["block"]["none"]["rel_slope"]
chk("N6", score(D), {"P8": ["FALSIFIED"]})
chk("N7", score({}), {k: ["FALSIFIED"] for k in ALL})
D = build()
for r in D["um"]: r["KL"] = 0.2
chk("N8", score(D), {"P1": ["NON-DIAGNOSTIC"]})
D = build()
for u in D["dr"]: D["dr"][u]["mixture_heads"] = []
chk("N9", score(D), {"P6": ["CONFIRMED"]})
D = build(); D["ca"] = {u: {"sets": {}} for u in D["ca"]}
chk("N10", score(D), {"P3": ["FALSIFIED"], "P4": ["FALSIFIED"]})
json.dump(checks, open(os.path.join(LG, "preflight_f2.json"), "w"), indent=1)
for k, v in checks.items(): print(k, "OK" if v["ok"] else "FAIL", v["summary"])
sys.exit(0 if all(v["ok"] for v in checks.values()) else 1)
