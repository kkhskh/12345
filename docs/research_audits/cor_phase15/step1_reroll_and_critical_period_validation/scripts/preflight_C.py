"""Preflight of score_C.py: P0 the C-C1 (development) record relabelled as C-C2 -> both CONFIRMED (it is the development basis); N1 k=1 weak ->
R5 FALSIFIED; N2 decoder carries the effect -> R6 FALSIFIED; N3 record missing -> REFUSED; N4 incomplete record -> REFUSED."""
import json, os, sys, subprocess, tempfile, copy
HERE = os.path.dirname(os.path.abspath(__file__)); dev = json.load(open(os.path.join(HERE, "..", "logs", "c_audit.json")))["uP160:s:23"]
def run(rec):
    d = tempfile.mkdtemp()
    if rec is not None: json.dump({"uP165:s:21": rec}, open(os.path.join(d, "c_audit.json"), "w"))
    p = subprocess.run([sys.executable, os.path.join(HERE, "score_C.py"), d], capture_output=True, text=True)
    return p.returncode, (json.load(open(os.path.join(d, "score_C.json")))["summary"] if os.path.exists(os.path.join(d, "score_C.json")) else None)
C = {}; rc, o = run(copy.deepcopy(dev)); C["P0"] = dict(o=o, ok=o == dict(R5="CONFIRMED", R6="CONFIRMED"))
r = copy.deepcopy(dev); r["C4"]["1"]["SUFF_slope"] = 0.3; rc, o = run(r); C["N1"] = dict(o=o, ok=o is not None and o["R5"] == "FALSIFIED")
r = copy.deepcopy(dev); r["C2"]["geometry"]["DEC_m"]["SUFF_slope"] = 0.7; rc, o = run(r); C["N2"] = dict(o=o, ok=o is not None and o["R6"] == "FALSIFIED")
rc, o = run(None); C["N3"] = dict(rc=rc, ok=rc == 2 and o is None)
r = copy.deepcopy(dev); del r["C4"]; rc, o = run(r); C["N4"] = dict(rc=rc, ok=rc == 2 and o is None)
json.dump(C, open(os.path.join(HERE, "..", "logs", "preflight_C.json"), "w"), indent=1); [print(k, "OK" if v["ok"] else "FAIL", v.get("o")) for k, v in C.items()]; sys.exit(0 if all(v["ok"] for v in C.values()) else 1)
