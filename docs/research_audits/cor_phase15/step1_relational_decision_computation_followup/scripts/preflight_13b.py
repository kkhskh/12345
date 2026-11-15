"""Scorer preflight (XIII-B): (1) score development units; (2) score each NULL unit ALONE (min_units overridden to 1) and require that no
prediction is confirmed by any single null unit; (3) report arm-level passes on null units (arms that do not discriminate that null on their own)."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
subprocess.run([sys.executable, os.path.join(HERE, "build_dev13b.py")], check=True)
spec = json.load(open(os.path.join(L, "predictions_13b.json"))); s1 = dict(spec); s1["predictions"] = [dict(P, min_units=1) for P in spec["predictions"]]
json.dump(s1, open("/tmp/spec13b_min1.json", "w"))
TARGETS = {"NULLNOOP": ["Q3", "Q4", "Q5", "Q6"], "NULLUNIFORM": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"], "NULLPOE": ["Q1"], "NULLA1": ["Q1", "Q4", "Q5"],
           "NULLSEEDSWITCH": ["Q2"], "NULLDECOYALL": ["Q4"]}   # predictions each null scenario is meant to defeat (frozen with the preflight)
bad, armp, other = [], [], []
for u in [l.strip() for l in open(os.path.join(L, "units_null13b.txt")) if l.strip()]:
    open("/tmp/u1.txt", "w").write(u + "\n")
    subprocess.run([sys.executable, os.path.join(HERE, "score_13b.py"), "--units", "/tmp/u1.txt", "--sfx", "_null13b"], env=dict(os.environ, SCORE_SPEC="/tmp/spec13b_min1.json"), capture_output=True)
    S = json.load(open(os.path.join(L, "score_13b_null13b.json")))
    for pid, p in S["predictions"].items():
        if p["verdict"] == "confirmed": (bad if pid in TARGETS[u.split(":")[1]] else other).append((pid, u))
        for a, x in p["arms"].items():
            if x["status"] == "pass": armp.append((pid, a, u.split(":")[1]))
r = subprocess.run([sys.executable, os.path.join(HERE, "score_13b.py"), "--units", os.path.join(L, "units_dev13b.txt"), "--sfx", "_dev13b"], capture_output=True, text=True)
json.dump(dict(ok=not bad, targets=TARGETS, targeted_predictions_confirmed_by_a_null_unit=bad, untargeted_confirmations=other, arm_passes_on_null_units=armp, dev_scoring=r.stdout), open(os.path.join(L, "preflight_13b.json"), "w"), indent=1)
print(r.stdout); print("PREFLIGHT", "OK" if not bad else f"FAIL {bad}"); print("arm passes on nulls:", armp)
