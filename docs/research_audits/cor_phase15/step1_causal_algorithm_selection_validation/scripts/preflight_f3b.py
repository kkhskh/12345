"""Preflight of score_f3b.py: P0 consistent -> CONFIRMED; N1 W-only cell early -> FALSIFIED; N2 reference fails -> NON-DIAGNOSTIC; N3 empty ->
FALSIFIED; N4 D-only cell not early -> FALSIFIED."""
import json, os, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
E, N = dict(lik_RO1=0.7, mix_mass=0.2, KL=0.01), dict(lik_RO1=0.02, mix_mass=0.95, KL=0.005)
base = {"uP164:fx:20:21:21:stop1211": N, "uP164:fx:21:20:21:stop1211": N, "uP164:fx:21:21:20:stop1211": E, "uP164:s:21": E, "uP164:s:20": N}
def sc(m):
    d = tempfile.mkdtemp(); open(os.path.join(d, "unit_metrics_f3b.jsonl"), "w").write("".join(json.dumps(dict(v, unit=u, step=1211)) + "\n" for u, v in m.items()))
    subprocess.run([sys.executable, os.path.join(HERE, "score_f3b.py"), d], capture_output=True); return json.load(open(os.path.join(d, "score_f3b.json")))["summary"]["P10"]
C = dict(P0=sc(base) == "CONFIRMED", N1=sc(dict(base, **{"uP164:fx:20:21:21:stop1211": E})) == "FALSIFIED", N2=sc(dict(base, **{"uP164:s:21": N})) == "NON-DIAGNOSTIC",
         N3=sc({}) == "FALSIFIED", N4=sc(dict(base, **{"uP164:fx:21:21:20:stop1211": N})) == "FALSIFIED")
json.dump(C, open(os.path.join(HERE, "..", "logs", "preflight_f3b.json"), "w")); print(C); sys.exit(0 if all(C.values()) else 1)
