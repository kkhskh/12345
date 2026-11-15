"""Preflight of score_f3a.py: P0 prediction-consistent synthetic files -> all CONFIRMED; N1 supply fails at lr3 -> P5a,P5b FALSIFIED; N2 supRO fails
-> P5b FALSIFIED only; N3 L1only repairs -> P5c FALSIFIED; N4 empty -> all FALSIFIED; N5 malformed file -> all FALSIFIED (no crash)."""
import json, os, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); U = "uP165_s_20"
def mk(lr1, lr3, raw=None):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "adapt4_f3a"))
    for lr, A in (("0.001", lr1), ("0.003", lr3)):
        if A is None: continue
        open(os.path.join(d, "adapt4_f3a", f"{U}_b0.6_lr{lr}.json"), "w").write(raw if raw else json.dumps({a: dict(acc=0.75, J_DM=dict(slope=s)) for a, s in A.items()}))
    subprocess.run([sys.executable, os.path.join(HERE, "score_f3a.py"), d], capture_output=True); return json.load(open(os.path.join(d, "score_f3a.json")))["summary"]
G1 = dict(supL2=0.02, L0only=0.9, L1only=0.0, L0attn=0.5, L0mlp=0.1); G3 = dict(supRO=0.8, supQS=0.8, supL2=0.9, L2=0.0)
C = {}
def chk(n, s, w): C[n] = dict(summary=s, ok=all(s[k] == v for k, v in w.items()))
chk("P0", mk(G1, G3), dict(P5a="CONFIRMED", P5b="CONFIRMED", P5c="CONFIRMED"))
chk("N1", mk(G1, dict(G3, supRO=0.02, supQS=0.02, supL2=0.03)), dict(P5a="FALSIFIED", P5b="FALSIFIED", P5c="CONFIRMED"))
chk("N2", mk(G1, dict(G3, supRO=0.05)), dict(P5a="CONFIRMED", P5b="FALSIFIED", P5c="CONFIRMED"))
chk("N3", mk(dict(G1, L1only=0.8), G3), dict(P5c="FALSIFIED"))
chk("N4", mk(None, None), dict(P5a="FALSIFIED", P5b="FALSIFIED", P5c="FALSIFIED"))
chk("N5", mk(G1, G3, raw="{not json"), dict(P5a="FALSIFIED", P5b="FALSIFIED", P5c="FALSIFIED"))
json.dump(C, open(os.path.join(HERE, "..", "logs", "preflight_f3a.json"), "w"), indent=1)
for k, v in C.items(): print(k, "OK" if v["ok"] else "FAIL", v["summary"])
sys.exit(0 if all(v["ok"] for v in C.values()) else 1)
