"""Preflight of score_f1.py: P0 prediction-consistent synthetic logs -> all CONFIRMED; N1 information lost -> P2a FALSIFIED; N2 100-step masks
flip -> P2b FALSIFIED; N3 information suppressed -> P2c FALSIFIED; N4 empty -> all FALSIFIED; N5 all runs non-competent -> all NON-DIAGNOSTIC."""
import json, os, sys, subprocess, tempfile, copy
HERE = os.path.dirname(os.path.abspath(__file__))
U = dict(F1a="uP160:fx:21:21:21:ro0T100", F1b="uP160:fx:21:21:21:ro0T250", F1c="uP160:fx:23:23:23:l2qT100", F1d="uP160:fx:23:23:23:l2qT250",
         F1e="uP160:fx:21:21:21:ro0T0_br546_w546-1146", F1f="uP160:fx:23:23:23:l2qT0_br546_w546-1146")
CLS = dict(F1a="early", F1b="mixture", F1c="mixture", F1d="hybrid", F1e="mixture", F1f="mixture")
def write(d, um, rs, rr):
    for f, rows in (("unit_metrics_f1.jsonl", um), ("route_status_f1.jsonl", rs), ("route_status_f1_react.jsonl", rr)):
        open(os.path.join(d, f), "w").write("".join(json.dumps(r) + "\n" for r in rows))
def build(mod=None):
    um, rs, rr = [], [], []
    for k, u in U.items():
        c = CLS[k]; l, m = dict(early=(0.8, 0.2), mixture=(0.01, 0.95), hybrid=(0.4, 0.9))[c]
        um.append(dict(unit=u, step=4000, lik_RO1=l, mix_mass=m, KL=0.003))
        for s, pl, pp in ((246, 0.14, 0.82), (546, 0.06, 0.8), (4000, l if c != "mixture" else 0.005, 0.84)): rs.append(dict(unit=f"{u}@{s}", lik_RO1=pl if s < 4000 else (l if c != "mixture" else 0.005), present_pooled_RO1=pp))
        rr.append(dict(unit=u, present_pooled_RO1=0.84, lik_RO1=0.005 if c == "mixture" else l, react=dict(before_acc=0.34, after_acc=0.56), react_ro1_removed=dict(before_acc=0.34, after_acc=0.34)))
    if mod: mod(um, rs, rr)
    return um, rs, rr
def score(um, rs, rr):
    d = tempfile.mkdtemp(); write(d, um, rs, rr); subprocess.run([sys.executable, os.path.join(HERE, "score_f1.py"), d], capture_output=True); return json.load(open(os.path.join(d, "score_f1.json")))["summary"]
checks = {}
def chk(nm, s, want): checks[nm] = dict(summary=s, ok=all(s[k] in v for k, v in want.items()))
chk("P0", score(*build()), {k: ["CONFIRMED"] for k in ("P2a", "P2b", "P2c")})
def lost(um, rs, rr):
    for r in rr: r["present_pooled_RO1"] = 0.1; r["react"]["after_acc"] = 0.35
chk("N1", score(*build(lost)), {"P2a": ["FALSIFIED"]})
def flip(um, rs, rr):
    for r in um:
        if r["unit"] == U["F1a"]: r.update(lik_RO1=0.01, mix_mass=0.95)
        if r["unit"] == U["F1c"]: r.update(lik_RO1=0.8, mix_mass=0.2)
chk("N2", score(*build(flip)), {"P2b": ["FALSIFIED"]})
def supp(um, rs, rr):
    for r in rs:
        if r["unit"].endswith("@4000"): r["present_pooled_RO1"] = 0.3
chk("N3", score(*build(supp)), {"P2c": ["FALSIFIED"]})
chk("N4", score([], [], []), {k: ["FALSIFIED"] for k in ("P2a", "P2b", "P2c")})
def incomp(um, rs, rr):
    for r in um: r["KL"] = 0.2
chk("N5", score(*build(incomp)), {k: ["NON-DIAGNOSTIC"] for k in ("P2a", "P2b", "P2c")})
ok = all(v["ok"] for v in checks.values()); json.dump(dict(ok=ok, checks=checks), open(os.path.join(HERE, "..", "logs", "preflight_f1.json"), "w"), indent=1)
for k, v in checks.items(): print(k, "OK" if v["ok"] else "FAIL", v["summary"])
sys.exit(0 if ok else 1)
