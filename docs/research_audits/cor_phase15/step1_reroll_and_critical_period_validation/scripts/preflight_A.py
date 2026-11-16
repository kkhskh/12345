"""Preflight of score_A.py on SYNTHETIC outputs in a temporary tree (no real output is read).
P0 complete data, outcomes independent across contexts (re-roll) -> verdicts produced, R1-W and R1-L not TRANSFER
P1 complete data with a strong transferable W component -> R1-W TRANSFER (FALSIFIED)
N1 one stop_done.json missing -> REFUSED (exit 2, no score file)
N2 one run's metrics missing -> REFUSED
N3 configuration mismatch (wrong W in factors) -> REFUSED
N4 continuation manifest missing -> REFUSED
N5 continuation set violating the rule -> REFUSED
N6 one continuation final.json missing -> REFUSED
N7 malformed metrics line -> REFUSED
Writes logs/preflight_A.json; exit 1 if any check fails."""
import json, os, sys, shutil, subprocess, tempfile, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); LG = os.path.join(HERE, "..", "logs")
MAN = json.load(open(os.path.join(LG, "manifest_A.json"))); D = MAN["design"]
def build(td, mode="null", drop=None):
    lg, ck = os.path.join(td, "logs"), os.path.join(td, "ft"); os.makedirs(lg); os.makedirs(ck)
    json.dump(MAN, open(os.path.join(lg, "manifest_A.json"), "w"))
    r = np.random.default_rng(7); wq = {s: r.uniform() for s in D["W_seeds"]}
    um = []
    for x in D["runs"]:
        d = os.path.join(ck, x["tag"]); os.makedirs(d)
        if drop != ("stop", x["tag"]): json.dump(dict(stop=1211), open(os.path.join(d, "stop_done.json"), "w"))
        for f in ("step1211.npz", "opt_step1211.pkl"): open(os.path.join(d, f), "w").write("x")
        mix = (0.1 + 0.85 * (wq[x["W"]] > 0.5) + r.normal(0, 0.03)) if (mode == "transfer" and x["type"] == "W") else float(r.choice([0.15, 0.95, 0.6]))
        lik = 0.7 if mix < 0.5 else 0.02
        if drop != ("um", x["tag"]): um.append(json.dumps(dict(unit=x["spec"], step=1211, factors=[x["parent"], x["L"], x["W"] + (1 if drop == ("cfg", x["tag"]) else 0), x["D"], "stop1211"], lik_RO1=lik, mix_mass=mix, KL=0.01, acc=0.85, route_Q=0.1)))
    if drop == "malformed": um.append("{not json")
    open(os.path.join(lg, "unit_metrics_A.jsonl"), "w").write("\n".join(um) + "\n")
    cont = D["continuation"]["fixed"] + [x["tag"] for x in D["runs"] if x["tag"] not in D["continuation"]["fixed"]][:2]
    if drop == "rule": cont = cont[:-1]
    if drop != "nocont": json.dump(dict(continuation_final=cont), open(os.path.join(lg, "manifest_A_cont.json"), "w"))
    uc = []
    for t in cont:
        x = next(y for y in D["runs"] if y["tag"] == t); bd = os.path.join(ck, t.replace("_stop1211", "") + "_br1211"); os.makedirs(bd)
        if drop != ("final", t): open(os.path.join(bd, "final.json"), "w").write("{}")
        uc.append(json.dumps(dict(unit=x["spec"].replace(":stop1211", ":br1211"), step=4000, lik_RO1=0.7, mix_mass=0.2, KL=0.004)))
    open(os.path.join(lg, "unit_metrics_Acont.jsonl"), "w").write("\n".join(uc) + "\n")
    return lg, ck
def run(mode="null", drop=None):
    td = tempfile.mkdtemp(); lg, ck = build(td, mode, drop)
    p = subprocess.run([sys.executable, os.path.join(HERE, "score_A.py"), lg, ck], capture_output=True, text=True)
    out = json.load(open(os.path.join(lg, "score_A.json"))) if os.path.exists(os.path.join(lg, "score_A.json")) else None
    shutil.rmtree(td); return p.returncode, out
t0 = D["runs"][5]["tag"]; ct = D["continuation"]["fixed"][0]; C = {}
rc, o = run(); C["P0"] = dict(rc=rc, summary=o and o["summary"], ok=rc == 0 and o is not None and o["R1"]["R1-W"]["outcome"] != "TRANSFER")
rc, o = run("transfer"); C["P1"] = dict(rc=rc, summary=o and o["summary"], ok=rc == 0 and o is not None and o["R1"]["R1-W"]["outcome"] == "TRANSFER")
for nm, dr in (("N1", ("stop", t0)), ("N2", ("um", t0)), ("N3", ("cfg", t0)), ("N4", "nocont"), ("N5", "rule"), ("N6", ("final", ct)), ("N7", "malformed")):
    rc, o = run(drop=dr); C[nm] = dict(rc=rc, ok=rc == 2 and o is None)
json.dump(C, open(os.path.join(LG, "preflight_A.json"), "w"), indent=1, default=str)
for k, v in C.items(): print(k, "OK" if v["ok"] else "FAIL", v.get("summary", ""), "rc", v["rc"])
sys.exit(0 if all(v["ok"] for v in C.values()) else 1)
