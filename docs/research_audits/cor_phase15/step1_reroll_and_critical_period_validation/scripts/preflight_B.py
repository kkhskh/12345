"""Preflight of score_B.py on synthetic outputs (temporary tree).  P0 route-specific, specifically-early pattern -> R3 and R4 CONFIRMED;
N1 matched noise flips like the real mask -> R3 FALSIFIED; N2 late disturbances flip -> R4 FALSIFIED; N3 sham not bitwise -> R3 not CONFIRMED;
N4 a late final.json missing -> REFUSED; N5 a 1211 metric missing -> REFUSED; N6 malformed metrics -> REFUSED; N7 continuation missing ->
REFUSED.  Writes logs/preflight_B.json."""
import json, os, sys, shutil, subprocess, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); LG = os.path.join(HERE, "..", "logs"); D = json.load(open(os.path.join(LG, "B_design.json")))
EARLY, MIX = dict(lik_RO1=0.75, mix_mass=0.18), dict(lik_RO1=0.02, mix_mass=0.95)
def outcome(r, mode):
    base = EARLY if r["base"] == "E" else MIX; other = MIX if r["base"] == "E" else EARLY
    if r["cond"].startswith("real"): return other
    if r["cond"] == "dist100" and mode == "distflip": return other
    if r["kind"] == "late" and mode == "lateflip": return other
    return base
def build(td, mode="ok", drop=None):
    lg, ck = os.path.join(td, "logs"), os.path.join(td, "ft"); os.makedirs(lg); os.makedirs(ck); json.dump(D, open(os.path.join(lg, "B_design.json"), "w"))
    u1, u4 = [], []
    for r in D["runs"]:
        d = os.path.join(ck, r["tag"]); os.makedirs(d)
        if r["kind"] == "short":
            json.dump(dict(stop=1211), open(os.path.join(d, "stop_done.json"), "w"))
            val = 1.0 if (r["cond"] == "sham100" and mode == "shamdiff") else 0.0
            np.savez(os.path.join(d, "step1211.npz"), np.array([val])); m = outcome(r, mode)
            if drop != ("um", r["tag"]): u1.append(dict(unit=r["spec"], step=1211, factors=[r["parent"], r["L"], r["W"], r["D"], ""], KL=0.01, **m))
        else:
            if drop != ("final", r["tag"]): open(os.path.join(d, "final.json"), "w").write("{}")
            np.savez(os.path.join(d, "step4000.npz"), np.array([0.0])); u4.append(dict(unit=r["spec"], step=4000, factors=[r["parent"], r["L"], r["W"], r["D"], ""], KL=0.005, **outcome(r, mode)))
    for b, sp in D["anchor_baselines"].items():
        m = EARLY if b == "E" else MIX; u1.append(dict(unit=sp, step=1211, KL=0.01, **m)); u4.append(dict(unit=sp, step=4000, KL=0.004, **m))
    for t in D["continuation"]:
        r = next(x for x in D["runs"] if x["tag"] == t); ct = f'{r["parent"]}_fx_L{r["L"]}_W{r["W"]}_D{r["D"]}_br1211{r["suffix"]}'
        os.makedirs(os.path.join(ck, ct))
        if drop != "cont": open(os.path.join(ck, ct, "final.json"), "w").write("{}")
        u4.append(dict(unit=f'{r["parent"]}:fx:{r["L"]}:{r["W"]}:{r["D"]}:br1211{r["suffix"]}', step=4000, KL=0.004, **outcome(r, mode)))
    s1 = "\n".join(json.dumps(x) for x in u1) + ("\n{bad" if drop == "malformed" else "") + "\n"
    open(os.path.join(lg, "unit_metrics_B.jsonl"), "w").write(s1); open(os.path.join(lg, "unit_metrics_Bfinal.jsonl"), "w").write("\n".join(json.dumps(x) for x in u4) + "\n")
    return lg, ck
def run(mode="ok", drop=None):
    td = tempfile.mkdtemp(); lg, ck = build(td, mode, drop); p = subprocess.run([sys.executable, os.path.join(HERE, "score_B.py"), lg, ck], capture_output=True, text=True)
    o = json.load(open(os.path.join(lg, "score_B.json")))["summary"] if os.path.exists(os.path.join(lg, "score_B.json")) else None; shutil.rmtree(td); return p.returncode, o, p.stderr[-300:]
late0 = next(r["tag"] for r in D["runs"] if r["kind"] == "late"); short0 = next(r["tag"] for r in D["runs"] if r["kind"] == "short"); C = {}
rc, o, e = run(); C["P0"] = dict(o=o, ok=o == dict(R3="CONFIRMED", R4="CONFIRMED"), err=e)
rc, o, e = run("distflip"); C["N1"] = dict(o=o, ok=o is not None and o["R3"] == "FALSIFIED", err=e)
rc, o, e = run("lateflip"); C["N2"] = dict(o=o, ok=o is not None and o["R4"] == "FALSIFIED", err=e)
rc, o, e = run("shamdiff"); C["N3"] = dict(o=o, ok=o is not None and o["R3"] != "CONFIRMED", err=e)
for nm, dr in (("N4", ("final", late0)), ("N5", ("um", short0)), ("N6", "malformed"), ("N7", "cont")):
    rc, o, e = run(drop=dr); C[nm] = dict(rc=rc, ok=rc == 2 and o is None)
json.dump(C, open(os.path.join(LG, "preflight_B.json"), "w"), indent=1, default=str)
for k, v in C.items(): print(k, "OK" if v["ok"] else "FAIL", v.get("o"), v.get("err", "")[:200] if not v["ok"] else "")
sys.exit(0 if all(v["ok"] for v in C.values()) else 1)
