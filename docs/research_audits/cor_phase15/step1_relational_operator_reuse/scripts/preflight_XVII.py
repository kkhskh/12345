"""XVII preflight (run before any confirmation output exists): score_XVII.py on a SYNTHETIC package with planted analysis results.
Scenario 1 plants: every model evaluable; transfer, late-less-dependent, routing-insufficient, code keying and the learning pattern all present
-> expect pooled STAGGERED, FOLLOWS_CODE, TRANSFER, LATE_LESS_CODE_DEPENDENT, ROUTING_INSUFFICIENT, CODE_KEYING_PERSISTS, INTERFERENCE_NO_ACCELERATION.
Scenario 2 plants the opposite pattern -> SYNCHRONOUS (no partial checkpoints) and NOT_EVALUABLE elsewhere; scenario 3 plants evaluable
models without transfer and with sufficient routing -> NO_TRANSFER, ROUTING_SUFFICIENT, NOT_PERSISTENT, NOT_SUPPORTED.
Refusals: missing analysis file, blind = false, missing snapshot, changed frozen file, missing learning arm.  -> logs/preflight_XVII.json"""
import json, os, sys, shutil, subprocess, tempfile, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src"))
import numpy as np, jax, train_tasks as TT
D = json.load(open(os.path.join(PKG, "logs", "XVII_design.json")))
hsh = lambda p: hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(p))).hexdigest(); IH = {r["seed"]: hsh(TT.init("gate", r["seed"])) for r in D["runs"]}
def analysis(r, sc):
    A = dict(cond=r["cond"], run=r["dir"], t1=500)
    if sc == 2: A.update(status="no_partial_checkpoint", slot_acquisition_step={str(q): 800 for q in range(6)}); return A
    good = sc == 1; sel = 900; Fsel = [1, 1, 1, 0, 0, 0]
    pairs = [[i, j] for i in (0, 1, 2) for j in (3, 4, 5)]; n = len(pairs); f = lambda v: [v] * n
    bat = {"0-1": dict(recipient=f(0.5), donor=f(0.0)), "1-2": dict(recipient=f(0.0), donor=f(0.6)), "2-3": dict(recipient=f(0.0), donor=f(0.95 if good else 0.1)),
           "0-2": dict(recipient=f(0.95 if good else 0.1), donor=f(0.0)), "1-3": dict(recipient=f(0.0), donor=f(0.5)), "0-3": dict(recipient=f(1.0), donor=f(0.0)),
           "UU_control_0-2": dict(recipient=f(0.0)), "route_L1": dict(recipient=f(0.05 if good else 0.7)), "route_L01": dict(recipient=f(0.1 if good else 0.8))}
    late = {"0-1": dict(recipient=f(0.6 if good else 0.95), donor=f(0.6 if good else 0.95)), "1-3": dict(recipient=f(0.6 if good else 0.95), donor=f(0.6 if good else 0.95)),
            "0-2": dict(recipient=f(0.95), donor=f(0.95)), "2-3": dict(recipient=f(0.9), donor=f(0.9))}
    A.update(status="evaluable", selected=sel, A=[0, 1, 2], U=[3, 4, 5], pairs=pairs, battery=bat, per_slot={str(sel): dict(F_G=Fsel)},
             slot_acquisition_step={str(q): 800 for q in range(6)}, late=1300, late_battery=late); return A
def build(T, sc):
    P = os.path.join(T, "pkg"); os.makedirs(P)
    for sub in ("logs", "src", "scripts"): shutil.copytree(os.path.join(PKG, sub), os.path.join(P, sub), ignore=shutil.ignore_patterns("__pycache__", "claims", "run", "expl", "analysis"))
    os.makedirs(os.path.join(P, "logs", "analysis"))
    for sib in os.listdir(ROOT):
        if sib.startswith("step1") and sib != os.path.basename(PKG): os.symlink(os.path.join(ROOT, sib), os.path.join(T, sib))
    for r in D["runs"]:
        d = os.path.join(P, r["dir"]); os.makedirs(d)
        json.dump(dict(cond=r["cond"], seed=r["seed"], sched=r["sched"], steps=r["steps"], diag_every=r["diag_every"], snap_every=20, init=None, init_sha256=IH[r["seed"]], blind=True), open(os.path.join(d, "done.json"), "w"))
        open(os.path.join(d, "traj.jsonl"), "w").write("".join(json.dumps(dict(step=s, F_G=0.0)) + "\n" for s in range(0, r["steps"] + 1, r["diag_every"])))
        for k in range(20, r["steps"] + 1, 20): open(os.path.join(d, f"step{k}.npz"), "w").close()
        json.dump(analysis(r, sc), open(os.path.join(P, "logs", "analysis", r["name"] + ".json"), "w"))
    if sc != 2:
        ev3 = sorted([r for r in D["runs"] if r["arity"] == 3], key=lambda r: r["seed"]); ev2 = sorted([r for r in D["runs"] if r["arity"] == 2], key=lambda r: r["seed"])[:2]
        for r in ev3 + ev2:
            for arm in ("targeted", "full"):
                d = os.path.join(P, "learning", f"{r['name']}_{arm}"); os.makedirs(d); good = sc == 1
                fin = ([0.3, 0.3, 0.3, 0.0, 0.0, 0.0] if arm == "targeted" else [1.0] * 6) if good else ([1.0] * 6 if arm == "targeted" else [1.0] * 6)
                json.dump(dict(steps=400, ckpt=os.path.join(r["dir"], "step900.npz"), traj=[dict(step=s, F_G=fin) for s in range(0, 401, 100)]), open(os.path.join(d, "slot_traj.json"), "w"))
    return P
res = dict(scenarios={}, refusals={})
expect = {1: dict(P0="STAGGERED", P1="FOLLOWS_CODE", P2="TRANSFER", P3="LATE_LESS_CODE_DEPENDENT", P4="ROUTING_INSUFFICIENT", P5="CODE_KEYING_PERSISTS", PL="INTERFERENCE_NO_ACCELERATION"),
          2: dict(P0="SYNCHRONOUS", P1="NOT_EVALUABLE", P2="NOT_EVALUABLE", P3="NOT_EVALUABLE", P4="NOT_EVALUABLE", P5="NOT_EVALUABLE", PL="NOT_EVALUABLE"),
          3: dict(P0="STAGGERED", P1="FOLLOWS_CODE", P2="NO_TRANSFER", P3="OPPOSITE", P4="ROUTING_SUFFICIENT", P5="NOT_PERSISTENT", PL="NOT_SUPPORTED")}
for sc in (1, 2, 3):
    T = tempfile.mkdtemp(); P = build(T, sc); o = os.path.join(T, "o.json")
    rc = subprocess.run([sys.executable, os.path.join(P, "scripts", "score_XVII.py"), P, o], capture_output=True, text=True)
    got = {k: v["call"] for k, v in json.load(open(o))["verdicts"]["pooled"].items()} if os.path.exists(o) else rc.stdout[-300:] + rc.stderr[-1500:]
    res["scenarios"][sc] = dict(expected=expect[sc], got=got, ok=got == expect[sc]); shutil.rmtree(T); print(sc, got == expect[sc], got, flush=True)
def refusal(name, mut):
    T = tempfile.mkdtemp(); P = build(T, 1); mut(P); o = os.path.join(T, "o.json")
    rc = subprocess.run([sys.executable, os.path.join(P, "scripts", "score_XVII.py"), P, o], capture_output=True, text=True)
    res["refusals"][name] = dict(rc=rc.returncode, wrote=os.path.exists(o), msg=rc.stdout.strip()[-150:]); shutil.rmtree(T)
def unblind(P):
    f = os.path.join(P, "confirm_outputs/A2_s703/done.json"); j = json.load(open(f)); j["blind"] = False; json.dump(j, open(f, "w"))
refusal("missing_analysis", lambda P: os.remove(os.path.join(P, "logs/analysis/A3_s802.json")))
refusal("not_blind", unblind)
refusal("missing_snapshot", lambda P: os.remove(os.path.join(P, "confirm_outputs/A2_s705/step1000.npz")))
refusal("changed_frozen_file", lambda P: open(os.path.join(P, "src/xviislot.py"), "a").write("\n#x\n"))
refusal("missing_learning_arm", lambda P: shutil.rmtree(os.path.join(P, "learning/A3_s803_full")))
res["all_pass"] = all(v["ok"] for v in res["scenarios"].values()) and all(v["rc"] == 2 and not v["wrote"] for v in res["refusals"].values())
json.dump(res, open(os.path.join(PKG, "logs", "preflight_XVII.json"), "w"), indent=1); print({k: (v["rc"], v["wrote"]) for k, v in res["refusals"].items()}, "ALL_PASS", res["all_pass"])
