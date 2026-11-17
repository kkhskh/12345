"""XVI preflight [run before any confirmation output exists]: the frozen scorer is exercised on SYNTHETIC trajectories with planted
outcomes (no model is trained or read), and its fail-closed refusals are tested.  Writes logs/preflight_XVI.json.
Synthetic package: a temporary copy of logs/ + src/ + scripts/ with sibling links; every design run gets a done.json with the true
init hash, empty snapshot files and a planted trajectory.  Planted outcomes:
  WA fresh t_A = 2000, 3000, censored, 5000 (expect P1 BROAD);  continuations: 4 of 6 acquire at 300-800 post-restart (CONT_BROAD,
  FAST_UNLOCK);  WB low 1500/1400/1600/censored vs high 800/700/900/1000 (BENEFIT_SPEEDS);  every acquirer has a floor-verified stage
  (STAGE_ROBUST).  Refusal tests: a missing trajectory line, blind = false, a leftover resume.pkl, a missing snapshot, a changed frozen
  file; each must exit 2 and write nothing."""
import json, os, sys, shutil, subprocess, tempfile, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src")); sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, optax, train_tasks as TT, gatelib as GL, xvidiag as XD
D = json.load(open(os.path.join(PKG, "logs", "XVI_design.json")))
hsh = lambda p: hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(p))).hexdigest()
plant = {}
for s, t in zip((401, 402, 403, 404), (2000, 3000, None, 5000)): plant[f"confirm_outputs/WA_const_s{s}"] = t
for d, t in zip(("C_s201", "C_s202", "C_s203", "D_s201", "D_s202", "D_s204"), (300, 500, None, 800, None, 600)): plant[f"confirm_outputs/WA_cont_{d}"] = t
for s, tl, th in zip((501, 502, 503, 504), (1500, 1400, 1600, None), (800, 700, 900, 1000)): plant[f"confirm_outputs/WB_low_s{s}"] = tl; plant[f"confirm_outputs/WB_high_s{s}"] = th
expect = dict(P1="BROAD", P2="CONT_BROAD", P3="BENEFIT_SPEEDS", P5="STAGE_ROBUST")
REF = {c: XD.Diag(c, D["diag_stream"], D["diag_n"], os.path.join(PKG, "logs")).ref_scores() for c in {r["cond"] for r in D["runs"]}}
def build(T):
    os.makedirs(os.path.join(T, "pkg"))
    for sub in ("logs", "src", "scripts"): shutil.copytree(os.path.join(PKG, sub), os.path.join(T, "pkg", sub), ignore=shutil.ignore_patterns("__pycache__", "claims", "run"))
    for sib in os.listdir(ROOT):
        if sib.startswith("step1") and sib != os.path.basename(PKG): os.symlink(os.path.join(ROOT, sib), os.path.join(T, sib))
    for r in D["runs"]:
        d = os.path.join(T, "pkg", r["dir"]); os.makedirs(d); R = REF[r["cond"]]; tA = plant[r["dir"]]
        p = TT.init("gate", r["seed"]) if r["init"] is None else GL.load(os.path.normpath(os.path.join(PKG, r["init"])))
        json.dump(dict(cond=r["cond"], seed=r["seed"], sched=r["sched"], steps=r["steps"], diag_every=r["diag_every"], init=r["init"], init_sha256=hsh(p), blind=True), open(os.path.join(d, "done.json"), "w"))
        for k in list(range(500, r["steps"] + 1, 500)) + [r["steps"]]: open(os.path.join(d, f"step{k}.npz"), "w").close()
        with open(os.path.join(d, "traj.jsonl"), "w") as f:
            for st in range(0, r["steps"] + 1, r["diag_every"]):
                on = tA is not None and st >= tA; stage = st >= 100 and not globals().get("NOSTAGE")
                x = dict(step=st, KL=(0.005 if on else R["ordera"]["KL"]), F_G=(1.0 if on else 0.0), A_conf=(0.0 if on else (R["ordera"]["A_conf"] if stage else 0.0)),
                         cum_lr=st * 1e-3, G=(1.0 if on else 0.0), A_p=0.0, conf=0.8, sB=0.6)
                if not stage and not on: x["KL"] = R["order1"]["KL"]
                f.write(json.dumps(x) + "\n")
    return os.path.join(T, "pkg")
res = dict(planted=plant, expected=expect, refusals={})
T = tempfile.mkdtemp(); P = build(T); o = os.path.join(T, "o.json")
rc = subprocess.run([sys.executable, os.path.join(P, "scripts", "score_XVI.py"), P, o], capture_output=True, text=True); res["scorer_rc"] = rc.returncode; res["scorer_out"] = rc.stdout[-500:] + rc.stderr[-2000:]
got = json.load(open(o))["verdicts"] if os.path.exists(o) else None; res["got"] = got
S = json.load(open(o)) if os.path.exists(o) else {}
res["t_A_recovered"] = {k: v["t_A"] for k, v in S.get("models", {}).items()}
res["t_A_correct"] = all(res["t_A_recovered"].get(k) == v for k, v in plant.items()) if S else False
def refusal(name, mutate):
    T2 = tempfile.mkdtemp(); P2 = build(T2); mutate(P2); o2 = os.path.join(T2, "o.json")
    rc = subprocess.run([sys.executable, os.path.join(P2, "scripts", "score_XVI.py"), P2, o2], capture_output=True, text=True)
    res["refusals"][name] = dict(rc=rc.returncode, wrote=os.path.exists(o2), msg=rc.stdout.strip()[-200:]); shutil.rmtree(T2)
def drop_line(P2):
    f = os.path.join(P2, "confirm_outputs/WB_high_s503/traj.jsonl"); L = open(f).readlines(); open(f, "w").writelines(L[:-1])
def unblind(P2):
    f = os.path.join(P2, "confirm_outputs/WA_const_s402/done.json"); j = json.load(open(f)); j["blind"] = False; json.dump(j, open(f, "w"))
refusal("missing_trajectory_line", drop_line); refusal("not_blind", unblind)
refusal("leftover_resume", lambda P2: open(os.path.join(P2, "confirm_outputs/WA_cont_D_s202/resume.pkl"), "w").close())
refusal("missing_snapshot", lambda P2: os.remove(os.path.join(P2, "confirm_outputs/WA_const_s404/step7500.npz")))
refusal("changed_frozen_file", lambda P2: open(os.path.join(P2, "src/xvidiag.py"), "a").write("\n# changed\n"))
shutil.rmtree(T)
# scenario 2: opposite branches (P1 RARE, P2 CONT_RARE, P3 NO_EFFECT, P5 STAGE_NOT_ROBUST: acquirers without a stage)
plant1 = dict(plant)
for k in plant: plant[k] = None
plant["confirm_outputs/WA_const_s401"] = 9000; plant["confirm_outputs/WA_cont_C_s202"] = 2500
for s, t in zip((501, 502, 503, 504), (1000, 1100, 900, 1200)): plant[f"confirm_outputs/WB_low_s{s}"] = t; plant[f"confirm_outputs/WB_high_s{s}"] = t + 50
NOSTAGE = True
T = tempfile.mkdtemp(); P = build(T); o = os.path.join(T, "o.json")
subprocess.run([sys.executable, os.path.join(P, "scripts", "score_XVI.py"), P, o], capture_output=True, text=True)
got2 = json.load(open(o))["verdicts"] if os.path.exists(o) else None; shutil.rmtree(T)
expect2 = dict(P1="RARE", P2="CONT_RARE", P3="NO_EFFECT", P5="STAGE_NOT_ROBUST"); res["scenario2"] = dict(planted=dict(plant), expected=expect2, got=got2); plant.update(plant1)
res["all_pass"] = bool(got2 == expect2 and got == expect and res["t_A_correct"] and all(v["rc"] == 2 and not v["wrote"] for v in res["refusals"].values()))
json.dump(res, open(os.path.join(PKG, "logs", "preflight_XVI.json"), "w"), indent=1); print("scenario2", got2, "got", got, "t_A_correct", res["t_A_correct"], {k: (v["rc"], v["wrote"]) for k, v in res["refusals"].items()}, "ALL_PASS", res["all_pass"])
