"""XVII frozen scorer (FAIL-CLOSED).  Usage: score_XVII.py [PKG] [OUT=logs/score_XVII.json]
Refuses (exit 2, writes nothing) unless every file in logs/FREEZE_XVII_HASHES.txt matches; every design run has done.json matching the design
(cond, seed, sched, steps, diag_every, blind = true, snap_every = 20, init hash recomputed) and a complete traj.jsonl; every run has its
analysis file (logs/analysis/<run>.json, produced by scripts/xvii_confirm_analysis.py) with a valid status; every learning-intervention arm
required by the design rule (below) exists and is complete.

Model-level tests (models whose analysis status is 'evaluable' have a selected partial checkpoint; the replication unit is the model):
  P0 STAGGER   an acquiring model (status != no_gate_response, slots reach F_G >= 0.8) has a partial checkpoint (status evaluable).
               Cohort call: STAGGERED if >= 75% of acquiring models, SYNCHRONOUS if <= 25%, else MIXED.
  P1 CODE      full-window relabel [0,3): mean recipient F_G >= 0.8 and mean donor F_G <= 0.2.  FOLLOWS_CODE if >= 90% of evaluable models,
               else NOT_ALWAYS.   (Expected by Theorem P; a numerical verification.)
  P2 TRANSFER  early-code relabel [0,2): mean recipient F_G >= 0.5 and >= (U->U control mean, or the baseline mean recipient F_G = 0 if
               |U| = 1, using the selected-checkpoint F_G) + 0.4.  TRANSFER if >= 75% of evaluable models, NO_TRANSFER if <= 25%, else MIXED.
  P3 LATE      donor mean F_G under [2,3) minus donor mean under [0,2) >= 0.4.  LATE_LESS_CODE_DEPENDENT if >= 75%, OPPOSITE if <= 25%, else MIXED.
  P4 ROUTING   routing transplant in layers 0-1: mean recipient F_G < 0.3 while [0,2) relabel recipient mean >= 0.5.
               ROUTING_INSUFFICIENT if >= 75% of evaluable models, ROUTING_SUFFICIENT if routing mean >= 0.5 in >= 75%, else MIXED.
  P5 KEYED     late checkpoint: min over mixing windows {[0,1), [1,3)} of mean(recipient, donor) F_G <= 0.8 and mean under [0,2) >= mean under
               [0,1) + 0.15.  CODE_KEYING_PERSISTS if >= 75% of models with a late checkpoint, NOT_PERSISTENT if <= 25%, else MIXED.
  PL LEARNING  (models in the learning subset: every evaluable arity-3 model and the first two evaluable arity-2 models by seed) targeted arm
               (all parameters, loss on slot j0 = min(U) only, 400 steps) against full arm (all slots' loss, 400 steps), from the selected
               checkpoint: final mean F_G over A <= 0.5 in the targeted arm while >= 0.9 in the full arm (INTERFERENCE), and final F_G of j0 in the
               targeted arm <= its final F_G in the full arm + 0.1 (NO_ACCELERATION).  INTERFERENCE_NO_ACCELERATION if >= 75% of the subset pass,
               NOT_SUPPORTED if <= 25%, else MIXED.
Each verdict is reported for arity 2, arity 3 and pooled; the frozen primary verdict is the pooled one.  Exact Clopper-Pearson intervals."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "logs", "score_XVII.json")
def refuse(m): print("REFUSED:", m); sys.exit(2)
H = os.path.join(PKG, "logs", "FREEZE_XVII_HASHES.txt")
if not os.path.exists(H): refuse("freeze hash list missing")
for l in open(H):
    h, p = l.split(); fp = os.path.join(PKG, p)
    if not os.path.exists(fp) or hashlib.sha256(open(fp, "rb").read()).hexdigest() != h: refuse(f"frozen file changed or missing: {p}")
D = json.load(open(os.path.join(PKG, "logs", "XVII_design.json")))
sys.path.insert(0, os.path.join(PKG, "..", "step1_foundational_atlas_continuation", "src"))
import numpy as np, jax
from scipy.stats import beta as Beta
import train_tasks as TT
hsh = lambda p: hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(p))).hexdigest()
AN = {}
for r in D["runs"]:
    d = os.path.join(PKG, r["dir"]); f = os.path.join(d, "done.json")
    if not os.path.exists(f): refuse(f"run not finished: {r['dir']}")
    j = json.load(open(f))
    if (j.get("cond"), j.get("seed"), j.get("sched"), j.get("steps"), j.get("diag_every"), j.get("snap_every")) != (r["cond"], r["seed"], r["sched"], r["steps"], r["diag_every"], 20) or j.get("blind") is not True: refuse(f"config mismatch: {r['dir']}")
    if os.path.exists(os.path.join(d, "resume.pkl")): refuse(f"resume state left: {r['dir']}")
    if j["init_sha256"] != hsh(TT.init("gate", r["seed"])): refuse(f"init hash mismatch: {r['dir']}")
    L = [json.loads(x) for x in open(os.path.join(d, "traj.jsonl"))]
    if [x["step"] for x in L] != list(range(0, r["steps"] + 1, r["diag_every"])): refuse(f"trajectory incomplete: {r['dir']}")
    if [k for k in range(20, r["steps"] + 1, 20) if not os.path.exists(os.path.join(d, f"step{k}.npz"))]: refuse(f"snapshots missing: {r['dir']}")
    a = os.path.join(PKG, "logs", "analysis", r["name"] + ".json")
    if not os.path.exists(a): refuse(f"analysis missing: {r['name']}")
    A = json.load(open(a))
    if A.get("status") not in ("no_gate_response", "no_partial_checkpoint", "evaluable") or A.get("run") != r["dir"]: refuse(f"analysis invalid: {r['name']}")
    if A["status"] == "evaluable" and ("battery" not in A or not A["pairs"]): refuse(f"battery missing: {r['name']}")
    AN[r["name"]] = (r, A)
ev = lambda arity: sorted([n for n, (r, A) in AN.items() if A["status"] == "evaluable" and r["arity"] == arity], key=lambda n: AN[n][0]["seed"])
LSUB = ev(3) + ev(2)[:2]
LR = {}
for n in LSUB:
    for arm in ("targeted", "full"):
        f = os.path.join(PKG, "learning", f"{n}_{arm}", "slot_traj.json")
        if not os.path.exists(f): refuse(f"learning arm missing: {n}_{arm}")
        T = json.load(open(f))
        if T["steps"] != 400 or [x["step"] for x in T["traj"]] != list(range(0, 401, 100)) or T["ckpt"] != os.path.join(AN[n][0]["dir"], f"step{AN[n][1]['selected']}.npz"): refuse(f"learning arm invalid: {n}_{arm}")
        LR[(n, arm)] = T
m = lambda v: float(np.mean(v)) if v else float("nan")
def tests(n):
    r, A = AN[n]; out = {}
    acquiring = A["status"] != "no_gate_response" and any(v is not None for v in A.get("slot_acquisition_step", {}).values())
    out["acquiring"] = acquiring; out["P0"] = A["status"] == "evaluable" if acquiring else None
    if A["status"] != "evaluable": return out
    B = A["battery"]; sel = A["per_slot"][str(A["selected"])]["F_G"]
    out["P1"] = m(B["0-3"]["recipient"]) >= 0.8 and m(B["0-3"]["donor"]) <= 0.2
    ctrl = m(B["UU_control_0-2"]["recipient"]) if B["UU_control_0-2"]["recipient"] else m([sel[j] for _, j in A["pairs"]])
    out["P2"] = m(B["0-2"]["recipient"]) >= 0.5 and m(B["0-2"]["recipient"]) >= ctrl + 0.4; out["P2_values"] = dict(transfer=m(B["0-2"]["recipient"]), control=ctrl)
    out["P3"] = m(B["2-3"]["donor"]) - m(B["0-2"]["donor"]) >= 0.4
    out["P4"] = m(B["route_L01"]["recipient"]) < 0.3 and m(B["0-2"]["recipient"]) >= 0.5; out["P4_routing_suff"] = m(B["route_L01"]["recipient"]) >= 0.5
    if "late_battery" in A:
        lb = A["late_battery"]; mm = lambda w: m(lb[w]["recipient"] + lb[w]["donor"])
        out["P5"] = min(mm("0-1"), mm("1-3")) <= 0.8 and mm("0-2") >= mm("0-1") + 0.15
    if n in LSUB:
        t, f = LR[(n, "targeted")]["traj"][-1]["F_G"], LR[(n, "full")]["traj"][-1]["F_G"]; j0 = min(A["U"])
        out["PL"] = m([t[q] for q in A["A"]]) <= 0.5 and m([f[q] for q in A["A"]]) >= 0.9 and t[j0] <= f[j0] + 0.1
        out["PL_values"] = dict(targeted_A=m([t[q] for q in A["A"]]), full_A=m([f[q] for q in A["A"]]), targeted_j0=t[j0], full_j0=f[j0])
    return out
TS = {n: tests(n) for n in AN}
cp = lambda k, n: [float(Beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(Beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0] if n else None
def call(key, names, hi=0.75, lo=0.25, labels=("PASS", "FAIL", "MIXED")):
    vals = [TS[n][key] for n in names if TS[n].get(key) is not None]; k = sum(vals); n = len(vals)
    if n == 0: return dict(call="NOT_EVALUABLE", k=0, n=0)
    return dict(call=labels[0] if k >= hi * n else (labels[1] if k <= lo * n else labels[2]), k=k, n=n, cp95=cp(k, n))
out = dict(models=TS, learning_subset=LSUB, verdicts={})
for scope, names in (("pooled", list(AN)), ("arity2", [n for n in AN if AN[n][0]["arity"] == 2]), ("arity3", [n for n in AN if AN[n][0]["arity"] == 3])):
    v = {}
    v["P0"] = call("P0", names, labels=("STAGGERED", "SYNCHRONOUS", "MIXED"))
    v["P1"] = call("P1", names, hi=0.9, lo=-1, labels=("FOLLOWS_CODE", "NOT_ALWAYS", "NOT_ALWAYS"))
    v["P2"] = call("P2", names, labels=("TRANSFER", "NO_TRANSFER", "MIXED"))
    v["P3"] = call("P3", names, labels=("LATE_LESS_CODE_DEPENDENT", "OPPOSITE", "MIXED"))
    p4 = call("P4", names, labels=("ROUTING_INSUFFICIENT", "X", "MIXED")); rs = call("P4_routing_suff", names, labels=("ROUTING_SUFFICIENT", "X", "MIXED"))
    if p4["call"] == "X": p4["call"] = "ROUTING_SUFFICIENT" if rs["call"] == "ROUTING_SUFFICIENT" else "MIXED"
    v["P4"] = p4
    v["P5"] = call("P5", names, labels=("CODE_KEYING_PERSISTS", "NOT_PERSISTENT", "MIXED"))
    v["PL"] = call("PL", [n for n in names if n in LSUB], labels=("INTERFERENCE_NO_ACCELERATION", "NOT_SUPPORTED", "MIXED"))
    out["verdicts"][scope] = v
json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps({k: v["call"] for k, v in out["verdicts"]["pooled"].items()}))
