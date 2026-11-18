"""XVII frozen per-model analysis battery (run after training, before scoring; reads a run directory produced by scripts/xvii_train.py).
Usage: xvii_confirm_analysis.py COND RUNDIR OUT.json
1. Window from the inline pooled trajectory (traj.jsonl): t1 = first diagnostic step with pooled F_G >= 0.05; t9 = first step >= t1 with
   pooled F_G >= 0.95 (else the last step).  Window = [t1 - 200, t9 + 200] clipped to the run; every 20-step snapshot in it is scored per
   slot with src/xviislot.SlotDiag (stream 99_800_000, 600 episodes; per-slot gate flips).
2. Slot acquisition time: first snapshot at which the slot's F_G >= 0.8 at it and the next snapshot.  Partial checkpoints: snapshots with
   >= 1 slot F_G >= 0.8 and >= 1 slot F_G <= 0.2.  Selected partial checkpoint: maximise min(|A|, |U|) (A = slots >= 0.8, U = slots <= 0.2),
   ties to the earliest.  None: the model has no resolvable partial state (recorded; it is not evaluable for P1-P5 and the learning test).
3. Battery at the selected checkpoint, for every pair (i in A, j in U):
   relabel windows (layers) [0,1) [1,2) [2,3) [0,2) [1,3) [0,3): recipient F_G (slot j) and donor F_G (slot i);
   U->U control: window [0,2) exchange between j and k = the next unacquired slot (cyclic in U, k != j), recipient F_G at j (if |U| >= 2);
   routing transplant (attention rows of all 7 tokens of j, all heads, re-indexed from donor i) in layer 1, and in layers 0 and 1.
4. Late checkpoint: the first snapshot at which every slot has F_G >= 0.9, plus 200 steps (nearest snapshot at or after; the last snapshot if
   beyond the window, else the run's next available snapshot); windows [0,1) [1,3) [0,2) [2,3) for the same pairs (i in A, j in U).
Outputs every raw number; scoring is done only by scripts/score_XVII.py."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, gatelib as GL, xviislot as XS
cond, rd, outp = sys.argv[1], sys.argv[2], sys.argv[3]
L = [json.loads(l) for l in open(os.path.join(rd, "traj.jsonl"))]; steps = [x["step"] for x in L]
t1 = next((x["step"] for x in L if x["F_G"] >= 0.05), None)
res = dict(cond=cond, run=rd, t1=t1)
if t1 is None:
    res.update(status="no_gate_response"); json.dump(res, open(outp, "w"), indent=1); sys.exit(0)
t9 = next((x["step"] for x in L if x["step"] >= t1 and x["F_G"] >= 0.95), steps[-1]); res["t9"] = t9
snaps = GL.snap_steps(rd); W = [k for k in snaps if t1 - 200 <= k <= t9 + 200]; D = XS.SlotDiag(cond, n=600); res["balance"] = D.balance
res["window"] = W; slot = {}
for k in W:
    slot[k] = D.score(GL.load(os.path.join(rd, f"step{k}.npz"))); print(k, [round(v, 2) for v in slot[k]["F_G"]], flush=True)
res["per_slot"] = {str(k): v for k, v in slot.items()}
acq = {}
for q in range(6):
    acq[q] = next((W[n] for n in range(len(W) - 1) if slot[W[n]]["F_G"][q] >= 0.8 and slot[W[n + 1]]["F_G"][q] >= 0.8), None)
res["slot_acquisition_step"] = acq
part = [k for k in W if max(slot[k]["F_G"]) >= 0.8 and min(slot[k]["F_G"]) <= 0.2]; res["partial_checkpoints"] = part
if not part:
    res.update(status="no_partial_checkpoint"); json.dump(res, open(outp, "w"), indent=1); sys.exit(0)
score = lambda k: min(sum(v >= 0.8 for v in slot[k]["F_G"]), sum(v <= 0.2 for v in slot[k]["F_G"]))
sel = sorted(part, key=lambda k: (-score(k), k))[0]; A = [q for q in range(6) if slot[sel]["F_G"][q] >= 0.8]; U = [q for q in range(6) if slot[sel]["F_G"][q] <= 0.2]
res.update(status="evaluable", selected=sel, A=A, U=U); p = GL.load(os.path.join(rd, f"step{sel}.npz")); pairs = list(itertools.product(A, U))
WIN = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]; bat = {f"{lo}-{hi}": dict(recipient=[], donor=[]) for lo, hi in WIN}
bat["UU_control_0-2"] = dict(recipient=[]); bat["route_L1"] = dict(recipient=[]); bat["route_L01"] = dict(recipient=[])
for i, j in pairs:
    for lo, hi in WIN:
        s = D.score(p, (i, j, lo, hi))["F_G"]; bat[f"{lo}-{hi}"]["recipient"].append(s[j]); bat[f"{lo}-{hi}"]["donor"].append(s[i])
    if len(U) >= 2:
        k = U[(U.index(j) + 1) % len(U)]; bat["UU_control_0-2"]["recipient"].append(D.score(p, (k, j, 0, 2))["F_G"][j])
    bat["route_L1"]["recipient"].append(D.score(p, pat=(i, j, (1,), (0, 1, 2, 3), "all"))["F_G"][j])
    bat["route_L01"]["recipient"].append(D.score(p, pat=(i, j, (0, 1), (0, 1, 2, 3), "all"))["F_G"][j])
    print("pair", i, j, {k: round(v["recipient"][-1], 2) for k, v in bat.items() if v["recipient"]}, flush=True)
res["pairs"] = pairs; res["battery"] = bat
full = next((k for k in W if min(slot[k]["F_G"]) >= 0.9), None); res["all_acquired_step"] = full
if full is not None:
    later = [k for k in snaps if k >= full + 200]; late = later[0] if later else snaps[-1]; res["late"] = late; pl = GL.load(os.path.join(rd, f"step{late}.npz"))
    res["late_base"] = D.score(pl)["F_G"]; lb = {}
    for lo, hi in [(0, 1), (1, 3), (0, 2), (2, 3)]:
        rr = [D.score(pl, (i, j, lo, hi))["F_G"] for i, j in pairs]; lb[f"{lo}-{hi}"] = dict(recipient=[r[j] for r, (i, j) in zip(rr, pairs)], donor=[r[i] for r, (i, j) in zip(rr, pairs)])
    res["late_battery"] = lb; print("late", late, {k: round(float(np.mean(v["recipient"] + v["donor"])), 2) for k, v in lb.items()}, flush=True)
json.dump(res, open(outp, "w"), indent=1)
