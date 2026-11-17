"""XV Workstream E frozen scorer (FAIL-CLOSED): DeepSets T4 from scratch, seeds 301-304.  Usage: score_T4.py [ROOT] [OUT]
Refuses unless frozen hashes match, every run has done.json (blind, seed, 6000 steps, init hash recomputed) and every snapshot, and every
trajectory JSON (stream 98_800_000, n 3000) has one entry per snapshot.
Per model: t_G4 = first snapshot with G4 >= 0.5 at it and the next;  source stage = snapshot with KL(own || model) <= 0.03 and G4 <= 0.1;
competent = final KL(Bayes || model) <= 0.02 and final G4 >= 0.6 (G4 is attenuated: dev seed 300 reached 0.84 at KL 0.001).
TRAINABILITY GATE: >= 3 of 4 competent, else verdict TRAINABILITY_FAILED (no learning-order conclusion).
P6 (source stage first): among competent models, >= 4 consecutive source-stage snapshots (>= 200 steps) before t_G4.
   CONFIRMED if this holds in all competent models (>= 3); FALSIFIED if it fails in >= 2; else INCONCLUSIVE.
Also reported: transition width (last G4 <= 0.1 before t_G4 to first G4 >= 0.5)."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "logs", "score_T4.json")
def refuse(m): print("REFUSED:", m); sys.exit(2)
for l in open(os.path.join(PKG, "logs", "FREEZE_XV_HASHES.txt")):
    h, p = l.split()
    if hashlib.sha256(open(os.path.join(PKG, p), "rb").read()).hexdigest() != h: refuse(f"frozen file changed: {p}")
SEEDS = (301, 302, 303, 304); SN = list(range(50, 3001, 50)) + list(range(3250, 6001, 250))
sys.path.insert(0, os.path.join(PKG, "..", "step1_latent_reliability", "src"))
import numpy as np, jax, nets as N, bayes as B
env = B.Env("T4", K=3, Q=6)
for s in SEEDS:
    d = os.path.join(PKG, "confirm_outputs", f"T4ds_s{s}"); f = os.path.join(d, "done.json")
    if not os.path.exists(f): refuse(f"run not finished: T4ds_s{s}")
    j = json.load(open(f)); ih = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(N.init_base("deepsets", "M", env, s)))).hexdigest()
    if j["seed"] != s or j["steps"] != 6000 or not j["blind"] or j["init_sha256"] != ih: refuse(f"config mismatch: T4ds_s{s}")
    if [k for k in SN if not os.path.exists(os.path.join(d, f"step{k}.npz"))]: refuse(f"snapshots missing: T4ds_s{s}")
    t = os.path.join(PKG, "confirm_outputs", f"traj_T4ds_s{s}.json")
    if not os.path.exists(t): refuse(f"trajectory missing: T4ds_s{s}")
    T = json.load(open(t))
    if T["stream"] != 98_800_000 or T["n"] != 3000 or [x["step"] for x in T["traj"]] != SN: refuse(f"trajectory mismatch: T4ds_s{s}")
out = dict(models={}, verdicts={})
for s in SEEDS:
    tr = json.load(open(os.path.join(PKG, "confirm_outputs", f"traj_T4ds_s{s}.json")))["traj"]; n = len(tr)
    tG = next((tr[i]["step"] for i in range(n - 1) if tr[i]["G4"] >= 0.5 and tr[i + 1]["G4"] >= 0.5), None)
    src = [x["KL_own"] <= 0.03 and x["G4"] <= 0.1 and (tG is None or x["step"] < tG) for x in tr]; run = best = 0
    for o in src: run = run + 1 if o else 0; best = max(best, run)
    w = None
    if tG is not None:
        i = [x["step"] for x in tr].index(tG); lo = max([tr[j]["step"] for j in range(i + 1) if tr[j]["G4"] <= 0.1], default=0); w = tG - lo
    fin = tr[-1]; out["models"][f"s{s}"] = dict(t_G4=tG, longest_source_run=best, width=w, competent=bool(fin["KL_bayes"] <= 0.02 and fin["G4"] >= 0.6), final=fin)
comp = [m for m in out["models"].values() if m["competent"]]
if len(comp) < 3: out["verdicts"]["P6"] = "TRAINABILITY_FAILED"
else:
    ok = [m["longest_source_run"] >= 4 for m in comp]
    out["verdicts"]["P6"] = "CONFIRMED" if all(ok) else ("FALSIFIED" if sum(not o for o in ok) >= 2 else "INCONCLUSIVE")
out["n_competent"] = len(comp); json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps(out["verdicts"]), len(comp))
