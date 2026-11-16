"""XIV frozen cross-family prediction X1 (interaction-order staircase) scorer, FAIL-CLOSED.  Usage: score_X.py [CKPT_ROOT] [OUT]
Models: untouched GATE seeds 3 and 4 (scripts/g_train.py, snapshots every 25 steps to 1500, then 1600..2000 by 100, 2500..4000 by 500).
Refuses (exit 2, writes nothing) unless the frozen files match logs/FREEZE_X_HASHES.txt and, for both seeds, hist.json exists and every
snapshot is present.  Evaluation stream 98_500_000 (3000 episodes = 18000 questions), never used before.
Per snapshot: NLL (cross-entropy to the exact posterior), acc (vs labels), KL(Bayes||model),
  G(t) = gate-flip fidelity slope: <dm, dB> / <dB, dB> over all gate flips (x1 redrawn to switch the gate; dm, dB = changes of the
         centred log-probabilities of model and Bayes).  Every rule of interaction order <= 2 has G = 0 exactly; Bayes 1.
  A(t) = pairwise-agreement index: slope of the model's change under the swap x3 <-> x5 on the change of the exact gate-free (order-2)
         rule, over swaps where that rule changes (max |d| > 0.05).  The order-1 rule is exactly invariant to the swap (A = 0); the
         gate-free rule has A = 1.
Frozen predictions (logs/FREEZE_X.md), per competent model (competent: final KL <= 0.02 and final G >= 0.9):
  X1a order-1 phase first: >= 4 consecutive snapshots (step <= 1500) with acc >= 0.75, A <= 0.3 and G <= 0.1.
  X1b pairwise before gate: some snapshot with G <= 0.1 and A >= 0.5.
  Verdict per prediction: CONFIRMED if both models are competent and satisfy it; FALSIFIED if a competent model fails it;
  INCONCLUSIVE if fewer than 2 models are competent.  Bootstrap (episodes, 1000) intervals for A and G at the reported snapshots."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); sys.path.insert(0, os.path.join(PKG, "src"))
CK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "checkpoints", "gate"); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "logs", "score_X.json")
def refuse(m): print("REFUSED:", m); sys.exit(2)
for l in open(os.path.join(PKG, "logs", "FREEZE_X_HASHES.txt")):
    h, p = l.split()
    if hashlib.sha256(open(os.path.join(PKG, p), "rb").read()).hexdigest() != h: refuse(f"frozen file changed: {p}")
SNAPS = sorted(set(range(25, 1501, 25)) | set(range(1600, 2001, 100)) | set(range(2500, 4001, 500))); SEEDS = (3, 4)
for s in SEEDS:
    d = os.path.join(CK, f"s{s}")
    if not os.path.exists(os.path.join(d, "hist.json")): refuse(f"seed {s} training not finished")
    miss = [k for k in SNAPS if not os.path.exists(os.path.join(d, f"step{k}.npz"))]
    if miss: refuse(f"seed {s}: {len(miss)} snapshots missing")
import numpy as np, gatelib as GL
rng = np.random.default_rng(98_500_000); dd = GL.TX.sample_gate(3000, rng); X, Y = dd["X"], dd["Y"]; Xf = GL.gate_flip(X, rng); Xw = GL.swap35(X)
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
PB, PBf = GL.rule("bayes", X), GL.rule("bayes", Xf); GF, GFw = GL.rule("gatefree", X), GL.rule("gatefree", Xw)
dB = cen(PBf) - cen(PB); dG = cen(GFw) - cen(GF); mw = np.abs(dG).max(-1) > 0.05
def slope(dm, dr, m=None):
    if m is None: m = np.ones(dm.shape[:2], bool)
    return float((dm[m] * dr[m]).sum() / (dr[m] ** 2).sum())
def boot(dm, dr, m, n=1000):
    r = np.random.default_rng(0); E = dm.shape[0]; v = []
    for _ in range(n):
        I = r.integers(0, E, E); v.append(slope(dm[I], dr[I], m[I]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = dict(models={}, verdicts={})
for s in SEEDS:
    traj = []
    for k in SNAPS:
        p = GL.load(os.path.join(CK, f"s{s}", f"step{k}.npz")); P0, Pf, Pw = GL.probs(p, X), GL.probs(p, Xf), GL.probs(p, Xw)
        dmf, dmw = cen(Pf) - cen(P0), cen(Pw) - cen(P0)
        traj.append(dict(step=k, NLL=float(-(PB * np.log(P0 + 1e-12)).sum(-1).mean()), acc=float((P0.argmax(-1) == Y).mean()), KL_bayes=GL.kl(PB, P0),
                         KL_gatefree=GL.kl(GF, P0), KL_sourcewise=GL.kl(GL.rule("sourcewise", X), P0), G=slope(dmf, dB), A=slope(dmw, dG, mw)))
        traj[-1]["_d"] = (dmf, dmw)
    fin = traj[-1]; comp = fin["KL_bayes"] <= 0.02 and fin["G"] >= 0.9
    ok1 = [t["step"] <= 1500 and t["acc"] >= 0.75 and t["A"] <= 0.3 and t["G"] <= 0.1 for t in traj]; run = best = 0; first_run = None
    for i, o in enumerate(ok1):
        run = run + 1 if o else 0
        if run >= 4 and first_run is None: first_run = traj[i - 3]["step"]
        best = max(best, run)
    x1b = [t for t in traj if t["G"] <= 0.1 and t["A"] >= 0.5]
    rep = {}
    for name, t in (("first_order1_snapshot", next((t for t, o in zip(traj, ok1) if o), None)), ("first_X1b_snapshot", x1b[0] if x1b else None), ("final", fin)):
        if t is None: rep[name] = None; continue
        dmf, dmw = t["_d"]; rep[name] = dict(step=t["step"], G=t["G"], G_ci=boot(dmf, dB, np.ones(dmf.shape[:2], bool)), A=t["A"], A_ci=boot(dmw, dG, mw), acc=t["acc"], NLL=t["NLL"])
    for t in traj: t.pop("_d")
    tG = next((t["step"] for t in traj if t["G"] >= 0.5), None); tA = next((t["step"] for t in traj if t["A"] >= 0.5), None)
    out["models"][f"s{s}"] = dict(competent=comp, X1a=best >= 4, X1a_longest_run=best, X1a_first_run_start=first_run, X1b=bool(x1b), t_G_half=tG, t_A_half=tA, report=rep, trajectory=traj)
comp = [m for m in out["models"].values() if m["competent"]]
for k in ("X1a", "X1b"):
    out["verdicts"][k] = "INCONCLUSIVE" if len(comp) < 2 else ("CONFIRMED" if all(m[k] for m in comp) else "FALSIFIED")
out["references"] = json.load(open(os.path.join(PKG, "logs", "g_order_ceilings.json")))
json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps(out["verdicts"]))
