"""XIV pre-registered AMENDMENT to X1 (FREEZE_X_AMEND.md), FAIL-CLOSED; written after mathematical review A (F1, F4, F5) and before any
confirmation output was inspected.  The original frozen X1a / X1b verdicts (score_X.py) are unchanged and reported as frozen.
Amended criteria, per competent model:
  X1a' : >= 4 consecutive snapshots (step <= 1500) with  0.75 <= acc <= 0.83,  NLL <= 0.50,  A_p <= 0.3  and  G <= 0.1.
         (The accuracy band and NLL clause exclude the review-A competitor 'joint learning at small logit scale', which has the Bayes
         argmax, accuracy ~0.90, and high NLL.)
  X1b' : some snapshot with G <= 0.1 and A_p >= 0.5, where A_p is the PARTIAL agreement index: the coefficient on the gate-free rule's
         swap response in the joint least-squares regression of the model's swap response on (gate-free response, Bayes response).
         Exact references (logs/score_X_amend_refs.json): order-1 rule 0, order-2 rule ~0.96, Bayes 0, gate-free rule 1.
Same stream, models, competence rule and refusals as score_X.py.  Usage: score_X_amend.py [CKPT_ROOT] [OUT]"""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); sys.path.insert(0, os.path.join(PKG, "src"))
CK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "checkpoints", "gate"); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "logs", "score_X_amend.json")
def refuse(m): print("REFUSED:", m); sys.exit(2)
for f in ("FREEZE_X_HASHES.txt", "FREEZE_X_AMEND_HASHES.txt"):
    for l in open(os.path.join(PKG, "logs", f)):
        h, p = l.split()
        if hashlib.sha256(open(os.path.join(PKG, p), "rb").read()).hexdigest() != h: refuse(f"frozen file changed: {p}")
SNAPS = sorted(set(range(25, 1501, 25)) | set(range(1600, 2001, 100)) | set(range(2500, 4001, 500))); SEEDS = (3, 4)
for s in SEEDS:
    d = os.path.join(CK, f"s{s}")
    if not os.path.exists(os.path.join(d, "hist.json")): refuse(f"seed {s} training not finished")
    if [k for k in SNAPS if not os.path.exists(os.path.join(d, f"step{k}.npz"))]: refuse(f"seed {s}: snapshots missing")
import numpy as np, gatelib as GL
rng = np.random.default_rng(98_500_000); dd = GL.TX.sample_gate(3000, rng); X, Y = dd["X"], dd["Y"]; Xf = GL.gate_flip(X, rng); Xw = GL.swap35(X)
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
PB, PBf, PBw = GL.rule("bayes", X), GL.rule("bayes", Xf), GL.rule("bayes", Xw); GF, GFw = GL.rule("gatefree", X), GL.rule("gatefree", Xw)
dB = cen(PBf) - cen(PB); dG = cen(GFw) - cen(GF); dBw = cen(PBw) - cen(PB); mw = np.abs(dG).max(-1) > 0.05
def slope(dm, dr, m=None):
    m = np.ones(dm.shape[:2], bool) if m is None else m; return float((dm[m] * dr[m]).sum() / (dr[m] ** 2).sum())
def partial(dm):
    Z = np.stack([dG[mw].ravel(), dBw[mw].ravel()], 1); c, *_ = np.linalg.lstsq(Z, dm[mw].ravel(), rcond=None); return float(c[0])
out = dict(models={}, verdicts={})
for s in SEEDS:
    traj = []
    for k in SNAPS:
        p = GL.load(os.path.join(CK, f"s{s}", f"step{k}.npz")); P0, Pf, Pw = GL.probs(p, X), GL.probs(p, Xf), GL.probs(p, Xw)
        traj.append(dict(step=k, NLL=float(-(PB * np.log(P0 + 1e-12)).sum(-1).mean()), acc=float((P0.argmax(-1) == Y).mean()), KL_bayes=GL.kl(PB, P0),
                         G=slope(cen(Pf) - cen(P0), dB), A=slope(cen(Pw) - cen(P0), dG, mw), A_p=partial(cen(Pw) - cen(P0))))
    fin = traj[-1]; comp = fin["KL_bayes"] <= 0.02 and fin["G"] >= 0.9
    ok = [t["step"] <= 1500 and 0.75 <= t["acc"] <= 0.83 and t["NLL"] <= 0.50 and t["A_p"] <= 0.3 and t["G"] <= 0.1 for t in traj]; run = best = 0
    for o in ok: run = run + 1 if o else 0; best = max(best, run)
    b = [t["step"] for t in traj if t["G"] <= 0.1 and t["A_p"] >= 0.5]
    out["models"][f"s{s}"] = dict(competent=comp, X1a_prime=best >= 4, X1a_prime_longest_run=best, X1b_prime=bool(b), X1b_prime_steps=b, trajectory=traj)
comp = [m for m in out["models"].values() if m["competent"]]
for k in ("X1a_prime", "X1b_prime"):
    out["verdicts"][k] = "INCONCLUSIVE" if len(comp) < 2 else ("CONFIRMED" if all(m[k] for m in comp) else "FALSIFIED")
json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps(out["verdicts"]))
