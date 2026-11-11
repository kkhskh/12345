"""Mechanistic review check (arrays only): route decomposition in absolute units, additivity of patch/block,
Bayes-aligned vs residual (error) parts of the R response per route, and per-unit-total-count Rq response."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "..", "logs", "conf")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); import cri as R; E = R.E
G = np.load(os.path.join(D, "generator.npz")); X, q = G["X"], G["qsel"]; idx = np.arange(len(X))
names = ["cT61", "cT62", "cT63", "cT64", "cT65", "cT67", "cP61", "cP62", "cP63", "cP64", "cP66", "cP67"]
dtot = {F: (E.loo(G[f"X_{F}"])[idx, q] - E.loo(X)[idx, q]).sum(-1) for F in ("N", "R")}
print("total-count change |dSum|: N", np.unique(np.abs(dtot["N"][G["ok_N"]]), return_counts=True), " R", np.unique(np.abs(dtot["R"][G["ok_R"]]), return_counts=True))
nrm = lambda a: float(np.sqrt((a ** 2).sum(-1)).mean())
proj = lambda a, b: float((a * b).sum() / (b * b).sum())
print("model | N: |full| |Rq-only| |Rq-blk| addit.err | R: |full| |Rq-only| |Rq-blk| addit.err | R Bayes-beta, err share of R full, Rq-only Bayes-aligned frac (R) | err-norm R: nat -> Rq-blocked | |Rq-only| per unit |dSum| N, R")
for m in names:
    z = np.load(os.path.join(D, f"model_{m}.npz")); out = [m]
    for F in ("N", "R"):
        mk = G[f"ok_{F}"].astype(bool); b = z["c_base"][mk]; full = z[f"c_{F}"][mk] - b
        pr = z[f"c_{F}_patch_Rq"][mk] - b; bl = z[f"c_{F}_block_Rq"][mk] - b
        pq, po = z[f"c_{F}_patch_Qs"][mk] - b, z[f"c_{F}_patch_Oth"][mk] - b
        add = nrm(full - pr - pq - po) / nrm(full)
        out += [f"{F}: {nrm(full):.3f} {nrm(pr):.3f} {nrm(bl):.3f} add3={add:.2f} (pr+bl-full)/full={nrm(pr + bl - full) / nrm(full):.2f}"]
        if F == "R":
            dB = (G["cB_R"] - G["cB_base"])[mk]; beta = proj(full, dB); err = full - beta * dB
            errb = bl - beta * dB
            # Bayes-aligned fraction of the Rq-only response: R^2 of pr on dB
            bA = proj(pr, dB); fr = 1 - ((pr - bA * dB) ** 2).sum() / (pr ** 2).sum()
            out += [f"beta={beta:.2f} err/full={nrm(err) / nrm(full):.2f} RqOnly_BayesR2={fr:.2f} errR {nrm(err):.3f}->{nrm(errb):.3f} err-share-via-Rq={proj(pr - bA*dB, err):.2f}"]
        u = np.abs(dtot[F][mk]); s = u > 0
        out += [f"Rq/|dSum|={float((np.sqrt((pr[s] ** 2).sum(-1)) / u[s]).mean()):.3f}"]
    print(" | ".join(out))
