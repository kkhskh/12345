"""Arrays only: is the N response a small residual of two large, cancelling route contributions?  Per model: norms of
Rq-only, Qs-only, Oth-only and full N responses; cosine(Rq-only, Rq-blocked) per pair (mean); interaction term
||full - patch_Rq - block_Rq|| / ||full||; and the same for R."""
import os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "conf"); G = np.load(os.path.join(D, "generator.npz"))
nrm = lambda a: np.sqrt((a ** 2).sum(-1))
for m in ["cT61", "cT62", "cT63", "cT64", "cT65", "cT67", "cP61", "cP62", "cP63", "cP64", "cP66", "cP67"]:
    z = np.load(os.path.join(D, f"model_{m}.npz")); s = [m]
    for F in ("N", "R"):
        mk = G[f"ok_{F}"].astype(bool); b = z["c_base"][mk]; f = z[f"c_{F}"][mk] - b
        p = {k: z[f"c_{F}_patch_{k}"][mk] - b for k in ("Rq", "Qs", "Oth")}; bl = z[f"c_{F}_block_Rq"][mk] - b
        cos = ((p["Rq"] * bl).sum(-1) / (nrm(p["Rq"]) * nrm(bl) + 1e-9)).mean()
        s.append(f"{F}: |full| {nrm(f).mean():.3f} |Rq| {nrm(p['Rq']).mean():.3f} |Qs| {nrm(p['Qs']).mean():.3f} |Oth| {nrm(p['Oth']).mean():.3f} |blkRq| {nrm(bl).mean():.3f} cos(Rq,blkRq) {cos:+.2f} interaction {nrm(f - p['Rq'] - bl).mean() / nrm(f).mean():.2f}")
    print(" | ".join(s))
