"""Reference-dependence check of the head-level pattern/value split (NATIVE §2.3): dd(a) u_00 vs dd(a) u_AP for xP81 heads 0-3.
Run from the package root: python3 scripts/ref_dependence_check.py  (prints per-head shares; output archived in logs/ref_dependence_check.txt)"""
import sys, os; sys.path.insert(0, "src"); sys.argv = ["x", "-", "xP81:full_c1:0:"]
exec(open("scripts/alt_ablation.py").read().split("out = {}\nfor sp")[0])
import numpy as np, jax.numpy as jnp
M = F.load_model("xP81", "full_c1", 0); Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6_R.npy")))
R = {}
for c, Xc in cells.items(): r = {}; F.forward(M, jnp.asarray(Xc), rec=r); R[c] = {k: np.asarray(v) for k, v in r.items()}
lay = {k: np.asarray(v) for k, v in M["layers"][1].items()}
def ln(x, g, b): mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True); return (x - mu) / np.sqrt(v + 1e-5) * g + b
T = tok(qq); q7E = lambda Z: Z[ii[:, None], T].reshape(N_, -1) @ Q
dd0 = sum(s * q7E(R[c]["resid2"]) for c, s in (("xAP", 1), ("xA", -1), ("xP", -1), ("x00", 1)))
for h in range(4):
    u = {c: np.einsum("ntd,de,ef->ntf", ln(R[c]["resid1"], lay["g1"], lay["b1"]), lay["WV"][h], lay["WO"][h]) for c in cells}
    A = {c: R[c]["pattern1"][:, h] for c in cells}
    o = lambda a, uu: np.einsum("nts,nsd->ntd", a, uu)
    recon = o(A["xAP"], u["xAP"]); err = np.abs(recon - R["xAP"]["heads1"][:, h]).max()
    ddA = A["xAP"] - A["xA"] - A["xP"] + A["x00"]
    pat_u00 = q7E(o(ddA, u["x00"])); pat_uAP = q7E(o(ddA, u["xAP"]))
    tot = q7E(sum(s * R[c]["heads1"][:, h] for c, s in (("xAP", 1), ("xA", -1), ("xP", -1), ("x00", 1))))
    sh = lambda Z: float((Z * dd0).sum() / (dd0 ** 2).sum())
    print(h, "recon err", err, "tot", round(sh(tot), 3), "pat_u00", round(sh(pat_u00), 3), "pat_uAP", round(sh(pat_uAP), 3))
