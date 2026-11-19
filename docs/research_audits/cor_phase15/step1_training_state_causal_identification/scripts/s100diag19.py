"""Step-100 diagnostics of window states (pre-review A3, A8, B3, C; diagnostic only, not mediation):
for each window state X against the matched B and I states: plain-forward target-route share (own0 on E / own2 on M, head mean, readout
rows) and other-question share, KL / accuracy / route necessity, total and layer-0 parameter-distance ratio to I's, and the cosine between
X's layer-0 displacement from B and I's.  Usage: s100diag19.py BASE(E|M) STREAM BDIR IDIR OUT.json X...   (dirs containing step100.npz)"""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import xviiim as XM, xix_score as XS
base, stream, bd, idr, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]; L = 0 if base == "E" else 2
names = json.load(open(os.path.join(H, "..", "logs", "param_names19.json"))); l0 = [k for k, n in enumerate(names) if n.startswith("pre/0/")]
X, q, PB = XM.nat(stream, 1500)
def diag(d):
    M = XM.load_dir(d, "uP160", 100); s = XM.shares(M, X); n = XS.necessity(M, stream)
    rec = {}; XM.F.forward(M, XM.jnp.asarray(X), rec=rec); A = np.asarray(rec[f"pattern{L}"])[:, :, 36:42, :36]
    own = np.zeros((6, 36), bool)
    for qq in range(6): own[qq, qq * 6:(qq + 1) * 6] = True
    return dict(T_share=float((A * own).sum(-1).mean()), oq_share=float((A * ~own).sum(-1).mean()), **n)
z = lambda d: np.load(os.path.join(d, "step100.npz")); zB, zI = z(bd), z(idr)
dI = np.concatenate([(zI[f"arr_{k}"] - zB[f"arr_{k}"]).ravel() for k in range(len(names))]); dI0 = np.concatenate([(zI[f"arr_{k}"] - zB[f"arr_{k}"]).ravel() for k in l0])
R = dict(B=diag(bd), I=diag(idr), X={})
for x in sys.argv[6:]:
    zX = z(x); dX = np.concatenate([(zX[f"arr_{k}"] - zB[f"arr_{k}"]).ravel() for k in range(len(names))]); dX0 = np.concatenate([(zX[f"arr_{k}"] - zB[f"arr_{k}"]).ravel() for k in l0])
    R["X"][os.path.basename(os.path.normpath(x))] = dict(**diag(x), dist_ratio=float(np.linalg.norm(dX) / np.linalg.norm(dI)), L0_dist_ratio=float(np.linalg.norm(dX0) / np.linalg.norm(dI0)),
                                                         L0_cos_with_block=float(dX0 @ dI0 / (np.linalg.norm(dX0) * np.linalg.norm(dI0) + 1e-30)))
json.dump(R, open(out, "w"), indent=1)
for k in ("B", "I"): print(k.ljust(44), {a: round(b, 3) for a, b in R[k].items()})
for k, v in R["X"].items(): print(k[22:66].ljust(44), {a: round(b, 3) for a, b in v.items()})
