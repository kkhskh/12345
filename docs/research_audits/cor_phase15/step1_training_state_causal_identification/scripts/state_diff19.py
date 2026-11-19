"""XIX: inventory of the step-100 learned-state difference between matched baseline (B) and intervention (I) trajectories.
Per real parameter group: ||theta_I - theta_B||, ||theta_B - theta_0|| (training movement), their ratio; Adam mu/nu differences per group;
SHA-256 of every state file.  Usage: state_diff19.py BDIR IDIR PARENT L W OUT.json"""
import os, sys, json, pickle, hashlib
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xix_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace('OUT = os.path.join(HERE, "..", "checkpoints", "ft", tag); os.makedirs(OUT, exist_ok=True)', "OUT = None")
import numpy as np
BD, ID, parent, L, W, OUTF = sys.argv[1:7]
for k in list(os.environ):
    if k.startswith("FX_"): os.environ.pop(k)
sys.argv = ["x", parent, L, W, "801"]; g = {"__file__": os.path.join(H, "xix_train.py"), "__name__": "sd"}; exec(SRC, g); import jax
def group(name):
    p = name.split("/")
    if p[0] == "emb": return "emb"
    if p[0] in ("gf", "bf"): return "final_LN"
    if p[0] == "WA": return "WA"
    lay = "L" + p[1] if p[0] == "pre" else "L2"; t = p[-1]
    return lay + "_" + ("QK" if t in ("WQ", "WK") else "OV" if t in ("WV", "WO") else "LN" if t in ("g1", "b1", "g2", "b2") else "MLP")
names = ["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in p) for p, _ in jax.tree_util.tree_flatten_with_path(g["t"])[0]]
t0 = [np.asarray(x) for x in jax.tree_util.tree_leaves(g["t"])]
ld = lambda D: [np.load(os.path.join(D, "step100.npz"))[f"arr_{k}"] for k in range(len(names))]
tB, tI = ld(BD), ld(ID); RB, RI = (pickle.load(open(os.path.join(D, "opt_step100.pkl"), "rb")) for D in (BD, ID))
stl = jax.tree_util.tree_leaves(g["st"]); nmu = len(names)
# optax adamw state leaves: [count, mu..., nu..., (schedule count)] -- locate mu/nu blocks by shape matching
shapes = [x.shape for x in t0]; ol = [np.asarray(a) for a in RB["opt"]]; idx = [i for i in range(len(ol) - 2 * nmu + 1) if [o.shape for o in ol[i:i + nmu]] == shapes and [o.shape for o in ol[i + nmu:i + 2 * nmu]] == shapes]
assert idx, "could not locate Adam moments"; i0 = idx[0]
oB, oI = [np.asarray(a) for a in RB["opt"]], [np.asarray(a) for a in RI["opt"]]
G = {}
for k, n in enumerate(names):
    gr = group(n); e = G.setdefault(gr, dict(d2=0.0, m2=0.0, b2=0.0, mu_d2=0.0, mu_b2=0.0, nu_d2=0.0, nu_b2=0.0, tensors=[]))
    e["d2"] += float(((tI[k] - tB[k]) ** 2).sum()); e["m2"] += float(((tB[k] - t0[k]) ** 2).sum()); e["b2"] += float((tB[k] ** 2).sum()); e["tensors"].append(n)
    e["mu_d2"] += float(((oI[i0 + k] - oB[i0 + k]) ** 2).sum()); e["mu_b2"] += float((oB[i0 + k] ** 2).sum())
    e["nu_d2"] += float(((oI[i0 + nmu + k] - oB[i0 + nmu + k]) ** 2).sum()); e["nu_b2"] += float((oB[i0 + nmu + k] ** 2).sum())
out = dict(groups={gr: dict(diff=e["d2"] ** .5, movement_B=e["m2"] ** .5, ratio_diff_to_movement=(e["d2"] / max(e["m2"], 1e-30)) ** .5, rel_to_norm=(e["d2"] / max(e["b2"], 1e-30)) ** .5,
                            mu_rel_diff=(e["mu_d2"] / max(e["mu_b2"], 1e-30)) ** .5, nu_rel_diff=(e["nu_d2"] / max(e["nu_b2"], 1e-30)) ** .5, tensors=e["tensors"]) for gr, e in G.items()},
           counts=dict(B=[int(x) for x in oB if x.size == 1], I=[int(x) for x in oI if x.size == 1]), rng_equal=RB["rng"] == RI["rng"],
           sha256={os.path.basename(D) + "/" + f: hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest() for D in (BD, ID) for f in ("step100.npz", "opt_step100.pkl")})
json.dump(out, open(OUTF, "w"), indent=1)
for gr, v in sorted(out["groups"].items()): print(gr.ljust(9), f"diff {v['diff']:.4f}  B-movement {v['movement_B']:.4f}  ratio {v['ratio_diff_to_movement']:.2f}  mu {v['mu_rel_diff']:.2f}  nu {v['nu_rel_diff']:.2f}")
print("counts", out["counts"], "rng equal", out["rng_equal"])
