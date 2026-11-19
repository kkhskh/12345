"""XIX state composition for transplant experiments.
Builds a synthetic step-100 training state (step100.npz + opt_step100.pkl) in checkpoints/ft/<OUTTAG>/ from two matched step-100 states
B (baseline) and I (intervention) of the same parent, (L, W) and data order:
  --params-from-I GROUPS   parameter groups taken from I (others from B); groups as in state_diff19.group(), or 'all', or 'none'
  --opt MODE               Adam moments: 'B' | 'I' | 'follow' (each group's mu/nu from the same source as its parameters) | 'fresh' (zeros)
  --alpha A               instead of I's values, theta_B + A (theta_I - theta_B) on the selected groups (interpolation diagnostic)
  --perturb SEED           instead of I's values, B + a random direction with, per group, the norm ||theta_I - theta_B||_G (matched-norm control)
The step counter, learning-rate schedule position, data-stream RNG state and loss history are B's (identical to I's: verified, matched data).
Writes provenance.json (sources, groups, hashes of inputs and outputs).  Usage: compose19.py BDIR IDIR OUTTAG --params-from-I G1,G2 --opt MODE [--perturb SEED]"""
import os, sys, json, pickle, hashlib, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("bdir"); ap.add_argument("idir"); ap.add_argument("outtag"); ap.add_argument("--params-from-I", default="none")
ap.add_argument("--opt", default="follow"); ap.add_argument("--alpha", type=float, default=None); ap.add_argument("--foreign-donor", action="store_true"); ap.add_argument("--perturb", type=int, default=None); ap.add_argument("--names", default=None)
a = ap.parse_args()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
names = json.load(open(a.names or os.path.join(H, "..", "logs", "param_names19.json")))
def group(name):
    p = name.split("/")
    if p[0] == "emb": return "emb"
    if p[0] in ("gf", "bf"): return "final_LN"
    if p[0] == "WA": return "WA"
    lay = "L" + p[1] if p[0] == "pre" else "L2"; t = p[-1]
    return lay + "_" + ("QK" if t in ("WQ", "WK") else "OV" if t in ("WV", "WO") else "LN" if t in ("g1", "b1", "g2", "b2") else "MLP")
def sel(spec):
    if spec == "none": return set()
    allg = {group(n) for n in names}
    if spec == "all": return allg
    out = set()
    for s in spec.split(","):
        out |= {g for g in allg if g == s or g.startswith(s + "_")}          # 'L0' selects L0_QK, L0_OV, L0_MLP, L0_LN
    if not out: raise SystemExit(f"no group matches {spec}")
    return out
G = sel(a.params_from_I)
zB, zI = np.load(os.path.join(a.bdir, "step100.npz")), np.load(os.path.join(a.idir, "step100.npz")); n = len(names); assert len(zB.files) == n == len(zI.files)
RB, RI = pickle.load(open(os.path.join(a.bdir, "opt_step100.pkl"), "rb")), pickle.load(open(os.path.join(a.idir, "opt_step100.pkl"), "rb"))
assert RB["step"] == RI["step"] == 100 and (RB["rng"] == RI["rng"] or a.foreign_donor), "states are not matched (step or data-stream RNG differ)"   # --foreign-donor: donor from another data order (control); rng stays B's
tB, tI = [zB[f"arr_{k}"] for k in range(n)], [zI[f"arr_{k}"] for k in range(n)]
oB, oI = [np.asarray(x) for x in RB["opt"]], [np.asarray(x) for x in RI["opt"]]
shapes = [x.shape for x in tB]; i0 = [i for i in range(len(oB) - 2 * n + 1) if [o.shape for o in oB[i:i + n]] == shapes and [o.shape for o in oB[i + n:i + 2 * n]] == shapes][0]
new_t = [tI[k].copy() if group(names[k]) in G else tB[k].copy() for k in range(n)]
if a.alpha is not None:                                                # XIX: interpolation theta_B + alpha (theta_I - theta_B) on the selected groups
    assert a.opt in ("B", "I", "fresh"), "interpolated states need an explicit optimiser source"
    new_t = [(tB[k] + a.alpha * (tI[k] - tB[k])).astype(tB[k].dtype) if group(names[k]) in G else tB[k].copy() for k in range(n)]
if a.perturb is not None:
    rng = np.random.default_rng(a.perturb); new_t = [x.copy() for x in tB]
    for g_ in sorted(G):                                               # pre-review B4: deterministic order (set order depends on PYTHONHASHSEED)
        ks = [k for k in range(n) if group(names[k]) == g_]; dn = np.sqrt(sum(((tI[k] - tB[k]) ** 2).sum() for k in ks))
        r = [rng.standard_normal(tB[k].shape) for k in ks]; rn = np.sqrt(sum((x ** 2).sum() for x in r))
        for k, x in zip(ks, r): new_t[k] = (tB[k] + x * (dn / rn)).astype(tB[k].dtype)
new_o = [x.copy() for x in oB]
for k in range(n):
    src = {"B": "B", "I": "I", "fresh": "fresh", "follow": ("I" if (group(names[k]) in G and a.perturb is None) else "B")}[a.opt]
    for off in (0, n):
        new_o[i0 + off + k] = np.zeros_like(oB[i0 + off + k]) if src == "fresh" else (oI if src == "I" else oB)[i0 + off + k].copy()
D = os.path.join(H, "..", "checkpoints", "ft", a.outtag)
if os.path.exists(os.path.join(D, "provenance.json")):                 # pre-review B4: never silently overwrite a composed state
    import tempfile; T_ = tempfile.mkdtemp(); np.savez(os.path.join(T_, "step100.npz"), *new_t); pickle.dump(dict(step=100, rng=RB["rng"], hist=RB["hist"], opt=new_o), open(os.path.join(T_, "opt_step100.pkl"), "wb"))
    old_ = json.load(open(os.path.join(D, "provenance.json")))["outputs"]
    if all(old_.get(f) == sha(os.path.join(T_, f)) == sha(os.path.join(D, f)) for f in ("step100.npz", "opt_step100.pkl")): print(a.outtag, "already composed (identical)"); sys.exit(0)
    raise SystemExit(f"{a.outtag}: an existing composed state differs from the recomposition; refusing to overwrite")
os.makedirs(D, exist_ok=True)
np.savez(os.path.join(D, "step100.npz"), *new_t)
pickle.dump(dict(step=100, rng=RB["rng"], hist=RB["hist"], opt=new_o), open(os.path.join(D, "opt_step100.pkl"), "wb"))
prov = dict(B=os.path.basename(os.path.normpath(a.bdir)), I=os.path.basename(os.path.normpath(a.idir)), params_from_I=sorted(G) if a.perturb is None else [], perturbed_groups=sorted(G) if a.perturb is not None else [],
            perturb_seed=a.perturb, alpha=a.alpha, foreign_donor=a.foreign_donor, opt=a.opt, inputs={f"{lab}/{f}": sha(os.path.join(d, f)) for lab, d in (("B", a.bdir), ("I", a.idir)) for f in ("step100.npz", "opt_step100.pkl")},
            outputs={f: sha(os.path.join(D, f)) for f in ("step100.npz", "opt_step100.pkl")},
            param_dist_to_B=float(np.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(new_t, tB)))), param_dist_to_I=float(np.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(new_t, tI)))))
json.dump(prov, open(os.path.join(D, "provenance.json"), "w"), indent=1); print(a.outtag, "groups", prov["params_from_I"] or prov["perturbed_groups"], "opt", a.opt, "dB", round(prov["param_dist_to_B"], 3), "dI", round(prov["param_dist_to_I"], 3))
