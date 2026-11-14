"""Workstream B2: algorithm identity along training (behavioural, on-support discriminating edits; development models only).
For each saved checkpoint of a full_c1 run, the mean response distance to rules B, A1, A2, A3, A0 on D1, D2, F45_R1, F45_R2 (definitions as
scripts/algorithm_discrimination.py, eval_dev).  -> logs/traj_alg.json"""
import json, os, sys, re, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F
src = open(os.path.join(HERE, "algorithm_discrimination.py")).read(); src = src[:src.index("out = dict(n=")]
EVP = os.environ.get("TRAJ_EVAL", "dev"); MODELS = os.environ.get("TRAJ_MODELS", "xP81:full_c1:0:,yP130:full_c1:0:_eval_conf9,yP131:full_c1:0:_eval_conf9,yP132:full_c1:0:_eval_conf9,yP134:full_c1:0:_eval_conf9").split(","); sys.argv = ["x", EVP]; g = {"__file__": os.path.join(HERE, "algorithm_discrimination.py"), "__name__": "alg"}; exec(compile(src, "alg[def]", "exec"), g)
fams, pred, ALGS = g["fams"], g["pred"], g["ALGS"]
out = {}
for sp in MODELS:
    parent, spec, seed, suf = sp.split(":"); D = F.model_dir(f"{parent}_{spec}_s4000_seed{seed}{suf}")
    steps = sorted(int(re.findall(r"step(\d+)", os.path.basename(f))[0]) for f in glob.glob(os.path.join(D, "step*.npz"))); rows = []
    for k in steps:
        M = F.load_model(parent, spec, seed=int(seed), suffix=suf, step=k); fj = jax.jit(lambda X_: F.forward(M, X_)); r = {"step": k}
        for f in ("D1", "D2", "F45_R1", "F45_R2"):
            Xb, Xa, qq = fams[f]; i = np.arange(len(qq)); D_ = F.centred(F.batched(fj, Xa)[i, qq]) - F.centred(F.batched(fj, Xb)[i, qq])
            dist = {a: float(np.abs(D_ - pred[f][a]).mean()) for a in ALGS}; r[f] = dict(dist, closest=min(ALGS, key=lambda a: dist[a]))
        rows.append(r); print(sp, k, {f: (r[f]["closest"], round(r[f]["B"], 2), round(r[f]["A1"], 2), round(r[f]["A3"], 2)) for f in ("D1", "D2", "F45_R1")}, flush=True)
    out[sp] = rows
json.dump(out, open(os.path.join(HERE, "..", "logs", "traj_alg" + os.environ.get("TRAJ_TAG", "") + ".json"), "w"), indent=1)
