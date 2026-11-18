"""XVIII frozen H3 metric (pre-review A addendum A4, B addendum fix 3): the IMMEDIATE loss change of an arm's training-time forward pass at
the paired baseline's step-100 parameters, on a fixed batch of 1024 episodes (stream 95_400_000 + 17): loss(arm forward) - loss(plain forward).
Arms: vrg and sha_hi_rolled (target = the baseline schedule at step 99), for every E and M pair of the design.
Usage: h3_immediate18.py DESIGN.json OUT.json"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xviii_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace('OUT = os.path.join(HERE, "..", "checkpoints", "ft", tag); os.makedirs(OUT, exist_ok=True)', "OUT = None")
import numpy as np
D = json.load(open(sys.argv[1])); P = os.path.join(H, ".."); out = {}
def env(args, envstr):
    for k in list(os.environ):
        if k.startswith("FX_"): os.environ.pop(k)
    for kv in envstr.split():
        k, v = kv.split("=", 1); os.environ[k] = v
    sys.argv = ["x"] + args; g = {"__file__": os.path.join(H, "xviii_train.py"), "__name__": "h3"}; exec(SRC, g); return g
for b, B in D["bases"].items():
    for Dd in D["orders"]:
        args = [D["parent"], str(B["L"]), str(B["W"]), str(Dd)]; bdir = os.path.join(P, D["runs"][f"{b}|{Dd}|baseline"]["dir"]); res = {}
        for arm in ("vrg", "sha_hi_rolled"):
            g = env(args, D["runs"][f"{b}|{Dd}|{arm}"]["env"].replace("FX_SHA_SCHED=", "FX_SHA_SCHED=" + P + "/")); import jax, jax.numpy as jnp
            z = np.load(os.path.join(bdir, "step100.npz")); t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(g["t"]), [jnp.asarray(z[f"arr_{k}"]) for k in range(len(z.files))])
            dd = g["NL"].Bm.sample_fast(g["NL"].ENV, 1024, np.random.default_rng(D["stream"] + 17)); X, Y = jnp.asarray(dd["X"]), jnp.asarray(dd["Y"])
            res[arm] = float(g["lossf"](t, X, Y, True, jnp.int32(99)) - g["lossf"](t, X, Y, False))
        out[f"{b}|{Dd}"] = res; print(b, Dd, res, flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
