"""Post-hoc adapter ablation for trained radapt/lca repairs (written after the freeze; does not modify the frozen repair_train.py).
Executes repair_train.py's model definition (everything before its training loop) unchanged, loads the trained parameters, sets the
adapter's output projection to zero, and evaluates.  Usage: repair_ablate.py PHASE PARENT SPEC -> checkpoints/PHASE/<parent>_<spec>_ablate.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "repair_train.py")).read(); src = src[:src.index("rng = np.random.default_rng(stream)")]
g = {"__file__": os.path.join(HERE, "repair_train.py"), "__name__": "repair_def"}; exec(compile(src, "repair_train.py[def]", "exec"), g)
np, jax, jnp, V, ER = g["np"], g["jax"], g["jnp"], g["V"], g["ER"]
OUT, tag, spec = g["OUT"], g["tag"], g["spec"]
t = jax.tree_util.tree_map(jnp.asarray, V.load_tree(os.path.join(OUT, tag + ".npz")))
res = {"trained": ER.headline(ER.run(g["fn_of"](t), g["ev"]))}
if spec == "radapt": t["ra"]["out"]["W"] = jnp.zeros_like(t["ra"]["out"]["W"]); t["ra"]["out"]["b"] = jnp.zeros_like(t["ra"]["out"]["b"])
else: t["ad"][-1]["W"] = jnp.zeros_like(t["ad"][-1]["W"]); t["ad"][-1]["b"] = jnp.zeros_like(t["ad"][-1]["b"])
res["adapter_ablated"] = ER.headline(ER.run(g["fn_of"](t), g["ev"]))
for k, v in res.items(): print(tag, k, " ".join(f"{a} {b:+.3f}" for a, b in v.items()), flush=True)
json.dump(res, open(os.path.join(OUT, tag + "_ablate.json"), "w"), indent=1, default=float)
