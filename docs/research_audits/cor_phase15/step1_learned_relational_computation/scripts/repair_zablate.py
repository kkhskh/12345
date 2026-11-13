"""Post-hoc Z-ablation for trained radapt repairs (exploratory, after the freeze): keep the adapter, replace the frozen parent
representation z by its mean over the evaluation batch (per position), so the downstream block receives no episode-specific
information from Z.  Usage: repair_zablate.py PHASE PARENT SPEC -> checkpoints/PHASE/<parent>_<spec>_zablate.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "repair_train.py")).read(); src = src[:src.index("rng = np.random.default_rng(stream)")]
g = {"__file__": os.path.join(HERE, "repair_train.py"), "__name__": "repair_def"}; exec(compile(src, "repair_train.py[def]", "exec"), g)
np, jax, jnp, V, ER = g["np"], g["jax"], g["jnp"], g["V"], g["ER"]
t = jax.tree_util.tree_map(jnp.asarray, V.load_tree(os.path.join(g["OUT"], g["tag"] + ".npz")))
z0 = g["zfun"]; g["zfun"] = lambda tt, X: jnp.broadcast_to(z0(tt, X).mean(0, keepdims=True), z0(tt, X).shape)
res = {"z_mean_ablated": ER.headline(ER.run(g["fn_of"](t), g["ev"]))}
print(g["tag"], "z_mean_ablated", " ".join(f"{a} {b:+.3f}" for a, b in res["z_mean_ablated"].items()), flush=True)
json.dump(res, open(os.path.join(g["OUT"], g["tag"] + "_zablate.json"), "w"), indent=1, default=float)
