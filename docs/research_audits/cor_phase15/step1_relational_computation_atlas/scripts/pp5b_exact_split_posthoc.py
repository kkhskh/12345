"""POST HOC complement to the failed PP5: an EXACT function-preserving head split.  Add a 5th last-layer head to seed 11
that copies head 2's Q/K/V; halve head 2's W_O and give the copy the other half.  Earlier layers keep 4 heads (the
forward pass handles any per-layer head count).  Then re-run the frozen replication evaluation."""
import json, os, subprocess, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C
p = C.load(os.path.join(LR, "checkpoints/replication/T4_M_K3_s11_fromT1.npz")); L = len(p["layers"]) - 1
q = jax.tree_util.tree_map(np.array, p); lay = q["layers"][L]
for W in ["WQ", "WK", "WV"]: lay[W] = np.concatenate([lay[W], lay[W][2:3]], 0)
lay["WO"] = np.concatenate([lay["WO"], lay["WO"][2:3] / 2], 0); lay["WO"][2] = lay["WO"][2] / 2
ck = os.path.join(HERE, "..", "checkpoints", "pp5b_s11_exact_split5.npz"); C.save(ck, q)
X = G.sample(G.Fam(), 4000, np.random.default_rng(7_200_000))["X"]
pr = lambda P_: np.asarray(jax.nn.softmax(N.forward(jax.tree_util.tree_map(jnp.asarray, P_), jnp.asarray(X))["answer"], -1))
po, pq = pr(p), pr(q)
res = dict(note="POST HOC", max_abs_prob_diff=float(np.abs(po - pq).max()), argmax_agree=float((po.argmax(-1) == pq.argmax(-1)).mean()))
ev = os.path.join(HERE, "..", "logs", "pp5b_replicate_eval_exact_split.json")
if not os.path.exists(ev):
  r = subprocess.run([sys.executable, os.path.join(HERE, "replicate_t4_lastH_copy.py"), ck, os.path.join(LR, "checkpoints/replication/T1_M_K3_s11.npz"), ev], capture_output=True, text=True)
  if r.returncode: print(r.stderr[-2000:])
e = json.load(open(ev)); res.update(route_heads=e["route_heads"], control_heads=e["control_heads"], criteria=e["criteria"],
       route_frozen_iia=e["tests"]["joint_route_frozen"]["iia"], control_frozen_iia=e["tests"]["joint_control_frozen"]["iia"], head_r2=e["head_group_mass_r2_eval"])
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp5b_exact_split_posthoc.json"), "w"), indent=1); print(res)
