"""PP5 (frozen prediction): head splitting changes a head-count-dependent criterion while (nearly) preserving function.
Construct from replication seed 11: copy route head h_src into the inert head h_dst (identical W_Q, W_K, W_V), halve
both heads' W_O.  The new network equals the original minus head h_dst's original contribution (exactly).
Then re-run the frozen replication evaluation (scripts/replicate_t4.py, evaluation stream) on the constructed model.
Usage: pp5_headsplit.py OUT.json"""
import json, os, subprocess, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C

src_ck = os.path.join(LR, "checkpoints/replication/T4_M_K3_s11_fromT1.npz")
p = C.load(src_ck); L = len(p["layers"]) - 1
h_src, h_dst = 2, 1           # head 2: largest Shapley path share in seed 11 (logs/mech/s11.json); head 1: inert (share ~0.03)
q = jax.tree_util.tree_map(np.array, p); lay = q["layers"][L]
for W in ["WQ", "WK", "WV"]:
    lay[W][h_dst] = lay[W][h_src]
lay["WO"][h_src] = lay["WO"][h_src] / 2; lay["WO"][h_dst] = lay["WO"][h_src].copy()
# reference: original with head h_dst ablated (its W_O zeroed) -- the construction must equal this exactly
r = jax.tree_util.tree_map(np.array, p); r["layers"][L]["WO"][h_dst] = 0 * r["layers"][L]["WO"][h_dst]
out_ck = os.path.join(HERE, "..", "checkpoints", "pp5_s11_headsplit.npz"); C.save(out_ck, q)
X = G.sample(G.Fam(), 4000, np.random.default_rng(7_200_000))["X"]; Y = G.sample(G.Fam(), 4000, np.random.default_rng(7_200_000))["Y"]
pr = lambda P_: np.asarray(jax.nn.softmax(N.forward(jax.tree_util.tree_map(jnp.asarray, P_), jnp.asarray(X))["answer"], -1))
po, pq, pa = pr(p), pr(q), pr(r)
res = dict(construction_equals_ablation_max_abs=float(np.abs(pq - pa).max()),
           vs_original=dict(max_abs_prob=float(np.abs(pq - po).max()), argmax_agree=float((pq.argmax(-1) == po.argmax(-1)).mean()),
                            mean_kl=float((po * (np.log(po + 1e-12) - np.log(pq + 1e-12))).sum(-1).mean())),
           acc_original=float((po.argmax(-1) == Y).mean()), acc_constructed=float((pq.argmax(-1) == Y).mean()))
print(res, flush=True)
ev = os.path.join(HERE, "..", "logs", "pp5_replicate_eval_constructed.json")
subprocess.run([sys.executable, os.path.join(LR, "scripts", "replicate_t4.py"), out_ck, os.path.join(LR, "checkpoints/replication/T1_M_K3_s11.npz"), ev], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
e = json.load(open(ev)); o = json.load(open(os.path.join(LR, "logs/replication/eval_s11.json")))
res["original_R3"] = o["criteria"]["R3"]; res["constructed_R3"] = e["criteria"]["R3"]
res["constructed_route_heads"] = e["route_heads"]; res["constructed_control_heads"] = e["control_heads"]
res["constructed_route_frozen_iia"] = e["tests"]["joint_route_frozen"]["iia"]; res["original_route_frozen_iia"] = o["tests"]["joint_route_frozen"]["iia"]
res["constructed_criteria"] = e["criteria"]
res["PP5_pass"] = bool(res["original_R3"] and not res["constructed_R3"] and res["vs_original"]["argmax_agree"] >= 0.98)
json.dump(res, open(sys.argv[1], "w"), indent=1); print(res)
