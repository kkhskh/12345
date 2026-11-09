"""Natural accuracy and ECE of the 10 learned models (T4 natural, stream 6_150_000, 3000 episodes), joined with the PP1
sign agreement from logs/pp1_calibration.json; plus the max edit/natural density ratio over count-pattern classes."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src")); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp, gen_family as G, nets as N, ckpt as C
f4 = G.Fam(pc=1.0, rho=1.0); Dn = G.sample(f4, 3000, np.random.default_rng(6_150_000)); P = G.posterior(f4, Dn["X"])["post_y"]
pc = json.load(open(os.path.join(HERE, "..", "logs", "pp1_calibration.json")))
def ece(Pm, Y, nb=10):
    conf = Pm.max(-1).ravel(); cor = (Pm.argmax(-1) == Y).ravel(); e = 0
    for j in range(nb):
        m = (conf > j / nb) & (conf <= (j + 1) / nb)
        if m.any(): e += m.mean() * abs(conf[m].mean() - cor[m].mean())
    return float(e)
paths = {f"tf_s{s}": ("tf", f"checkpoints/posthoc_curriculum/T4_M_K3_s{s}_fromT1.npz") for s in [0, 1, 2]}
paths.update({f"tf_s{s}": ("tf", f"checkpoints/replication/T4_M_K3_s{s}_fromT1.npz") for s in [10, 11, 12, 13]})
paths.update({f"ds_s{s}": ("ds", f"checkpoints/main/T4_M_K3_s{s}_deepsets.npz") for s in [0, 1, 2]})
out = {}
for nm, (k, pth) in paths.items():
    pth = os.path.join(LR, pth)
    if k == "tf": p = jax.tree_util.tree_map(jnp.asarray, C.load(pth)); f = lambda X: N.forward(p, X)
    else: p = jax.tree_util.tree_map(jnp.asarray, C.load_base(pth)); fb = N.make_forward("deepsets", 3); f = lambda X: fb(p, X)
    l = np.asarray(jax.nn.log_softmax(f(jnp.asarray(Dn["X"]))["answer"], -1)); Pm = np.exp(l)
    out[nm] = dict(acc=float((l.argmax(-1) == Dn["Y"]).mean()), ECE=ece(Pm, Dn["Y"]), KL=pc[nm]["mean_KL_natural"], sign=pc[nm]["sign_agreement_scored"],
                   C1=pc[nm]["cert_C1_pointwise_margin_frac"])
    print(nm, {a: round(b, 4) for a, b in out[nm].items()}, flush=True)
v = np.array([[o["ECE"], o["KL"], o["sign"]] for o in out.values()])
out["spearman_ECE_vs_sign"] = float(np.corrcoef(np.argsort(np.argsort(v[:, 0])), np.argsort(np.argsort(v[:, 2])))[0, 1])
out["spearman_KL_vs_sign"] = float(np.corrcoef(np.argsort(np.argsort(v[:, 1])), np.argsort(np.argsort(v[:, 2])))[0, 1])
print(out["spearman_ECE_vs_sign"], out["spearman_KL_vs_sign"])
json.dump(out, open(os.path.join(HERE, "..", "logs", "pp1_learned_ece.json"), "w"), indent=1)
