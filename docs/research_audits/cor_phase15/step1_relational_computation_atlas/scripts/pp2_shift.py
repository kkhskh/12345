"""PP2 (frozen): under shifted generators, do models side with the training-distribution Bayes plug-in or with the
shifted generator's true Bayes? Extra (not frozen): T1 models on the same shifts, against the T1 plug-in.
Usage: pp2_shift.py OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C

shifts = {"rho0.9": G.Fam(pc=1.0, rho=0.9), "rho0.7": G.Fam(pc=1.0, rho=0.7), "pc0_T1data": G.Fam(pc=0.0)}
plug4, plug1 = G.Fam(pc=1.0, rho=0.999), G.Fam(pc=0.0)


def model_lp(kind, path, X):
    if kind == "tf":
        p = jax.tree_util.tree_map(jnp.asarray, C.load(path)); fwd = lambda Xb: N.forward(p, Xb)
    else:
        p = jax.tree_util.tree_map(jnp.asarray, C.load_base(path)); fb = N.make_forward("deepsets", 3); fwd = lambda Xb: fb(p, Xb)
    jf = jax.jit(lambda Xb: fwd(Xb)["answer"])
    return np.concatenate([np.asarray(jf(jnp.asarray(X[i:i + 3000]))) for i in range(0, len(X), 3000)])


m4 = {f"tf_curr_s{s}": ("tf", os.path.join(LR, "checkpoints/posthoc_curriculum", f"T4_M_K3_s{s}_fromT1.npz")) for s in [0, 1, 2]}
m4.update({f"tf_curr_s{s}": ("tf", os.path.join(LR, "checkpoints/replication", f"T4_M_K3_s{s}_fromT1.npz")) for s in [10, 11, 12, 13]})
m4.update({f"deepsets_T4_s{s}": ("ds", os.path.join(LR, "checkpoints/main", f"T4_M_K3_s{s}_deepsets.npz")) for s in [0, 1, 2]})
m1 = {f"tf_T1_s{s}": ("tf", os.path.join(LR, "checkpoints/main", f"T1_M_K3_s{s}.npz")) for s in [0, 1, 2]}
if __name__ == "__main__":
    res = {}
    for si, (sn, f) in enumerate(shifts.items()):
        d = G.sample(f, 6000, np.random.default_rng(6_200_000 + si)); X, Y = d["X"], d["Y"]
        tb = G.posterior(f, X)["post_y"].argmax(-1)
        pp4 = G.posterior(plug4, X)["post_y"]; pp1 = G.posterior(plug1, X)["post_y"]
        ok4 = np.isfinite(pp4).all(-1); pb4 = np.nan_to_num(pp4, nan=0).argmax(-1); pb1 = pp1.argmax(-1)
        R = dict(acc_true_bayes=float((tb == Y).mean()), acc_plugin4=float((pb4 == Y)[ok4].mean()), plugin4_defined=float(ok4.mean()),
                 acc_plugin1=float((pb1 == Y).mean()))
        dis4 = ok4 & (pb4 != tb); dis1 = pb1 != tb
        R["n_disagree_true_vs_plugin4"] = int(dis4.sum()); R["n_disagree_true_vs_plugin1"] = int(dis1.sum())
        for name, (kind, path) in list(m4.items()) + list(m1.items()):
            a = model_lp(kind, path, X).argmax(-1)
            dis, pb = (dis4, pb4) if name in m4 else (dis1, pb1)
            R[name] = dict(acc=float((a == Y).mean()), side_with_plugin=float((a == pb)[dis].mean()) if dis.any() else None,
                           side_with_true=float((a == tb)[dis].mean()) if dis.any() else None,
                           acc_where_plugin_undefined=float((a == Y)[~ok4].mean()) if (~ok4).any() else None)
        res[sn] = R
        print(sn, {k: v for k, v in R.items() if not isinstance(v, dict)}, flush=True)
        for name in list(m4) + list(m1):
            print("   ", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[name].items()}, flush=True)
    pp2 = {}
    for sn in shifts:
        if sn.startswith("pc0") or True:
            votes = [res[sn][n]["side_with_plugin"] >= 0.7 for n in m4 if res[sn][n]["side_with_plugin"] is not None]
            pp2[sn] = dict(models_siding_with_plugin=int(sum(votes)), n=len(votes))
    res["PP2_summary"] = pp2
    res["PP2_H1strong_falsified"] = any(v["models_siding_with_plugin"] <= v["n"] / 2 for v in pp2.values())
    json.dump(res, open(sys.argv[1], "w"), indent=1); print(pp2, "H1-strong falsified:", res["PP2_H1strong_falsified"])
