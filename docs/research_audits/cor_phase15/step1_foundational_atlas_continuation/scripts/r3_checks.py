"""Review R3 verification: (a) GATE with all last-layer head outputs zeroed (and each layer's heads zeroed);
(b) P-count 1000-step frozen-feature fine-tune (the dynamics protocol); (c) readout-token probes of count_loo for P-count vs T1.
Writes logs/r3_checks.json."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp, tasks_ext as T, fwd as F, ckpt as C, train_ext as TE, gen_family as G
from sklearn.linear_model import LogisticRegression
out = {}
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(HERE, "..", "checkpoints", "gate_s0.npz"))); p.pop("WX", None)
d = T.sample_gate(6000, np.random.default_rng(9_200_000)); X, Y = d["X"], d["Y"]; pb = T.bayes_gate(X)
s = np.sort(pb, -1); uq = s[..., -1] - s[..., -2] > 1e-6
for L in (0, 1, 2):
    def hk(n, a, L=L): return a * 0 if n == f"head_out{L}" else a
    P = np.asarray(jax.jit(lambda Z: jax.nn.softmax(F.forward(p, Z, hk)["answer"], -1))(jnp.asarray(X)))
    out[f"gate_zero_heads_layer{L}"] = dict(acc=float((P.argmax(-1) == Y).mean()), agree_bayes_unique=float((P.argmax(-1) == pb.argmax(-1))[uq].mean()))
print(out, flush=True)
q0 = C.load(os.path.join(HERE, "..", "checkpoints", "pre_pcount.npz")); q0.pop("WX", None)
pt, _, _ = TE.train("T4", 0, 1000, p0=q0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True)
out["pcount_transfer_1000steps"] = TE.evaluate(pt, "T4", 5000, 6_300_000)["acc"]; print(out["pcount_transfer_1000steps"], flush=True)
f4 = G.Fam(); D = G.sample(f4, 2500, np.random.default_rng(9_100_000))["X"]
cnt = np.stack([(D[:, :, a] == D[:, :, b]).sum(1, keepdims=True) - (D[:, :, a] == D[:, :, b]) for a, b in G.GROUPS], -1)  # N,Q,3
for nm, path in (("P-count", os.path.join(HERE, "..", "checkpoints", "pre_pcount.npz")), ("P-reliability(T1)", os.path.join(LR, "checkpoints", "main", "T1_M_K3_s0.npz"))):
    q = C.load(path); q.pop("WX", None); q = jax.tree_util.tree_map(jnp.asarray, q)
    def hk2(n, a, rec={}):
        return a
    rec = {}
    def hk(n, a):
        if n == "resid_pre2": rec["r"] = a
        return a
    F.forward(q, jnp.asarray(D), hk); R = np.asarray(rec["r"])[:, 36:42]            # readout tokens
    Z = R.reshape(-1, R.shape[-1]); accs = []
    for g in range(3):
        t = cnt[:, :, g].reshape(-1); tr = np.arange(len(t)) < 1500 * 6
        m = LogisticRegression(max_iter=1000).fit(Z[tr], t[tr]); accs.append(float(m.score(Z[~tr], t[~tr])))
    out[f"readout_count_loo_acc_{nm}"] = accs; print(nm, accs, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "r3_checks.json"), "w"), indent=1)
