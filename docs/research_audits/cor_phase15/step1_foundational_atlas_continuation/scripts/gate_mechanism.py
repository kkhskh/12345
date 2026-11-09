"""GATE mechanism localisation (mandate §11): where does the gate relation x0 + x1 = 0 (mod K) live when the last layer
routes?  Intervention: flip the gate in question q (re-draw x1), keeping all evidence sources.  Scored on cases where the
Bayes argmax changes AND both Bayes argmaxes are unique (margin > 1e-6).  Patches (from the flipped run into the base run):
 at resid_pre{1,2}: gate tokens (6q, 6q+1), evidence tokens (6q+2..6q+5), readout token (36+q), combinations.
Also the input-level flip itself (model vs Bayes).  Stream 9_250_000.  Writes logs/gate_mechanism.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, tasks_ext as T, fwd as F, ckpt as C
K, Q = 3, 6
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(HERE, "..", "checkpoints", "gate_s0.npz"))); p.pop("WX", None)
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n.startswith("resid_pre"): rec[n] = a
        return a
    o = F.forward(p, X, hk); return jax.nn.softmax(o["answer"], -1), rec["resid_pre1"], rec["resid_pre2"]
def patched(X, src, site, mask):
    @jax.jit
    def f(X, src, mask):
        def hk(n, a):
            return a + (src - a) * mask[None, :, None] if n == site else a
        return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
    return np.asarray(f(jnp.asarray(X), jnp.asarray(src), jnp.asarray(mask)))
rng = np.random.default_rng(9_250_000); d = T.sample_gate(4000, rng); X = d["X"]
res = {}; agg = {}
for q in range(Q):
    Xf = X.copy(); op = (X[:, q, 0] + X[:, q, 1]) % K == 0
    Xf[:, q, 1] = np.where(op, ((-X[:, q, 0]) % K + 1 + rng.integers(0, K - 1, len(X))) % K, (-X[:, q, 0]) % K)
    pb, pf = T.bayes_gate(X)[:, q], T.bayes_gate(Xf)[:, q]
    def uniq(P): s = np.sort(P, -1); return s[:, -1] - s[:, -2] > 1e-6
    sel = (pb.argmax(-1) != pf.argmax(-1)) & uniq(pb) & uniq(pf)
    P0, r1, r2 = map(np.asarray, acts(jnp.asarray(X))); Pf, f1, f2 = map(np.asarray, acts(jnp.asarray(Xf)))
    agg.setdefault("n_scored", 0); agg["n_scored"] += int(sel.sum())
    agg.setdefault("input_flip_model_follows_bayes", []).append((Pf[:, q].argmax(-1) == pf.argmax(-1))[sel])
    agg.setdefault("base_model_agrees_bayes", []).append((P0[:, q].argmax(-1) == pb.argmax(-1))[sel])
    for site, src in (("resid_pre1", f1), ("resid_pre2", f2)):
        for nm, pos in (("gate", [6 * q, 6 * q + 1]), ("evidence", list(range(6 * q + 2, 6 * q + 6))), ("readout", [36 + q]),
                        ("gate+evidence", list(range(6 * q, 6 * q + 6))), ("evidence+readout", list(range(6 * q + 2, 6 * q + 6)) + [36 + q]),
                        ("all_q_tokens+readout", list(range(6 * q, 6 * q + 6)) + [36 + q])):
            m = np.zeros(43, np.float32); m[pos] = 1; o = patched(X, src, site, m)
            agg.setdefault(f"{site}|{nm}", []).append((o[:, q].argmax(-1) == pf.argmax(-1))[sel])
res = {k: (float(np.concatenate(v).mean()) if isinstance(v, list) else v) for k, v in agg.items()}
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(HERE, "..", "logs", "gate_mechanism.json"), "w"), indent=1)
