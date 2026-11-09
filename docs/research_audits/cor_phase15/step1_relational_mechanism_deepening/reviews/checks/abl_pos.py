import os, sys, json
B0 = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, B0 + "/src")
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import fwd3 as F3, ckpt as C, gen_family as G, analysis as A, bayes as B
ck = sys.argv[1]; Q, T, K = 6, 43, 3
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(B0, ck))); p.pop("WX", None); env = B.Env("T4"); f4 = G.Fam(); d = 64
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n == "resid_pre2": rec["r"] = a
        return a
    o = F3.forward(p, X, hk); return rec["r"], jax.nn.softmax(o["answer"], -1)
@jax.jit
def patched(X, tgt, P, mask): return jax.nn.softmax(F3.forward(p, X, lambda n, a: a + ((tgt - a) @ P) * mask[None, :, None] if n == "resid_pre2" else a)["answer"], -1)
SITES = {"readout": list(range(36, 42)), "source": list(range(36))}
def mk(pos): m = np.zeros(T, np.float32); m[pos] = 1; return jnp.asarray(m)
Di = G.sample(f4, 800, np.random.default_rng(9_890_000))["X"]; ai = np.asarray(acts(jnp.asarray(Di))[0]); dk = {s: [] for s in SITES}
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(9_890_010 + q)); dd = np.asarray(acts(jnp.asarray(Xp))[0]) - ai
    dk["readout"].append(dd[:, 36 + q]); dk["source"].append(dd[:, [t for t in range(36) if t // 6 != q]].reshape(-1, d))
bases = {s: A.contrast_basis(np.concatenate(v), 4) for s, v in dk.items()}
D = G.sample(f4, 1000, np.random.default_rng(9_900_000)); X, Y = D["X"], D["Y"]; a0, P0 = map(np.asarray, acts(jnp.asarray(X)))
own = G.posterior(f4, X)["own_y"]; ac = (P0.argmax(-1) == Y).mean(); ao = (own.argmax(-1) == Y).mean(); res = {}
rng = np.random.default_rng(3)
for s, pos in SITES.items():
    Pc = A.proj(bases[s]).astype(np.float32); Pr = A.proj(A.random_basis(d, 4, rng)).astype(np.float32)
    pooled = np.broadcast_to(a0[:, pos].reshape(-1, d).mean(0), a0.shape); perpos = np.broadcast_to(a0.mean(0, keepdims=True), a0.shape)
    Z = a0[:, pos] - a0[:, pos].mean(0, keepdims=True); Z = Z.reshape(-1, d); Zo = Z - Z @ Pc; Pv = A.proj(np.linalg.svd(Zo, full_matrices=False)[2][:4].T).astype(np.float32)
    for mn, M in (("pooled_mean", pooled), ("per_position_mean", perpos)):
        for nm, Pm in (("count", Pc), ("random", Pr), ("varmatched_perpos_centred", Pv)):
            o = np.asarray(patched(jnp.asarray(X), jnp.asarray(M, jnp.float32), jnp.asarray(Pm), mk(pos))); a = (o.argmax(-1) == Y).mean()
            res[f"{s}|{mn}|{nm}"] = round(float((ac - a) / (ac - ao)), 2)
print(ck, json.dumps(res))
