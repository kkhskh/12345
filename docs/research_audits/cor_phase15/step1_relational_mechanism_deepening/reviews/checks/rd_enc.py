import os, sys, json
B0 = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, B0 + "/src")
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import fwd3 as F3, ckpt as C, gen_family as G, analysis as A, bayes as B
ck = sys.argv[1]; NE = 1000; Q, T, K = 6, 43, 3
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(B0, ck))); p.pop("WX", None)
env = B.Env("T4", K=3, Q=6); f4 = G.Fam(); d = p["Eread"].shape[0]
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n == "resid_pre2": rec["r"] = a
        return a
    o = F3.forward(p, X, hk); return rec["r"], jax.nn.softmax(o["answer"], -1)
@jax.jit
def added(X, delta):
    return jax.nn.softmax(F3.forward(p, X, lambda n, a: a + delta if n == "resid_pre2" else a)["answer"], -1)
def loo(Z): Ag = np.stack([(Z[:, :, a] == Z[:, :, b]) for a, b in G.GROUPS], -1).astype(int); return Ag.sum(1, keepdims=True) - Ag
def oh(n): return np.eye(6)[n].reshape(n.shape[:-1] + (18,))
def fit(F, Aa): return np.linalg.lstsq(np.c_[np.ones(len(F)), F], Aa, rcond=None)[0][1:]
Dn = G.sample(f4, 3000, np.random.default_rng(9_910_000))["X"]; an = np.asarray(acts(jnp.asarray(Dn))[0]); Fn = oh(loo(Dn))
Wsh = fit(Fn.reshape(-1, 18), an[:, 36:42].reshape(-1, d)); Wq = [fit(Fn[:, q], an[:, 36 + q]) for q in range(Q)]
Wsl = [fit(Fn[:, t // 6], an[:, t]) for t in range(36)]
D = G.sample(f4, NE, np.random.default_rng(9_900_000)); X = D["X"]; a0, P0 = map(np.asarray, acts(jnp.asarray(X))); res = {}
for q in range(Q):
    Xp = A.pairs(env, X, q, "cond", np.random.default_rng(9_900_100 + q)); ap, pp = map(np.asarray, acts(jnp.asarray(Xp)))
    inf = P0[:, q].argmax(-1) != pp[:, q].argmax(-1); F0, F1 = oh(loo(X)), oh(loo(Xp)); dd = {}
    for nm, W in (("readout_shared", Wsh), ("readout_perq", Wq[q])):
        dl = np.zeros((len(X), T, d), np.float32); dl[:, 36 + q] = (F1[:, q] - F0[:, q]) @ W; dd[nm] = dl
    dl = np.zeros((len(X), T, d), np.float32)
    for t in range(36): dl[:, t] = (F1[:, t // 6] - F0[:, t // 6]) @ Wsl[t]
    dd["source_perslot"] = dl
    for nm, dl in dd.items():
        o = np.asarray(added(jnp.asarray(X), jnp.asarray(dl))); res.setdefault(nm, []).append((o[:, q].argmax(-1) == pp[:, q].argmax(-1))[inf])
print(ck, {k: round(float(np.concatenate(v).mean()), 3) for k, v in res.items()})
