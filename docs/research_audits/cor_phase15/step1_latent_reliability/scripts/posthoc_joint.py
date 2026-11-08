"""POST HOC: query-side vs key-side reliability at the input of the last layer.
Patches (cond pairs) at resid_pre{L-1}: readout-token contrast subspace (k=|R|-1, query side), source-token tag
contrast direction (k_tag, key side), and both jointly. Usage: posthoc_joint.py CKPT TASK OUT.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import bayes as B, nets as N, ckpt as C, analysis as A

ck, task, outp = sys.argv[1:4]
p = C.load(ck); env = B.Env(task, K=3, Q=6); Q = env.Q; L = len(p["layers"]); d = p["Eread"].shape[0]; T = A.T_of(env)
site = f"resid_pre{L - 1}"; k_r = len(env.R_values) - 1; k_t = 2 if task == "T4" else 1
DB = 999_000 + 80_000
Df = B.sample_fast(env, 1200, np.random.default_rng(DB))
dr, dt = [], []
for q in range(Q):
    Xp = A.pairs(env, Df["X"], q, "cond", np.random.default_rng(DB + 10 + q))
    a1, _ = A.acts(p, Df["X"], [site]); a2, _ = A.acts(p, Xp, [site]); D = a2[site] - a1[site]
    dr.append(D[:, A.rd_pos(env, q)]); dt.append(D[:, 6 * q:6 * q + 6].reshape(-1, d))
Ur = A.contrast_basis(np.concatenate(dr), k_r); Ut = A.contrast_basis(np.concatenate(dt), k_t)


@partial(jax.jit, static_argnames=("site",))
def two(p, X, site, m1, P1, m2, P2, src):
    def hk(n, a):
        if n == site:
            dl = src - a
            a = a + (dl @ P1) * m1[None, :, None] + (dl @ P2) * m2[None, :, None]
        return a
    return jax.nn.softmax(N.forward(p, X, hk)["answer"], -1)


CB = 777_000 + 80_000
Db = B.sample_fast(env, 1500, np.random.default_rng(CB)); Xb = Db["X"]
_, ob = A.acts(p, Xb, [site]); pb = A.probs(ob)
zero = np.zeros((d, d)); res = dict(ckpt=ck, task=task, site=site, note="POST HOC")
cfg = {"query_readout_D2": (Ur, None), "key_tag_D2": (None, Ut), "joint": (Ur, Ut), "joint_full_q_tokens": ("I", "I")}
for nm, (U1, U2) in cfg.items():
    PE, PEp, PP = [], [], []
    for q in range(Q):
        Xp = A.pairs(env, Xb, q, "cond", np.random.default_rng(CB + 10 + q)); cp, op = A.acts(p, Xp, [site])
        m1 = np.zeros(T); m1[A.rd_pos(env, q)] = 1; m2 = np.zeros(T); m2[6 * q:6 * q + 6] = 1
        P1 = zero if U1 is None else (np.eye(d) if isinstance(U1, str) else A.proj(U1))
        P2 = zero if U2 is None else (np.eye(d) if isinstance(U2, str) else A.proj(U2))
        o = [np.asarray(two(p, jnp.asarray(Xb[i:i + A.CH]), site, jnp.asarray(m1, jnp.float32), jnp.asarray(P1, jnp.float32),
                            jnp.asarray(m2, jnp.float32), jnp.asarray(P2, jnp.float32), jnp.asarray(cp[site][i:i + A.CH]))) for i in range(0, len(Xb), A.CH)]
        PE.append(pb[:, q]); PEp.append(A.probs(op)[:, q]); PP.append(np.concatenate(o)[:, q])
    res[nm] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
    # target specificity
    PE, PEp, PP = [], [], []
    for q in range(Q):
        Xp = A.pairs(env, Xb, q, "target", np.random.default_rng(CB + 30 + q)); cp, op = A.acts(p, Xp, [site])
        m1 = np.zeros(T); m1[A.rd_pos(env, q)] = 1; m2 = np.zeros(T); m2[6 * q:6 * q + 6] = 1
        P1 = zero if U1 is None else (np.eye(d) if isinstance(U1, str) else A.proj(U1))
        P2 = zero if U2 is None else (np.eye(d) if isinstance(U2, str) else A.proj(U2))
        o = [np.asarray(two(p, jnp.asarray(Xb[i:i + A.CH]), site, jnp.asarray(m1, jnp.float32), jnp.asarray(P1, jnp.float32),
                            jnp.asarray(m2, jnp.float32), jnp.asarray(P2, jnp.float32), jnp.asarray(cp[site][i:i + A.CH]))) for i in range(0, len(Xb), A.CH)]
        PE.append(pb[:, q]); PEp.append(A.probs(op)[:, q]); PP.append(np.concatenate(o)[:, q])
    res[nm + "_target_change"] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))["change_rate"]
json.dump(res, open(outp, "w"), indent=1, default=float)
print(task, ck.split("/")[-1], {k: (round(v["iia"], 2), round(v["kl_recovered"], 2)) if isinstance(v, dict) else round(v, 3) for k, v in res.items() if k not in ("ckpt", "task", "site", "note")})
