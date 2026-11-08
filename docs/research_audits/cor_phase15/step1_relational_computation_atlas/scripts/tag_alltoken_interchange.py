"""POST HOC test of the unifying hypothesis suggested by OP-5: the 2-D source-token reliability subspace (identified at
question q's tokens) is the common variable; implementations differ in WHERE the last layer reads it.  Interchange the
subspace from cond pairs at (a) q's 6 source tokens only, (b) the 30 source tokens of the OTHER questions only, (c) all 36.
Streams 7,600,000.  Usage: tag_alltoken_interchange.py CKPT OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
ck, outp = sys.argv[1:3]
p = jax.tree_util.tree_map(jnp.asarray, C.load(ck)); env = B.Env("T4", K=3, Q=6); Q, T = 6, 43
L = len(p["layers"]) - 1; site = f"resid_pre{L}"; d = p["Eread"].shape[0]; f4 = G.Fam()
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n == site: rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec[site], jax.nn.softmax(o["answer"], -1)
@jax.jit
def patch(X, src, P, mask):
    def hk(n, a):
        if n == site: a = a + ((src - a) @ P) * mask[None, :, None]
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ai, _ = acts(jnp.asarray(Di)); ai = np.asarray(ai); dk = []
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); ap, _ = acts(jnp.asarray(Xp)); dk.append((np.asarray(ap) - ai)[:, 6 * q:6 * q + 6].reshape(-1, d))
P = A.proj(A.contrast_basis(np.concatenate(dk), 2)); Pr = A.proj(A.random_basis(d, 2, np.random.default_rng(5)))
X = G.sample(f4, 1200, np.random.default_rng(7_600_000))["X"]; _, pb = acts(jnp.asarray(X)); pb = np.asarray(pb)
res = dict(ckpt=ck)
for nm, P_ in [("tag", P), ("random2d", Pr)]:
    for loc in ["own_q", "other_q", "all"]:
        PE, PEp, PP = [], [], []
        for q in range(Q):
            Xp = A.pairs(env, X, q, "cond", np.random.default_rng(7_600_100 + q)); ap, pp = acts(jnp.asarray(Xp))
            m = np.zeros(T)
            if loc in ("own_q", "all"): m[6 * q:6 * q + 6] = 1
            if loc in ("other_q", "all"):
                for k in range(Q):
                    if k != q: m[6 * k:6 * k + 6] = 1
            o = patch(jnp.asarray(X), ap, jnp.asarray(P_, jnp.float32), jnp.asarray(m, jnp.float32))
            PE.append(pb[:, q]); PEp.append(np.asarray(pp)[:, q]); PP.append(np.asarray(o)[:, q])
        mm = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); res[f"{nm}|{loc}"] = dict(iia=mm["iia"], kl=mm["kl_recovered"])
json.dump(res, open(outp, "w"), indent=1); print(ck.split("/")[-1], {k: (round(v["iia"], 3), round(v["kl"], 3)) for k, v in res.items() if k != "ckpt"})
