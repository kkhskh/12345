"""Is the last-layer attention output at readout q a *relation* site or a *candidate-evidence/answer* site?
Patch it (full, and the 5-dim cond-contrast subspace) from cond pairs (relation changes, X_q fixed), countpres pairs
(relation fixed, aggregates change), and target pairs (X_q values shifted, relation fixed).  A pure reliability variable
should transfer cond and NOT target; an evidence/answer site transfers both.  Also the last-layer readout-row attention
PATTERN patched from the same pairs (the routing operation).  Streams 7,300,000.  Usage: site_specificity.py CKPT OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
ck, outp = sys.argv[1:3]
p = jax.tree_util.tree_map(jnp.asarray, C.load(ck)); env = B.Env("T4", K=3, Q=6); Q, K, T = 6, 3, 43
L = len(p["layers"]) - 1; Lh, Lp = f"head_out{L}", f"pattern{L}"; H = p["layers"][0]["WQ"].shape[0]; d = p["Eread"].shape[0]


@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n in (Lh, Lp): rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec, jax.nn.softmax(o["answer"], -1)


@jax.jit
def patched(X, ho_src, row, P, pat_src, prow):
    def hk(n, a):
        if n == Lh: a = a + ((ho_src - a) @ P) * row[None, None, :, None]
        if n == Lp: a = a * (1 - prow[None, None, :, None]) + pat_src * prow[None, None, :, None]
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)


def countpres(X, q, rng):
    Xp = X.copy(); n = len(X)
    for k in range(Q):
        if k == q: continue
        for a, b in G.GROUPS:
            ag = X[:, k, a] == X[:, k, b]; v1 = rng.integers(0, K, n); off = rng.integers(1, K, n); Xp[:, k, a], Xp[:, k, b] = v1, np.where(ag, v1, (v1 + off) % K)
    return Xp


f4 = G.Fam(); Xi = G.sample(f4, 800, np.random.default_rng(7_300_000))["X"]; Xb = G.sample(f4, 1000, np.random.default_rng(7_300_001))["X"]
ri, _ = acts(jnp.asarray(Xi)); dh = []
for q in range(Q):
    Xp = A.pairs(env, Xi, q, "cond", np.random.default_rng(7_300_010 + q)); rp, _ = acts(jnp.asarray(Xp))
    dh.append(np.asarray((rp[Lh] - ri[Lh])[:, :, 6 * Q + q].sum(1)))
P5 = A.proj(A.contrast_basis(np.concatenate(dh), 5)); I = np.eye(d); zr = np.zeros(T)
rb, pb = acts(jnp.asarray(Xb)); pb = np.asarray(pb)
res = dict(ckpt=ck)
for kind in ["cond", "countpres", "target"]:
    for nm, P_, use_pat in [("attn_out_full", I, False), ("attn_out_5d", P5, False), ("pattern_row", np.zeros((d, d)), True)]:
        PE, PEp, PP = [], [], []
        for q in range(Q):
            rng = np.random.default_rng(7_300_100 + 10 * ["cond", "countpres", "target"].index(kind) + q)
            Xp = countpres(Xb, q, rng) if kind == "countpres" else A.pairs(env, Xb, q, kind, rng)
            rp, pp = acts(jnp.asarray(Xp)); row = zr.copy(); row[6 * Q + q] = 1
            pt = patched(jnp.asarray(Xb), rp[Lh], jnp.asarray(row * (not use_pat), jnp.float32), jnp.asarray(P_, jnp.float32), rp[Lp], jnp.asarray(row * use_pat, jnp.float32))
            PE.append(pb[:, q]); PEp.append(np.asarray(pp)[:, q]); PP.append(np.asarray(pt)[:, q])
        m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP)))
        res[f"{kind}|{nm}"] = dict(iia=m["iia"], kl=m["kl_recovered"], change_rate=m["change_rate"], n_inf=m["n_inf"])
json.dump(res, open(outp, "w"), indent=1)
print(ck.split("/")[-1], {k: (round(v["iia"], 3), round(v["change_rate"], 3)) for k, v in res.items() if k != "ckpt"})
