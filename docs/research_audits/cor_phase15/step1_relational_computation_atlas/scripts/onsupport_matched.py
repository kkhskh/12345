"""On-support matched-aggregate relation intervention for T4 (Direction G).
'pairswap': in every other question k != q, exchange the two-source report pairs of the TRUE reliable group and the TRUE
noise group (generator knowledge; colluder untouched).  Every per-question histogram is unchanged, every hybrid is on the
generator's support (both roles give positive probability to any pair), but which group agrees in which question changes.
Reports: support check; model vs Bayes argmax change rate at q; joint key+query tag patch IIA on these pairs (mediation),
and the readout-row pattern patch.  Stream 7,400,000.  Usage: onsupport_matched.py CKPT OUT.json [ds]"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, nets as N, ckpt as C, analysis as A, bayes as B
ck, outp = sys.argv[1:3]; ds = len(sys.argv) > 3
f4 = G.Fam(); Q, K, T = 6, 3, 43; env = B.Env("T4", K=3, Q=6)
D = G.sample(f4, 1500, np.random.default_rng(7_400_000)); X, cfg = D["X"], D["cfg"]
cf = f4.configs
rel = np.array([cf[c][0] for c in cfg]); col = np.array([cf[c][1] for c in cfg]); noi = 3 - rel - col


def pairswap(X, q):
    Xp = X.copy(); n = np.arange(len(X))
    for k in range(Q):
        if k == q: continue
        for s in range(2):
            a = 2 * rel + s; b = 2 * noi + s
            va, vb = X[n, k, a].copy(), X[n, k, b].copy(); Xp[n, k, a], Xp[n, k, b] = vb, va
    return Xp


if ds:
    p = jax.tree_util.tree_map(jnp.asarray, C.load_base(ck)); fb = N.make_forward("deepsets", 3); run = jax.jit(lambda Z: jax.nn.softmax(fb(p, Z)["answer"], -1))
else:
    p = jax.tree_util.tree_map(jnp.asarray, C.load(ck)); run = jax.jit(lambda Z: jax.nn.softmax(F.forward(p, Z)["answer"], -1))
pb = np.asarray(run(jnp.asarray(X))); bb = G.posterior(f4, X)["post_y"]
res = dict(ckpt=ck)
chm, chb, und, pairs = [], [], 0, {}
for q in range(Q):
    Xp = pairswap(X, q); pairs[q] = Xp
    pp = np.asarray(run(jnp.asarray(Xp))); bp = G.posterior(f4, Xp)["post_y"]
    ok = np.isfinite(bp[:, q]).all(-1); und += int((~ok).sum())
    chm.append(pp[:, q].argmax(-1) != pb[:, q].argmax(-1)); chb.append(np.nan_to_num(bp[:, q], nan=0).argmax(-1) != bb[:, q].argmax(-1))
res["bayes_undefined_frac"] = und / (Q * len(X))
res["model_change_rate"] = float(np.mean(np.concatenate(chm))); res["bayes_change_rate"] = float(np.mean(np.concatenate(chb)))
res["model_bayes_change_agreement"] = float((np.concatenate(chm) == np.concatenate(chb)).mean())
if not ds:
    L = len(p["layers"]) - 1; site, Lp = f"resid_pre{L}", f"pattern{L}"; d = p["Eread"].shape[0]; H = p["layers"][0]["WQ"].shape[0]

    @jax.jit
    def acts(Z):
        rec = {}
        def hk(n, a):
            if n in (site, Lp): rec[n] = a
            return a
        o = F.forward(p, Z, hk); return rec, jax.nn.softmax(o["answer"], -1)

    @jax.jit
    def pt(Z, src, mk, Pk, mq, Pq, pat, prow):
        def hk(n, a):
            if n == site: dl = src - a; a = a + (dl @ Pk) * mk[None, :, None] + (dl @ Pq) * mq[None, :, None]
            if n == Lp: a = a * (1 - prow[None, None, :, None]) + pat * prow[None, None, :, None]
            return a
        return jax.nn.softmax(F.forward(p, Z, hk)["answer"], -1)
    Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ri, _ = acts(jnp.asarray(Di)); dk, dq = [], []
    for q in range(Q):
        Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); rp, _ = acts(jnp.asarray(Xp)); Dd = np.asarray(rp[site] - ri[site])
        dk.append(Dd[:, 6 * q:6 * q + 6].reshape(-1, d)); dq.append(Dd[:, 6 * Q + q])
    Pk = A.proj(A.contrast_basis(np.concatenate(dk), 2)); Pq = A.proj(A.contrast_basis(np.concatenate(dq), 5)); Z = np.zeros((d, d))
    rb, _ = acts(jnp.asarray(X))
    for nm, (PK, PQ, usepat) in {"joint_tag": (Pk, Pq, False), "key_only": (Pk, Z, False), "pattern_row": (Z, Z, True)}.items():
        PE, PEp, PP = [], [], []
        for q in range(Q):
            rp, pp = acts(jnp.asarray(pairs[q])); mk = np.zeros(T); mk[6 * q:6 * q + 6] = 1; mq = np.zeros(T); mq[6 * Q + q] = 1
            o = pt(jnp.asarray(X), rp[site], jnp.asarray(mk * (not usepat), jnp.float32), jnp.asarray(PK, jnp.float32), jnp.asarray(mq * (not usepat), jnp.float32),
                   jnp.asarray(PQ, jnp.float32), rp[Lp], jnp.asarray(mq * usepat, jnp.float32))
            PE.append(pb[:, q]); PEp.append(np.asarray(pp)[:, q]); PP.append(np.asarray(o)[:, q])
        m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); res[nm] = dict(iia=m["iia"], kl=m["kl_recovered"], n_inf=m["n_inf"])
json.dump(res, open(outp, "w"), indent=1); print(ck.split("/")[-1], {k: v for k, v in res.items() if k != "ckpt"})
