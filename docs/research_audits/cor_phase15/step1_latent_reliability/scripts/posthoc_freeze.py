"""POST HOC: joint key+query patch (cond) with an arbitrary set of last-layer heads frozen to base patterns.
Same identification as replicate_t4.py (identification stream 2,100,000); evaluation on a fresh stream 5,700,000.
Usage: posthoc_freeze.py T4_CKPT OUT.json HEADSETS (e.g. "0,2,3;1;none")"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import bayes as B, nets as N, ckpt as C, analysis as A
ck, outp, sets = sys.argv[1:4]
p = C.load(ck); env = B.Env("T4", K=3, Q=6); Q = env.Q; L = len(p["layers"]); d = p["Eread"].shape[0]; T = A.T_of(env)
site, Lp = f"resid_pre{L-1}", f"pattern{L-1}"; H = p["layers"][0]["WQ"].shape[0]
Xi = B.sample_fast(env, 3000, np.random.default_rng(2_100_000))["X"][:1000]; ai, _ = A.acts(p, Xi, [site]); dk, dq = [], []
for q in range(Q):
    Xp = A.pairs(env, Xi, q, "cond", np.random.default_rng(2_100_000 + 10 + q)); ap, _ = A.acts(p, Xp, [site]); D = ap[site] - ai[site]
    dk.append(D[:, 6*q:6*q+6].reshape(-1, d)); dq.append(D[:, A.rd_pos(env, q)])
Pk, Pq = A.proj(A.contrast_basis(np.concatenate(dk), 2)), A.proj(A.contrast_basis(np.concatenate(dq), 5))
@partial(jax.jit, static_argnames=("site", "psite"))
def run(p, X, site, psite, mk, mq, Pk, Pq, src, pat, hm):
    def hk(n, a):
        if n == site: dl = src - a; a = a + (dl @ Pk) * mk[None, :, None] + (dl @ Pq) * mq[None, :, None]
        if n == psite: a = a * (1 - hm[None, :, None, None]) + pat * hm[None, :, None, None]
        return a
    return jax.nn.softmax(N.forward(p, X, hk)["answer"], -1)
Xb = B.sample_fast(env, 1500, np.random.default_rng(5_700_000))["X"]; cb, ob = A.acts(p, Xb, [site, Lp]); pb = A.probs(ob)
srcs = {}
for q in range(Q):
    Xp = A.pairs(env, Xb, q, "cond", np.random.default_rng(5_700_100 + q)); cp, op = A.acts(p, Xp, [site]); srcs[q] = (cp[site], A.probs(op))
res = {"note": "POST HOC"}
for hs in sets.split(";"):
    hm = np.zeros(H)
    if hs != "none": hm[[int(h) for h in hs.split(",")]] = 1
    PE, PEp, PP = [], [], []
    for q in range(Q):
        mk = np.zeros(T); mk[6*q:6*q+6] = 1; mq = np.zeros(T); mq[A.rd_pos(env, q)] = 1
        o = [np.asarray(run(p, jnp.asarray(Xb[i:i+A.CH]), site, Lp, *(jnp.asarray(z, jnp.float32) for z in (mk, mq, Pk, Pq)), jnp.asarray(srcs[q][0][i:i+A.CH]),
                            jnp.asarray(cb[Lp][i:i+A.CH]), jnp.asarray(hm, jnp.float32))) for i in range(0, len(Xb), A.CH)]
        PE.append(pb[:, q]); PEp.append(srcs[q][1][:, q]); PP.append(np.concatenate(o)[:, q])
    m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); res[hs] = dict(iia=m["iia"], kl=m["kl_recovered"])
json.dump(res, open(outp, "w"), indent=1); print(ck.split('/')[-1], {k: (round(v['iia'], 3), round(v['kl'], 3)) for k, v in res.items() if k != 'note'})
