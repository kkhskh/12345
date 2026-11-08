"""Method calibration on DeepSets (explicit shared bottleneck `pool`). Usage: analyze_deepsets.py CKPT TASK SPLIT OUT.json
Probe the pool for P(R|X); patch pool (full, probe subspace, contrast subspace, random) from cond / perm / target pairs."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import bayes as B, nets as N, ckpt as C, analysis as A

ck, task, split, outp = sys.argv[1:5]
BASE = {"dev": 999_000, "conf": 777_000}[split] + 70_000
p = C.load(ck); env = B.Env(task, K=3, Q=6); Q, K = env.Q, env.K
nR = len(env.R_values); k = nR - 1
fwd = N.make_forward("deepsets", K)


@partial(jax.jit)
def run(p, X):
    rec = {}
    def hk(n, a):
        if n == "pool": rec[n] = a
        return a
    o = fwd(p, X, hk); return rec["pool"][:, 0], jax.nn.softmax(o["answer"], -1)


@partial(jax.jit)
def run_patch(p, X, P, srcm):
    def hk(n, a):
        if n == "pool": return a + ((srcm[:, None] - a) @ P)
        return a
    return jax.nn.softmax(fwd(p, X, hk)["answer"], -1)


def data(off, n): return B.sample_fast(env, n, np.random.default_rng(BASE + off))
Df, Dt, Dp = data(0, 4000), data(1, 4000), data(2, 1800)
mf, _ = run(p, jnp.asarray(Df["X"])); mt, _ = run(p, jnp.asarray(Dt["X"])); mf, mt = np.asarray(mf), np.asarray(mt)
pf, pt = B.episode_posterior(env, Df["X"]), B.episode_posterior(env, Dt["X"])
W, mu, tm = A.ridge(mf, A.logit_target(pf["post_R"])); res = dict(ckpt=ck, task=task, split=split, k=k)
res["probe_r2_post_R"] = A.r2(mt, A.logit_target(pt["post_R"]), W, mu, tm)
U1 = A.basis_from_W(W, k)
mb, pb = run(p, jnp.asarray(Dp["X"])); mb, pb = np.asarray(mb), np.asarray(pb)
d = mb.shape[1]
# contrast basis from fit-set cond pairs
diffs = []
for q in range(Q):
    Xp = A.pairs(env, Df["X"][:1000], q, "cond", np.random.default_rng(BASE + 60 + q))
    a2, _ = run(p, jnp.asarray(Xp)); diffs.append(np.asarray(a2) - mf[:1000])
U2 = A.contrast_basis(np.concatenate(diffs), k)
res["overlap"] = float(np.linalg.svd(U1.T @ U2)[1].mean())
bases = {"probe": U1, "contrast": U2, "full": None}
for i in range(5): bases[f"rand{i}"] = A.random_basis(d, k, np.random.default_rng(BASE + 900 + i))
out = {}
for kind in ["cond", "perm", "target"]:
    srcs = []
    for q in range(Q):
        Xp = A.pairs(env, Dp["X"], q, kind, np.random.default_rng(BASE + 100 * (["cond", "perm", "target"].index(kind) + 1) + q))
        ms, ps = run(p, jnp.asarray(Xp)); srcs.append((np.asarray(ms), np.asarray(ps)))
    for nm, U in bases.items():
        P = np.eye(d) if U is None else A.proj(U)
        PE, PEp, PP = [], [], []
        for q in range(Q):
            ms, ps = srcs[q]; pp = np.asarray(run_patch(p, jnp.asarray(Dp["X"]), jnp.asarray(P, jnp.float32), jnp.asarray(ms)))
            PE.append(pb[:, q]); PEp.append(ps[:, q]); PP.append(pp[:, q])
        out[f"{nm}|{kind}"] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
res["tests"] = out
res["random_mean_cond_iia"] = float(np.mean([out[f"rand{i}|cond"]["iia"] for i in range(5)]))
json.dump(res, open(outp, "w"), indent=1, default=float); print("done", outp)
