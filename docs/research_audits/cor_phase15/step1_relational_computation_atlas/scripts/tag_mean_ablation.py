"""OP-5 (exploratory): natural necessity of the 2-D key tag without injecting wrong values.
At the input of the last layer, replace the tag-subspace component of every source token by (a) its dataset mean
(mean ablation) or (b) the mean conditional on the token's own reported value and group (conditional-mean ablation:
keeps value/identity information that lives in that subspace, removes the across-episode reliability variation), and
for comparison (c) resample ablation (another episode's tag).  Also a random 2-D subspace control.
Accuracy vs clean, own-question Bayes and Bayes.  Streams 7,500,000.  Usage: tag_mean_ablation.py CKPT OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
ck, outp = sys.argv[1:3]
p = jax.tree_util.tree_map(jnp.asarray, C.load(ck)); env = B.Env("T4", K=3, Q=6); Q, K, T = 6, 3, 43
L = len(p["layers"]) - 1; site = f"resid_pre{L}"; d = p["Eread"].shape[0]; f4 = G.Fam()


@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n == site: rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec[site], jax.nn.softmax(o["answer"], -1)


@jax.jit
def setsub(X, target, P, mask):
    def hk(n, a):
        if n == site: a = a + ((target - a) @ P) * mask[None, :, None]
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)


Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ai, _ = acts(jnp.asarray(Di)); ai = np.asarray(ai); dk = []
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); ap, _ = acts(jnp.asarray(Xp)); dk.append((np.asarray(ap) - ai)[:, 6 * q:6 * q + 6].reshape(-1, d))
U = A.contrast_basis(np.concatenate(dk), 2); P = A.proj(U)
Ur = A.random_basis(d, 2, np.random.default_rng(3)); Pr = A.proj(Ur)
D = G.sample(f4, 2000, np.random.default_rng(7_500_000)); X, Y = D["X"], D["Y"]
a, pb = acts(jnp.asarray(X)); a = np.asarray(a); pb = np.asarray(pb)
post = G.posterior(f4, X); res = dict(ckpt=ck, acc_clean=float((pb.argmax(-1) == Y).mean()), own=float((post["own_y"].argmax(-1) == Y).mean()),
                                      bayes=float((post["post_y"].argmax(-1) == Y).mean()))
mask = np.zeros(T); mask[:6 * Q] = 1
src_tok = a[:, :6 * Q]                                                    # N,36,d
mean_all = src_tok.reshape(-1, d).mean(0)
# conditional mean by (source position s, reported value): keeps value and source-identity content
vals = X.reshape(len(X), -1); pos = np.tile(np.arange(6), Q)
cond = np.zeros_like(src_tok)
for s in range(6):
    for v in range(K):
        sel = (pos[None, :] == s) & (vals == v); m = src_tok[sel].mean(0); cond[sel] = m
Dind = G.sample(f4, 2000, np.random.default_rng(7_500_001))["X"]; ar, _ = acts(jnp.asarray(Dind)); ar = np.asarray(ar)
def run(target_tok, P_):
    tgt = a.copy(); tgt[:, :6 * Q] = target_tok
    return float((np.asarray(setsub(jnp.asarray(X), jnp.asarray(tgt), jnp.asarray(P_, jnp.float32), jnp.asarray(mask, jnp.float32))).argmax(-1) == Y).mean())
res["mean_ablation"] = run(np.broadcast_to(mean_all, src_tok.shape), P)
res["conditional_mean_ablation"] = run(cond, P)
res["resample_ablation"] = run(ar[:, :6 * Q], P)
res["random_subspace_mean_ablation"] = run(np.broadcast_to(mean_all, src_tok.shape), Pr)
res["random_subspace_conditional_mean"] = run(cond, Pr)
gap = res["acc_clean"] - res["own"]
res["drop_frac"] = {k: (res["acc_clean"] - res[k]) / gap for k in ["mean_ablation", "conditional_mean_ablation", "resample_ablation", "random_subspace_mean_ablation", "random_subspace_conditional_mean"]}
json.dump(res, open(outp, "w"), indent=1); print(ck.split("/")[-1], {k: (round(v, 3) if isinstance(v, float) else {a_: round(b, 2) for a_, b in v.items()}) for k, v in res.items() if k != "ckpt"})
