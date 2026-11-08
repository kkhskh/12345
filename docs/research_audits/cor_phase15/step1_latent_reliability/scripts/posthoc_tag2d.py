"""POST HOC: k-dim tag (U_D2) vs agreement count and Bayes roles; log-attention of route heads on all tag dims.
Usage: posthoc_tag2d.py CKPT TASK TAG_DEV.json ROUTE_HEADS(comma) OUT.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import bayes as B, ckpt as C, analysis as A
ck, task, tj, heads, outp = sys.argv[1:6]; heads = [int(h) for h in heads.split(",")]
sel = json.load(open(tj)); s = sel["s_tag"]; k = sel["k_tag"]
p = C.load(ck); env = B.Env(task, K=3, Q=6); Q = env.Q; d = p["Eread"].shape[0]; L = len(p["layers"]); Lp = f"pattern{L-1}"
DB = 1_049_000; Df = B.sample_fast(env, 1000, np.random.default_rng(DB)); diffs = []
Df = B.sample_fast(env, 3000, np.random.default_rng(DB))
for q in range(Q):
    Xp = A.pairs(env, Df["X"][:1000], q, "cond", np.random.default_rng(DB + 60 + q))
    a1, _ = A.acts(p, Df["X"][:1000], [s]); a2, _ = A.acts(p, Xp, [s]); diffs.append((a2[s][:, 6*q:6*q+6] - a1[s][:, 6*q:6*q+6]).reshape(-1, d))
U = A.contrast_basis(np.concatenate(diffs), k)
D = B.sample_fast(env, 3000, np.random.default_rng(565_000)); c, o = A.acts(p, D["X"], [s, Lp]); post = B.episode_posterior(env, D["X"])
res = {"note": "POST HOC"}
coords = np.stack([c[s][:, 6*q:6*q+6] @ U for q in range(Q)], 1)            # N,Q,6,k
g_of = np.array([0, 0, 1, 1, 2, 2])
cnt = np.stack([A.count_feats(env, D["X"], q) for q in range(Q)], 1)[:, :, g_of]   # N,Q,6
res["tag_mean_by_count"] = {int(n): coords[cnt == n].mean(0).tolist() for n in range(Q)}
Rs = env.R_values
prel = np.stack([np.stack([sum(post["pi_minus"][:, q, i] for i, R in enumerate(Rs) if R[0] == g) for g in range(3)], -1) for q in range(Q)], 1)[:, :, g_of]
res["bayes_P_rel_by_count"] = {int(n): float(prel[cnt == n].mean()) for n in range(Q)}
# log attention of route heads on the k tag dims (within question, centred), and on the count alone
for h in heads:
    la = np.stack([np.log(np.maximum(o_, 1e-9)) for o_ in [c[Lp][:, h, A.rd_pos(env, q), 6*q:6*q+6] for q in range(Q)]], 1)   # N,Q,6
    y = (la - la.mean(-1, keepdims=True)).reshape(-1)
    Xc = (coords - coords.mean(2, keepdims=True)).reshape(-1, k)
    W, mu, tm = A.ridge(Xc, y[:, None], lam=1e-6); r_tag = A.r2(Xc, y[:, None], W, mu, tm)
    cc = (cnt - cnt.mean(-1, keepdims=True)).reshape(-1, 1).astype(float)
    W2, mu2, tm2 = A.ridge(cc, y[:, None], lam=1e-6); r_cnt = A.r2(cc, y[:, None], W2, mu2, tm2)
    lp = np.log(np.clip(prel, 1e-6, 1)); lp = (lp - lp.mean(-1, keepdims=True)).reshape(-1, 1)
    W3, mu3, tm3 = A.ridge(lp, y[:, None], lam=1e-6); r_b = A.r2(lp, y[:, None], W3, mu3, tm3)
    res[f"head{h}_logattn_R2"] = dict(tag_kdim=r_tag, count_linear=r_cnt, bayes_log_prel=r_b)
json.dump(res, open(outp, "w"), indent=1); print(json.dumps(res, indent=0)[:2000])
