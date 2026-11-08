"""POST HOC diagnostic (not a criterion): per-layer reliability routing. For every layer and head: group-mass R^2
(readout q -> groups of question q vs Bayes P(group reliable)); per-layer all-head readout-row pattern patch (cond);
full-residual patch of question q's tokens at every site. Stream base 5,500,000.
Usage: posthoc_layers.py T4_CKPT OUT.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import bayes as B, ckpt as C, analysis as A
ck, outp = sys.argv[1:3]
p = C.load(ck); env = B.Env("T4", K=3, Q=6); Q = env.Q; L = len(p["layers"]); T = A.T_of(env); d = p["Eread"].shape[0]
H = p["layers"][0]["WQ"].shape[0]; Rs = env.R_values
D = B.sample_fast(env, 1500, np.random.default_rng(5_500_000)); X = D["X"]
names = [f"pattern{l}" for l in range(L)] + A.sites(p)
cb, ob = A.acts(p, X, names); pb = A.probs(ob); post = B.episode_posterior(env, X)
prel = np.stack([sum(post["post_R"][:, i] for i, R in enumerate(Rs) if R[0] == g) for g in range(3)], -1)
def gm(pat, q):
    m = np.stack([pat[:, :, A.rd_pos(env, q), [6 * q + a for a in G]].sum(-1) for G in B.GROUPS], -1); return m / np.maximum(m.sum(-1, keepdims=True), 1e-9)
res = {"note": "POST HOC"}
src = {}
for q in range(Q):
    Xp = A.pairs(env, X, q, "cond", np.random.default_rng(5_500_100 + q)); cp, op = A.acts(p, Xp, names); src[q] = (cp, A.probs(op))
for l in range(L):
    res[f"layer{l}_group_mass_r2"] = [float(np.mean([np.corrcoef(gm(cb[f"pattern{l}"], q)[:, h].ravel(), prel.ravel())[0, 1] ** 2 for q in range(Q)])) for h in range(H)]
    PE, PEp, PP = [], [], []
    for q in range(Q):
        qm = np.zeros(T); qm[A.rd_pos(env, q)] = 1
        pt = A.patch_pattern(p, X, f"pattern{l}", qm, np.ones(H), src[q][0][f"pattern{l}"]); PE.append(pb[:, q]); PEp.append(src[q][1][:, q]); PP.append(pt[:, q])
    m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); res[f"layer{l}_readout_pattern_patch"] = dict(iia=m["iia"], kl=m["kl_recovered"])
for s in A.sites(p):
    for loc in ["q_tokens", "readout", "src_q"]:
        PE, PEp, PP = [], [], []
        for q in range(Q):
            m_ = np.zeros(T)
            if loc in ("q_tokens", "src_q"): m_[6 * q:6 * q + 6] = 1
            if loc in ("q_tokens", "readout"): m_[A.rd_pos(env, q)] = 1
            pt = A.patch(p, X, s, m_, np.eye(d), src[q][0][s]); PE.append(pb[:, q]); PEp.append(src[q][1][:, q]); PP.append(pt[:, q])
        m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); res[f"{s}|{loc}|full"] = dict(iia=m["iia"], kl=m["kl_recovered"])
json.dump(res, open(outp, "w"), indent=1)
for k, v in res.items(): print(k, v if not isinstance(v, list) else [round(x, 2) for x in v])
