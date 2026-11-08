"""POST HOC (exploratory, not preregistered): attacks on the tag-and-route mechanism (mandate §12) and the quantitative
check of the routing equations used in THEOREM.md. Usage: posthoc.py CKPT TASK TAG_CONF.json OUT.json
Uses the s_tag chosen on the dev split. Fresh data stream base 555000."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import bayes as B, nets as N, ckpt as C, analysis as A

ck, task, tagj, outp = sys.argv[1:5]
sel = json.load(open(tagj)); s_tag = sel["s_tag"]; ktag = sel["k_tag"]
BASE = 555_000
p = C.load(ck); env = B.Env(task, K=3, Q=6); Q, K = env.Q, env.K
L = len(p["layers"]); d = p["Eread"].shape[0]; T = A.T_of(env); Lp = f"pattern{L - 1}"
res = dict(ckpt=ck, task=task, s_tag=s_tag, note="POST HOC, exploratory")
def data(off, n, e=env): return B.sample_fast(e, n, np.random.default_rng(BASE + off))

# rebuild U_D2 exactly as in analyze_tag (dev-split stream for the basis, so the basis is the registered one)
DBASE = 999_000 + 50_000
Dfit = B.sample_fast(env, 3000, np.random.default_rng(DBASE + 0))
diffs = []
for q in range(Q):
    Xp = A.pairs(env, Dfit["X"][:1000], q, "cond", np.random.default_rng(DBASE + 60 + q))
    a1, _ = A.acts(p, Dfit["X"][:1000], [s_tag]); a2, _ = A.acts(p, Xp, [s_tag])
    diffs.append((a2[s_tag][:, 6 * q:6 * q + 6] - a1[s_tag][:, 6 * q:6 * q + 6]).reshape(-1, d))
U = A.contrast_basis(np.concatenate(diffs), ktag); P = A.proj(U)
def smask(q): m = np.zeros(T); m[6 * q:6 * q + 6] = 1; return m

# ---------------------------------------------------------------- 1 bypass: tag patch with last-layer pattern frozen to base
@partial(jax.jit, static_argnames=("site", "psite"))
def patch_freeze(p, X, site, psite, mask, P, srcx, pat, hmask):
    def hk(n, a):
        if n == site: a = a + ((srcx - a) @ P) * mask[None, :, None]
        if n == psite: a = a * (1 - hmask[None, :, None, None]) + pat * hmask[None, :, None, None]
        return a
    return jax.nn.softmax(N.forward(p, X, hk)["answer"], -1)
Db = data(2, 1500); Xb = Db["X"]
cb, ob = A.acts(p, Xb, [s_tag, Lp]); pb = A.probs(ob)
H = cb[Lp].shape[1]
def group_mass(pat, q):
    m = np.stack([pat[:, :, A.rd_pos(env, q), [6 * q + a for a in G]].sum(-1) for G in B.GROUPS], -1)
    return m / np.maximum(m.sum(-1, keepdims=True), 1e-9)
post_b = B.episode_posterior(env, Xb)
if env.R_values[0] is not None:
    prel = np.stack([sum(post_b["post_R"][:, i] for i, R in enumerate(env.R_values) if R[0] == g) for g in range(3)], -1)
    hr2 = [float(np.mean([np.corrcoef(group_mass(cb[Lp], q)[:, h].ravel(), prel.ravel())[0, 1] ** 2 for q in range(Q)])) for h in range(H)]
else:
    hr2 = [float("nan")] * H
route_heads = [int(h) for h in np.argsort(hr2)[::-1][:2]]
res["route_heads"] = route_heads; res["head_group_mass_r2"] = hr2
out = {}
srcs = {}
for q in range(Q):
    Xp = A.pairs(env, Xb, q, "cond", np.random.default_rng(BASE + 100 + q)); cp, op = A.acts(p, Xp, [s_tag]); srcs[q] = (Xp, cp, A.probs(op))
for nm, hm in [("none_frozen", np.zeros(H)), ("route_heads_frozen", np.eye(H)[route_heads].sum(0)), ("all_heads_frozen", np.ones(H))]:
    PE, PEp, PP = [], [], []
    for q in range(Q):
        Xp, cp, pp = srcs[q]; o = []
        for i in range(0, len(Xb), A.CH):
            o.append(np.asarray(patch_freeze(p, jnp.asarray(Xb[i:i + A.CH]), s_tag, Lp, jnp.asarray(smask(q), jnp.float32), jnp.asarray(P, jnp.float32),
                                             jnp.asarray(cp[s_tag][i:i + A.CH]), jnp.asarray(cb[Lp][i:i + A.CH]), jnp.asarray(hm, jnp.float32))))
        PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(np.concatenate(o)[:, q])
    out[nm] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
res["bypass_tag_patch_with_frozen_pattern"] = out

# ---------------------------------------------------------------- 2 content: relational vs non-relational surrogates
Dt = data(3, 3000); at, _ = A.acts(p, Dt["X"], [s_tag]); post_t = B.episode_posterior(env, Dt["X"])
Df2 = data(4, 3000); af2, _ = A.acts(p, Df2["X"], [s_tag]); post_f = B.episode_posterior(env, Df2["X"])
def feats(X, act, post):
    c, onehot_all, counts_own, counts_all, roles = [], [], [], [], []
    for q in range(Q):
        c.append((act[:, 6 * q:6 * q + 6] @ U)[..., 0].reshape(-1))
        oth = [k for k in range(Q) if k != q]
        oh = np.eye(K)[X[:, oth]].reshape(len(X), -1); ownoh = np.eye(K)[X[:, q]].reshape(len(X), -1)
        onehot_all.append(np.repeat(np.concatenate([oh, ownoh], 1), 6, 0))
        cnt = A.count_feats(env, X, q)                                    # N,3
        g_of = np.array([0, 0, 1, 1, 2, 2])
        counts_own.append(np.eye(Q)[cnt[:, g_of].reshape(-1).astype(int)])          # one-hot of the token's own-group count
        counts_all.append(np.repeat(np.concatenate([np.eye(Q)[cnt[:, g].astype(int)] for g in range(3)], 1), 6, 0) * 1.0)
        if env.R_values[0] is not None:
            pr = np.stack([sum(post["pi_minus"][:, q, i] for i, R in enumerate(env.R_values) if R[0] == g) for g in range(3)], -1)
            roles.append(np.log(np.clip(pr[:, g_of], 1e-6, 1)).reshape(-1) - np.log(np.clip(1 - pr[:, g_of], 1e-6, 1)).reshape(-1))
    cat = lambda v: np.concatenate(v) if v else None
    return dict(coord=cat(c), linear_onehot_all_reports=cat(onehot_all), own_group_count_onehot=cat(counts_own),
                all_group_counts_onehot=cat(counts_all), bayes_rel_logit=cat(roles))
Ff, Ft = feats(Df2["X"], af2[s_tag], post_f), feats(Dt["X"], at[s_tag], post_t)
cont = {}
for k_ in ["linear_onehot_all_reports", "own_group_count_onehot", "all_group_counts_onehot", "bayes_rel_logit"]:
    if Ff[k_] is None: continue
    X1 = Ff[k_] if Ff[k_].ndim > 1 else Ff[k_][:, None]; X2 = Ft[k_] if Ft[k_].ndim > 1 else Ft[k_][:, None]
    W, mu, tm = A.ridge(X1, Ff["coord"][:, None], lam=1e-4); cont[k_] = A.r2(X2, Ft["coord"][:, None], W, mu, tm)
res["tag_content_R2"] = cont
# tag as a function of own-group leave-one-out count (and, in T4, non-monotonicity)
cnt_own = np.argmax(Ft["own_group_count_onehot"], 1)
res["tag_mean_by_own_group_count"] = {int(c): float(Ft["coord"][cnt_own == c].mean()) for c in range(Q) if (cnt_own == c).any()}
if Ft["bayes_rel_logit"] is not None:
    res["tag_mean_bayes_rel_logit_by_count"] = {int(c): float(Ft["bayes_rel_logit"][cnt_own == c].mean()) for c in range(Q) if (cnt_own == c).any()}
    res["corr_tag_bayes_rel_logit"] = float(np.corrcoef(Ft["coord"], Ft["bayes_rel_logit"])[0, 1])
    res["corr_tag_own_count"] = float(np.corrcoef(Ft["coord"], cnt_own)[0, 1])

# ---------------------------------------------------------------- 3 routing equations (THEOREM.md check)
# (a) per route head: attention logit difference between groups vs tag difference ; (b) readout margin vs routed vote
eq = {}
coord_b = np.stack([(cb[s_tag][:, 6 * q:6 * q + 6] @ U)[..., 0] for q in range(Q)], 1)       # N,Q,6
for h in route_heads:
    xs, ys = [], []
    for q in range(Q):
        a = cb[Lp][:, h, A.rd_pos(env, q), 6 * q:6 * q + 6]                                   # N,6 attention to q's sources
        la = np.log(np.maximum(a, 1e-9)); ys.append((la - la.mean(1, keepdims=True)).ravel())
        t_ = coord_b[:, q]; xs.append((t_ - t_.mean(1, keepdims=True)).ravel())
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    beta = float((xs * ys).sum() / (xs ** 2).sum()); eq[f"head{h}_logattn_vs_tag"] = dict(beta=beta, r2=float(np.corrcoef(xs, ys)[0, 1] ** 2))
# routed vote: v_y = sum_s abar_s 1[x_s = y], abar = mean over route heads of within-question renormalised attention
vs, ms = [], []
for q in range(Q):
    a = np.mean([cb[Lp][:, h, A.rd_pos(env, q), 6 * q:6 * q + 6] for h in route_heads], 0); a = a / a.sum(1, keepdims=True)
    v = np.stack([(a * (Xb[:, q] == y)).sum(1) for y in range(K)], -1)
    lg = np.log(pb[:, q] + 1e-12); vs.append((v - v.mean(1, keepdims=True)).ravel()); ms.append((lg - lg.mean(1, keepdims=True)).ravel())
vs, ms = np.concatenate(vs), np.concatenate(ms)
eq["readout_logit_vs_routed_vote"] = dict(c=float((vs * ms).sum() / (vs ** 2).sum()), r2=float(np.corrcoef(vs, ms)[0, 1] ** 2))
# Bayes comparison: Bayes log posterior vs routed vote
bl = []
for q in range(Q):
    lb = np.log(post_b["post_y"][:, q] + 1e-12); bl.append((lb - lb.mean(1, keepdims=True)).ravel())
bl = np.concatenate(bl); eq["bayes_logpost_vs_routed_vote_r2"] = float(np.corrcoef(vs, bl)[0, 1] ** 2)
eq["model_logit_vs_bayes_logpost_r2"] = float(np.corrcoef(ms, bl)[0, 1] ** 2)
res["routing_equations"] = eq

# ---------------------------------------------------------------- 4 held-out source distributions (alpha shift), behaviour + D2 transfer
ho = {}
for al in [0.7, 0.9]:
    e2 = B.Env(task, K=3, Q=6, alpha=al)
    D2 = B.sample_fast(e2, 6000, np.random.default_rng(BASE + 40 + int(al * 10)))
    _, o2 = A.acts(p, D2["X"], [s_tag]); pr2 = A.probs(o2); po2 = B.episode_posterior(e2, D2["X"])
    rec = dict(acc=float((pr2.argmax(-1) == D2["Y"]).mean()), bayes=float((po2["post_y"].argmax(-1) == D2["Y"]).mean()),
               own=float((po2["own_y"].argmax(-1) == D2["Y"]).mean()))
    Xs = D2["X"][:1200]; cs, os_ = A.acts(p, Xs, [s_tag]); ps_ = A.probs(os_); PE, PEp, PP = [], [], []
    for q in range(Q):
        Xp = A.pairs(e2, Xs, q, "cond", np.random.default_rng(BASE + 70 + q)); cp, op = A.acts(p, Xp, [s_tag])
        pt = A.patch(p, Xs, s_tag, smask(q), P, cp[s_tag]); PE.append(ps_[:, q]); PEp.append(A.probs(op)[:, q]); PP.append(pt[:, q])
    rec["D2_cond"] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
    ho[f"alpha{al}"] = rec
res["held_out_alpha"] = ho
json.dump(res, open(outp, "w"), indent=1, default=float); print("done", outp)
