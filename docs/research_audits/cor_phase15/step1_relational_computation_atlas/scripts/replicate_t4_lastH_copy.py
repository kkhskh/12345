# POST HOC COPY of step1_latent_reliability/scripts/replicate_t4.py; only change: H read from the last layer (for the 5-head construction).
"""Frozen evaluation for REPLICATION_PROTOCOL_T4.md.
Usage: replicate_t4.py T4_CKPT T1_CKPT OUT.json [codecheck]
'codecheck' switches both streams to base 4,400,000 (debugging on old models only; not criteria)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "step1_latent_reliability", "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from sklearn.isotonic import IsotonicRegression
import bayes as B, nets as N, ckpt as C, analysis as A

ck4, ck1, outp = sys.argv[1:4]
cc = len(sys.argv) > 4 and sys.argv[4] == "codecheck"
IDB, EVB = (4_400_000, 4_500_000) if cc else (2_100_000, 3_300_000)
t0 = time.time()
p = C.load(ck4); env = B.Env("T4", K=3, Q=6); Q, K = env.Q, env.K
L = len(p["layers"]); d = p["Eread"].shape[0]; T = A.T_of(env); site = f"resid_pre{L - 1}"; Lp = f"pattern{L - 1}"
H = p["layers"][-1]["WQ"].shape[0]; k_key, k_q = 2, len(env.R_values) - 1
g_of = np.array([0, 0, 1, 1, 2, 2]); Rs = env.R_values
res = dict(t4_ckpt=ck4, t1_ckpt=ck1, codecheck=cc, site=site)


def smp(e, base, n): return B.sample_fast(e, n, np.random.default_rng(base))
def prel_of(pi): return np.stack([sum(pi[..., i] for i, R in enumerate(Rs) if R[0] == g) for g in range(3)], -1)
def gmass(pat, q):
    m = np.stack([pat[:, :, A.rd_pos(env, q), [6 * q + a for a in G]].sum(-1) for G in B.GROUPS], -1)
    return m / np.maximum(m.sum(-1, keepdims=True), 1e-9)


def countpres(X, q, rng):
    Xp = X.copy(); n = len(X)
    for k in range(Q):
        if k == q: continue
        for a, b in B.GROUPS:
            agree = X[:, k, a] == X[:, k, b]
            v1 = rng.integers(0, K, n); off = rng.integers(1, K, n); v2 = np.where(agree, v1, (v1 + off) % K)
            Xp[:, k, a], Xp[:, k, b] = v1, v2
    return Xp


def make_pairs(X, q, kind, rng):
    return countpres(X, q, rng) if kind == "countpres" else A.pairs(env, X, q, kind, rng)


# ------------------------------------------------------------------ R0 behaviour (both stages)
beh = {}
p1 = C.load(ck1)
beh["T1_stage"] = N.evaluate(p1, "T1", n=20000, seed=EVB + 1)
beh["T4_stage"] = N.evaluate(p, "T4", n=20000, seed=EVB + 2)
R0 = all(b["acc"] >= b["own_question_bayes_acc"] + 0.5 * (b["bayes_acc"] - b["own_question_bayes_acc"]) for b in beh.values())
res["behaviour"] = beh

# ------------------------------------------------------------------ identification (identification stream only)
Di = smp(env, IDB, 3000)
ci, _ = A.acts(p, Di["X"], [Lp]); pi_i = B.episode_posterior(env, Di["X"]); pr_i = prel_of(pi_i["post_R"])
head_r2 = [float(np.mean([np.corrcoef(gmass(ci[Lp], q)[:, h].ravel(), pr_i.ravel())[0, 1] ** 2 for q in range(Q)])) for h in range(H)]
route = [int(h) for h in np.argsort(head_r2)[::-1][:2]]; ctrl = [h for h in range(H) if h not in route]
dk, dq = [], []
Xi = Di["X"][:1000]; ai, _ = A.acts(p, Xi, [site])
for q in range(Q):
    Xp = A.pairs(env, Xi, q, "cond", np.random.default_rng(IDB + 10 + q)); ap, _ = A.acts(p, Xp, [site]); D = ap[site] - ai[site]
    dk.append(D[:, 6 * q:6 * q + 6].reshape(-1, d)); dq.append(D[:, A.rd_pos(env, q)])
Uk, Uq = A.contrast_basis(np.concatenate(dk), k_key), A.contrast_basis(np.concatenate(dq), k_q)
res.update(route_heads=route, control_heads=ctrl, head_group_mass_r2_ident=head_r2)


@partial(jax.jit, static_argnames=("site", "psite"))
def run(p, X, site, psite, mk, Pk, mq, Pq, src, pat, hm):
    rec = {}
    def hk(n, a):
        if n == site:
            dl = src - a; a = a + (dl @ Pk) * mk[None, :, None] + (dl @ Pq) * mq[None, :, None]
        if n == psite:
            a = a * (1 - hm[None, :, None, None]) + pat * hm[None, :, None, None]; rec["pat"] = a
        return a
    o = N.forward(p, X, hk); return jax.nn.softmax(o["answer"], -1), rec["pat"]


def runb(X, mk, Pk, mq, Pq, src, pat, hm):
    P_, A_ = [], []
    for i in range(0, len(X), A.CH):
        a, b = run(p, jnp.asarray(X[i:i + A.CH]), site, Lp, *(jnp.asarray(z, jnp.float32) for z in (mk, Pk, mq, Pq)),
                   jnp.asarray(src[i:i + A.CH]), jnp.asarray(pat[i:i + A.CH]), jnp.asarray(hm, jnp.float32))
        P_.append(np.asarray(a)); A_.append(np.asarray(b))
    return np.concatenate(P_), np.concatenate(A_)


# ------------------------------------------------------------------ evaluation stream
De = smp(env, EVB, 1800); Xb = De["X"]
cb, ob = A.acts(p, Xb, [site, Lp]); pb = A.probs(ob); post_b = B.episode_posterior(env, Xb)
res["bayes_comparison"] = dict(model_acc=float((pb.argmax(-1) == De["Y"]).mean()), bayes_acc=float((post_b["post_y"].argmax(-1) == De["Y"]).mean()),
                               argmax_agreement=float((pb.argmax(-1) == post_b["post_y"].argmax(-1)).mean()))
kinds = ["cond", "perm", "countpres", "target"]
src = {}
for kind in kinds:
    for q in range(Q):
        Xp = make_pairs(Xb, q, kind, np.random.default_rng(EVB + 100 * (kinds.index(kind) + 1) + q))
        cp, op = A.acts(p, Xp, [site, Lp]); src[(kind, q)] = (Xp, cp, A.probs(op), B.episode_posterior(env, Xp)["post_y"])
Z = np.zeros((d, d)); I = np.eye(d)
def mk_(q): m = np.zeros(T); m[6 * q:6 * q + 6] = 1; return m
def mq_(q): m = np.zeros(T); m[A.rd_pos(env, q)] = 1; return m
nofreeze = np.zeros(H)
rk = [A.random_basis(d, k_key, np.random.default_rng(EVB + 900 + i)) for i in range(8)]
rq = [A.random_basis(d, k_q, np.random.default_rng(EVB + 950 + i)) for i in range(8)]


def patch_eval(kind, Pk, Pq, hm=nofreeze, mismatch=False, want_shift=False):
    PE, PEp, PP, PB, num, den = [], [], [], [], 0.0, 0.0
    for q in range(Q):
        Xp, cp, pp, pbay = src[(kind, q)]
        s_ = cp[site]
        if mismatch:
            s_ = s_[np.random.default_rng(EVB + 77 + q).permutation(len(Xb))]
        ptc, pat = runb(Xb, mk_(q), Pk, mq_(q), Pq, s_, cb[Lp], hm)
        PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(ptc[:, q]); PB.append(pbay[:, q])
        if want_shift:
            gp, gb, ge = gmass(pat, q)[:, route], gmass(cb[Lp], q)[:, route], gmass(cp[Lp], q)[:, route]
            num += ((gp - gb) * (ge - gb)).sum(); den += ((ge - gb) ** 2).sum()
    PE, PEp, PP, PB = map(np.concatenate, (PE, PEp, PP, PB))
    m = A.interchange_metrics(PE, PEp, PP)
    inf = PE.argmax(-1) != PEp.argmax(-1)
    okb = inf & np.isfinite(PB).all(-1)          # zero-probability hybrids (Bayes undefined) are excluded
    m["iia_bayes"] = float((PP.argmax(-1) == PB.argmax(-1))[okb].mean()) if okb.any() else float("nan")
    m["n_bayes_undefined"] = int((~np.isfinite(PB).all(-1)).sum())
    if want_shift: m["routing_shift"] = float(num / max(den, 1e-12))
    return m


Pk, Pq = A.proj(Uk), A.proj(Uq)
fr_route = np.eye(H)[route].sum(0); fr_ctrl = np.eye(H)[ctrl].sum(0)
t = {}
t["joint"] = {kd: patch_eval(kd, Pk, Pq, want_shift=(kd == "cond")) for kd in kinds}
t["key_only"] = {kd: patch_eval(kd, Pk, Z) for kd in ["cond", "perm", "target"]}
t["query_only"] = {kd: patch_eval(kd, Z, Pq) for kd in ["cond", "perm", "target"]}
t["full_q_tokens"] = {"cond": patch_eval("cond", I, I)}
t["joint_mismatch"] = patch_eval("cond", Pk, Pq, mismatch=True)
t["joint_random"] = float(np.nanmean([patch_eval("cond", A.proj(a), A.proj(b))["iia"] for a, b in zip(rk, rq)]))
t["joint_route_frozen"] = patch_eval("cond", Pk, Pq, hm=fr_route)
t["joint_control_frozen"] = patch_eval("cond", Pk, Pq, hm=fr_ctrl)
t["joint_all_frozen"] = patch_eval("cond", Pk, Pq, hm=np.ones(H))
# frozen Type-B style pattern patch: all heads' readout-q row from E'
PE, PEp, PP = [], [], []
for q in range(Q):
    Xp, cp, pp, _ = src[("cond", q)]
    pt = A.patch_pattern(p, Xb, Lp, mq_(q), np.ones(H), cp[Lp]); PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(pt[:, q])
t["pattern_patch_all_heads"] = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP)))
res["tests"] = t
head_r2_eval = [float(np.mean([np.corrcoef(gmass(cb[Lp], q)[:, h].ravel(), prel_of(post_b["post_R"]).ravel())[0, 1] ** 2 for q in range(Q)])) for h in range(H)]
res["head_group_mass_r2_eval"] = head_r2_eval

# ------------------------------------------------------------------ R4 non-monotone use (best route head, evaluation stream)
Dn = smp(env, EVB + 5, 4000); cn, _ = A.acts(p, Dn["X"], [Lp, site]); pn = B.episode_posterior(env, Dn["X"])
h0 = route[0]
la = np.stack([np.log(np.maximum(cn[Lp][:, h0, A.rd_pos(env, q), 6 * q:6 * q + 6], 1e-9)) for q in range(Q)], 1)   # N,Q,6
la = la - la.mean(-1, keepdims=True)
cnt = np.stack([A.count_feats(env, Dn["X"], q) for q in range(Q)], 1).astype(int)            # N,Q,3 (LOO)
own = cnt[:, :, g_of]                                                                          # N,Q,6
prl = np.stack([prel_of(pn["pi_minus"][:, q]) for q in range(Q)], 1)[:, :, g_of]
blo = np.log(np.clip(prl, 1e-6, 1)) - np.log(np.clip(1 - prl, 1e-6, 1)); blo = blo - blo.mean(-1, keepdims=True)
rng = np.random.default_rng(EVB + 6)
def means(ix):
    o_, l_ = own[ix].ravel(), la[ix].ravel(); return np.array([l_[o_ == c].mean() if (o_ == c).any() else np.nan for c in range(Q)])
m_all = means(np.arange(len(Dn["X"])))
boots = np.array([means(rng.integers(0, len(Dn["X"]), len(Dn["X"]))) for _ in range(1000)])
drop = boots[:, 4] - boots[:, 5]; rise = boots[:, 4] - boots[:, 0]
bayes_means = [float(blo.ravel()[own.ravel() == c].mean()) for c in range(Q)]
half = len(Dn["X"]) // 2; tr, te = slice(0, half), slice(half, None)
y_tr, y_te = la[tr].ravel(), la[te].ravel()
def r2v(pred, y): return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
W = np.polyfit(blo[tr].ravel(), y_tr, 1); r2_bayes = r2v(np.polyval(W, blo[te].ravel()), y_te)
r2_mono = max(r2v(IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(own[tr].ravel(), y_tr).predict(own[te].ravel()), y_te) for inc in (True, False))
# flexible lookup over (own group, LOO count triple)
def keyidx(cn_, gidx): return gidx * 216 + cn_[..., 0] * 36 + cn_[..., 1] * 6 + cn_[..., 2]
gid = np.broadcast_to(g_of, own.shape)
ktr = keyidx(np.repeat(cnt[tr][:, :, None, :], 6, 2), gid[tr]).ravel(); kte = keyidx(np.repeat(cnt[te][:, :, None, :], 6, 2), gid[te]).ravel()
def lookup_fit(keys, y, nk=648):
    s_ = np.bincount(keys, y, nk); c_ = np.bincount(keys, None, nk); return np.where(c_ > 0, s_ / np.maximum(c_, 1), y.mean())
lut = lookup_fit(ktr, y_tr); r2_lookup = r2v(lut[kte], y_te)
R4 = dict(m=m_all.tolist(), bayes_logodds_by_count=bayes_means, drop=float(m_all[4] - m_all[5]), drop_ci=np.percentile(drop, [2.5, 97.5]).tolist(),
          rise=float(m_all[4] - m_all[0]), rise_ci=np.percentile(rise, [2.5, 97.5]).tolist(), r2_bayes=r2_bayes, r2_best_monotone_count=r2_mono,
          r2_flexible_count_lookup=r2_lookup, head=h0)
R4["pass"] = bool(np.percentile(drop, 2.5) > 0 and np.percentile(rise, 2.5) > 0 and r2_bayes - r2_mono >= 0.10)
res["R4"] = R4

# ------------------------------------------------------------------ R5 relational content of U_key coordinates
tag = np.stack([cn[site][:, 6 * q:6 * q + 6] @ Uk for q in range(Q)], 1)                     # N,Q,6,2
ytr, yte = tag[tr].reshape(-1, 2), tag[te].reshape(-1, 2)
def r2m(pred, y): return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
lut2 = np.stack([lookup_fit(ktr, ytr[:, j]) for j in range(2)], -1); r2_tag_lookup = r2m(lut2[kte], yte)
# flexible aggregate: one-hot histogram pattern of each other question (28 patterns) x token group
pats = {}
def hist_pat(Xq):
    h = np.stack([(Xq == c).sum(-1) for c in range(K)], -1); key = h[:, 0] * 7 + h[:, 1]
    return key
def agg_feats(X):
    N_ = len(X); F = []
    for q in range(Q):
        oth = [k for k in range(Q) if k != q]
        oh = np.concatenate([np.eye(49)[hist_pat(X[:, k])] for k in oth], 1)                   # N, 5*49
        F.append(np.repeat(oh, 6, 0))
    F = np.stack(F, 1).reshape(N_, Q, 6, -1)
    gid1 = np.eye(3)[np.broadcast_to(g_of, (N_, Q, 6))]
    return (F[..., :, None] * gid1[..., None, :]).reshape(N_ * Q * 6, -1)
Ftr, Fte = agg_feats(Dn["X"][tr]), agg_feats(Dn["X"][te])
Wa, mu, tm = A.ridge(Ftr, ytr, lam=1e-3); r2_agg = A.r2(Fte, yte, Wa, mu, tm)
res["R5"] = dict(r2_tag_from_count_lookup=r2_tag_lookup, r2_tag_from_flexible_aggregate=r2_agg,
                 tag_mean_by_own_count=[tag.reshape(-1, 2)[own.ravel() == c].mean(0).tolist() for c in range(Q)],
                 pass_=bool(r2_tag_lookup >= 0.7 and r2_tag_lookup >= r2_agg + 0.3))

# ------------------------------------------------------------------ criteria
j = t["joint"]
c = {}
c["R0"] = bool(R0)
c["R1"] = bool(max(head_r2_eval) >= 0.5 and t["pattern_patch_all_heads"]["iia"] >= 0.7 and t["pattern_patch_all_heads"]["kl_recovered"] >= 0.5)
c["R2"] = bool(j["cond"]["iia"] >= 0.7 and j["cond"]["kl_recovered"] >= 0.5 and j["cond"]["iia"] >= t["joint_mismatch"]["iia"] + 0.2
               and j["cond"]["iia"] >= t["joint_random"] + 0.3 and j["perm"]["iia"] >= 0.6 and j["countpres"]["change_rate"] <= 0.10
               and j["target"]["change_rate"] <= 0.10)
c["R3"] = bool(t["joint_route_frozen"]["iia"] <= 0.25 and t["joint_control_frozen"]["iia"] >= 0.7 * j["cond"]["iia"] and j["cond"]["routing_shift"] >= 0.5)
c["R4"] = R4["pass"]; c["R5"] = res["R5"]["pass_"]
ko, qo = t["key_only"]["cond"]["iia"], t["query_only"]["cond"]["iia"]
c["localisation"] = "key-localised" if ko >= 0.7 else ("query-localised" if qo >= 0.7 else ("distributed key/query" if j["cond"]["iia"] >= 0.7 else "none"))
res["criteria"] = c; res["seconds"] = time.time() - t0
json.dump(res, open(outp, "w"), indent=1, default=float)
print(json.dumps(c), round(res["seconds"]))
