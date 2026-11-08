"""Addendum analysis (PROTOCOL_ADDENDUM_TAG.md): source-token reliability tag + readout routing.
Usage: analyze_tag.py CKPT TASK SPLIT OUT.json [DEV_TAG.json]"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import bayes as B, nets as N, ckpt as C, analysis as A

ck, task, split, outp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sel = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
BASE = {"dev": 999_000, "conf": 777_000}[split] + 50_000          # disjoint from analyze_model.py streams
N_FIT, N_TEST, N_PAIR, N_RAND = 3000, 3000, 1800, 8
t0 = time.time()
p = C.load(ck); env = B.Env(task, K=3, Q=6); Q, K = env.Q, env.K
hasR = env.R_values[0] is not None
L = len(p["layers"]); d = p["Eread"].shape[0]; T = A.T_of(env)
cand_sites = [f"resid_pre{l}" for l in range(1, L)]
I = np.eye(d)
res = dict(ckpt=ck, task=task, split=split)


def data(off, n):
    return B.sample_fast(env, n, np.random.default_rng(BASE + off))


def role_logits(env, pi):
    """pi (N,nR) -> per-group centred log marginal role probabilities. T1-T3: (N,3,1) [rel vs noise];
    T4: (N,3,2) [rel, coll vs noise]."""
    Rs = env.R_values
    prel = np.stack([sum(pi[:, i] for i, R in enumerate(Rs) if R[0] == g) for g in range(3)], -1)
    if env.task == "T4":
        pcol = np.stack([sum(pi[:, i] for i, R in enumerate(Rs) if R[1] == g) for g in range(3)], -1)
        pno = np.clip(1 - prel - pcol, 1e-6, 1)
        lg = np.log(np.stack([np.clip(prel, 1e-6, 1), np.clip(pcol, 1e-6, 1), pno], -1))
        lg = lg - lg.mean(-1, keepdims=True)
        return lg[..., :2]
    pno = np.clip(1 - prel, 1e-6, 1)
    return (np.log(np.clip(prel, 1e-6, 1)) - np.log(pno))[..., None] / 2


ktag = (2 if task == "T4" else 1) if hasR else 1


def tag_rows(act, X, post, q, which="loo"):
    """rows = the 6 source tokens of question q. target = role logits of the token's own group."""
    Z = act[:, 6 * q:6 * q + 6].reshape(-1, d)
    if hasR:
        pi = post["pi_minus"][:, q] if which == "loo" else post["post_R"]
        rl = role_logits(env, pi)                                  # N,3,k
        Tt = np.repeat(rl, 2, axis=1).reshape(-1, rl.shape[-1])    # sources 2g,2g+1 -> group g
    else:
        cnt = A.count_feats(env, X, q)                             # N,3
        Tt = np.repeat(cnt, 2, axis=1).reshape(-1, 1)
    return Z, Tt


Dfit, Dtest = data(0, N_FIT), data(1, N_TEST)
pf, pt = B.episode_posterior(env, Dfit["X"]), B.episode_posterior(env, Dtest["X"])
init_p = N.init({16: "S", 64: "M", 128: "L"}[d], env, 12345)
af, _ = A.acts(p, Dfit["X"], cand_sites); at, _ = A.acts(p, Dtest["X"], cand_sites)
ai_f, _ = A.acts(init_p, Dfit["X"], cand_sites); ai_t, _ = A.acts(init_p, Dtest["X"], cand_sites)
probes = {}
Ws = {}
for s in cand_sites:
    for which in (["loo", "episode"] if hasR else ["loo"]):
        for tag, (F, Tt_) in {"trained": (af, at), "init": (ai_f, ai_t)}.items():
            Zf, Tf = map(np.concatenate, zip(*[tag_rows(F[s], Dfit["X"], pf, q, which) for q in range(Q)]))
            Zt, Tt = map(np.concatenate, zip(*[tag_rows(Tt_[s], Dtest["X"], pt, q, which) for q in range(Q)]))
            W, mu, tm = A.ridge(Zf, Tf); probes[f"{s}|{which}|{tag}"] = A.r2(Zt, Tt, W, mu, tm)
            if tag == "trained" and which == "loo":
                Ws[s] = W
res["tag_probes"] = probes
s_tag = max(cand_sites, key=lambda s: probes[f"{s}|loo|trained"]) if split == "dev" else sel["s_tag"]
res["s_tag"] = s_tag; res["k_tag"] = ktag
U_tag = A.basis_from_W(Ws[s_tag], ktag)
# reuse basis: fit on source tokens of questions 0-2 only
Zf, Tf = map(np.concatenate, zip(*[tag_rows(af[s_tag], Dfit["X"], pf, q) for q in range(3)]))
U_012 = A.basis_from_W(A.ridge(Zf, Tf)[0], ktag)

# pairs
Dp = data(2, N_PAIR); Xb = Dp["X"]
Lp = f"pattern{L - 1}"
cb, ob = A.acts(p, Xb, [s_tag, Lp]); pb = A.probs(ob)
kinds = ["cond", "perm", "splice", "target"]
src = {}
for kind in kinds:
    for q in range(Q):
        Xp = A.pairs(env, Xb, q, kind, np.random.default_rng(BASE + 100 * (kinds.index(kind) + 1) + q))
        cp, op = A.acts(p, Xp, [s_tag] + ([Lp] if kind in ("cond", "target") else []))
        src[(kind, q)] = (Xp, cp, A.probs(op), np.asarray(op["conf"]))
# D2_tag: output-free contrast basis on fit-set cond pairs, source tokens of q
diffs = []
for q in range(Q):
    Xp = A.pairs(env, Dfit["X"][:1000], q, "cond", np.random.default_rng(BASE + 60 + q))
    a1, _ = A.acts(p, Dfit["X"][:1000], [s_tag]); a2, _ = A.acts(p, Xp, [s_tag])
    diffs.append((a2[s_tag][:, 6 * q:6 * q + 6] - a1[s_tag][:, 6 * q:6 * q + 6]).reshape(-1, d))
U_D2 = A.contrast_basis(np.concatenate(diffs), ktag)
U_D2_012 = A.contrast_basis(np.concatenate(diffs[:3]), ktag)          # addendum 2: reuse variant of D2
res["overlap_tag_D2"] = float(np.linalg.svd(U_tag.T @ U_D2)[1].mean())


def smask(q):
    m = np.zeros(T); m[6 * q:6 * q + 6] = 1; return m


def inter(P, kind, qs=range(Q), site=None):
    site = site or s_tag
    PE, PEp, PP = [], [], []
    for q in qs:
        Xp, cp, pp, _ = src[(kind, q)]
        ptc = A.patch(p, Xb, site, smask(q), P, cp[site]); PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(ptc[:, q])
    return A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))


rb = [A.random_basis(d, ktag, np.random.default_rng(BASE + 900 + i)) for i in range(N_RAND)]
tests = {}
for name, P in [("tag", A.proj(U_tag)), ("tag_D2", A.proj(U_D2)), ("full_src_q", I)]:
    tests[name] = {kind: inter(P, kind) for kind in kinds}
tests["tag_fit_q012"] = {"cond_q345": inter(A.proj(U_012), "cond", qs=[3, 4, 5])}
tests["tag_D2_fit_q012"] = {"cond_q345": inter(A.proj(U_D2_012), "cond", qs=[3, 4, 5])}
rr = [{kind: inter(A.proj(U), kind) for kind in ["cond", "target"]} for U in rb]
tests["random"] = {kind: {m: float(np.nanmean([r[kind][m] for r in rr])) for m in ["iia", "kl_recovered", "change_rate"]} for kind in ["cond", "target"]}

# necessity: resample-ablate the tag at all 36 source tokens
Xind = data(5, N_PAIR)["X"]; ci, _ = A.acts(p, Xind, [s_tag])
mall = np.zeros(T); mall[:6 * Q] = 1
post_b = B.episode_posterior(env, Xb)
acc_clean = float((pb.argmax(-1) == Dp["Y"]).mean()); own = float((post_b["own_y"].argmax(-1) == Dp["Y"]).mean())
nec = dict(acc_clean=acc_clean, own_question_bayes=own, gap=acc_clean - own)
for name, P in [("tag", A.proj(U_tag)), ("tag_D2", A.proj(U_D2))]:
    nec[name] = float((A.patch(p, Xb, s_tag, mall, P, ci[s_tag]).argmax(-1) == Dp["Y"]).mean())
nec["random_mean"] = float(np.mean([(A.patch(p, Xb, s_tag, mall, A.proj(U), ci[s_tag]).argmax(-1) == Dp["Y"]).mean() for U in rb]))
nec["drop_frac"] = {kk: ((acc_clean - nec[kk]) / nec["gap"] if nec["gap"] >= 0.02 else float("nan")) for kk in ["tag", "tag_D2", "random_mean"]}
tests["necessity"] = nec

# content of the output-free D2_tag coordinates
at_s, _ = A.acts(p, Dtest["X"], [s_tag])
def coords(act, X, post, U):
    rows_, tg_, val_, ag_, hi_ = [], [], [], [], []
    for q in range(Q):
        Z, Tt = tag_rows(act, X, post, q); rows_.append(Z @ U); tg_.append(Tt)
        val_.append(np.eye(K)[X[:, q]].reshape(-1, K))
        A_ = np.stack([X[:, q, a] == X[:, q, b] for a, b in B.GROUPS], -1).astype(float); ag_.append(np.repeat(A_, 2, 1).reshape(-1, 1))
        hi_.append(np.repeat(A.hist_feats(env, X, q), 6, 0))
    return [np.concatenate(v) for v in (rows_, tg_, val_, ag_, hi_)]
cf = coords(af[s_tag], Dfit["X"], pf, U_D2); ct = coords(at_s[s_tag], Dtest["X"], pt, U_D2)
content = {}
for i, nm in [(1, "tag_target"), (2, "own_value_onehot"), (3, "own_pair_agreement"), (4, "other_question_histograms")]:
    W, mu, tm = A.ridge(cf[i], cf[0]); content[nm] = A.r2(ct[i], ct[0], W, mu, tm)
res["content_D2_tag"] = content

# routing mediation: under the tag patch (cond), does last-layer readout attention to groups move toward E'?
def group_mass(pat, q):
    m = np.stack([pat[:, :, A.rd_pos(env, q), [6 * q + a for a in G]].sum(-1) for G in B.GROUPS], -1)   # N,H,3
    return m / np.maximum(m.sum(-1, keepdims=True), 1e-9)

import jax, jax.numpy as jnp
from functools import partial
@partial(jax.jit, static_argnames=("site", "cap"))
def _patch_cap(p, X, site, cap, mask, P, srcx):
    rec = {}
    def hk(n, a):
        if n == site:
            a = a + ((srcx - a) @ P) * mask[None, :, None]
        if n == cap:
            rec[n] = a
        return a
    o = N.forward(p, X, hk); return rec[cap], o
def routing_shift(U):
    mov = []
    for q in range(Q):
        Xp, cp, pp, _ = src[("cond", q)]
        pats = []
        for i in range(0, len(Xb), A.CH):
            r_, _ = _patch_cap(p, jnp.asarray(Xb[i:i + A.CH]), s_tag, Lp, jnp.asarray(smask(q), jnp.float32),
                               jnp.asarray(A.proj(U), jnp.float32), jnp.asarray(cp[s_tag][i:i + A.CH]))
            pats.append(np.asarray(r_))
        gp = group_mass(np.concatenate(pats), q); gb = group_mass(cb[Lp], q); ge = group_mass(cp[Lp], q)
        mov.append((((gp - gb) * (ge - gb)).sum(), ((ge - gb) ** 2).sum()))
    return float(sum(a for a, _ in mov) / max(sum(b for _, b in mov), 1e-12))


res["routing_shift_fraction_under_tag_patch"] = routing_shift(U_tag)
res["routing_shift_fraction_under_tag_D2_patch"] = routing_shift(U_D2)
# pattern-patch specificity (target pairs): reliability unchanged, so routing should transfer little
qm_all = []
PE, PEp, PP = [], [], []
for q in range(Q):
    Xp, cp, pp, _ = src[("target", q)]
    qm = np.zeros(T); qm[A.rd_pos(env, q)] = 1
    pt_ = A.patch_pattern(p, Xb, Lp, qm, np.ones(cb[Lp].shape[1]), cp[Lp]); PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(pt_[:, q])
res["pattern_patch_target_pairs"] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))

# confidence head (T2)
if task == "T2":
    dm, dp_ = [], []
    for q in range(Q):
        Xp, cp, pp, cfp = src[("cond", q)]
        for nm, U in [("tag", U_tag), ("tag_D2", U_D2)]:
            _, cpt = A.patch_full(p, Xb, s_tag, smask(q), A.proj(U), cp[s_tag])
            dm.append((nm, cfp[:, q] - np.asarray(ob["conf"])[:, q])); dp_.append((nm, cpt[:, q] - np.asarray(ob["conf"])[:, q]))
    for nm in ["tag", "tag_D2"]:
        a_ = np.concatenate([v for n_, v in dm if n_ == nm]); b_ = np.concatenate([v for n_, v in dp_ if n_ == nm])
        tests[f"conf_head_{nm}"] = dict(corr=float(np.corrcoef(a_, b_)[0, 1]), slope=float((a_ * b_).sum() / max((a_ ** 2).sum(), 1e-12)))
res["tests"] = tests; res["seconds"] = time.time() - t0
json.dump(res, open(outp, "w"), indent=1, default=float)
print("done", outp, round(res["seconds"]), "s; s_tag", s_tag)
