"""Usage: analyze_model.py CKPT TASK SPLIT OUT.json [SELECTION.json]
SPLIT = dev | conf.  dev: chooses the site s* (max readout-token probe R^2 for the leave-one-out reliability posterior) and
writes it to OUT.json; conf: reads s* from SELECTION.json (the dev output of the same model) and evaluates on
fresh confirmation data.  All definitions frozen in PROTOCOL.md."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import jax
import bayes as B, nets as N, ckpt as C, analysis as A

ck, task, split, outp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sel = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
BASE = {"dev": 999_000, "conf": 777_000}[split]
N_FIT, N_TEST, N_PAIR, N_RAND = 4000, 4000, 1800, 8
t0 = time.time()
p = C.load(ck); env = B.Env(task, K=3, Q=6)
Q, K = env.Q, env.K
hasR = env.R_values[0] is not None
nR = len(env.R_values); k = nR - 1 if hasR else 3
S = A.sites(p); d = p["Eread"].shape[0]
res = dict(ckpt=ck, task=task, split=split, sites=S, k=k)
rng = np.random.default_rng(BASE + 4)


def data(off, n):
    return B.sample_fast(env, n, np.random.default_rng(BASE + off))


# ------------------------------------------------------------------ 1 behaviour
res["behaviour"] = N.evaluate(p, task, n=20000, seed=BASE + 3)

# ------------------------------------------------------------------ 2 probes (identification, output-free)
Dfit, Dtest = data(0, N_FIT), data(1, N_TEST)
pr_init = N.init({16: "S", 64: "M", 128: "L"}[d], env, 12345)   # random-init control, same architecture


def rows(acts_site, X, post, kind):
    """Stack rows over questions. kind: readout | src_q | episode."""
    Zs, T = [], {}
    tg = {"pi_minus": [], "pi_episode": [], "counts": [], "hist": [], "agree_own": [], "naive": []}
    for q in range(Q):
        if kind == "readout":
            Zs.append(acts_site[:, A.rd_pos(env, q)])
        elif kind == "src_q":
            Zs.append(acts_site[:, 6 * q:6 * q + 6].mean(1))
        elif kind == "episode":
            Zs.append(acts_site[:, A.ep_pos(env)])
        if hasR:
            tg["pi_minus"].append(A.logit_target(post["pi_minus"][:, q]))
            tg["pi_episode"].append(A.logit_target(post["post_R"]))
            cnt = A.count_feats(env, X, q)
            tg["naive"].append(cnt * np.log(1.98 / 0.51) - (cnt * np.log(1.98 / 0.51)).mean(-1, keepdims=True))
        tg["counts"].append(A.count_feats(env, X, q))
        tg["hist"].append(A.hist_feats(env, X, q))
        tg["agree_own"].append(np.stack([X[:, q, a] == X[:, q, b] for a, b in B.GROUPS], -1).astype(float))
    Z = np.concatenate(Zs)
    return Z, {kk: np.concatenate(v) for kk, v in tg.items() if v}


def probe_table(params, tag):
    af, _ = A.acts(params, Dfit["X"], S); at, _ = A.acts(params, Dtest["X"], S)
    pf, pt = B.episode_posterior(env, Dfit["X"]), B.episode_posterior(env, Dtest["X"])
    out, Ws = {}, {}
    for s in S:
        for kind in ["readout", "src_q", "episode"]:
            Zf, Tf = rows(af[s], Dfit["X"], pf, kind); Zt, Tt = rows(at[s], Dtest["X"], pt, kind)
            for tname in Tf:
                W, mu, tm = A.ridge(Zf, Tf[tname]); out[f"{s}|{kind}|{tname}"] = A.r2(Zt, Tt[tname], W, mu, tm)
                if kind == "readout":
                    Ws[(s, tname)] = W
    return out, Ws, af


probes, Ws, act_fit = probe_table(p, "trained")
probes_init, _, _ = probe_table(pr_init, "init")
res["probes"] = probes; res["probes_random_init"] = probes_init

# ------------------------------------------------------------------ 3 site selection (dev only)
tsel = "pi_minus" if hasR else "counts"
if split == "dev":
    s_star = max(S, key=lambda s: probes[f"{s}|readout|{tsel}"])
else:
    s_star = sel["s_star"]
res["s_star"] = s_star; res["selection_target"] = tsel

# ------------------------------------------------------------------ 4 candidate subspaces at s*, readout tokens
Dp = data(2, N_PAIR)
bases = {}
bases["D1_probe"] = A.basis_from_W(Ws[(s_star, tsel)], k)
bases["hist_surrogate"] = A.basis_from_W(Ws[(s_star, "hist")], k)
# D2: contrast PCA from on-manifold cond pairs on the FIT set (inputs only)
diffs = []
for q in range(Q):
    Xp = A.pairs(env, Dfit["X"][:1200], q, "cond", np.random.default_rng(BASE + 50 + q))
    a1, _ = A.acts(p, Dfit["X"][:1200], [s_star]); a2, _ = A.acts(p, Xp, [s_star])
    diffs.append(a2[s_star][:, A.rd_pos(env, q)] - a1[s_star][:, A.rd_pos(env, q)])
bases["D2_contrast"] = A.contrast_basis(np.concatenate(diffs), k)
# reuse variant: D1 fitted on q in {0,1,2} only
pf = B.episode_posterior(env, Dfit["X"])
if hasR:
    Z012 = np.concatenate([act_fit[s_star][:, A.rd_pos(env, q)] for q in range(3)])
    T012 = np.concatenate([A.logit_target(pf["pi_minus"][:, q]) for q in range(3)])
    bases["D1_fit_q012"] = A.basis_from_W(A.ridge(Z012, T012)[0], k)
rb = [A.random_basis(d, k, np.random.default_rng(BASE + 900 + i)) for i in range(N_RAND)]
I = np.eye(d)
# principal-angle overlap between bases
res["overlap_D1_D2"] = float(np.linalg.svd(bases["D1_probe"].T @ bases["D2_contrast"])[1].mean())

# ------------------------------------------------------------------ 5 causal tests
Xb = Dp["X"]; T = A.T_of(env)
cache_b, out_b = A.acts(p, Xb, [s_star] + [f"pattern{l}" for l in range(len(p["layers"]))])
pb = A.probs(out_b)
kinds = ["cond", "perm", "splice", "target"]
src = {}
for kind in kinds:
    for q in range(Q):
        Xp = A.pairs(env, Xb, q, kind, np.random.default_rng(BASE + 100 * (kinds.index(kind) + 1) + q))
        cp, op = A.acts(p, Xp, [s_star] + ([f"pattern{l}" for l in range(len(p["layers"]))] if kind == "cond" else []))
        src[(kind, q)] = (Xp, cp, A.probs(op), np.asarray(op["conf"]))
ll_nat = A.log_lik(env, Xb)
lo, hi = np.percentile(ll_nat, 1), np.percentile(ll_nat, 99)


def mask_q(q):
    m = np.zeros(T); m[A.rd_pos(env, q)] = 1; return m


def interchange(P, kind, qs=range(Q), manifold=None, site=None):
    site = site or s_star
    PE, PEp, PP = [], [], []
    for q in qs:
        Xp, cp, pp, _ = src[(kind, q)]
        sel_ = np.ones(len(Xb), bool)
        if manifold is not None:
            llp = A.log_lik(env, Xp); on = (llp >= lo) & (llp <= hi) & np.isfinite(llp)
            sel_ = on if manifold == "on" else ~on
        if sel_.sum() == 0:
            continue
        pt = A.patch(p, Xb[sel_], site, mask_q(q), P, cp[site][sel_])
        PE.append(pb[sel_, q]); PEp.append(pp[sel_, q]); PP.append(pt[:, q])
    if not PE:
        return dict(n=0)
    m = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP)); m["n"] = int(sum(len(x) for x in PE))
    return m


tests = {}
for name, U in list(bases.items()) + [("full_residual", None)]:
    P = I if U is None else A.proj(U)
    tests[name] = {kind: interchange(P, kind) for kind in kinds}
    if name in ("D1_probe", "D2_contrast", "full_residual"):
        tests[name]["perm_on"] = interchange(P, "perm", manifold="on")
        tests[name]["perm_off"] = interchange(P, "perm", manifold="off")
        tests[name]["splice_on"] = interchange(P, "splice", manifold="on")
        tests[name]["splice_off"] = interchange(P, "splice", manifold="off")
    if name == "D1_fit_q012":
        tests[name]["cond_q345"] = interchange(P, "cond", qs=[3, 4, 5])
rand = [{kind: interchange(A.proj(U), kind) for kind in ["cond", "perm", "target"]} for U in rb]
tests["random"] = {kind: {m: float(np.nanmean([r[kind][m] for r in rand])) for m in ["iia", "kl_recovered", "change_rate"]}
                   for kind in ["cond", "perm", "target"]}
tests["random"]["cond_iia_max"] = float(np.nanmax([r["cond"]["iia"] for r in rand]))

# site x location sweep (cond interchange): D1 refit per site/location, and full residual (upper bound for that location)
def qtok_mask(q):
    m = np.zeros(T); m[6 * q:6 * q + 6] = 1; m[A.rd_pos(env, q)] = 1; return m


sweep = {}
for s in S:
    cps = {q: A.acts(p, src[("cond", q)][0], [s])[0][s] for q in range(Q)}
    ab, _ = A.acts(p, Xb, [s])
    for loc in ["readout", "q_tokens"]:
        if loc == "readout":
            Zf = np.concatenate([act_fit[s][:, A.rd_pos(env, q)] for q in range(Q)])
            Tf = np.concatenate([A.logit_target(pf["pi_minus"][:, q]) if hasR else A.count_feats(env, Dfit["X"], q) for q in range(Q)])
        else:
            Zf = np.concatenate([act_fit[s][:, list(range(6 * q, 6 * q + 6)) + [A.rd_pos(env, q)]].reshape(-1, d) for q in range(Q)])
            Tf = np.concatenate([np.repeat(A.logit_target(pf["pi_minus"][:, q]) if hasR else A.count_feats(env, Dfit["X"], q), 7, 0) for q in range(Q)])
        U = A.basis_from_W(A.ridge(Zf, Tf)[0], k)
        for nm, P in [("D1", A.proj(U)), ("full", I)]:
            PE, PEp, PP = [], [], []
            for q in range(Q):
                msk = mask_q(q) if loc == "readout" else qtok_mask(q)
                pt = A.patch(p, Xb, s, msk, P, cps[q]); PE.append(pb[:, q]); PEp.append(src[("cond", q)][2][:, q]); PP.append(pt[:, q])
            mm = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
            sweep[f"{s}|{loc}|{nm}"] = dict(iia=mm["iia"], kl_recovered=mm["kl_recovered"])
    # episode token full-residual patch (does the episode token carry a used summary?)
    m = np.zeros(T); m[A.ep_pos(env)] = 1; PE, PEp, PP = [], [], []
    for q in range(Q):
        pt = A.patch(p, Xb, s, m, I, cps[q]); PE.append(pb[:, q]); PEp.append(src[("cond", q)][2][:, q]); PP.append(pt[:, q])
    mm = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
    sweep[f"{s}|episode|full"] = dict(iia=mm["iia"], kl_recovered=mm["kl_recovered"])
tests["site_location_sweep_cond"] = sweep

# cross-head reuse (confidence head) for cond interchange with D1
if hasR:
    dc_m, dc_p = [], []
    for q in range(Q):
        Xp, cp, pp, cfp = src[("cond", q)]
        _, cpt = A.patch_full(p, Xb, s_star, mask_q(q), A.proj(bases["D1_probe"]), cp[s_star])
        dc_m.append(cfp[:, q] - np.asarray(out_b["conf"])[:, q]); dc_p.append(cpt[:, q] - np.asarray(out_b["conf"])[:, q])
    dc_m, dc_p = np.concatenate(dc_m), np.concatenate(dc_p)
    tests["conf_head_D1"] = dict(corr=float(np.corrcoef(dc_m, dc_p)[0, 1]) if dc_m.std() > 1e-6 else float("nan"),
                                 slope=float((dc_m * dc_p).sum() / max((dc_m ** 2).sum(), 1e-12)), conf_std=float(dc_m.std()))

# necessity: resample-ablate subspace at ALL readout tokens with activations of independent episodes
Xind = data(5, N_PAIR)["X"]; cind, _ = A.acts(p, Xind, [s_star])
mall = np.zeros(T); mall[6 * Q:7 * Q] = 1
post_b = B.episode_posterior(env, Xb)
acc_clean = float((pb.argmax(-1) == Dp["Y"]).mean()); own = float((post_b["own_y"].argmax(-1) == Dp["Y"]).mean())
nec = dict(acc_clean=acc_clean, own_question_bayes=own, bayes=float((post_b["post_y"].argmax(-1) == Dp["Y"]).mean()))
for name, U in [("D1_probe", bases["D1_probe"]), ("D2_contrast", bases["D2_contrast"]), ("hist_surrogate", bases["hist_surrogate"]),
                ("full_readout", None)]:
    P = I if U is None else A.proj(U)
    pa = A.patch(p, Xb, s_star, mall, P, cind[s_star]); nec[name] = float((pa.argmax(-1) == Dp["Y"]).mean())
nec["random_mean"] = float(np.mean([(A.patch(p, Xb, s_star, mall, A.proj(U), cind[s_star]).argmax(-1) == Dp["Y"]).mean() for U in rb]))
gap = acc_clean - own
nec["gap"] = gap
nec["drop_frac"] = {kk: ((acc_clean - v) / gap if gap >= 0.02 else float('nan')) for kk, v in nec.items() if kk not in ("acc_clean", "own_question_bayes", "bayes", "gap")}
tests["necessity"] = nec

# restoration: corrupted = splice (other questions replaced), restore D1 at readout q from clean
rest = {}
for name, U in [("D1_probe", bases["D1_probe"]), ("D2_contrast", bases["D2_contrast"]), ("full_residual", None)]:
    P = I if U is None else A.proj(U); recs = []
    for q in range(Q):
        Xc = src[("splice", q)][0]; pc = src[("splice", q)][2]
        pr_ = A.patch(p, Xc, s_star, mask_q(q), P, cache_b[s_star])
        b = A.kl(pb[:, q], pc[:, q]); r_ = 1 - A.kl(pb[:, q], pr_[:, q]) / np.maximum(b, 1e-9); recs.append(r_[b > 1e-3])
    rest[name] = float(np.median(np.concatenate(recs)))
tests["restoration_kl_recovered"] = rest

# ------------------------------------------------------------------ 6 content test on D2 coordinates (output-free basis)
at_test, _ = A.acts(p, Dtest["X"], [s_star]); ptst = B.episode_posterior(env, Dtest["X"])
Zf = np.concatenate([act_fit[s_star][:, A.rd_pos(env, q)] for q in range(Q)]) @ bases["D2_contrast"]
Zt = np.concatenate([at_test[s_star][:, A.rd_pos(env, q)] for q in range(Q)]) @ bases["D2_contrast"]
feat = {
    "pi_minus": (lambda X, post: np.concatenate([A.logit_target(post["pi_minus"][:, q]) for q in range(Q)]) if hasR else None),
    "counts": (lambda X, post: np.concatenate([A.count_feats(env, X, q) for q in range(Q)])),
    "hist": (lambda X, post: np.concatenate([A.hist_feats(env, X, q) for q in range(Q)])),
    "own_question_onehot": (lambda X, post: np.concatenate([np.eye(K)[X[:, q]].reshape(len(X), -1) for q in range(Q)])),
}
content = {}
for fn, f in feat.items():
    Ff, Ft = f(Dfit["X"], pf), f(Dtest["X"], ptst)
    if Ff is None:
        continue
    W, mu, tm = A.ridge(Ff, Zf); content[fn] = A.r2(Ft, Zt, W, mu, tm)
res["content_D2"] = content

# ------------------------------------------------------------------ 7 Type-B attention analysis
att = {}
if hasR:
    preliab = np.stack([sum(post_b["post_R"][:, i] for i, R in enumerate(env.R_values) if R[0] == g) for g in range(3)], -1)
    for l in range(len(p["layers"])):
        pat = cache_b[f"pattern{l}"]                                    # N,H,T,T
        H = pat.shape[1]; best = []
        for h in range(H):
            r2s = []
            for q in range(Q):
                m = np.stack([pat[:, h, A.rd_pos(env, q), [6 * q + a for a in G]].sum(-1) for G in B.GROUPS], -1)
                m = m / np.maximum(m.sum(-1, keepdims=True), 1e-9)
                r2s.append(np.corrcoef(m.ravel(), preliab.ravel())[0, 1] ** 2)
            best.append(float(np.mean(r2s)))
        att[f"layer{l}_readout_group_mass_r2_per_head"] = best
        # pattern patching (cond pairs), readout-q query row, all heads then each head
        for hsel in ["all"] + list(range(H)):
            hm = np.ones(H) if hsel == "all" else np.eye(H)[hsel]
            PE, PEp, PP = [], [], []
            for q in range(Q):
                Xp, cp, pp, _ = src[("cond", q)]
                qm = np.zeros(T); qm[A.rd_pos(env, q)] = 1
                pt = A.patch_pattern(p, Xb, f"pattern{l}", qm, hm, cp[f"pattern{l}"])
                PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(pt[:, q])
            att[f"layer{l}_pattern_patch_{hsel}"] = A.interchange_metrics(np.concatenate(PE), np.concatenate(PEp), np.concatenate(PP))
res["attention"] = att
res["tests"] = tests
res["seconds"] = time.time() - t0
json.dump(res, open(outp, "w"), indent=1, default=float)
print("done", outp, round(res["seconds"]), "s; s*", s_star)
