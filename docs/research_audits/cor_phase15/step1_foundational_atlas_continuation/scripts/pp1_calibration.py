"""PP1 re-analysis (mandate §2): support taxonomy of the frozen PP1 edit population, pointwise calibration-to-sign
certificates, distribution-level bounds, stratification, and constructed counterexample models.
Edit population: identical to the frozen PP1 run (1500 base episodes, stream 6,100,000; every q, k != q, g).
Writes logs/pp1_calibration.json."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C

f4, f1 = G.Fam(pc=1.0, rho=1.0), G.Fam(pc=0.0); K, Q = 3, 6
NB = int(os.environ.get("NB", "1500"))
d = G.sample(f4, NB, np.random.default_rng(6_100_000)); X0, cfg = d["X"], d["cfg"]
bi, qq, kk, gg, Xe = [], [], [], [], []
for q in range(Q):
    for k in range(Q):
        if k == q: continue
        for g, (a, b) in enumerate(G.GROUPS):
            X = X0.copy(); ag = X[:, k, a] == X[:, k, b]; X[:, k, b] = np.where(ag, (X[:, k, a] + 1) % K, X[:, k, a])
            Xe.append(X); bi.append(np.arange(NB)); qq.append(np.full(NB, q)); kk.append(np.full(NB, k)); gg.append(np.full(NB, g))
Xe = np.concatenate(Xe); bi = np.concatenate(bi); qq = np.concatenate(qq); kk = np.concatenate(kk); gg = np.concatenate(gg); n = len(Xe)
creates = np.array([X0[bi[i], kk[i], G.GROUPS[gg[i]][0]] != X0[bi[i], kk[i], G.GROUPS[gg[i]][1]] for i in range(n)])  # edit creates agreement
A0 = np.stack([(X0[:, :, a] == X0[:, :, b]) for a, b in G.GROUPS], -1).astype(int); Ae = np.stack([(Xe[:, :, a] == Xe[:, :, b]) for a, b in G.GROUPS], -1).astype(int)
cb = A0[bi].sum(1)[np.arange(n), gg] - A0[bi, qq, gg]; ca = Ae.sum(1)[np.arange(n), gg] - Ae[np.arange(n), qq, gg]
boundary = ((cb == 4) & (ca == 5)) | ((cb == 5) & (ca == 4))

# ------------------------------------------------------------------ support taxonomy
cf = f4.configs
true_rel = np.array([cf[c][0] for c in cfg]); true_col = np.array([cf[c][1] for c in cfg])
post_e = G.posterior(f4, Xe); post_0 = G.posterior(f4, X0)
S1_marginal = np.isfinite(post_e["post_y"]).all((1, 2))                          # P_T4(x') > 0
# S2: P(x' | R_true) > 0  <=> the true colluder still agrees in every question (only the colluder role has zeros)
S2_conditional = (Ae[np.arange(n), :, true_col[bi]]).all(1)
# S3: naturally reachable by resampling ONE source's mechanism (source b of group g in question k) given (R_true, Y):
#     reliable and noise sources give every value positive probability; the colluder's second source must copy -> same as S2
S3_reachable = S2_conditional.copy()
logp_nat = np.log(np.exp(post_0["logM"].sum(1) - post_0["logM"].sum(1).max(-1, keepdims=True)).mean(-1)) + post_0["logM"].sum(1).max(-1)
lpe = post_e["logM"].sum(1); m_ = np.where(np.isfinite(lpe.max(-1)), lpe.max(-1), 0)
with np.errstate(divide="ignore"):
    logp_edit = np.log(np.exp(lpe - m_[:, None]).mean(-1)) + m_
res = dict(n_edits=int(n), support=dict(S1_marginal=int(S1_marginal.sum()), S2_conditional_on_true_latent=int(S2_conditional.sum()),
           S1_not_S2=int((S1_marginal & ~S2_conditional).sum()), S2_not_S1=int((S2_conditional & ~S1_marginal).sum()),
           S4_all_surgical=int(n)))
print("support", res["support"], flush=True)

# ------------------------------------------------------------------ Bayes margins
with np.errstate(divide="ignore", invalid="ignore"):
    lb0, lbe = np.log(post_0["post_y"]), np.log(post_e["post_y"])
order = np.argsort(-np.nan_to_num(lb0, nan=-1e9), -1); c1 = order[bi, qq, 0]; c2 = order[bi, qq, 1]
def margin(lp, rows, qv): return lp[rows, qv, c1] - lp[rows, qv, c2]
m0 = margin(lb0, bi, qq); me = margin(lbe, np.arange(n), qq); dm = me - m0
lb1e, lb10 = np.log(G.posterior(f1, Xe)["post_y"]), np.log(G.posterior(f1, X0)["post_y"])
dm1 = margin(lb1e, np.arange(n), qq) - margin(lb10, bi, qq)
scored = S1_marginal & (np.abs(dm) > 0.5)


def model_lp(kind, path, X):
    if kind == "tf":
        p = jax.tree_util.tree_map(jnp.asarray, C.load(path)); fwd = lambda Xb: N.forward(p, Xb)
    else:
        p = jax.tree_util.tree_map(jnp.asarray, C.load_base(path)); fb = N.make_forward("deepsets", 3); fwd = lambda Xb: fb(p, Xb)
    jf = jax.jit(lambda Xb: jax.nn.log_softmax(fwd(Xb)["answer"], -1))
    return np.concatenate([np.asarray(jf(jnp.asarray(X[i:i + 3000]))) for i in range(0, len(X), 3000)])


models = {f"tf_s{s}": ("tf", os.path.join(LR, "checkpoints/posthoc_curriculum", f"T4_M_K3_s{s}_fromT1.npz"), "A'/B" if s != 2 else "B") for s in [0, 1, 2]}
models.update({f"tf_s{s}": ("tf", os.path.join(LR, "checkpoints/replication", f"T4_M_K3_s{s}_fromT1.npz"), "C" if s == 12 else "A'") for s in [10, 11, 12, 13]})
models["tf_s0"] = (models["tf_s0"][0], models["tf_s0"][1], "A'"); models["tf_s1"] = (models["tf_s1"][0], models["tf_s1"][1], "A'")
models.update({f"ds_s{s}": ("ds", os.path.join(LR, "checkpoints/main", f"T4_M_K3_s{s}_deepsets.npz"), "A(pool)") for s in [0, 1, 2]})

# natural training-distribution reference for KL
Dn = G.sample(f4, 3000, np.random.default_rng(6_150_000)); pn = G.posterior(f4, Dn["X"])["post_y"]


def analyse(name, l0, le, ln=None, cls=""):
    mh0, mhe = margin(l0, bi, qq), margin(le, np.arange(n), qq); dmh = mhe - mh0
    e0, ee = mh0 - m0, mhe - me                                              # pointwise margin errors at both endpoints
    agree = np.sign(dmh) == np.sign(dm)
    cert = np.abs(dm) > np.abs(e0) + np.abs(ee)                                # Theorem C1 condition (sufficient)
    # log-probability sup-norm error at both endpoints (Theorem C2: |dm| > 2(delta0 + deltae))
    with np.errstate(invalid="ignore"):
        d0 = np.nanmax(np.abs(l0[bi, qq] - lb0[bi, qq]), -1); de = np.nanmax(np.abs(le[np.arange(n), qq] - lbe[np.arange(n), qq]), -1)
    cert2 = np.abs(dm) > 2 * (d0 + de)
    # TV and pi_min (Theorem C3): |log a - log b| <= tv / (pmin - tv) when tv < pmin
    P0, Pe = np.exp(lb0[bi, qq]), np.exp(lbe[np.arange(n), qq]); H0, He = np.exp(l0[bi, qq]), np.exp(le[np.arange(n), qq])
    tv0, tve = 0.5 * np.abs(P0 - H0).sum(-1), 0.5 * np.abs(Pe - He).sum(-1)
    pm0 = np.minimum(np.take_along_axis(P0, c1[:, None], 1), np.take_along_axis(P0, c2[:, None], 1))[:, 0]
    pme = np.minimum(np.take_along_axis(Pe, c1[:, None], 1), np.take_along_axis(Pe, c2[:, None], 1))[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = np.where(tv0 < pm0, 2 * tv0 / (pm0 - tv0), np.inf); be = np.where(tve < pme, 2 * tve / (pme - tve), np.inf)
    cert3 = np.abs(dm) > b0 + be
    r = dict(cls=cls)
    sel = scored
    r["sign_agreement_scored"] = float(agree[sel].mean())
    r["cert_C1_pointwise_margin_frac"] = float(cert[sel].mean()); r["agree_given_not_C1"] = float(agree[sel & ~cert].mean()) if (sel & ~cert).any() else None
    r["cert_C2_logprob_frac"] = float(cert2[sel].mean()); r["cert_C3_TV_frac"] = float(cert3[sel].mean())
    r["C1_violation_check(agree_given_C1)"] = float(agree[sel & cert].mean()) if (sel & cert).any() else None
    # distribution-level: mean KL on natural training episodes vs on the scored edit endpoints
    if ln is not None:
        r["mean_KL_natural"] = float((pn * (np.log(pn + 1e-12) - ln)).sum(-1).mean())
    r["mean_KL_edit_endpoints"] = float((Pe * (lbe[np.arange(n), qq] - le[np.arange(n), qq])).sum(-1)[sel].mean())
    r["p95_abs_margin_err_edit"] = float(np.percentile(np.abs(ee[sel]), 95))
    # stratification
    st = {}
    bins = [(0.5, 1), (1, 2), (2, 5), (5, 1e9)]
    for lo, hi in bins:
        m = sel & (np.abs(dm) >= lo) & (np.abs(dm) < hi); st[f"|dm| {lo}-{hi}"] = (int(m.sum()), float(agree[m].mean()) if m.any() else None, float(cert[m].mean()) if m.any() else None)
    for nm, m in [("boundary", sel & boundary), ("non-boundary", sel & ~boundary), ("creates agreement", sel & creates), ("breaks agreement", sel & ~creates),
                  ("S1 and S2", sel & S2_conditional), ("S1 not S2 (conditionally unreachable)", sel & ~S2_conditional)]:
        st[nm] = (int(m.sum()), float(agree[m].mean()) if m.any() else None, float(cert[m].mean()) if m.any() else None)
    # unscored / off-support behaviour (no Bayes reference): model margin change magnitude
    st["off_support_mean_abs_dm_model"] = float(np.abs(dmh[~S1_marginal]).mean())
    r["strata(n, sign_agree, C1_cert)"] = st
    # error-vs-agreement: does the sign fail where the pointwise errors exceed |dm|?
    return r


for name, (kind, path, cls) in models.items():
    l0, le, ln = model_lp(kind, path, X0), model_lp(kind, path, Xe), model_lp(kind, path, Dn["X"])
    res[name] = analyse(name, l0, le, ln, cls); print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items() if not k.startswith("strata")}, flush=True)

# ------------------------------------------------------------------ constructed counterexamples
lbn = np.log(pn)
def nat_acc(l): return float((l.argmax(-1) == Dn["Y"]).mean())
# CX1: argmax-preserving margin distortion: logits = s(x) * centred Bayes log-probs with s(x) = 1 / (0.05 + m_top(x)^2)
def distort(lp):
    lp = np.nan_to_num(lp, nan=-50.0, neginf=-50.0); srt = np.sort(lp, -1); mt = srt[..., -1] - srt[..., -2]
    s = 1.0 / (0.05 + mt ** 2); z = (lp - lp.mean(-1, keepdims=True)) * s[..., None]
    return z - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - z.max(-1, keepdims=True)
l0, le, ln = distort(lb0), distort(lbe), distort(lbn)
res["CX1_argmax_preserving_distortion"] = analyse("CX1", l0, le, ln, "constructed"); res["CX1_argmax_preserving_distortion"]["natural_acc"] = nat_acc(ln)
res["CX1_argmax_preserving_distortion"]["bayes_natural_acc"] = nat_acc(lbn)
# CX2: exact Bayes everywhere except a low-density region (generator log-prob below the natural 2nd percentile), where the
#      candidate logits are randomly permuted.  Average natural KL is small; edits landing in that region are corrupted.
thr = np.percentile(logp_nat, 2)
rng = np.random.default_rng(7)
def corrupt(lp, logpx):
    lp = np.nan_to_num(lp, nan=-50.0, neginf=-50.0).copy(); low = logpx < thr
    for i in np.where(low)[0]:
        lp[i] = lp[i][:, rng.permutation(K)]
    return lp
# natural reference episodes need their own log P(x)
pnn = G.posterior(f4, Dn["X"]); lpx_n = pnn["logM"].sum(1); mm = lpx_n.max(-1); logp_nat2 = np.log(np.exp(lpx_n - mm[:, None]).mean(-1)) + mm
l0, le, ln = corrupt(lb0, logp_nat), corrupt(lbe, logp_edit), corrupt(lbn, logp_nat2)
res["CX2_low_density_corruption"] = analyse("CX2", l0, le, ln, "constructed"); res["CX2_low_density_corruption"]["natural_acc"] = nat_acc(ln)
res["CX2_low_density_corruption"]["frac_edit_endpoints_in_corrupted_region"] = float((logp_edit < thr)[scored].mean())
res["CX2_low_density_corruption"]["frac_natural_in_corrupted_region"] = float((logp_nat2 < thr).mean())
for k_ in ["CX1_argmax_preserving_distortion", "CX2_low_density_corruption"]:
    print(k_, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k_].items() if not a.startswith("strata")})
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp1_calibration.json"), "w"), indent=1, default=float)
