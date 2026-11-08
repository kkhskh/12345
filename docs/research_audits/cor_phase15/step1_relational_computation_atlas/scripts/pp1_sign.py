"""PP1 (frozen in PROSPECTIVE_PREDICTIONS_AND_FALSIFIERS.md): sign of candidate-margin change under natural single-report
edits in other questions, model vs exact T4-Bayes vs monotone T1-Bayes plug-in. Usage: pp1_sign.py OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C

f4, f1 = G.Fam(pc=1.0, rho=1.0), G.Fam(pc=0.0)
NB = int(os.environ.get("PP1_NB", "500"))
d = G.sample(f4, NB, np.random.default_rng(6_100_000)); X0 = d["X"]; Q = 6; K = 3
# build all edits
base_idx, qs, ks, gs, Xe = [], [], [], [], []
for q in range(Q):
    for k in range(Q):
        if k == q: continue
        for g, (a, b) in enumerate(G.GROUPS):
            X = X0.copy()
            agree = X[:, k, a] == X[:, k, b]
            X[:, k, b] = np.where(agree, (X[:, k, a] + 1) % K, X[:, k, a])
            Xe.append(X); base_idx.append(np.arange(NB)); qs.append(np.full(NB, q)); ks.append(np.full(NB, k)); gs.append(np.full(NB, g))
Xe = np.concatenate(Xe); bi = np.concatenate(base_idx); qq = np.concatenate(qs); kk = np.concatenate(ks); gg = np.concatenate(gs)
n_edit = len(Xe)


def loo_count(X, q, g):
    a, b = G.GROUPS[g]; A = (X[:, :, a] == X[:, :, b]).astype(int); return A.sum(1) - A[np.arange(len(X)), q]
cnt0 = np.array([loo_count(X0[bi[i:i + 1]], qq[i], gg[i])[0] for i in range(0)]) if False else None
A0 = np.stack([(X0[:, :, a] == X0[:, :, b]) for a, b in G.GROUPS], -1).astype(int)       # NB,Q,3
Ae = np.stack([(Xe[:, :, a] == Xe[:, :, b]) for a, b in G.GROUPS], -1).astype(int)
c_before = A0[bi].sum(1)[np.arange(n_edit), gg] - A0[bi, qq, gg]
c_after = Ae.sum(1)[np.arange(n_edit), gg] - Ae[np.arange(n_edit), qq, gg]
boundary = ((c_before == 4) & (c_after == 5)) | ((c_before == 5) & (c_after == 4))


def bayes_logp(f, X):
    p = G.posterior(f, X)["post_y"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p)
lb4_0, lb4_e = bayes_logp(f4, X0), bayes_logp(f4, Xe)
lb1_0, lb1_e = bayes_logp(f1, X0), bayes_logp(f1, Xe)
# Bayes top-two candidates on the base episode at q
order = np.argsort(-np.nan_to_num(lb4_0, nan=-1e9), -1)
c1 = order[bi, qq, 0]; c2 = order[bi, qq, 1]
def margin(lp, idx_rows, q_, a_, b_):
    return lp[idx_rows, q_, a_] - lp[idx_rows, q_, b_]
dm4 = margin(lb4_e, np.arange(n_edit), qq, c1, c2) - margin(lb4_0, bi, qq, c1, c2)
dm1 = margin(lb1_e, np.arange(n_edit), qq, c1, c2) - margin(lb1_0, bi, qq, c1, c2)
defined = np.isfinite(dm4)


def model_logp(kind, path, X):
    if kind == "tf":
        p = jax.tree_util.tree_map(jnp.asarray, C.load(path)); fwd = lambda Xb: N.forward(p, Xb)
    else:
        p = jax.tree_util.tree_map(jnp.asarray, C.load_base(path)); fb = N.make_forward("deepsets", 3); fwd = lambda Xb: fb(p, Xb)
    jf = jax.jit(lambda Xb: jax.nn.log_softmax(fwd(Xb)["answer"], -1))
    return np.concatenate([np.asarray(jf(jnp.asarray(X[i:i + 3000]))) for i in range(0, len(X), 3000)])


models = {f"tf_curr_s{s}": ("tf", os.path.join(LR, "checkpoints/posthoc_curriculum", f"T4_M_K3_s{s}_fromT1.npz")) for s in [0, 1, 2]}
models.update({f"tf_curr_s{s}": ("tf", os.path.join(LR, "checkpoints/replication", f"T4_M_K3_s{s}_fromT1.npz")) for s in [10, 11, 12, 13]})
models.update({f"deepsets_s{s}": ("ds", os.path.join(LR, "checkpoints/main", f"T4_M_K3_s{s}_deepsets.npz")) for s in [0, 1, 2]})
res = dict(n_base=NB, n_edits=int(n_edit), n_bayes_undefined=int((~defined).sum()), n_boundary=int((boundary & defined).sum()))
sel = defined & (np.abs(dm4) > 0.5); selb = sel & boundary
res["n_scored"] = int(sel.sum()); res["n_scored_boundary"] = int(selb.sum())
res["T1plugin_vs_T4bayes_sign_agreement"] = dict(all=float((np.sign(dm1[sel]) == np.sign(dm4[sel])).mean()), boundary=float((np.sign(dm1[selb]) == np.sign(dm4[selb])).mean()))
for name, (kind, path) in models.items():
    l0, le = model_logp(kind, path, X0), model_logp(kind, path, Xe)
    dmm = margin(le, np.arange(n_edit), qq, c1, c2) - margin(l0, bi, qq, c1, c2)
    r = dict(sign_vs_T4bayes=float((np.sign(dmm[sel]) == np.sign(dm4[sel])).mean()),
             boundary_sign_vs_T4bayes=float((np.sign(dmm[selb]) == np.sign(dm4[selb])).mean()),
             boundary_sign_vs_T1plugin=float((np.sign(dmm[selb]) == np.sign(dm1[selb])).mean()),
             corr_dm_vs_T4bayes=float(np.corrcoef(dmm[sel], dm4[sel])[0, 1]),
             offsupport_edits_model_margin_change_mean_abs=float(np.abs(dmm[~defined]).mean()) if (~defined).any() else None)
    # review R1 additions (reported, not criteria): unfiltered agreement, and episode-cluster bootstrap CI for the boundary rate
    alld = defined & (dm4 != 0); allb = alld & boundary
    r["unfiltered_sign_vs_T4bayes"] = float((np.sign(dmm[alld]) == np.sign(dm4[alld])).mean())
    r["unfiltered_boundary_sign_vs_T4bayes"] = float((np.sign(dmm[allb]) == np.sign(dm4[allb])).mean())
    r["unfiltered_boundary_sign_vs_T1plugin"] = float((np.sign(dmm[allb]) == np.sign(dm1[allb])).mean())
    hit = (np.sign(dmm) == np.sign(dm4)).astype(float); rb_ = np.random.default_rng(9); bs = []
    ep = bi[selb]; hv = hit[selb]; ue = np.unique(ep)
    per = {e: hv[ep == e] for e in ue}
    for _ in range(300):
        smp = rb_.choice(ue, len(ue)); v = np.concatenate([per[e] for e in smp]); bs.append(v.mean())
    r["boundary_sign_vs_T4bayes_cluster_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    r["PP1_pass"] = bool(r["sign_vs_T4bayes"] >= 0.9 and r["boundary_sign_vs_T4bayes"] >= 0.8 and r["boundary_sign_vs_T1plugin"] <= 0.5)
    res[name] = r; print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print({k: v for k, v in res.items() if not isinstance(v, dict)}, res["T1plugin_vs_T4bayes_sign_agreement"])
