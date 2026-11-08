"""Direction A/F/G on saved T4 checkpoints (exploratory; no thresholds).  Usage: mechanism_atlas.py CKPT OUT.json [ds]
For a Transformer:
  0 forward-equivalence check (fwd.py vs nets.forward)
  1 input-level invariances (E2): argmax change rate at q under countpres / perm / cond edits of the other questions,
    with exact-Bayes reference rates
  2 head-subset analysis over all 16 subsets of last-layer heads:
      path    : patch ONLY the subset's head outputs at readout q from a cond pair (sufficiency of these paths)
      nec_out : resample-ablate the subset's head outputs at all readout tokens (natural necessity; accuracy)
      nec_pat : resample-ablate the subset's attention patterns, readout rows (natural necessity of the routing)
      med     : joint key+query tag patch with the subset's patterns frozen (mediation of the artificial patch)
    + Shapley values over heads for path-IIA and nec_out accuracy drop
  3 low-dimensional alignment of the last-layer attention output at readout q (contrast PCA, k = 2, 5, 10)
  4 reachability of patched states (Mahalanobis distance under the natural covariance) for the joint tag patch vs a
    mismatched-source patch vs an equal-norm random displacement
For DeepSets ('ds'): only 1.
Streams: identification 7,000,000; evaluation 7,100,000."""
import itertools, json, math, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import gen_family as G, fwd as F, nets as N, ckpt as C, analysis as A, bayes as B

ck, outp = sys.argv[1:3]; is_ds = len(sys.argv) > 3 and sys.argv[3] == "ds"
env = B.Env("T4", K=3, Q=6); f4 = G.Fam(); Q, K = 6, 3; T = 7 * Q + 1
res = dict(ckpt=ck)
Db = G.sample(f4, 1000, np.random.default_rng(7_100_000)); Xb, Yb = Db["X"], Db["Y"]
Dind = G.sample(f4, 1000, np.random.default_rng(7_100_001))["X"]


def countpres(X, q, rng):
    Xp = X.copy(); n = len(X)
    for k in range(Q):
        if k == q: continue
        for a, b in G.GROUPS:
            ag = X[:, k, a] == X[:, k, b]; v1 = rng.integers(0, K, n); off = rng.integers(1, K, n)
            Xp[:, k, a], Xp[:, k, b] = v1, np.where(ag, v1, (v1 + off) % K)
    return Xp


def mk_pairs(X, q, kind, seed):
    rng = np.random.default_rng(seed)
    return countpres(X, q, rng) if kind == "countpres" else A.pairs(env, X, q, kind, rng)


if is_ds:
    p = jax.tree_util.tree_map(jnp.asarray, C.load_base(ck)); fb = N.make_forward("deepsets", 3)
    run = jax.jit(lambda X: jax.nn.softmax(fb(p, X)["answer"], -1))
else:
    p = jax.tree_util.tree_map(jnp.asarray, C.load(ck))
    run = jax.jit(lambda X: jax.nn.softmax(F.forward(p, X)["answer"], -1))
    ref = np.asarray(jax.nn.softmax(N.forward(p, jnp.asarray(Xb))["answer"], -1))
    res["fwd_equivalence_max_abs"] = float(np.abs(np.asarray(run(jnp.asarray(Xb))) - ref).max())
pb = np.asarray(run(jnp.asarray(Xb))); post_b = G.posterior(f4, Xb)["post_y"]

# ---------------------------------------------------------------- 1 input-level invariances
inv = {}
for kind in ["cond", "perm", "countpres"]:
    ch_m, ch_b, und = [], [], 0
    for q in range(Q):
        Xp = mk_pairs(Xb, q, kind, 7_100_100 + 10 * ["cond", "perm", "countpres"].index(kind) + q)
        pp = np.asarray(run(jnp.asarray(Xp))); bp = G.posterior(f4, Xp)["post_y"]
        ch_m.append(pp[:, q].argmax(-1) != pb[:, q].argmax(-1))
        okb = np.isfinite(bp[:, q]).all(-1); und += int((~okb).sum())
        ch_b.append(np.where(okb, np.nan_to_num(bp[:, q], nan=0).argmax(-1) != post_b[:, q].argmax(-1), np.nan))
    inv[kind] = dict(model_change_rate=float(np.mean(np.concatenate(ch_m))), bayes_change_rate=float(np.nanmean(np.concatenate(ch_b))),
                     bayes_undefined_frac=und / (Q * len(Xb)))
res["input_invariances"] = inv
if is_ds:
    json.dump(res, open(outp, "w"), indent=1); print(json.dumps(res)); sys.exit()

L = len(p["layers"]); H = p["layers"][0]["WQ"].shape[0]; d = p["Eread"].shape[0]; Ll = L - 1
site, Lp, Lh = f"resid_pre{Ll}", f"pattern{Ll}", f"head_out{Ll}"
names = (site, Lp, Lh)


@partial(jax.jit, static_argnames=())
def acts(X):
    rec = {}
    def hk(n, a):
        if n in names: rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec, jax.nn.softmax(o["answer"], -1)


def acts_b(X):
    R = [acts(jnp.asarray(X[i:i + 500])) for i in range(0, len(X), 500)]
    return {k: np.concatenate([np.asarray(r[0][k]) for r in R]) for k in names}, np.concatenate([np.asarray(r[1]) for r in R])


@jax.jit
def surgery(X, src_site, mk, Pk, mq, Pq, pat_src, pat_hm, pat_rowmask, ho_src, ho_hm, ho_rowmask, ho_P):
    def hk(n, a):
        if n == site:
            dl = src_site - a; a = a + (dl @ Pk) * mk[None, :, None] + (dl @ Pq) * mq[None, :, None]
        if n == Lp:
            m = pat_hm[None, :, None, None] * pat_rowmask[None, None, :, None]; a = a * (1 - m) + pat_src * m
        if n == Lh:
            m = ho_hm[None, :, None, None] * ho_rowmask[None, None, :, None]
            a = a + ((ho_src - a) @ ho_P) * m
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)


Z, I = np.zeros((d, d)), np.eye(d)
zT, zH = np.zeros(T), np.zeros(H)
cb, _ = acts_b(Xb)


def S(X, src_site=None, mk=zT, Pk=Z, mq=zT, Pq=Z, pat_src=None, pat_hm=zH, pat_row=zT, ho_src=None, ho_hm=zH, ho_row=zT, ho_P=I):
    src_site = cb[site] if src_site is None else src_site; pat_src = cb[Lp] if pat_src is None else pat_src; ho_src = cb[Lh] if ho_src is None else ho_src
    f32 = lambda z: jnp.asarray(z, jnp.float32)
    return np.concatenate([np.asarray(surgery(jnp.asarray(X[i:i + 500]), f32(src_site[i:i + 500]), f32(mk), f32(Pk), f32(mq), f32(Pq), f32(pat_src[i:i + 500]),
                                              f32(pat_hm), f32(pat_row), f32(ho_src[i:i + 500]), f32(ho_hm), f32(ho_row), f32(ho_P))) for i in range(0, len(X), 500)])


# identification (inputs only): key/query contrast bases as in the replication protocol, own stream
Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ci, _ = acts_b(Di)
dk, dq, dh = [], [], []
srcs = {}
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); cp, _ = acts_b(Xp); D = cp[site] - ci[site]
    dk.append(D[:, 6 * q:6 * q + 6].reshape(-1, d)); dq.append(D[:, 6 * Q + q]); dh.append((cp[Lh] - ci[Lh])[:, :, 6 * Q + q].sum(1))
Pk = A.proj(A.contrast_basis(np.concatenate(dk), 2)); Pq = A.proj(A.contrast_basis(np.concatenate(dq), 5))
for q in range(Q):
    Xp = A.pairs(env, Xb, q, "cond", np.random.default_rng(7_100_200 + q)); cp, pp = acts_b(Xp); srcs[q] = (Xp, cp, pp)
cind, _ = acts_b(Dind)


def rowq(q): m = np.zeros(T); m[6 * Q + q] = 1; return m
def srcq(q): m = np.zeros(T); m[6 * q:6 * q + 6] = 1; return m
rd_all = np.zeros(T); rd_all[6 * Q:7 * Q] = 1
acc_clean = float((pb.argmax(-1) == Yb).mean()); own = float((post_b.argmax(-1) == Yb).mean())
own = float((G.posterior(f4, Xb)["own_y"].argmax(-1) == Yb).mean())


def iia_over_q(fn):
    PE, PEp, PP = [], [], []
    for q in range(Q):
        Xp, cp, pp = srcs[q]; pt = fn(q, cp); PE.append(pb[:, q]); PEp.append(pp[:, q]); PP.append(pt[:, q])
    m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); return dict(iia=m["iia"], kl=m["kl_recovered"])


subsets = [tuple(s) for r in range(H + 1) for s in itertools.combinations(range(H), r)]
tab = {}
for sub in subsets:
    hm = np.zeros(H); hm[list(sub)] = 1
    path = iia_over_q(lambda q, cp: S(Xb, ho_src=cp[Lh], ho_hm=hm, ho_row=rowq(q))) if sub else dict(iia=0.0, kl=0.0)
    nec_out = float((S(Xb, ho_src=cind[Lh], ho_hm=hm, ho_row=rd_all).argmax(-1) == Yb).mean())
    nec_pat = float((S(Xb, pat_src=cind[Lp], pat_hm=hm, pat_row=rd_all).argmax(-1) == Yb).mean())
    med = iia_over_q(lambda q, cp: S(Xb, src_site=cp[site], mk=srcq(q), Pk=Pk, mq=rowq(q), Pq=Pq, pat_hm=hm, pat_row=np.ones(T)))
    tab[",".join(map(str, sub)) or "none"] = dict(path_iia=path["iia"], path_kl=path["kl"], nec_out_acc=nec_out, nec_pat_acc=nec_pat, med_iia=med["iia"])
res["acc_clean"] = acc_clean; res["own_question_bayes"] = own; res["head_subsets"] = tab


def shapley(val):
    phi = np.zeros(H)
    for h in range(H):
        for sub in subsets:
            if h in sub: continue
            w = math.factorial(len(sub)) * math.factorial(H - len(sub) - 1) / math.factorial(H)
            a = ",".join(map(str, sub)) or "none"; b = ",".join(map(str, sorted(sub + (h,))))
            phi[h] += w * (val(b) - val(a))
    return phi.tolist()
res["shapley_path_iia"] = shapley(lambda s: tab[s]["path_iia"] if s != "none" else 0.0)
res["shapley_nec_out_drop"] = shapley(lambda s: acc_clean - tab[s]["nec_out_acc"])
res["shapley_nec_pat_drop"] = shapley(lambda s: acc_clean - tab[s]["nec_pat_acc"])

# ---------------------------------------------------------------- 3 low-dim alignment of last-layer attention output at readout q
Dh = np.concatenate(dh); al = {}
for k_ in [2, 5, 10]:
    Ph = A.proj(A.contrast_basis(Dh, k_))
    al[str(k_)] = iia_over_q(lambda q, cp: S(Xb, ho_src=cp[Lh], ho_hm=np.ones(H), ho_row=rowq(q), ho_P=Ph))
al["full"] = iia_over_q(lambda q, cp: S(Xb, ho_src=cp[Lh], ho_hm=np.ones(H), ho_row=rowq(q)))
res["attn_out_alignment"] = al

# ---------------------------------------------------------------- 4 reachability of the joint tag patch (source tokens of q)
nat = cb[site][:, :6 * Q].reshape(-1, d); mu = nat.mean(0); Cv = np.cov(nat.T) + 1e-3 * np.eye(d); Ci = np.linalg.inv(Cv)
def maha(Z_): z = Z_ - mu; return np.sqrt(np.einsum("nd,de,ne->n", z, Ci, z))
q = 0; Xp, cp, _ = srcs[q]
base = cb[site][:, 6 * q:6 * q + 6].reshape(-1, d)
patched = base + (cp[site][:, 6 * q:6 * q + 6].reshape(-1, d) - base) @ Pk
mis = cp[site][np.random.default_rng(1).permutation(len(Xb))][:, 6 * q:6 * q + 6].reshape(-1, d)
patched_mis = base + (mis - base) @ Pk
disp = np.linalg.norm(patched - base, axis=1, keepdims=True)
rnd = np.random.default_rng(2).normal(size=base.shape); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
patched_rnd = base + rnd * disp
nd = maha(nat); thr = np.percentile(nd, 99)
res["reachability"] = dict(natural_p99=float(thr), frac_beyond_p99=dict(cond_patch=float((maha(patched) > thr).mean()), mismatch_patch=float((maha(patched_mis) > thr).mean()),
                           random_equal_norm=float((maha(patched_rnd) > thr).mean())), median=dict(natural=float(np.median(nd)), cond_patch=float(np.median(maha(patched))),
                           mismatch_patch=float(np.median(maha(patched_mis))), random_equal_norm=float(np.median(maha(patched_rnd)))))
json.dump(res, open(outp, "w"), indent=1)
print(ck.split("/")[-1], "fwd_eq", res["fwd_equivalence_max_abs"], "acc", round(acc_clean, 4), "own", round(own, 4))
print("  invariances", {k: {a: round(b, 3) for a, b in v.items()} for k, v in inv.items()})
for s_, v in tab.items(): print("  heads", s_, {a: round(b, 3) for a, b in v.items()})
print("  shapley path", [round(x, 3) for x in res["shapley_path_iia"]], "nec_out", [round(x, 3) for x in res["shapley_nec_out_drop"]], "nec_pat", [round(x, 3) for x in res["shapley_nec_pat_drop"]])
print("  attn_out alignment", {k: round(v["iia"], 3) for k, v in al.items()}); print("  reach", res["reachability"])
