"""Re-analyses requested by review R2 (in-package versions of the reviewer's checks).
(a) Reachability with PER-POSITION covariance (36 source-token positions), held-out natural reference.
(b) Relation-specific natural necessity: resample-ablate all last-layer head outputs at readout q with outputs from a
    cond PARTNER episode (same X_q, other questions resampled from P(X_{-q}|X_q)) instead of an independent episode,
    so that only the cross-question (relation) information is removed.
Streams 7,700,000.  Usage: fixes_r2.py CKPT OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
ck, outp = sys.argv[1:3]
p = jax.tree_util.tree_map(jnp.asarray, C.load(ck)); env = B.Env("T4", K=3, Q=6); Q, T = 6, 43
L = len(p["layers"]) - 1; site, Lh = f"resid_pre{L}", f"head_out{L}"; d = p["Eread"].shape[0]; H = p["layers"][L]["WQ"].shape[0]; f4 = G.Fam()
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n in (site, Lh): rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec, jax.nn.softmax(o["answer"], -1)
@jax.jit
def ho_patch(X, src, row):
    def hk(n, a):
        if n == Lh: a = a + (src - a) * row[None, None, :, None]
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
def A_(X):
    R = [acts(jnp.asarray(X[i:i + 500])) for i in range(0, len(X), 500)]
    return {k: np.concatenate([np.asarray(r[0][k]) for r in R]) for k in (site, Lh)}, np.concatenate([np.asarray(r[1]) for r in R])
# identification of the key tag (same as elsewhere)
Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ai, _ = A_(Di); dk = []
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); ap, _ = A_(Xp); dk.append((ap[site] - ai[site])[:, 6 * q:6 * q + 6].reshape(-1, d))
P = A.proj(A.contrast_basis(np.concatenate(dk), 2))
res = dict(ckpt=ck)
# (a) reachability, per-position covariance
Dn = G.sample(f4, 3000, np.random.default_rng(7_700_000))["X"]; an, _ = A_(Dn); nat = an[site][:, :6 * Q]           # N,36,d
Dh = G.sample(f4, 1000, np.random.default_rng(7_700_001)); Xb = Dh["X"]; ab, pb = A_(Xb)
mus = nat.mean(0); Cis = [np.linalg.inv(np.cov(nat[:, t].T) + 1e-3 * np.eye(d)) for t in range(6 * Q)]
def maha_t(Z, t): z = Z - mus[t]; return np.sqrt(np.einsum("nd,de,ne->n", z, Cis[t], z))
q = 0; Xp = A.pairs(env, Xb, q, "cond", np.random.default_rng(7_700_100)); ap, _ = A_(Xp)
mis = ap[site][np.random.default_rng(1).permutation(len(Xb))]
out = {"natural_heldout": [], "cond_patch": [], "mismatch_patch": [], "random_equal_norm": []}
for s in range(6):
    t = 6 * q + s; thr = np.percentile(maha_t(nat[:, t], t), 99)
    base = ab[site][:, t]; pat = base + (ap[site][:, t] - base) @ P; pm = base + (mis[:, t] - base) @ P
    rnd = np.random.default_rng(2 + s).normal(size=base.shape); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    pr = base + rnd * np.linalg.norm(pat - base, axis=1, keepdims=True)
    for k_, Z in [("natural_heldout", base), ("cond_patch", pat), ("mismatch_patch", pm), ("random_equal_norm", pr)]:
        out[k_].append(float((maha_t(Z, t) > thr).mean()))
res["reachability_per_position_frac_beyond_p99"] = {k: float(np.mean(v)) for k, v in out.items()}
# (b) relation-specific natural necessity (partner resample of all last-layer head outputs at readout q)
acc_clean = float((pb.argmax(-1) == Dh["Y"]).mean()); own = float((G.posterior(f4, Xb)["own_y"].argmax(-1) == Dh["Y"]).mean())
Dind = G.sample(f4, 1000, np.random.default_rng(7_700_002))["X"]; aind, _ = A_(Dind)
acc_partner, acc_indep = [], []
for q in range(Q):
    Xp = A.pairs(env, Xb, q, "cond", np.random.default_rng(7_700_200 + q)); ap, _ = A_(Xp); row = np.zeros(T); row[6 * Q + q] = 1
    o1 = np.asarray(ho_patch(jnp.asarray(Xb), jnp.asarray(ap[Lh]), jnp.asarray(row, jnp.float32)))
    o2 = np.asarray(ho_patch(jnp.asarray(Xb), jnp.asarray(aind[Lh]), jnp.asarray(row, jnp.float32)))
    acc_partner.append((o1[:, q].argmax(-1) == Dh["Y"][:, q]).mean()); acc_indep.append((o2[:, q].argmax(-1) == Dh["Y"][:, q]).mean())
res["necessity_all_lastlayer_heads"] = dict(acc_clean=acc_clean, own_question_bayes=own, acc_partner_resample=float(np.mean(acc_partner)),
                                            acc_independent_resample=float(np.mean(acc_indep)), relation_share_drop=acc_clean - float(np.mean(acc_partner)))
json.dump(res, open(outp, "w"), indent=1); print(ck.split("/")[-1], json.dumps({k: v for k, v in res.items() if k != "ckpt"}))
