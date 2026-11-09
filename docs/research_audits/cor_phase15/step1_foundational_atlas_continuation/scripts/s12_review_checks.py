"""Review-response checks for seed 12 / seed 10 (R2-causal F2, F7): (a) interchange of all 36 source tokens + all 6 readout
tokens (+ episode token) at the last-layer input; (b) content-only interchange at other30 (complement of the tag subspace);
(c) tag mean-ablation (pooled mean, all 36 source tokens) with the last layer's attention patterns frozen at clean values.
Same streams as s12_operations.py (7_700_000).  Usage: s12_review_checks.py SEED OUT.json"""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp, gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
seed, outp = int(sys.argv[1]), sys.argv[2]
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(LR, "checkpoints/replication", f"T4_M_K3_s{seed}_fromT1.npz")))
env = B.Env("T4", K=3, Q=6); Q, T = 6, 43; d = p["Eread"].shape[0]; f4 = G.Fam()
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n in ("resid_pre2", "pattern2"): rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec["resid_pre2"], rec["pattern2"], jax.nn.softmax(o["answer"], -1)
@jax.jit
def run(X, src, P, mask, pat, use_pat):
    def hk(n, a):
        if n == "resid_pre2": a = a + ((src - a) @ P) * mask[None, :, None]
        if n == "pattern2": a = use_pat * pat + (1 - use_pat) * a
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ai = np.asarray(acts(jnp.asarray(Di))[0]); dk = []
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); dk.append((np.asarray(acts(jnp.asarray(Xp))[0]) - ai)[:, 6 * q:6 * q + 6].reshape(-1, d))
Pt = A.proj(A.contrast_basis(np.concatenate(dk), 2)).astype(np.float32); I = np.eye(d, dtype=np.float32)
D = G.sample(f4, 1000, np.random.default_rng(7_700_000)); X, Y = D["X"], D["Y"]; ra, pa, pb = map(np.asarray, acts(jnp.asarray(X)))
cfa = [list(map(np.asarray, acts(jnp.asarray(A.pairs(env, X, q, "cond", np.random.default_rng(7_700_100 + q)))))) for q in range(Q)]
res = dict(seed=seed)
def iia(P_, maskfn):
    PE, PEp, PP = [], [], []
    for q in range(Q):
        m = maskfn(q); o = run(jnp.asarray(X), jnp.asarray(cfa[q][0]), jnp.asarray(P_), jnp.asarray(m), jnp.asarray(pa), 0.0)
        PE.append(pb[:, q]); PEp.append(cfa[q][2][:, q]); PP.append(np.asarray(o)[:, q])
    mm = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); return dict(iia=float(mm["iia"]), kl_median=float(mm["kl_recovered"]))
def mk(pos):
    m = np.zeros(T, np.float32); m[pos] = 1; return m
res["full@all36+readout_q+episode"] = iia(I, lambda q: mk(list(range(36)) + [36 + q, 42]))
res["full@all36+all6readouts"] = iia(I, lambda q: mk(list(range(42))))
res["full@all36+all6readouts+episode"] = iia(I, lambda q: mk(list(range(43))))
res["content@other30 (I - Pt)"] = iia(I - Pt, lambda q: mk([t for t in range(36) if t // 6 != q]))
res["tag@other30"] = iia(Pt, lambda q: mk([t for t in range(36) if t // 6 != q]))
post = G.posterior(f4, X); acc_own = float((post["own_y"].argmax(-1) == Y).mean()); acc_clean = float((pb.argmax(-1) == Y).mean())
tgt = ra.copy(); tgt[:, :36] = ra[:, :36].reshape(-1, d).mean(0)
for nm, up in [("tag_meanabl_live_patterns", 0.0), ("tag_meanabl_frozen_patterns", 1.0)]:
    o = np.asarray(run(jnp.asarray(X), jnp.asarray(tgt), jnp.asarray(Pt), jnp.asarray(mk(list(range(36)))), jnp.asarray(pa), up))
    res[nm] = dict(acc=float((o.argmax(-1) == Y).mean()), drop_frac=float((acc_clean - (o.argmax(-1) == Y).mean()) / (acc_clean - acc_own)))
res.update(acc_clean=acc_clean, acc_own=acc_own); print(json.dumps(res)); json.dump(res, open(outp, "w"), indent=1)
