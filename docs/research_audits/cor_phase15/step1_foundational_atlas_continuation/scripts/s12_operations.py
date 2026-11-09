"""Seed 12 operation-level reconstruction (mandate §5), with seed 10 as control.  Site: input of the last layer
(resid_pre2) and the last layer's per-head outputs at readout q.  Counterfactual: cond pairs (X'_q = X_q, context
resampled on-manifold).  IIA = argmax agreement at q of the patched run with the counterfactual run (analysis.interchange_metrics).
Conditions (union masks over token positions; 'tag' = 2-D contrast subspace, 'full' = whole residual):
  tag@all36, full@other30, full@other30+tag@own6, full@all36, full@readout_q, full@readout_q+all36, full@all43 (sanity)
  per-head: head_out2[h] at readout q from the counterfactual run, single heads and pairs.
Mean-ablation reconciliation: with the tag mean-ablated at all 36 source tokens, agreement of the model's argmax with the
own-question Bayes answer (own_y) and with full Bayes, and the model's response to own-report ('target') edits.
Usage: s12_operations.py SEED OUT.json   (streams 7_700_000)"""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
seed, outp = int(sys.argv[1]), sys.argv[2]
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(LR, "checkpoints/replication", f"T4_M_K3_s{seed}_fromT1.npz")))
env = B.Env("T4", K=3, Q=6); Q, T, H = 6, 43, 4; L = len(p["layers"]) - 1; site = f"resid_pre{L}"; hsite = f"head_out{L}"; d = p["Eread"].shape[0]; f4 = G.Fam()
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n in (site, hsite): rec[n] = a
        return a
    o = F.forward(p, X, hk); return rec[site], rec[hsite], jax.nn.softmax(o["answer"], -1)
@jax.jit
def patch(X, src, P, mask, hsrc, hmask):
    def hk(n, a):
        if n == site: a = a + ((src - a) @ P) * mask[None, :, None]
        if n == hsite: a = a + (hsrc - a) * hmask[None, :, :, None]
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
Di = G.sample(f4, 800, np.random.default_rng(7_000_000))["X"]; ai = np.asarray(acts(jnp.asarray(Di))[0]); dk = []
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(7_000_010 + q)); dk.append((np.asarray(acts(jnp.asarray(Xp))[0]) - ai)[:, 6 * q:6 * q + 6].reshape(-1, d))
Pt = A.proj(A.contrast_basis(np.concatenate(dk), 2)).astype(np.float32); I = np.eye(d, dtype=np.float32)
STREAM = int(os.environ.get("STREAM", "7700000")); X = G.sample(f4, int(os.environ.get("NX", "1000")), np.random.default_rng(STREAM))["X"]; ra, ha, pb = map(np.asarray, acts(jnp.asarray(X)))
cf = [A.pairs(env, X, q, "cond", np.random.default_rng(7_700_100 + q)) for q in range(Q)]; cfa = [list(map(np.asarray, acts(jnp.asarray(c)))) for c in cf]
def run(spec):
    """spec(q) -> list of (P, positions) residual patches, and list of (head, position) head patches"""
    PE, PEp, PP = [], [], []
    for q in range(Q):
        rp, hp = spec(q); src = cfa[q][0]; out = None
        # compose multiple residual subspace patches by sequential application at the same site via a combined projector per token
        Pm = np.zeros((T, d, d), np.float32)
        for P_, pos in rp:
            for t in pos:   # projectors used are nested (tag subspace inside I): the union is the larger one
                Pm[t] = P_ if (not Pm[t].any() or np.allclose(P_, I)) else Pm[t]
        hm = np.zeros((H, T), np.float32)
        for h, t in hp: hm[h, t] = 1
        # token-wise projector: implement as patch with per-token P by looping distinct projectors
        Xb = jnp.asarray(X); base_src = jnp.asarray(src); res_ = None
        uniq = {}
        for t in range(T):
            key = Pm[t].tobytes()
            uniq.setdefault(key, (Pm[t], []))[1].append(t)
        # apply via a single forward: build effective target = a + sum over groups ((src-a) @ P_g) * mask_g
        groups = [(P_, ts) for P_, ts in uniq.values() if P_.any()]
        o = patch_multi(Xb, base_src, groups, jnp.asarray(cfa[q][1]), jnp.asarray(hm))
        PE.append(pb[:, q]); PEp.append(cfa[q][2][:, q]); PP.append(np.asarray(o)[:, q])
    m = A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP))); return dict(iia=float(m["iia"]), kl=float(m["kl_recovered"]))
def patch_multi(X, src, groups, hsrc, hm):
    Ps = jnp.asarray(np.stack([g[0] for g in groups]) if groups else np.zeros((1, d, d), np.float32))
    Ms = np.zeros((max(len(groups), 1), T), np.float32)
    for i, (_, ts) in enumerate(groups): Ms[i, ts] = 1
    return _pm(X, src, Ps, jnp.asarray(Ms), hsrc, hm)
@jax.jit
def _pm(X, src, Ps, Ms, hsrc, hm):
    def hk(n, a):
        if n == site: a = a + jnp.einsum("gntd,g t->ntd", jnp.einsum("ntd,gde->gnte", src - a, Ps), Ms) if False else a + sum_groups(src - a, Ps, Ms)
        if n == hsite: a = a + (hsrc - a) * hm[None, :, :, None]
        return a
    return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
def sum_groups(diff, Ps, Ms):
    return jnp.einsum("gnte,gt->nte", jnp.einsum("ntd,gde->gnte", diff, Ps), Ms)
own = lambda q: list(range(6 * q, 6 * q + 6)); oth = lambda q: [t for t in range(36) if t // 6 != q]; rq = lambda q: [36 + q]
ONLY = os.environ.get("ONLY")
C_ = {"tag@all36": lambda q: ([(Pt, list(range(36)))], []),
      "full@other30": lambda q: ([(I, oth(q))], []),
      "full@other30+tag@own6": lambda q: ([(I, oth(q)), (Pt, own(q))], []),
      "tag@own6": lambda q: ([(Pt, own(q))], []),
      "full@own6": lambda q: ([(I, own(q))], []),
      "full@all36": lambda q: ([(I, list(range(36)))], []),
      "full@readout_q": lambda q: ([(I, rq(q))], []),
      "full@readout_q+all36": lambda q: ([(I, list(range(36)) + rq(q))], []),
      "full@all43": lambda q: ([(I, list(range(43)))], [])}
for h in range(H): C_[f"head{h}@readout_q"] = (lambda h_: lambda q: ([], [(h_, 36 + q)]))(h)
for a_, b_ in itertools.combinations(range(H), 2): C_[f"heads{a_}{b_}@readout_q"] = (lambda a1, b1: lambda q: ([], [(a1, 36 + q), (b1, 36 + q)]))(a_, b_)
C_["all_heads@readout_q"] = lambda q: ([], [(h, 36 + q) for h in range(H)])
res = dict(seed=seed)
for k, fn in C_.items():
    if ONLY and k not in ONLY.split(","): continue
    res[k] = run(fn); print(k, res[k], flush=True)
# attention mass of each last-layer head at readout q on own-q sources / other-q sources / readouts+episode
@jax.jit
def pat(X):
    rec = {}
    def hk(n, a):
        if n == f"pattern{L}": rec["a"] = a
        return a
    F.forward(p, X, hk); return rec["a"]
A_ = np.asarray(pat(jnp.asarray(X)))
res["attn_mass"] = {f"head{h}": dict(own=float(np.mean([A_[:, h, 36 + q, 6 * q:6 * q + 6].sum(-1).mean() for q in range(Q)])),
                                     other=float(np.mean([A_[:, h, 36 + q, :36].sum(-1).mean() - A_[:, h, 36 + q, 6 * q:6 * q + 6].sum(-1).mean() for q in range(Q)])))
                    for h in range(H)}
# mean-ablation reconciliation
a_src = ra[:, :36]; mean_all = a_src.reshape(-1, d).mean(0); tgt = ra.copy(); tgt[:, :36] = mean_all
m36 = np.zeros((1, T), np.float32); m36[0, :36] = 1
oab = np.asarray(_pm(jnp.asarray(X), jnp.asarray(tgt), jnp.asarray(Pt[None]), jnp.asarray(m36), jnp.asarray(ha), jnp.zeros((H, T))))
post = G.posterior(f4, X); own_y, full_y = post["own_y"].argmax(-1), post["post_y"].argmax(-1)
res["meanabl"] = dict(agree_own_bayes=float((oab.argmax(-1) == own_y).mean()), agree_full_bayes=float((oab.argmax(-1) == full_y).mean()),
                      clean_agree_own=float((pb.argmax(-1) == own_y).mean()), clean_agree_full=float((pb.argmax(-1) == full_y).mean()),
                      KL_to_own_bayes=float((post["own_y"] * (np.log(post["own_y"] + 1e-12) - np.log(oab + 1e-12))).sum(-1).mean()),
                      KL_clean_to_own=float((post["own_y"] * (np.log(post["own_y"] + 1e-12) - np.log(pb + 1e-12))).sum(-1).mean()))
# per-head mean ablation of last-layer head outputs at the readouts (necessity of each channel); accuracy drop fraction
Y = G.sample(f4, int(os.environ.get("NX", "1000")), np.random.default_rng(STREAM))["Y"]
acc_clean = float((pb.argmax(-1) == Y).mean()); acc_own = float((own_y == Y).mean()); res["head_meanabl"] = dict(acc_clean=acc_clean, acc_own_bayes=acc_own)
hmean = ha.mean(0, keepdims=True)
for hs in [(0,), (1,), (2,), (3,), (0, 3), (0, 1), (2, 3)]:
    hm = np.zeros((H, T), np.float32)
    for h in hs: hm[h, 36:42] = 1
    o = np.asarray(_pm(jnp.asarray(X), jnp.asarray(ra), jnp.zeros((1, d, d)), jnp.zeros((1, T)), jnp.asarray(np.broadcast_to(hmean, ha.shape)), jnp.asarray(hm)))
    res["head_meanabl"]["heads" + "".join(map(str, hs))] = dict(acc=float((o.argmax(-1) == Y).mean()), drop_frac=float((acc_clean - (o.argmax(-1) == Y).mean()) / (acc_clean - acc_own)),
                                                                 agree_own_bayes=float((o.argmax(-1) == own_y).mean()))
print(res["attn_mass"], res["meanabl"], res["head_meanabl"])
json.dump(res, open(outp, "w"), indent=1)
