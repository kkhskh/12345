"""R2 reviewer null controls for the per-slot source-encoder alignment (CA1), on one checkpoint.
Same setup as src_enc.py (resid_pre2, additive delta at the 36 source tokens, 'cond' pairs, IIA on informative pairs),
with fewer episodes. Variants (all per-slot, 36 separate linear encoders):
  slot|loo                : reference (leave-q one-hot counts, 21 features), as in src_enc.py
  null|permfit            : encoder fitted to counts of a PERMUTED episode pairing (counts of episode pi(i) vs acts of i)
  null|mismatch_target    : true encoder, but the counterfactual counts are taken from a random OTHER pair (wrong target)
  null|wrong_groups       : counts recomputed with the wrong source pairing [(0,2),(1,4),(3,5)] (fit and apply)
  coarse|argmax_group     : 3-valued one-hot of argmax_g leave-q count (a much coarser function of counts)
  oracle|answer_label     : one-hot of the model's own answer at question t//6 (label leakage; expressiveness ceiling)
  full_source_patch       : counterfactual activations at all 36 source tokens
Also reports ||delta|| / ||true source patch|| and cosine for slot|loo on informative pairs."""
import os, sys, json
B0 = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, B0 + "/src")
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import fwd3 as F3, ckpt as C, gen_family as G, analysis as A, bayes as B
ck = sys.argv[1]; NE = int(sys.argv[2]) if len(sys.argv) > 2 else 600; NF = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
Q, T, K = 6, 43, 3
p = jax.tree_util.tree_map(jnp.asarray, C.load(os.path.join(B0, ck))); p.pop("WX", None)
env = B.Env("T4", K=3, Q=6); f4 = G.Fam(); d = p["Eread"].shape[0]
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n == "resid_pre2": rec["r"] = a
        return a
    o = F3.forward(p, X, hk); return rec["r"], jax.nn.softmax(o["answer"], -1)
@jax.jit
def added(X, delta):
    return jax.nn.softmax(F3.forward(p, X, lambda n, a: a + delta if n == "resid_pre2" else a)["answer"], -1)
WRONG = [(0, 2), (1, 4), (3, 5)]
def agree(Z, pr=G.GROUPS): return np.stack([(Z[:, :, a] == Z[:, :, b]) for a, b in pr], -1).astype(int)
def loo(Z, pr=G.GROUPS): Ag = agree(Z, pr); return Ag.sum(1, keepdims=True) - Ag
def oh(n, m=7): return np.eye(m)[n].reshape(n.shape[:-1] + (n.shape[-1] * m,))
def fit(F, Aa): return np.linalg.lstsq(np.c_[np.ones(len(F)), F], Aa, rcond=None)[0][1:]
rng0 = np.random.default_rng(9_910_000)
Dn = G.sample(f4, NF, rng0)["X"]; an_, Pn = map(np.asarray, acts(jnp.asarray(Dn)))
def feats(Z, P):
    L = loo(Z)
    return {"slot|loo": oh(L), "null|wrong_groups": oh(loo(Z, WRONG)),
            "coarse|argmax_group": np.eye(3)[L.argmax(-1)], "oracle|answer_label": np.eye(3)[P.argmax(-1)]}
Fn = feats(Dn, Pn); perm = np.random.default_rng(1).permutation(NF)
enc = {nm: [fit(F[:, t // 6], an_[:, t]) for t in range(36)] for nm, F in Fn.items()}
enc["null|permfit"] = [fit(Fn["slot|loo"][perm, t // 6], an_[:, t]) for t in range(36)]
D = G.sample(f4, NE, np.random.default_rng(9_900_000)); X = D["X"]; a0, P0 = map(np.asarray, acts(jnp.asarray(X)))
res, geo = {}, {"norm_ratio": [], "cos": []}
def slot_delta(F0, F1, E):
    dl = np.zeros((len(F0), T, d), np.float32)
    for t in range(36): dl[:, t] = (F1[:, t // 6] - F0[:, t // 6]) @ E[t]
    return dl
for q in range(Q):
    Xp = A.pairs(env, X, q, "cond", np.random.default_rng(9_900_100 + q)); ap, pp = map(np.asarray, acts(jnp.asarray(Xp)))
    inf = P0[:, q].argmax(-1) != pp[:, q].argmax(-1)
    F0, F1 = feats(X, P0), feats(Xp, pp)
    deltas = {nm: slot_delta(F0[nm], F1[nm], enc[nm]) for nm in F0}
    deltas["null|permfit"] = slot_delta(F0["slot|loo"], F1["slot|loo"], enc["null|permfit"])
    sh = np.random.default_rng(77 + q).permutation(len(X))
    deltas["null|mismatch_target"] = slot_delta(F0["slot|loo"], F1["slot|loo"][sh], enc["slot|loo"])
    dl = np.zeros((len(X), T, d), np.float32); dl[:, :36] = ap[:, :36] - a0[:, :36]; deltas["full_source_patch"] = dl
    u = deltas["slot|loo"][inf, :36].reshape(inf.sum(), -1); v = dl[inf, :36].reshape(inf.sum(), -1)
    geo["norm_ratio"].append(np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1))
    nu = np.linalg.norm(u, axis=1); ok = nu > 0; geo.setdefault("frac_inf_counts_unchanged", []).append(~ok)
    geo["cos"].append((u[ok] * v[ok]).sum(1) / nu[ok] / np.linalg.norm(v[ok], axis=1))
    for nm, dl in deltas.items():
        o = np.asarray(added(jnp.asarray(X), jnp.asarray(dl)))
        tgt = pp[:, q].argmax(-1)
        if nm == "null|mismatch_target":  # success = matching the mismatched pair's answer would be unfair; report both
            res.setdefault(nm + "(vs true cf)", []).append((o[:, q].argmax(-1) == tgt)[inf])
            res.setdefault(nm + "(any flip)", []).append((o[:, q].argmax(-1) != P0[:, q].argmax(-1))[inf])
        else:
            res.setdefault(nm, []).append((o[:, q].argmax(-1) == tgt)[inf])
out = {k: (round(float(np.concatenate(v).mean()), 3), int(np.concatenate(v).size)) for k, v in res.items()}
out["geom_slot_loo_vs_true_patch"] = {k: [round(float(np.median(np.concatenate(v).astype(float))), 3),
                                          round(float(np.mean(np.concatenate(v))), 3)] for k, v in geo.items()}
print(ck, "NE", NE, "NF", NF, json.dumps(out, indent=0))
