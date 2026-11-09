"""Reviewer check: is the T1 source-token abstract-count negative result robust to encoder specification?
Variants: (a) original shared-slot linear encoder, leave-k counts; (b) per-(k,s) slot linear encoder, leave-k counts;
(c) per-slot encoder, FULL-episode counts; (d) per-slot encoder on 'own-group' count only; (e) per-slot joint-count
lookup (conditional mean over the 216 count configs, backoff to (b)); (f) per-slot encoder, only q's own 6 tokens patched."""
import os, sys, json
B0 = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, B0 + "/src")
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import fwd3 as F3, ckpt as C, gen_family as G, analysis as A, bayes as B
ck = sys.argv[1]; NE = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
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
def agree(Z): return np.stack([(Z[:, :, a] == Z[:, :, b]) for a, b in G.GROUPS], -1).astype(int)
def loo(Z): Ag = agree(Z); return Ag.sum(1, keepdims=True) - Ag
def full(Z): Ag = agree(Z); return np.broadcast_to(Ag.sum(1, keepdims=True), Ag.shape)
def oh(n, m=7): return np.eye(m)[n].reshape(n.shape[:-1] + (n.shape[-1] * m,))
Dn = G.sample(f4, 3000, np.random.default_rng(9_910_000))["X"]; an = np.asarray(acts(jnp.asarray(Dn))[0])
def fit(F, Aa):  # F (N,f), Aa (N,d)
    return np.linalg.lstsq(np.c_[np.ones(len(F)), F], Aa, rcond=None)[0][1:]
cnt = {"loo": loo, "full": full}
enc = {}
for cn, fn in cnt.items():
    Fq = oh(fn(Dn))                         # N,Q,21
    enc[("shared", cn)] = fit(np.repeat(Fq, 6, 1).reshape(-1, Fq.shape[-1]), an[:, :36].reshape(-1, d))
    enc[("slot", cn)] = [fit(Fq[:, t // 6], an[:, t]) for t in range(36)]
# own-group count only (per slot)
own = [fit(oh(loo(Dn)[:, t // 6, (t % 6) // 2][:, None]), an[:, t]) for t in range(36)]
# joint lookup per slot
def key(n): return n[..., 0] * 49 + n[..., 1] * 7 + n[..., 2]
Kn = key(loo(Dn)); look = []
for t in range(36):
    k = Kn[:, t // 6]; tab = {}
    for v in np.unique(k):
        m = k == v
        if m.sum() >= 5: tab[int(v)] = an[m, t].mean(0)
    look.append(tab)
D = G.sample(f4, NE, np.random.default_rng(9_900_000)); X = D["X"]; a0, P0 = map(np.asarray, acts(jnp.asarray(X)))
res = {}
for q in range(Q):
    Xp = A.pairs(env, X, q, "cond", np.random.default_rng(9_900_100 + q)); ap, pp = map(np.asarray, acts(jnp.asarray(Xp)))
    inf = P0[:, q].argmax(-1) != pp[:, q].argmax(-1)
    deltas = {}
    for cn, fn in cnt.items():
        F0, F1 = oh(fn(X)), oh(fn(Xp))
        dl = np.zeros((len(X), T, d), np.float32)
        for k in range(Q): dl[:, 6 * k:6 * k + 6] = ((F1[:, k] - F0[:, k]) @ enc[("shared", cn)])[:, None]
        deltas[f"shared|{cn}|all"] = dl
        dl = np.zeros((len(X), T, d), np.float32)
        for t in range(36): dl[:, t] = (F1[:, t // 6] - F0[:, t // 6]) @ enc[("slot", cn)][t]
        deltas[f"slot|{cn}|all"] = dl
        dq = np.zeros_like(dl); dq[:, 6 * q:6 * q + 6] = dl[:, 6 * q:6 * q + 6]; deltas[f"slot|{cn}|own_q_only"] = dq
    N0, N1 = loo(X), loo(Xp); dl = np.zeros((len(X), T, d), np.float32)
    for t in range(36):
        g = (t % 6) // 2; dl[:, t] = (oh(N1[:, t // 6, g][:, None]) - oh(N0[:, t // 6, g][:, None])) @ own[t]
    deltas["slot|own_group_loo|all"] = dl
    K0, K1 = key(N0), key(N1); dl = np.zeros((len(X), T, d), np.float32); miss = 0
    for t in range(36):
        k = t // 6
        for i in range(len(X)):
            a_, b_ = look[t].get(int(K0[i, k])), look[t].get(int(K1[i, k]))
            if a_ is None or b_ is None: miss += 1; continue
            dl[i, t] = b_ - a_
    deltas["slot|joint_lookup_loo|all"] = dl
    # oracle-ish ceiling: full counterfactual activation at source tokens
    dl = np.zeros((len(X), T, d), np.float32); dl[:, :36] = ap[:, :36] - a0[:, :36]; deltas["full_source_patch"] = dl
    for nm, dl in deltas.items():
        o = np.asarray(added(jnp.asarray(X), jnp.asarray(dl)))
        res.setdefault(nm, []).append((o[:, q].argmax(-1) == pp[:, q].argmax(-1))[inf])
    res.setdefault("_lookup_miss_frac", []).append(np.array([miss / (36 * len(X))]))
out = {k: (round(float(np.concatenate(v).mean()), 3), int(np.concatenate(v).size)) for k, v in res.items()}
print(ck, json.dumps(out, indent=0))
