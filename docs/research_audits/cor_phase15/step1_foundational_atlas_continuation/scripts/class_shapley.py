"""Split- and gauge-robust attribution (mandate §6).  Game: v(S) = median KL-recovered when the last-layer head outputs in
S (at readout q) are patched from the cond counterfactual.  Head-level Shapley vs CLASS-level Shapley, where classes are
connected components of heads whose readout attention patterns have correlation > tau (tau = 0.99 exact; 0.9 coarse).
Then an exact Net2Net split of one head (duplicate WQ, WK, WV; halve WO) and a random within-head gauge transform
(WV -> WV G, WO -> G^{-1} WO), and recompute.  Streams 7_800_000.  Usage: class_shapley.py SEED SPLIT_HEAD OUT.json"""
import json, os, sys, itertools, math
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, fwd as F, ckpt as C, analysis as A, bayes as B
seed, sh, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
p0 = C.load(os.path.join(LR, "checkpoints/replication", f"T4_M_K3_s{seed}_fromT1.npz"))
env = B.Env("T4", K=3, Q=6); Q, T = 6, 43; f4 = G.Fam(); L = len(p0["layers"]) - 1
def split(p, h):
    p = jax.tree_util.tree_map(np.array, p); lay = p["layers"][L]
    for k in ("WQ", "WK", "WV"): lay[k] = np.concatenate([lay[k], lay[k][h:h + 1]], 0)
    lay["WO"] = np.concatenate([lay["WO"], lay["WO"][h:h + 1] * 0.5], 0); lay["WO"][h] *= 0.5
    return p
def gauge(p, rng):
    p = jax.tree_util.tree_map(np.array, p); lay = p["layers"][L]
    for h in range(lay["WV"].shape[0]):
        e = lay["WV"].shape[-1]; Gm = rng.normal(size=(e, e)) + 3 * np.eye(e); lay["WV"][h] = lay["WV"][h] @ Gm; lay["WO"][h] = np.linalg.solve(Gm, lay["WO"][h])
    return p
X = G.sample(f4, 600, np.random.default_rng(7_800_000))["X"]; cfs = [A.pairs(env, X, q, "cond", np.random.default_rng(7_800_100 + q)) for q in range(Q)]
def game(p):
    p = jax.tree_util.tree_map(jnp.asarray, p); H = p["layers"][L]["WQ"].shape[0]
    @jax.jit
    def acts(Xb):
        rec = {}
        def hk(n, a):
            if n in (f"head_out{L}", f"pattern{L}"): rec[n] = a
            return a
        o = F.forward(p, Xb, hk); return rec[f"head_out{L}"], rec[f"pattern{L}"], jax.nn.softmax(o["answer"], -1)
    @jax.jit
    def patched(Xb, hsrc, hm):
        def hk(n, a):
            return a + (hsrc - a) * hm[None, :, :, None] if n == f"head_out{L}" else a
        return jax.nn.softmax(F.forward(p, Xb, hk)["answer"], -1)
    base = [acts(jnp.asarray(X))]; cfa = [acts(jnp.asarray(c)) for c in cfs]
    pats = np.asarray(base[0][1])[:, :, 36:42].reshape(len(X), H, -1).transpose(1, 0, 2).reshape(H, -1)
    corr = np.corrcoef(pats)
    cache = {}
    def v(S):
        S = tuple(sorted(S))
        if S in cache: return cache[S]
        if not S: cache[S] = 0.0; return 0.0
        PE, PEp, PP = [], [], []
        for q in range(Q):
            hm = np.zeros((H, T), np.float32)
            for h in S: hm[h, 36 + q] = 1
            o = patched(jnp.asarray(X), cfa[q][0], jnp.asarray(hm))
            PE.append(np.asarray(base[0][2])[:, q]); PEp.append(np.asarray(cfa[q][2])[:, q]); PP.append(np.asarray(o)[:, q])
        cache[S] = float(A.interchange_metrics(*map(np.concatenate, (PE, PEp, PP)))["kl_recovered_mean"]); return cache[S]
    return H, corr, v
def shapley(players, v):
    n = len(players); phi = {}
    for i, pl in enumerate(players):
        others = [x for x in players if x != pl]; s = 0.0
        for r in range(n):
            for Sub in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                s += w * (v([h for g in Sub for h in g] + list(pl)) - v([h for g in Sub for h in g]))
        phi[str(pl)] = s
    return phi
def classes(corr, tau):
    H = len(corr); lab = list(range(H))
    for a in range(H):
        for b in range(H):
            if corr[a, b] > tau:
                la, lb = lab[a], lab[b]; lab = [la if x == lb else x for x in lab]
    out = {}
    for h, l in enumerate(lab): out.setdefault(l, []).append(h)
    return [tuple(v) for v in out.values()]
res = dict(seed=seed, split_head=sh)
for nm, p in [("original", p0), ("split", split(p0, sh)), ("gauge", gauge(p0, np.random.default_rng(1))), ("split+gauge", gauge(split(p0, sh), np.random.default_rng(2)))]:
    H, corr, v = game(p); r = dict(H=H, v_all=v(list(range(H))))
    r["head_shapley"] = shapley([(h,) for h in range(H)], v)
    for tau in (0.99, 0.9):
        cl = classes(corr, tau); r[f"classes_tau{tau}"] = [list(c) for c in cl]; r[f"class_shapley_tau{tau}"] = shapley(cl, v)
    res[nm] = r; print(nm, json.dumps({k: (val if not isinstance(val, dict) else {a: round(b, 3) for a, b in val.items()}) for k, val in r.items()}), flush=True)
json.dump(res, open(outp, "w"), indent=1)
