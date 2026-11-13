"""Workstream D: generalization of learned relational readers on the role-model generator (src/genlib.py), with exact brute-force Bayes.
Per-group ingredient features (primitive evidence, as in Investigation VIII Condition R): one-hot n_g(q) (Q values), one-hot q-pair report
(K*K), and optionally a group-id one-hot (Gmax).  Readers (all output 3 logits):
  ds_sum   phi (MLP) per group -> SUM -> rho (MLP)          (Theorem-1-shaped, identity-free, size-generic)
  ds_mean  phi per group -> MEAN -> rho                     (identity-free; mean pooling)
  attn     2 blocks of group self-attention, mean pool      (identity-free unless ID)
  relnet   pairwise phi summed over ordered pairs + psi     (identity-free unless ID)
  mlp      concatenation MLP (needs fixed G; identity implicit in slot)
Training: G_train groups, C colluders, beta; generator labels; optional held-out pair: reject training episodes in which groups 0 and 1 are
BOTH all-agreeing (so the eligible pair {0,1} never occurs in training).
Tests (exact Bayes per setting): KL (all decisions), KL on decisions with S >= 2 (two or more eligible groups: the relational regime),
for: in-distribution; held-out pair {0,1} (S>=2 with both 0,1 eligible); group permutation invariance; G in {3,4,5,6} (size-generic readers);
partial copying beta = 0.7 (relational semantics change: hard gate -> soft gate, Theorem 3).
Usage: gen_readers.py READER [ID=0|1] [HOLD=0|1] [STEPS=12000] [SEED=0] [C=1] [GTRAIN=3] -> checkpoints/gen/<tag>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import genlib as GL
np.seterr(all="ignore")
reader = sys.argv[1]; useid = int(sys.argv[2]) if len(sys.argv) > 2 else 0; hold = int(sys.argv[3]) if len(sys.argv) > 3 else 0
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 12000; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
C = int(sys.argv[6]) if len(sys.argv) > 6 else 1; GT = int(sys.argv[7]) if len(sys.argv) > 7 else 3
K, Q, GMAX = 3, 6, 6
tag = f"{reader}_id{useid}_hold{hold}_C{C}_G{GT}_s{STEPS}_seed{seed}"; OUT = os.path.join(HERE, "..", "checkpoints", "gen" if os.environ.get("GEN_EVAL_OFFSET", "0") == "0" else "gen_conf"); os.makedirs(OUT, exist_ok=True)
F = Q + K * K + (GMAX if useid else 0)
def feats(X, q, G):
    n = len(X); idx = np.arange(n); cnt = GL.counts(X, G)[idx, q]; a_ = X.reshape(n, Q, G, 2)[idx, q]
    f = [np.eye(Q)[cnt], np.eye(K * K)[a_[..., 0] * K + a_[..., 1]]]
    if useid: f.append(np.broadcast_to(np.eye(GMAX)[:G][None], (n, G, GMAX)))
    return np.concatenate(f, -1).astype(np.float32)                                   # n,G,F
def dense(k, a, b): return dict(W=jax.random.normal(k, (a, b)) * a ** -0.5, b=jnp.zeros(b))
def mlp_init(k, dims): ks = jax.random.split(k, len(dims)); return [dense(kk, a, b) for kk, a, b in zip(ks, dims[:-1], dims[1:])]
def mlp(ps, x):
    for i, l in enumerate(ps):
        x = x @ l["W"] + l["b"]
        if i < len(ps) - 1: x = jax.nn.gelu(x)
    return x
def ln(x): mu = x.mean(-1, keepdims=True); return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
H = 128; k = jax.random.split(jax.random.PRNGKey(seed), 8)
if reader in ("ds_sum", "ds_mean"): p = dict(phi=mlp_init(k[0], [F, H, H]), rho=mlp_init(k[1], [H, H, H, 3]))
elif reader == "relnet": p = dict(phi=mlp_init(k[0], [2 * F, H, H]), psi=mlp_init(k[1], [F, H, H]), rho=mlp_init(k[2], [2 * H, H, 3]))
elif reader == "attn":
    fk = lambda a, b: jax.random.fold_in(k[1], 10 * a + b)
    p = dict(emb=dense(k[0], F, 64), out=dense(k[6], 64, 3), blocks=[dict(Wq=dense(fk(i, 0), 64, 64)["W"], Wk=dense(fk(i, 1), 64, 64)["W"], Wv=dense(fk(i, 2), 64, 64)["W"],
             Wo=dense(fk(i, 3), 64, 64)["W"], ff=mlp_init(fk(i, 4), [64, H, 64])) for i in range(2)])
elif reader == "mlp": p = dict(m=mlp_init(k[0], [GT * F, 256, 256, 3]))
def apply(p, f):
    if reader == "ds_sum": return mlp(p["rho"], mlp(p["phi"], f).sum(1))
    if reader == "ds_mean": return mlp(p["rho"], mlp(p["phi"], f).mean(1))
    if reader == "relnet":
        G = f.shape[1]; P = 0
        for a in range(G):
            for b in range(G):
                if a != b: P = P + mlp(p["phi"], jnp.concatenate([f[:, a], f[:, b]], -1))
        return mlp(p["rho"], jnp.concatenate([P, mlp(p["psi"], f).sum(1)], -1))
    if reader == "attn":
        x = f @ p["emb"]["W"] + p["emb"]["b"]
        for b in p["blocks"]:
            z = ln(x); a = jax.nn.softmax(jnp.einsum("ngd,nhd->ngh", z @ b["Wq"], z @ b["Wk"]) / 8.0, -1)
            x = x + jnp.einsum("ngh,nhd->ngd", a, z @ b["Wv"]) @ b["Wo"]; x = x + mlp(b["ff"], ln(x))
        return x.mean(1) @ p["out"]["W"] + p["out"]["b"]
    if reader == "mlp": return mlp(p["m"], f.reshape(len(f), -1))
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(300, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, f, y):
    def loss(p): lp = jax.nn.log_softmax(apply(p, f), -1); return (-jnp.take_along_axis(lp, y[:, None], -1)[:, 0]).mean()
    l, g = jax.value_and_grad(loss)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
fwd = jax.jit(apply)
env_tr = GL.Env(G=GT, C=C); rng = np.random.default_rng(40_000_000 + 1000 * seed + 17 * C + GT); t0 = time.time()
def train_batch(n=512):
    keep_X, keep_Y = [], []
    while sum(len(x) for x in keep_X) < n:
        d = GL.sample(env_tr, 2 * n, rng); X = d["X"]
        if hold:
            a_ = X.reshape(len(X), Q, GT, 2); allag = (a_[..., 0] == a_[..., 1]).all(1); ok = ~(allag[:, 0] & allag[:, 1]); X, Y = X[ok], d["Y"][ok]
        else: Y = d["Y"]
        keep_X.append(X); keep_Y.append(Y)
    X = np.concatenate(keep_X)[:n]; Y = np.concatenate(keep_Y)[:n]; q = rng.integers(0, Q, n); return feats(X, q, GT), Y[np.arange(n), q]
hist = []
for i in range(STEPS):
    f, y = train_batch(); p, st, l = step(p, st, jnp.asarray(f), jnp.asarray(y))
    if (i + 1) % max(STEPS // 20, 1) == 0: hist.append((i + 1, float(l)))
EOFF = int(os.environ.get("GEN_EVAL_OFFSET", 0))     # confirmation: 5_000_000 (evaluation streams never used in development)
def evaluate(env, n=6000, sd=0, perm=False, sel=None):
    sd = sd + EOFF
    d = GL.sample(env, n, np.random.default_rng(41_000_000 + sd)); X = d["X"]; q = np.random.default_rng(41_500_000 + sd).integers(0, Q, n)
    if perm:                                                     # swap group slots 0 <-> 1 (a symmetry of the generator)
        Xp = X.reshape(n, Q, env.G, 2).copy(); Xp[:, :, [0, 1]] = Xp[:, :, [1, 0]]; X = Xp.reshape(n, Q, 2 * env.G)
    pb = GL.posterior_brute(env, X, q); lg = np.asarray(fwd(p, jnp.asarray(feats(X, q, env.G)))); pm = np.exp(lg - lg.max(-1, keepdims=True)); pm /= pm.sum(-1, keepdims=True)
    kl = (pb * (np.log(pb + 1e-12) - np.log(pm + 1e-12))).sum(-1); _, e = GL.eligibility(X, env.G, q); S = e.sum(1)
    out = dict(KL=float(kl.mean()), KL_S2=float(kl[S >= 2].mean()) if (S >= 2).any() else None, n_S2=int((S >= 2).sum()), acc=float((pm.argmax(-1) == d["Y"][np.arange(n), q]).mean()))
    if env.G >= 2:
        both = e[:, 0] & e[:, 1]; out["KL_pair01"] = float(kl[both].mean()) if both.any() else None; out["n_pair01"] = int(both.sum())
        oth = (S >= 2) & ~both; out["KL_S2_not01"] = float(kl[oth].mean()) if oth.any() else None
    return out, lg
res = {}
res["indist"], lg0 = evaluate(GL.Env(G=GT, C=C), sd=1)
res["perm01"], lgp = evaluate(GL.Env(G=GT, C=C), sd=1, perm=True)
res["perm01"]["mean_abs_logit_change"] = float(np.abs((lgp - lgp.mean(-1, keepdims=True)) - (lg0 - lg0.mean(-1, keepdims=True))).mean())
if reader != "mlp":
    for G in (3, 4, 5, 6):
        if G != GT and G >= C + 1: res[f"G{G}"], _ = evaluate(GL.Env(G=G, C=C), sd=10 + G)
res["beta0.7"], _ = evaluate(GL.Env(G=GT, C=C, beta=0.7), sd=3)
res["indist_bigN"], _ = evaluate(GL.Env(G=GT, C=C), n=30000, sd=5)        # more S>=2 / pair01 cases
json.dump(dict(reader=reader, id=useid, hold=hold, C=C, Gtrain=GT, steps=STEPS, seed=seed, seconds=time.time() - t0, hist=hist, results=res), open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
print(tag, json.dumps(res), flush=True)
