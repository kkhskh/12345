"""Compositional generalization (Condition R readers).  Usage: gen_holdout.py READER STEPS SEED
Training distribution: the T4 generator with every episode in which groups 0 AND 1 both agree on all six questions REJECTED (rejection
sampling of the natural generator; support-valid, distribution-shifted).  Such episodes are the only source of decisions where the two
perfect groups are {0,1}.  Training labels: generator labels.  Test: development F45 pairs split by the pair {edited group g, existing
perfect group h*}: held-out {0,1} vs seen {0,2},{1,2}; plus the nested S45 and F54o families split the same way.
READER attn|relnet|mlp|lookup.  'lookup' = tabular estimator over the exact Condition-R feature key (counts and q-reports of all groups),
Laplace-smoothed empirical label frequencies from the same number of training decisions; unseen keys -> uniform.
Invariance tests on all development natural decisions (for the trained model): group permutation (relabel group slots: sources of group 0
<-> group 1, which leaves the generator distribution invariant) and candidate relabelling (y -> (y+1) mod 3 applied to every report):
mean |logit change| after undoing the relabelling.  Writes checkpoints/gen/<tag>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import v8lib as V, readers as RD
sys.path.insert(0, V.V7); import nnlib as NL, ckpt as C
L, E, B = V.L, V.E, V.B
reader, STEPS, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); tag = f"{reader}_holdout01_s{STEPS}_seed{seed}"
OUT = os.path.join(HERE, "..", "checkpoints", "gen"); os.makedirs(OUT, exist_ok=True)
def sample(rng, n=256):
    keep = []
    while sum(len(k["X"]) for k in keep) < n:
        d = NL.Bm.sample_fast(NL.ENV, n, rng); A = E.agree(d["X"]).all(1)          # N,3: group agrees on all questions
        ok = ~(A[:, 0] & A[:, 1]); keep.append(dict(X=d["X"][ok], Y=d["Y"][ok]))
    return dict(X=np.concatenate([k["X"] for k in keep])[:n], Y=np.concatenate([k["Y"] for k in keep])[:n])
ev = dict(np.load(os.path.join(V.V7, "..", "logs", "eval_dev.npz"), allow_pickle=True))
rng = np.random.default_rng(21_000_000 + 1000 * seed); t0 = time.time()
if reader == "lookup":
    table = {}
    for i in range(STEPS):
        d = sample(rng); f = RD.features(d["X"]).reshape(len(d["X"]), 6, -1)
        for e_ in range(len(d["X"])):
            for q in range(6):
                k = f[e_, q].tobytes(); table.setdefault(k, np.ones(3))[d["Y"][e_, q]] += 1
    def fn(X):
        f = RD.features(X).reshape(len(X), 6, -1); out = np.zeros((len(X), 6, 3))
        for e_ in range(len(X)):
            for q in range(6):
                c = table.get(f[e_, q].tobytes(), np.ones(3)); out[e_, q] = np.log(c / c.sum())
        return out
    npar = len(table) * 3
else:
    kind = reader; p = RD.init(kind, 15, seed, H={"relnet": 64}.get(kind, 128), d=40); npar = RD.count_params(p)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(p)
    @jax.jit
    def step(p, st, f, Y):
        def loss(p): lp = jax.nn.log_softmax(RD.apply(kind, p, f), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
        l, g = jax.value_and_grad(loss)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    for i in range(STEPS):
        d = sample(rng); p, st, l = step(p, st, jnp.asarray(RD.features(d["X"])), jnp.asarray(d["Y"]))
    fwd = jax.jit(lambda p, f: RD.apply(kind, p, f)); fn = lambda X: np.asarray(fwd(p, jnp.asarray(RD.features(X))))
def cl(c): return c - c.mean(-1, keepdims=True)
res = dict(reader=reader, steps=STEPS, seed=seed, params=npar)
# split F45 by pair {g, h*}
idx = np.arange(len(ev["X"])); q = ev["q"]; Xb, Xa = ev["X"], ev["X_F45"]
nb = E.loo(Xb)[idx, q]; na = E.loo(Xa)[idx, q]; g = np.where((na != nb).any(1), (na != nb).argmax(1), -1)
Vb = L.votes(Xb, q); hs = np.full(len(idx), -1)
for i in idx:
    c = [h for h in range(3) if h != g[i] and nb[i, h] == 5 and Vb[i, h] >= 0]; hs[i] = c[0] if c else -1
pair01 = ((g == 0) & (hs == 1)) | ((g == 1) & (hs == 0))
lb, la = cl(fn(Xb)[idx, q]), cl(fn(Xa)[idx, q]); v = ev["vote_F45"]
dm = B.oriented_margin(la, v) - B.oriented_margin(lb, v); db = B.oriented_margin(ev["cB_F45"], v) - B.oriented_margin(ev["cB_base"], v)
for rg in ("R1", "R2", "R3"):
    for nm, sel in (("heldout01", pair01), ("seen", ~pair01)):
        s = ev["orient_F45"] & (ev["reg_F45"] == rg) & sel
        if s.sum() >= 5: res[f"{rg}_{nm}"] = dict(n=int(s.sum()), E=float((dm - db)[s].mean()), absE=float(np.abs(dm - db)[s].mean()), dmM=float(dm[s].mean()), dmB=float(db[s].mean()),
                                                   frac_sign_correct=float((np.sign(dm[s]) == np.sign(db[s]))[np.abs(db[s]) > 1e-9].mean()))
# invariances on natural decisions
X = ev["X"]; L0 = cl(fn(X)); perm = np.arange(6); a0, b0 = L.GROUPS[0]; a1, b1 = L.GROUPS[1]; perm[[a0, b0, a1, b1]] = [a1, b1, a0, b0]
Lg = cl(fn(X[:, :, perm])); Lc = cl(fn((X + 1) % 3)); Lc_back = np.roll(Lc, -1, axis=-1)
res["invariance"] = dict(group_perm_mean_abs=float(np.abs(Lg - L0).mean()), candidate_relabel_mean_abs=float(np.abs(Lc_back - L0).mean()), logit_scale=float(np.abs(L0).mean()))
res["seconds"] = time.time() - t0
json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
print(tag, {k: (round(v["E"], 3), round(v["absE"], 3), v["n"]) if isinstance(v, dict) and "E" in v else v for k, v in res.items() if k not in ("reader",)})
