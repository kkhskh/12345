"""PP2 structural signatures, computed ONLY on inputs derived from the T4 training generator (never on p_c = 0 episodes).
  rep(model, X)        : internal episode representation. TF: [mean of final-layer readout states (pos 36-41), episode
                         token (pos 42)]; DeepSets: the mean-pooled summary m.
  CCP_j                : colluder-code persistence. A multinomial logistic probe for the colluder group h is fitted on
                         natural T4 episodes; CCP_j = mean probe probability of the true h after the colluder's agreement is
                         broken in j questions (j = 1..3).  Global (configuration-constrained) computation predicts the
                         code persists (some group must be the colluder); a local factorised computation predicts decay.
  LPI_deg              : behavioural minimal probe: on T4 episodes with the colluder broken in 3 questions, the fraction of
                         global-vs-local plug-in disagreements in which the model sides with the local plug-in.
Streams: probe fit 6_400_000, degradation 6_410_000."""
import os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
for p_ in [os.path.join(AT, "src"), os.path.join(LR, "src")]: sys.path.insert(0, p_)
import numpy as np, jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C, local_bayes as LB, fwd as F
from sklearn.linear_model import LogisticRegression
f4 = G.Fam(pc=1.0, rho=1.0); plug4 = G.Fam(pc=1.0, rho=0.999); K, Q = 3, 6

def load(kind, path):
    if kind == "tf":
        p = jax.tree_util.tree_map(jnp.asarray, C.load(path))
        def run(X):
            st = {}
            def hk(n, a):
                if n == "final": st["f"] = a
                return a
            o = F.forward(p, X, hk); f = st["f"]
            return o["answer"], jnp.concatenate([f[:, 36:42].mean(1), f[:, 42]], -1)
    else:
        p = jax.tree_util.tree_map(jnp.asarray, C.load_base(path)); fb = N.make_forward("deepsets", 3)
        def run(X):
            st = {}
            def hk(n, a):
                if n == "pool": st["m"] = a
                return a
            o = fb(p, X, hk); return o["answer"], st["m"][:, 0]
    jr = jax.jit(run)
    def apply(X):
        outs = [jr(jnp.asarray(X[i:i + 2000])) for i in range(0, len(X), 2000)]
        return np.concatenate([np.asarray(o[0]) for o in outs]), np.concatenate([np.asarray(o[1]) for o in outs])
    return apply

def degrade(X, cfg, j, rng):
    X = X.copy(); cf = f4.configs; h = np.array([cf[c][1] for c in cfg]); n = len(X)
    for i in range(n):
        ks = rng.choice(np.arange(1, Q), j, replace=False)        # never question 0 (scored query for LPI)
        a, b = G.GROUPS[h[i]]
        for k in ks: X[i, k, b] = (X[i, k, a] + 1 + rng.integers(2)) % K
    return X, h

def signatures(apply):
    d = G.sample(f4, 4000, np.random.default_rng(6_400_000)); cf = f4.configs; h = np.array([cf[c][1] for c in d["cfg"]])
    _, r = apply(d["X"]); mu, sd = r.mean(0), r.std(0) + 1e-6
    pr = LogisticRegression(max_iter=2000, C=1.0).fit((r - mu) / sd, h)
    out = dict(probe_train_acc=float(pr.score((r - mu) / sd, h)))
    t = G.sample(f4, 2000, np.random.default_rng(6_410_000)); rng = np.random.default_rng(6_410_001)
    _, rt = apply(t["X"]); out["probe_test_acc_j0"] = float(pr.score((rt - mu) / sd, np.array([cf[c][1] for c in t["cfg"]])))
    for j in (1, 2, 3):
        Xd, hd = degrade(t["X"], t["cfg"], j, rng); ad, rd = apply(Xd)
        out[f"CCP_{j}"] = float(pr.predict_proba((rd - mu) / sd)[np.arange(len(hd)), hd].mean())
        if j == 3:
            gp = np.nan_to_num(G.posterior(plug4, Xd)["post_y"], nan=0).argmax(-1); lp = LB.local_post(plug4, Xd).argmax(-1)
            a = ad.argmax(-1); dis = gp != lp
            out["LPI_deg"] = float((a == lp)[dis].mean()); out["n_deg_disagree"] = int(dis.sum())
    return out

def pc0_local_index(apply, support_split=True):
    f1 = G.Fam(pc=0.0); d = G.sample(f1, 6000, np.random.default_rng(6_200_002)); X = d["X"]
    gp = np.nan_to_num(G.posterior(plug4, X)["post_y"], nan=0).argmax(-1); lp = LB.local_post(plug4, X).argmax(-1); tb = G.posterior(f1, X)["post_y"].argmax(-1)
    a = apply(X)[0].argmax(-1); dis = gp != lp
    inT4 = np.stack([(X[:, :, a_] == X[:, :, b_]).all(1) for a_, b_ in G.GROUPS], -1).any(-1)   # some group agrees in all 6 questions
    r = dict(local_index=float((a == lp)[dis].mean()), acc=float((a == d["Y"]).mean()), frac_episodes_in_T4_support=float(inT4.mean()))
    for nm, m in [("inT4", inT4), ("offT4", ~inT4)]:
        mm = dis & m[:, None]
        r[f"local_index_{nm}"] = float((a == lp)[mm].mean()) if mm.any() else None; r[f"n_dis_{nm}"] = int(mm.sum())
        r[f"agree_true_{nm}"] = float((a == tb)[m].mean()); r[f"agree_global_{nm}"] = float((a == gp)[m].mean())
    return r
