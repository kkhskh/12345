"""ADAPT latent-state representation (F3): at readout token q (final-layer state), decode the group that is reliable AT q
(changing latent state) and, for comparison, the group reliable at question 0 (episode-initial state) and whether a change
occurred.  Logistic probes, held-out; compared with the exact-Bayes MAP of the same quantity (statistical ceiling) and with
the static-model MAP (assumes no change).  Usage: adapt_state_probe.py CKPT NAME"""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, tasks3 as T3, fwd3 as F3, ckpt as C, gen_family as G
from sklearn.linear_model import LogisticRegression
ck, name = sys.argv[1], sys.argv[2]; p = jax.tree_util.tree_map(jnp.asarray, C.load(ck))
d = T3.sample_adapt(6000, np.random.default_rng(9_720_000)); X = d["X"]; cfg = np.array([T3.ADAPT_CFGS[c] for c in d["cfg"]])
@jax.jit
def rep(Xb):
    rec = {}
    def hk(n, a):
        if n == "final": rec["f"] = a
        return a
    F3.forward(p, Xb, hk); return rec["f"][:, 36:42]
R = np.concatenate([np.asarray(rep(jnp.asarray(X[i:i + 2000]))) for i in range(0, 6000, 2000)])     # N,6,d
active = np.stack([np.where(q < cfg[:, 2], cfg[:, 0], cfg[:, 1]) for q in range(6)], 1)
pc = T3.bayes_adapt(X)["post_cfg"]; act_cfg = np.array([[g1 if q < c else g2 for q in range(6)] for (g1, g2, c) in T3.ADAPT_CFGS])   # C,6
map_active = np.stack([np.stack([pc[:, act_cfg[:, q] == g].sum(1) for g in range(3)], -1).argmax(-1) for q in range(6)], 1)
st = G.posterior(G.Fam(pc=0.0), X); static_map = np.repeat(np.log(np.clip(st["post_y"], 1e-12, 1)).shape[0:1], 1)  # placeholder
cnt = np.stack([(X[:, :, a] == X[:, :, b]).sum(1) for a, b in G.GROUPS], -1); static_active = np.repeat(cnt.argmax(-1)[:, None], 6, 1)
tr = np.arange(6000) < 4000; Z = R.reshape(-1, R.shape[-1]); m = np.repeat(tr, 6); mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-6; Zs = (Z - mu) / sd
out = dict(ckpt=ck)
t = active.reshape(-1); out["probe_active_group_at_q"] = float(LogisticRegression(max_iter=3000).fit(Zs[m], t[m]).score(Zs[~m], t[~m]))
t0 = np.repeat(cfg[:, 0], 6); out["probe_initial_group"] = float(LogisticRegression(max_iter=3000).fit(Zs[m], t0[m]).score(Zs[~m], t0[~m]))
out["exact_MAP_active_at_q"] = float((map_active == active)[~tr].mean()); out["static_MAP_active_at_q"] = float((static_active == active)[~tr].mean())
# where the active group differs from the initial group (after a change): probe vs ceilings
after = (active != cfg[:, :1]).reshape(-1)
pr = LogisticRegression(max_iter=3000).fit(Zs[m], t[m]).predict(Zs)
out["after_change"] = dict(n=int((after & ~m).sum()), probe=float((pr == t)[after & ~m].mean()), exact_MAP=float((map_active.reshape(-1) == t)[after & ~m].mean()),
                           static_MAP=float((static_active.reshape(-1) == t)[after & ~m].mean()))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "..", "logs", "adapt", f"state_probe_{name}.json"), "w"), indent=1)
